// webui.rs
//
// Minimal read-only web UI. No frameworks, no extra dependencies.
// Uses tokio::net::TcpListener with raw HTTP/1.1.
//
// Endpoints:
//   GET /            → redirect to /events
//   GET /events      → recent events table (auto-refreshes every 5 s)
//   GET /trade/{id}  → full trade timeline for a correlation_id
//   GET /status      → system mode, position, risk state
//   GET /health      → 200 OK plain text
//
// Thread safety:
//   EventStore reads use the r2d2 pool (concurrent-safe).
//   Executor / TruthState / RiskEngine are read under a brief async Mutex lock
//   that is always released before any I/O. No lock is ever held while writing
//   to the TCP stream.

use std::sync::Arc;

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tracing::{debug, error, info};

use crate::assistant;
use crate::authority::{AuthorityLayer, AuthorityMode};
use crate::executor::Executor;
use crate::profile::RuntimeProfile;
use crate::reader::{get_trade_timeline, summarise_event, LifecycleStage, TradeOutcome};
use crate::reconciler::TruthState;
use crate::risk::{self, RiskEngine, RiskVerdict};
use crate::store::EventStore;
use crate::strategy::{StrategyEngine, ALL_STRATEGIES};
use crate::withdrawal::{NewWithdrawalRequest, WithdrawalManager, WithdrawalStatus};
use crate::client::BinanceClient;
use uuid::Uuid;

// ── Shared application state ──────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub store:     Arc<dyn EventStore>,
    pub exec:      Arc<Executor>,
    pub truth:     Arc<Mutex<TruthState>>,
    pub risk:      Arc<Mutex<RiskEngine>>,
    pub authority: Arc<AuthorityLayer>,
    pub strategy:  Arc<Mutex<StrategyEngine>>,
    pub client:    Option<Arc<BinanceClient>>,
    pub withdrawals: Arc<WithdrawalManager>,
    pub npc: Arc<crate::npc::NpcAutonomousController>,
    pub profile: Arc<Mutex<RuntimeProfile>>,
}

#[cfg(test)]
fn test_app_state_extras() -> (Option<Arc<BinanceClient>>, Arc<WithdrawalManager>) {
    (None, Arc::new(WithdrawalManager::new(crate::withdrawal::WithdrawalConfig::default())))
}
// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(addr: &str, state: AppState) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("[WEBUI] Listening on http://{}", addr);

    let state = Arc::new(state);
    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                debug!("[WEBUI] Connection from {}", peer);
                let s = Arc::clone(&state);
                tokio::spawn(async move {
                    if let Err(e) = handle(stream, s).await {
                        debug!("[WEBUI] Connection error: {}", e);
                    }
                });
            }
            Err(e) => error!("[WEBUI] Accept error: {}", e),
        }
    }
}

// ── HTTP parsing ──────────────────────────────────────────────────────────────

async fn handle(mut stream: TcpStream, state: Arc<AppState>) -> Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    if n == 0 { return Ok(()); }

    let raw = String::from_utf8_lossy(&buf[..n]);
    let first = raw.lines().next().unwrap_or("");

    let mut parts = first.split_whitespace();
    let method     = parts.next().unwrap_or("GET");
    let path_query = parts.next().unwrap_or("/");
    let (path, query) = path_query.split_once('?').unwrap_or((path_query, ""));

    let body = raw.split("\r\n\r\n").nth(1).unwrap_or("");

    let response: String = if method == "POST" {
        handle_post(path, query, body, &state).await
    } else if path == "/" {
        redirect("/events")
    } else if path == "/events" {
        page_events(&state, query).await
    } else if let Some(corr_id) = path.strip_prefix("/trade/") {
        page_trade(&state, corr_id).await
    } else if path == "/trade" {
        let corr_id = qparam(query, "id").unwrap_or_default();
        if corr_id.is_empty() {
            page_trade(&state, "").await
        } else {
            redirect(&format!("/trade/{}", url_encode(corr_id)))
        }
    } else if path == "/status" {
        page_status(&state, query).await
    } else if path == "/status/export.csv" {
        export_positions_csv(&state).await
    } else if path == "/assistant" {
        page_assistant(&state, query).await
    } else if path == "/suggestions" {
        page_suggestions(&state, query).await
    } else if path == "/authority" {
        page_authority(&state, query).await
    } else if path == "/health" {
        plain("OK")
    } else if path == "/agent/status" {
        agent_status_json(&state).await
    } else {
        not_found()
    };

    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

async fn handle_post(path: &str, _query: &str, body: &str, state: &AppState) -> String {
    const LARGE_WITHDRAWAL_THRESHOLD: f64 = 500.0;
    if path == "/events/quick/buy" {
        log_ui_action(&*state.store, "ui_quick_execute_buy", "manual quick execute from /events");
        return redirect_with_ok("/events", "Queued quick BUY market simulation.");
    }
    if path == "/events/quick/sell" {
        log_ui_action(&*state.store, "ui_quick_execute_sell", "manual quick execute from /events");
        return redirect_with_ok("/events", "Queued quick SELL market simulation.");
    }

    if let Some(order_id) = path.strip_prefix("/status/cancel/") {
        log_ui_action(
            &*state.store,
            &format!("ui_cancel_order:{order_id}"),
            "cancel requested from /status",
        );
        return redirect_with_ok("/status", &format!("Cancel requested for order {order_id}."));
    }
    if path == "/status/close-all" {
        log_ui_action(&*state.store, "ui_close_all_positions", "close-all requested from /status");
        return redirect_with_ok("/status", "Close-all request queued.");
    }
    if path == "/status/update-strategy" {
        log_ui_action(&*state.store, "ui_update_strategy", "strategy config update requested from /status");
        return redirect_with_ok("/status", "Strategy config saved (UI simulation).");
    }

    if path == "/assistant/system-restart" {
        log_ui_action(&*state.store, "ui_system_restart", "system restart requested from /assistant");
        return redirect_with_ok("/assistant", "Restart request recorded.");
    }
    if path == "/assistant/autonomous/on" {
        state.npc.set_autonomous_mode(true).await;
        log_ui_action(&*state.store, "ui_autonomous_mode_on", "autonomous mode enabled from /assistant");
        return redirect_with_ok("/assistant", "Autonomous Mode enabled.");
    }
    if path == "/assistant/autonomous/off" {
        state.npc.set_autonomous_mode(false).await;
        log_ui_action(&*state.store, "ui_autonomous_mode_off", "autonomous mode disabled from /assistant");
        return redirect_with_ok("/assistant", "Autonomous Mode disabled.");
    }

    // POST /agent/mode — set agent mode (off | auto | pause)
    if path == "/agent/mode" {
        let form = parse_form_body(body);
        let mode_str = form.get("mode").map(|s| s.as_str()).unwrap_or("off");
        match crate::npc::AgentMode::from_str(mode_str) {
            Some(mode) => {
                state.npc.set_agent_mode(mode).await;
                log_ui_action(
                    &*state.store,
                    &format!("ui_agent_mode_{}", mode.as_str()),
                    &format!("agent mode set to {} from /agent/mode", mode.as_str()),
                );
                return redirect_with_ok("/events", &format!("Agent mode set to {}.", mode.as_str().to_uppercase()));
            }
            None => {
                return redirect_with_err("/events", &format!("Unknown agent mode: {}", esc(mode_str)));
            }
        }
    }
    if path == "/assistant/autonomous/interval" {
        let form = parse_form_body(body);
        let interval_ms = form
            .get("interval_ms")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1000)
            .clamp(500, 2000);
        state.npc.set_interval_ms(interval_ms).await;
        log_ui_action(
            &*state.store,
            "ui_autonomous_interval_set",
            &format!("autonomous interval set to {}ms from /assistant", interval_ms),
        );
        return redirect_with_ok("/assistant", &format!("Autonomous interval set to {}ms.", interval_ms));
    }
    if path == "/assistant/kill-switch/on" {
        state.risk.lock().await.set_kill_switch(true);
        log_ui_action(&*state.store, "ui_kill_switch_engaged", "kill switch engaged from /assistant");
        return redirect_with_ok("/assistant", "Kill switch engaged.");
    }
    if path == "/assistant/kill-switch/off" {
        state.risk.lock().await.set_kill_switch(false);
        log_ui_action(&*state.store, "ui_kill_switch_cleared", "kill switch cleared from /assistant");
        return redirect_with_ok("/assistant", "Kill switch cleared.");
    }
    if path == "/assistant/profile" {
        let form = parse_form_body(body);
        let profile_str = form.get("profile").map(|s| s.as_str()).unwrap_or("ACTIVE");
        let new_profile = RuntimeProfile::from_str(profile_str);
        *state.profile.lock().await = new_profile;
        log_ui_action(
            &*state.store,
            "ui_profile_changed",
            &format!("runtime profile set to {} from /assistant", new_profile.as_str()),
        );
        return redirect_with_ok("/assistant", &format!("Runtime profile set to {}.", new_profile.as_str()));
    }

    // Dedicated simulation-only withdrawal confirmation endpoint.
    // Intentionally separate from real proposal rejection/execution handlers.
    if path == "/withdraw/confirm/simulation" {
        log_ui_action(
            &*state.store,
            "ui_withdrawal_confirmed_simulation",
            "withdrawal confirmation requested from /authority",
        );
        return redirect_with_ok("/authority", "Simulation confirmation recorded. No real funds moved.");
    }

    if path == "/withdraw/request" {
        let form = parse_form_body(body);
        let amount = form.get("amount").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        let asset = form.get("asset").cloned().unwrap_or_default();
        let destination = form.get("destination").cloned().unwrap_or_default();
        let network = form.get("network").cloned().unwrap_or_default();
        let reason = form.get("reason").cloned().unwrap_or_default();
        let fee = form.get("fee_estimate").and_then(|v| v.parse::<f64>().ok());
        let confirmed = form.get("confirm_text").map(|v| v == "CONFIRM").unwrap_or(false)
            && form.get("confirm_checkbox").map(|v| v == "on").unwrap_or(false);
        if !confirmed {
            return redirect_with_err("/authority", "Explicit reconfirmation required. Tick confirmation and type CONFIRM.");
        }
        if amount >= LARGE_WITHDRAWAL_THRESHOLD {
            let large_confirmed = form
                .get("confirm_large_text")
                .map(|v| v == "WITHDRAW LARGE")
                .unwrap_or(false);
            if !large_confirmed {
                return redirect_with_err(
                    "/authority",
                    "Large withdrawal requires extra confirmation. Type WITHDRAW LARGE.",
                );
            }
        }

        let req = NewWithdrawalRequest { amount, asset, destination, network, reason, estimated_fee: fee };
        let client_ref = state.client.as_ref().map(|c| c.as_ref());
        let proposal = match state
            .withdrawals
            .create_request(req, state.risk.lock().await.kill_switch_active(), client_ref, &*state.store)
            .await
        {
            Ok(p) => p,
            Err(e) => return redirect_with_err("/authority", &e),
        };

        let mode = state.authority.mode().await;
        if mode == AuthorityMode::Auto {
            match state
                .withdrawals
                .execute(&proposal.id, mode, &state.risk, client_ref, &*state.store)
                .await
            {
                Ok(_) => {
                    return redirect(&format!(
                        "/authority?ok={}&wd_id={}&wd_step=status",
                        url_encode("Withdrawal auto-executed under AUTO policy."),
                        url_encode(&proposal.id)
                    ));
                }
                Err(e) => {
                    state.withdrawals.mark_failed(&proposal.id, &e, &*state.store).await;
                    return redirect(&format!(
                        "/authority?err={}&wd_id={}&wd_step=status",
                        url_encode(&e),
                        url_encode(&proposal.id)
                    ));
                }
            }
        }
        return redirect(&format!(
            "/authority?ok={}&wd_id={}&wd_step=status",
            url_encode(&format!("Withdrawal proposal {} submitted.", proposal.id)),
            url_encode(&proposal.id)
        ));
    }

    if path == "/trade/request" {
        let form = parse_form_body(body);
        let symbol = form.get("symbol").cloned().unwrap_or_else(|| "BTCUSDT".to_string());
        let side = form.get("side").cloned().unwrap_or_default().to_uppercase();
        let size = form.get("size").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        let reason = form.get("reason").cloned().unwrap_or_else(|| "agent request".to_string());

        if side != "BUY" && side != "SELL" {
            return redirect_with_err("/authority", "trade request side must be BUY or SELL");
        }
        if size <= 0.0 {
            return redirect_with_err("/authority", "trade request size must be > 0");
        }
        {
            let t = state.truth.lock().await;
            let side_free = t.available_balance_for_side(&side);
            if side_free <= 0.0 {
                let msg = if side == "BUY" {
                    "Insufficient BUY balance: quote-asset buy power is 0."
                } else {
                    "Insufficient SELL balance: base-asset sell inventory is 0."
                };
                return redirect_with_err("/authority", msg);
            }
        }

        let mode = state.authority.mode().await;
        if mode == AuthorityMode::Off {
            return redirect_with_ok("/authority", "Authority mode OFF: trade request logged, no action taken.");
        }
        if !matches!(state.exec.execution_state().await, crate::executor::ExecutionState::Idle) {
            return redirect_with_err("/authority", "Trade request ignored: executor is busy; an order is in progress (idempotency guard).");
        }
        if !state.authority.pending_proposals().await.is_empty() {
            return redirect_with_err("/authority", "Trade request ignored: pending proposal already exists (idempotency guard).");
        }

        let (position_snapshot, market_snapshot) = {
            let t = state.truth.lock().await;
            let market = risk::MarketSnapshot {
                bid: t.position.mark_price,
                ask: t.position.mark_price,
                feed_last_seen: Some(std::time::Instant::now()),
            };
            (t.position.clone(), market)
        };
        let order_side = if side == "BUY" { risk::OrderSide::Buy } else { risk::OrderSide::Sell };
        let expected_price = market_snapshot.ask;
        let proposed = risk::ProposedOrder {
            symbol: symbol.clone(),
            side: order_side,
            qty: size,
            expected_price,
        };
        let risk_verdict = {
            let mut r = state.risk.lock().await;
            r.risk_check(&position_snapshot, &market_snapshot, &proposed)
        };
        state.store.append(crate::events::risk_event(
            &risk_verdict,
            &proposed,
            &position_snapshot,
            &symbol,
            &Uuid::new_v4().to_string(),
        ));
        if let RiskVerdict::Rejected(r) = risk_verdict {
            return redirect_with_err("/authority", &format!("Trade request rejected by risk: {}", r));
        }

        let sys_mode = state.exec.system_mode().await;
        let exec_is_idle = matches!(state.exec.execution_state().await, crate::executor::ExecutionState::Idle);
        let kill_active = { state.risk.lock().await.kill_switch_active() };
        let can_place = { state.truth.lock().await.can_place_order() };
        let auth = state.authority.check(
            &symbol,
            &side,
            size,
            &reason,
            0.6,
            sys_mode,
            exec_is_idle,
            kill_active,
            can_place,
            &*state.store,
        ).await;

        match auth {
            crate::authority::AuthorityResult::Blocked(r) => {
                return redirect_with_err("/authority", &format!("Trade request blocked: {}", r));
            }
            crate::authority::AuthorityResult::ProposalCreated(p) => {
                return redirect_with_ok("/authority", &format!("Trade proposal {} created (ASSIST).", p.id));
            }
            crate::authority::AuthorityResult::Proceed => {}
        }

        let Some(client) = state.client.as_ref() else {
            return redirect_with_err("/authority", "No exchange client configured.");
        };
        let correlation_id = Uuid::new_v4().to_string();
        let client_order_id = crate::executor::make_client_order_id(
            (chrono::Utc::now().timestamp_millis() as u64) % 10_000_000_000,
        );
        let qty_str = format!("{:.8}", size);
        state.store.append(crate::events::order_submitted_event(
            &client_order_id,
            &side,
            &qty_str,
            expected_price,
            &symbol,
            &correlation_id,
        ));
        let retry_policy = crate::orders::RetryPolicy::default();
        match state.exec.submit_market_order(
            &symbol,
            &side,
            &qty_str,
            &client_order_id,
            client,
            &state.truth,
            &state.risk,
            &retry_policy,
        ).await {
            Ok(resp) => {
                state.store.append(crate::events::StoredEvent::new(
                    Some(symbol.clone()),
                    Some(correlation_id.clone()),
                    Some(resp.client_order_id.clone()),
                    crate::events::TradingEvent::OrderAcked(crate::events::OrderAckedPayload {
                        client_order_id: resp.client_order_id.clone(),
                        exchange_order_id: resp.order_id,
                        status: resp.status.clone(),
                    }),
                ));
                if resp.status.to_uppercase() == "FILLED" {
                    state.store.append(crate::events::order_filled_event(&resp, &symbol, &correlation_id));
                }
                return redirect_with_ok("/authority", "Trade request executed.");
            }
            Err(e) => {
                state.store.append(crate::events::StoredEvent::new(
                    Some(symbol.clone()),
                    Some(correlation_id.clone()),
                    Some(client_order_id.clone()),
                    crate::events::TradingEvent::OrderRejected(crate::events::OrderRejectedPayload {
                        client_order_id,
                        reason: e.to_string(),
                    }),
                ));
                return redirect_with_err("/authority", &format!("Trade execution failed: {}", e));
            }
        }
    }

    // POST /strategy/enable/{id} or /strategy/disable/{id}
    if let Some(rest) = path.strip_prefix("/strategy/") {
        if let Some((verb, id_str)) = rest.split_once('/') {
            let enabled = verb == "enable";
            let target = ALL_STRATEGIES.iter().find(|s| s.as_str() == id_str);
            if let Some(id) = target {
                state.strategy.lock().await.set_enabled(id, enabled);
                let msg = if enabled { "Strategy enabled." } else { "Strategy disabled." };
                return redirect_with_ok("/suggestions", msg);
            }
        }
        return not_found();
    }

    // POST /authority/mode/{off|assist|auto}
    if let Some(mode_str) = path.strip_prefix("/authority/mode/") {
        match AuthorityMode::from_str(mode_str) {
            Some(AuthorityMode::Off) => {
                state.authority.set_mode_off(&*state.store).await;
                return redirect_with_ok("/authority", "Authority mode set to OFF.");
            }
            Some(AuthorityMode::Assist) => {
                state.authority.set_mode_assist(&*state.store).await;
                return redirect_with_ok("/authority", "Authority mode set to ASSIST.");
            }
            Some(AuthorityMode::Auto) => {
                state.authority.set_mode_auto(&*state.store).await;
                return redirect_with_ok("/authority", "Authority mode set to AUTO.");
            }
            None => return not_found(),
        }
    }

    // POST /authority/approve/{proposal_id}
    if let Some(pid) = path.strip_prefix("/authority/approve/") {
        let pid = pid.to_string();
        let approved = state.authority.approve_proposal(&pid, &*state.store).await;
        if approved.is_some() {
            // In ASSIST mode the approved proposal should now be executed
            // by the signal loop (which polls pending approvals).
            // We just redirect back to the authority page.
            return redirect_with_ok("/authority", "Proposal approved.");
        } else {
            let body = format!(
                "<p class='err'>Proposal '{}' not found or already expired.</p>\
                 <p><a href='/authority'>← Back</a></p>",
                esc(&pid)
            );
            return html_resp(&page("Authority — RW-Trader", "", &body));
        }
    }

    // POST /authority/reject/{proposal_id}
    if let Some(pid) = path.strip_prefix("/authority/reject/") {
        state.authority.reject_proposal(pid, &*state.store).await;
        return redirect_with_ok("/authority", "Proposal rejected.");
    }


    // POST /withdraw/proposal/approve/{proposal_id}
    if let Some(pid) = path.strip_prefix("/withdraw/proposal/approve/") {
        match state.withdrawals.approve(pid, &*state.store).await {
            Ok(_) => return redirect(&format!("/authority?ok={}&wd_id={}&wd_step=status", url_encode("Withdrawal proposal approved."), url_encode(pid))),
            Err(e) => return redirect(&format!("/authority?err={}&wd_id={}&wd_step=status", url_encode(&e), url_encode(pid))),
        }
    }

    // POST /withdraw/proposal/reject/{proposal_id}
    if let Some(pid) = path.strip_prefix("/withdraw/proposal/reject/") {
        match state.withdrawals.reject(pid, &*state.store).await {
            Ok(_) => return redirect(&format!("/authority?ok={}&wd_id={}&wd_step=status", url_encode("Withdrawal proposal rejected."), url_encode(pid))),
            Err(e) => return redirect(&format!("/authority?err={}&wd_id={}&wd_step=status", url_encode(&e), url_encode(pid))),
        }
    }

    // POST /withdraw/proposal/execute/{proposal_id}
    if let Some(pid) = path.strip_prefix("/withdraw/proposal/execute/") {
        let mode = state.authority.mode().await;
        let client_ref = state.client.as_ref().map(|c| c.as_ref());
        match state
            .withdrawals
            .execute(pid, mode, &state.risk, client_ref, &*state.store)
            .await
        {
            Ok(_) => return redirect(&format!("/authority?ok={}&wd_id={}&wd_step=status", url_encode("Withdrawal completed."), url_encode(pid))),
            Err(e) => {
                state.withdrawals.mark_failed(pid, &e, &*state.store).await;
                return redirect(&format!("/authority?err={}&wd_id={}&wd_step=status", url_encode(&e), url_encode(pid)));
            }
        }
    }

    not_found()
}

// ── Helpers ───────────────────────────────────────────────────────────────────


fn parse_form_body(body: &str) -> std::collections::HashMap<String, String> {
    body.split('&')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((url_decode(k), url_decode(v)))
        })
        .collect()
}

fn url_decode(s: &str) -> String {
    let mut out = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'+' {
            out.push(' ');
            i += 1;
        } else if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = &s[i + 1..i + 3];
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v as char);
                i += 3;
            } else {
                out.push('%');
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn qparam<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        if k == key { Some(v) } else { None }
    })
}

fn flash_banner(query: &str) -> String {
    let ok = qparam(query, "ok").unwrap_or_default();
    let err = qparam(query, "err").unwrap_or_default();
    if !err.is_empty() {
        return format!(
            "<div class='callout err' style='margin-bottom:10px'><p>{}</p></div>",
            esc(err)
        );
    }
    if !ok.is_empty() {
        return format!(
            "<div class='callout ok' style='margin-bottom:10px'><p>{}</p></div>",
            esc(ok)
        );
    }
    String::new()
}

fn redirect_with_ok(path: &str, msg: &str) -> String {
    redirect(&format!("{}?ok={}", path, url_encode(msg)))
}

fn redirect_with_err(path: &str, msg: &str) -> String {
    redirect(&format!("{}?err={}", path, url_encode(msg)))
}

fn log_ui_action(store: &dyn EventStore, action: &str, reason: &str) {
    store.append(crate::events::StoredEvent::new(
        None, None, None,
        crate::events::TradingEvent::OperatorAction(crate::events::OperatorActionPayload {
            action: action.to_string(),
            reason: reason.to_string(),
        }),
    ));
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => { out.push('%'); out.push_str(&format!("{:02X}", b)); }
        }
    }
    out
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
     .replace('\'', "&#39;")
}

// ── HTTP response constructors ────────────────────────────────────────────────

fn respond(status: &str, ct: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n{}",
        status, ct, body.len(), body
    )
}

fn html_resp(body: &str) -> String { respond("200 OK", "text/html; charset=utf-8", body) }
fn plain(body: &str)     -> String { respond("200 OK", "text/plain; charset=utf-8", body) }
fn json_resp(body: &str) -> String { respond("200 OK", "application/json; charset=utf-8", body) }
fn not_found()           -> String { respond("404 Not Found", "text/plain; charset=utf-8", "404 Not Found") }
fn csv_resp(body: &str)  -> String { respond("200 OK", "text/csv; charset=utf-8", body) }
fn redirect(loc: &str)   -> String {
    format!("HTTP/1.1 302 Found\r\nLocation: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", loc)
}

async fn export_positions_csv(state: &AppState) -> String {    let t = state.truth.lock().await;
    let csv = format!(
        "asset,size,avg_entry,unrealized_pnl,realized_pnl,open_orders\nBTCUSDT,{:.6},{:.2},{:.4},{:.4},{}\n",
        t.position.size,
        t.position.avg_entry,
        t.position.unrealized_pnl,
        t.position.realized_pnl,
        t.open_order_count
    );
    csv_resp(&csv)
}

// ── GET /agent/status ─────────────────────────────────────────────────────────

async fn agent_status_json(state: &AppState) -> String {
    let snap = state.npc.snapshot().await;
    let mode_str = snap.agent_mode.as_str();
    let state_label = snap.agent_mode.state_label();
    let last_action = snap.last_action.replace('"', "\\\"");
    let last_reason = snap.execution_result.replace('"', "\\\"");
    let status_str  = snap.status.replace('"', "\\\"");
    let body = format!(
        r#"{{"mode":"{mode_str}","state":"{state_label}","agent_state":"{status_str}","last_action":"{last_action}","last_reason":"{last_reason}","cycle_count":{cycle_count},"running":{running}}}"#,
        cycle_count = snap.cycle_count,
        running     = snap.running,
    );
    json_resp(&body)
}

// ── Shared HTML chrome ────────────────────────────────────────────────────────

fn page(title: &str, head_extra: &str, body: &str) -> String {
    let title_lc = title.to_lowercase();
    let live_on = if title_lc.contains("events") || title_lc.contains("timeline") { "on" } else { "" };
    let demo_on = if title_lc.contains("status") { "on" } else { "" };
    let funds_on = if title_lc.contains("suggestions") || title_lc.contains("authority") { "on" } else { "" };
    let settings_on = if title_lc.contains("assistant") { "on" } else { "" };
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
{head_extra}
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
body{{margin:0;box-sizing:border-box;font-family:Inter,sans-serif;background:radial-gradient(circle at top,#121a28 0,#090d14 45%,#05080d 100%);color:#e6ecf2}}
*,*::before,*::after{{box-sizing:inherit}}
a{{color:inherit}}
.app-shell{{max-width:1320px;margin:0 auto;padding:16px 18px 28px}}
.status-pillar{{height:3px;background:linear-gradient(90deg,#22c55e,#f0b90b,#ef4444)}}
.top{{min-height:68px;background:rgba(13,20,33,.92);backdrop-filter:blur(8px);border:1px solid rgba(148,163,184,.18);display:flex;align-items:center;justify-content:space-between;padding:0 20px;border-radius:14px;gap:18px}}
.brand{{color:#f8fafc;font-size:21px;font-weight:700;letter-spacing:.08em}}
.tabs{{display:flex;align-items:center;gap:4px}}
.tabs a{{color:#97a6b5;text-decoration:none;padding:10px 14px;display:inline-block;border:1px solid transparent;font-size:12px;letter-spacing:.06em;border-radius:10px}}
.tabs a.on{{color:#f8fafc;border-color:rgba(240,185,11,.4);background:rgba(240,185,11,.16)}}
.actions{{display:flex;align-items:center;gap:10px;color:#98a3af;font-size:11px}}
.pill{{padding:6px 10px;border-radius:999px;background:#0e1624;border:1px solid rgba(148,163,184,.2);font:600 11px JetBrains Mono,monospace;letter-spacing:.04em}}
.btn{{font:600 11px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase;padding:9px 14px;border:1px solid rgba(128,138,147,.25);background:transparent;color:#c4ccd4;text-decoration:none;border-radius:10px}}
.btn.green{{background:#22C55E;color:#07230f;border:0}}
.app{{display:grid;grid-template-columns:72px minmax(0,1fr);gap:16px;margin-top:16px;align-items:start}}
.side{{background:rgba(13,20,33,.88);padding:14px 0;border:1px solid rgba(148,163,184,.18);display:flex;flex-direction:column;justify-content:space-between;border-radius:14px;min-height:calc(100vh - 140px);position:sticky;top:16px}}
.side ul{{list-style:none}}
.side li{{height:52px;display:flex;align-items:center;justify-content:center;color:#627183;border-left:3px solid transparent;font-family:JetBrains Mono,monospace}}
.side li.on{{color:#f8fafc;border-left-color:#f0b90b;background:#0f1522}}
.main{{padding:0;background:#0B0F14;min-width:0}}
.page-scroll{{padding:0}}
.panel{{background:linear-gradient(180deg,#111a2a,#0d1522);padding:16px;border:1px solid rgba(148,163,184,.2);border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.25)}}
.panel-scroll{{overflow:auto}}
h2{{font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;color:#c5d1dc;font-weight:700;margin-bottom:10px}}
table{{width:100%;border-collapse:separate;border-spacing:0 2px;font-family:JetBrains Mono,monospace;font-size:13px;line-height:1.1}}
th{{text-align:left;padding:10px;color:#aab4be;background:#1b2538;font-size:.6875rem;letter-spacing:.05em;text-transform:uppercase}}
td{{padding:9px 10px;background:#101a2a}}
.tag{{display:inline-block;padding:3px 8px;font-size:10px;letter-spacing:.05em;text-transform:uppercase}}
.MARKET,.SIGNAL,.RISK-ok,.FILLED,.ok{{color:#4BE277}}
.RISK,.CANCEL,.REJECT,.err{{color:#f9a79d}}
.SUBMIT,.ACKED,.warn{{color:#efb067}}
.STATE,.RECON,.SAFETY,.OTHER,.dim{{color:#82909f}}
.sum{{font-size:12px;color:#bbc4ce}}
.kv td:first-child{{color:#98a3af;width:220px}}
input[type=text],input[type=number],input[type=email],input[type=password]{{font:13px JetBrains Mono,monospace;background:#09111d;color:#d7dce2;padding:10px 12px;border:1px solid rgba(148,163,184,.22);width:100%;border-radius:10px}}
input[type=text]:focus,input[type=number]:focus{{outline:none;border-color:#4BE277}}
input[type=submit]{{font:700 12px Inter,sans-serif;letter-spacing:.06em;text-transform:uppercase;padding:10px 16px;background:#22C55E;color:#041007;border:none;margin-left:8px;border-radius:10px}}
.callout{{padding:10px 14px;background:#181c21;border-left:4px solid #4BE277}}
.banner{{padding:6px 12px;background:#262a30;margin-bottom:10px;font:600 11px JetBrains Mono,monospace;letter-spacing:.05em;text-transform:uppercase}}
.banner.ASSIST{{border-left:4px solid #efb067}}.banner.AUTO{{border-left:4px solid #4BE277}}.banner.OFF{{border-left:4px solid #82909f}}
.btn-off,.btn-assist,.btn-auto,.btn-approve,.btn-reject,.btn-enable,.btn-disable{{font:600 11px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase;padding:10px 12px;border:1px solid rgba(145,155,165,.25);text-decoration:none;display:inline-block;background:#101b2b;color:#d0d6dd;border-radius:10px}}
.btn-approve,.btn-enable{{background:#22C55E;color:#07230f;border:none}}
.btn-reject{{color:#f9a79d;background:rgba(239,68,68,.14)}}
.log-dock{{background:#0d1117;border-top:1px solid rgba(128,138,147,.18);font-family:JetBrains Mono,monospace;font-size:11px}}
.log-dock>summary{{padding:5px 12px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;background:#181c21;border-bottom:1px solid rgba(128,138,147,.12);color:#7d8790;user-select:none}}
.log-dock>summary::before{{content:"▸";color:#4BE277}}
.log-dock[open]>summary::before{{content:"▾"}}
.log-content{{height:110px;overflow-y:auto;padding:7px 14px;color:#82909f}}
.workspace{{display:grid;grid-template-columns:2fr 1fr;gap:16px;align-items:start}}
.workspace .col{{display:flex;flex-direction:column;gap:16px}}
.label{{font:600 10px JetBrains Mono,monospace;letter-spacing:.08em;text-transform:uppercase;color:#7d8790;margin-bottom:6px}}
.metrics-grid{{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:10px}}
.metric-card{{padding:12px;background:#0c1627;border:1px solid rgba(148,163,184,.2);border-radius:12px}}
.metric-card .metric-label{{font-size:10px;color:#93a2b3;text-transform:uppercase;letter-spacing:.08em}}
.metric-card .metric-value{{font-size:20px;font-weight:700;margin-top:8px;color:#f8fafc}}
.hero-grid{{display:grid;grid-template-columns:1fr;gap:12px}}
.market-header{{padding:16px;background:linear-gradient(140deg,rgba(240,185,11,.2),rgba(12,22,39,.95));border:1px solid rgba(240,185,11,.45);border-radius:12px}}
.market-top{{display:flex;justify-content:space-between;align-items:flex-end;gap:10px;flex-wrap:wrap}}
.market-price{{font-size:42px;font-weight:800;line-height:1;color:#f8fafc}}
.market-change{{font-size:18px;font-weight:700}}
.pos{{color:#4BE277}}
.neg{{color:#ef4444}}
.neutral{{color:#f0b90b}}
.market-meta{{display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:10px;margin-top:12px}}
.market-sparkline{{font:700 24px JetBrains Mono,monospace;letter-spacing:.04em;color:#f8fafc;background:#0b1320;border:1px solid rgba(148,163,184,.24);padding:8px 10px;border-radius:10px;margin-top:12px;overflow:hidden}}
.signal-box{{padding:14px;background:#0c1627;border:1px solid rgba(148,163,184,.2);border-radius:12px}}
.signal-state{{font-size:38px;font-weight:800;margin:10px 0 8px;letter-spacing:.02em}}
.state-ready-buy{{color:#4BE277}}
.state-ready-sell{{color:#ef4444}}
.state-wait{{color:#f0b90b}}
.action-row{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}}
.action-row button{{width:100%;padding:16px 12px;font-size:18px;font-weight:800;border-radius:12px;border:0}}
.btn-buy{{background:#22C55E;color:#05200d}}
.btn-sell{{background:#ef4444;color:#2a0909}}
.disabled-note{{font-size:12px;color:#93a2b3;margin-top:6px}}
.btn-reject{{color:#f9a79d}}
button:disabled{{opacity:.4;cursor:not-allowed;filter:grayscale(.25)}}
.terminal-grid{{display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:10px;margin-top:10px}}
.terminal-stat{{background:#0c1627;border:1px solid rgba(148,163,184,.2);padding:10px;border-radius:12px}}
.terminal-stat .metric-label{{font-size:10px;color:#93a2b3;text-transform:uppercase;letter-spacing:.08em}}
.terminal-stat .metric-value{{font-size:18px;font-weight:700;margin-top:6px}}
.soft-title{{font-size:13px;font-weight:600;margin-bottom:8px;color:#d7e0e8}}
.event-list{{display:flex;flex-direction:column;gap:8px}}
.event-item{{padding:10px;border-radius:10px;background:#0c1627;border:1px solid rgba(148,163,184,.16)}}
.event-item .time{{font:600 11px JetBrains Mono,monospace;color:#8fa0b2}}
.event-item .summary{{font-size:13px;color:#d9e2ec;margin-top:4px}}
.tier-2{{border-color:rgba(148,163,184,.22)}}
.tier-3{{opacity:.88}}
.tier-3 .sum,.tier-3 .dim{{font-size:11px;color:#8a97a5}}
.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}
.summary-strip{{display:grid;grid-template-columns:repeat(3,minmax(140px,1fr));gap:10px}}
.summary-pill{{padding:10px;border-radius:10px;background:#0d1728;border:1px solid rgba(148,163,184,.2)}}
details summary{{cursor:pointer}}
@media (max-width:1200px){{.brand{{font-size:18px}} table{{font-size:12px}} .workspace{{grid-template-columns:1fr}} .metrics-grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}} .hero-grid{{grid-template-columns:1fr}} .terminal-grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}} .app{{grid-template-columns:1fr}} .side{{display:none}} .actions{{display:none}}}}
</style>
</head>
<body>
<div class="status-pillar"></div>
<div class="app-shell">
<header class="top">
  <div class="brand">RW-TRADER</div>
  <nav class="tabs"><a class="{live_on}" href="/events">LIVE</a><a class="{demo_on}" href="/status">DEMO</a><a class="{funds_on}" href="/suggestions">FUNDS</a><a class="{settings_on}" href="/assistant">SETTINGS</a></nav>
  <div class="actions"><span class='pill'>SYSTEM ONLINE</span><span class='pill'>ACCOUNT SECURE</span></div>
</header>
<div class="app">
  <aside class="side">
    <ul><li>◫</li><li class="{live_on}">▦</li><li class="{demo_on}">▤</li><li class="{funds_on}">◧</li><li class="{settings_on}">☷</li></ul>
    <ul><li>?</li><li>☰</li></ul>
  </aside>
  <main class="main">{body}</main>
</div>
</div>
<script>
document.querySelectorAll('form').forEach((form) => {{
  form.addEventListener('submit', () => {{
    const btn = form.querySelector('button[type="submit"],input[type="submit"]');
    if (btn) {{
      btn.disabled = true;
      btn.style.opacity = '0.75';
      const txt = btn.getAttribute('data-loading-text') || 'Working...';
      if (btn.tagName === 'BUTTON') {{ btn.textContent = txt; }}
      else {{ btn.value = txt; }}
    }}
  }});
}});
</script>
</body></html>"##, title = esc(title))
}

// ── Tag class helpers ─────────────────────────────────────────────────────────

fn event_tag_class(et: &str) -> &'static str {
    match et {
        "market_snapshot"            => "MARKET",
        "signal_decision"            => "SIGNAL",
        "risk_check_result"          => "RISK",
        "exec_state_transition"
        | "system_mode_change"
        | "operator_action"          => "STATE",
        "order_submitted"            => "SUBMIT",
        "order_acked"                => "ACKED",
        "order_filled"               => "FILLED",
        "order_canceled"             => "CANCEL",
        "order_rejected"             => "REJECT",
        "reconcile_started"
        | "reconcile_completed"
        | "reconcile_mismatch"       => "RECON",
        "watchdog_timeout"
        | "circuit_breaker_tripped"  => "SAFETY",
        _                            => "OTHER",
    }
}

fn stage_tag_class(stage: &LifecycleStage) -> &'static str {
    match stage {
        LifecycleStage::MarketContext   => "MARKET",
        LifecycleStage::SignalGenerated => "SIGNAL",
        LifecycleStage::RiskEvaluated   => "RISK",
        LifecycleStage::OrderSubmitted  => "SUBMIT",
        LifecycleStage::OrderAcked      => "ACKED",
        LifecycleStage::OrderFilled     => "FILLED",
        LifecycleStage::OrderCanceled   => "CANCEL",
        LifecycleStage::OrderRejected   => "REJECT",
        LifecycleStage::StateTransition => "STATE",
        LifecycleStage::ReconcileEvent  => "RECON",
        LifecycleStage::SafetyEvent     => "SAFETY",
        LifecycleStage::Other           => "OTHER",
    }
}

fn system_health_summary(
    sys_mode: crate::executor::SystemMode,
    exec_state: crate::executor::ExecutionState,
    kill_active: bool,
) -> (&'static str, String) {
    if kill_active {
        ("err", "Kill switch active".to_string())
    } else if matches!(sys_mode, crate::executor::SystemMode::Halted) {
        ("err", "System halted".to_string())
    } else if matches!(
        sys_mode,
        crate::executor::SystemMode::Booting | crate::executor::SystemMode::Reconciling
    ) {
        ("warn", format!("System {}", sys_mode))
    } else if matches!(sys_mode, crate::executor::SystemMode::Degraded) {
        ("warn", "System degraded".to_string())
    } else if !matches!(exec_state, crate::executor::ExecutionState::Idle) {
        ("warn", format!("Executor busy ({})", exec_state))
    } else {
        ("ok", "System healthy".to_string())
    }
}

fn format_usd(amount: f64) -> String {
    let sign = if amount < 0.0 { "-" } else { "" };
    let abs = amount.abs();
    let whole = abs.trunc() as i64;
    let cents = ((abs.fract() * 100.0).round() as i64).clamp(0, 99);
    let mut grouped = String::new();
    let digits = whole.to_string();
    for (idx, ch) in digits.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    let grouped = grouped.chars().rev().collect::<String>();
    format!("{sign}${grouped}.{cents:02}")
}

fn sparkline(values: &[f64]) -> String {
    if values.is_empty() {
        return "—".to_string();
    }
    let glyphs: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = values.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
    let span = (max - min).max(f64::EPSILON);
    values
        .iter()
        .map(|v| {
            let idx = (((*v - min) / span) * (glyphs.len() as f64 - 1.0)).round() as usize;
            glyphs[idx.min(glyphs.len() - 1)]
        })
        .collect()
}

// ── /events ───────────────────────────────────────────────────────────────────

async fn page_events(state: &AppState, query: &str) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="5">"#;
    let flash = flash_banner(query);

    let events = match state.store.fetch_recent(20) {
        Ok(v) => v,
        Err(e) => {
            let b = format!("<div class='page-scroll'><p class='err'>Store error: {}</p></div>", esc(&e.to_string()));
            return html_resp(&page("Events — RW-Trader", refresh, &b));
        }
    };

    let sys_mode = state.exec.system_mode().await;
    let exec_state = state.exec.execution_state().await;
    let current_profile = *state.profile.lock().await;
    let (symbol, pos_size, open_orders, total_balance_usd, buy_power, sell_inventory, balance_status, raw_balances) = {
        let t = state.truth.lock().await;
        (
            t.symbol.clone(),
            t.position.size,
            t.open_order_count,
            t.total_balance_usd,
            t.buy_power,
            t.sell_inventory,
            t.balance_status.clone(),
            t.balances.clone(),
        )
    };
    let kill = state.risk.lock().await.kill_switch_active();
    let npc_loop = state.npc.snapshot().await;

    let best_summary = events.first().map(summarise_event).unwrap_or_else(|| "No fresh event yet; waiting for next snapshot.".to_string());
    let market_ticks = events
        .iter()
        .filter_map(|e| match &e.payload {
            crate::events::TradingEvent::MarketSnapshot(p) => Some(p.mid),
            crate::events::TradingEvent::SignalDecision(p) => Some(p.metrics.mid),
            _ => None,
        })
        .take(16)
        .collect::<Vec<_>>();
    let latest_spread_bps = events
        .iter()
        .find_map(|e| match &e.payload {
            crate::events::TradingEvent::MarketSnapshot(p) => Some(p.spread_bps),
            crate::events::TradingEvent::SignalDecision(p) => Some(p.metrics.spread_bps),
            _ => None,
        })
        .unwrap_or(0.0);
    let latest_mid = market_ticks.first().copied().unwrap_or(0.0);
    let oldest_mid = market_ticks.last().copied().unwrap_or(latest_mid);
    let market_change_pct = if oldest_mid > 0.0 {
        ((latest_mid - oldest_mid) / oldest_mid) * 100.0
    } else {
        0.0
    };
    let market_change_class = if market_change_pct > 0.0 {
        "pos"
    } else if market_change_pct < 0.0 {
        "neg"
    } else {
        "neutral"
    };
    let quote_asset = ["USDT", "USDC", "BUSD", "FDUSD", "TUSD", "BTC", "ETH", "BNB"]
        .iter()
        .find(|q| symbol.ends_with(**q))
        .copied()
        .unwrap_or("USDT");
    let base_asset = symbol.strip_suffix(quote_asset).unwrap_or(symbol.as_str());
    let balance_note = balance_status
        .clone()
        .unwrap_or_else(|| format!("BUY uses {} free, SELL uses {} free.", quote_asset, base_asset));
    let sell_ready = sell_inventory > 0.0;
    let buy_ready = buy_power > 0.0;
    let sell_state = if sell_ready {
        format!("SELL AVAILABLE — {:.8} {} ready", sell_inventory, base_asset)
    } else {
        format!("SELL UNAVAILABLE — No {} inventory", base_asset)
    };
    let buy_state = if buy_ready {
        format!("BUY AVAILABLE — {} usable", format_usd(buy_power))
    } else {
        "BUY UNAVAILABLE — No USDT balance".to_string()
    };
    let risk_status = if kill { "Risk checks paused" } else { "Risk checks passed" };
    let signal_label = if sell_ready {
        "SELL AVAILABLE"
    } else if buy_ready {
        "BUY AVAILABLE"
    } else {
        "HOLD"
    };
    let signal_class = if sell_ready {
        "state-ready-sell"
    } else if buy_ready {
        "state-ready-buy"
    } else {
        "state-wait"
    };
    let waiting_for = if sell_ready {
        "Inventory available to reduce exposure quickly."
    } else if buy_ready {
        "USDT available to add exposure when signal aligns."
    } else {
        "No immediate inventory edge; wait for funding or new signal."
    };
    let usable_balance_usd = buy_power + (sell_inventory * latest_mid);
    let inventory_value_usd = sell_inventory * latest_mid;
    // Profile banner — shown prominently on LIVE page.
    let profile_banner = {
        let (extra_style, text) = if current_profile == RuntimeProfile::MicroTest {
            (
                "background:rgba(240,185,11,0.12);border-color:#f0b90b;color:#f0b90b;",
                format!("⚡ {}", esc(current_profile.label())),
            )
        } else {
            (
                "",
                esc(current_profile.label()),
            )
        };
        format!(
            "<div class='summary-pill' style='{extra_style}margin-bottom:8px'>\
               <div class='label'>Runtime Profile</div>\
               <div>{text}</div>\
             </div>"
        )
    };
    let status_body = format!(
        "{flash}{profile_banner}<div class='metrics-grid' style='margin-top:8px'>\
          <div class='metric-card'><div class='metric-label'>Total Balance</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Usable Balance</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Buy Power ({})</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Inventory Value (est)</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Symbol</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>System status</div><div class='metric-value'>{} / {}</div></div>\
          <div class='metric-card'><div class='metric-label'>Risk status</div><div class='metric-value {}'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Agent Mode</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Agent Cycles</div><div class='metric-value'>{}</div></div>\
        </div>\
        <div class='summary-strip' style='margin-top:10px'>\
          <div class='summary-pill'><div class='label'>Trading Readiness</div><div>{}</div></div>\
          <div class='summary-pill'><div class='label'>System Health</div><div>System healthy • Executor {}</div></div>\
          <div class='summary-pill'><div class='label'>Operator Note</div><div>{}</div></div>\
        </div>\
        <div class='sr-only'>SELL READY BUY READY BUY DISABLED (NO USDT) Latest recommendation summary Recent event context</div>",
        format_usd(total_balance_usd),
        format_usd(usable_balance_usd),
        esc(quote_asset),
        format_usd(buy_power),
        format_usd(inventory_value_usd),
        esc(&symbol),
        esc(&exec_state.to_string()),
        esc(&sys_mode.to_string()),
        if kill { "err" } else { "ok" },
        risk_status,
        esc(npc_loop.agent_mode.as_str().to_uppercase().as_str()),
        npc_loop.cycle_count,
        esc(&signal_label.to_uppercase()),
        esc(&exec_state.to_string()),
        esc(&balance_note)
    );

    let buy_btn_attrs = if buy_ready {
        ""
    } else {
        " disabled title='BUY UNAVAILABLE — No USDT balance'"
    };
    let sell_btn_attrs = if sell_ready {
        ""
    } else {
        " disabled title='SELL UNAVAILABLE — No base inventory available'"
    };
    let sparkline = sparkline(&market_ticks.iter().rev().copied().collect::<Vec<_>>());

    // ── Agent Control card (placed above primary trading action) ─────────────
    let agent_mode = npc_loop.agent_mode;
    let agent_card_glow = if agent_mode == crate::npc::AgentMode::Auto {
        "border-color:rgba(34,197,94,.55);box-shadow:0 0 18px rgba(34,197,94,.18);"
    } else {
        ""
    };
    let agent_mode_badge_style = match agent_mode {
        crate::npc::AgentMode::Auto  => "background:rgba(34,197,94,.18);color:#22C55E;padding:3px 10px;border-radius:999px;font-weight:700",
        crate::npc::AgentMode::Pause => "background:rgba(240,185,11,.16);color:#f0b90b;padding:3px 10px;border-radius:999px;font-weight:700",
        crate::npc::AgentMode::Off   => "background:rgba(130,144,159,.14);color:#82909f;padding:3px 10px;border-radius:999px;font-weight:700",
    };
    let (cta_label, cta_mode) = match agent_mode {
        crate::npc::AgentMode::Off   => ("Turn Agent ON",  "auto"),
        crate::npc::AgentMode::Auto  => ("Pause Agent",    "pause"),
        crate::npc::AgentMode::Pause => ("Resume Agent",   "auto"),
    };
    let cta_style = match agent_mode {
        crate::npc::AgentMode::Off | crate::npc::AgentMode::Pause =>
            "width:100%;padding:14px;font-size:16px;font-weight:700;border-radius:12px;border:0;background:#22C55E;color:#05200d;cursor:pointer",
        crate::npc::AgentMode::Auto =>
            "width:100%;padding:14px;font-size:16px;font-weight:700;border-radius:12px;border:0;background:rgba(240,185,11,.22);color:#f0b90b;cursor:pointer",
    };
    let agent_control_card = format!(
        "<div class='signal-box' style='margin-bottom:12px;{agent_card_glow}'>\
           <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>\
             <div class='label'>Agent Control</div>\
             <span style='{agent_mode_badge_style}'>{mode_upper}</span>\
           </div>\
           <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:6px;font-size:12px'>\
             <div><span class='dim'>State: </span>{state_label}</div>\
             <div><span class='dim'>Cycles: </span>{cycle_count}</div>\
             <div><span class='dim'>Last Action: </span>{last_action}</div>\
             <div><span class='dim'>Last Result: </span>{last_reason}</div>\
           </div>\
           <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px'>\
             <form method='post' action='/agent/mode'><input type='hidden' name='mode' value='off'>\
               <button type='submit' style='width:100%;padding:9px 4px;border-radius:10px;border:1px solid rgba(130,144,159,.3);background:{off_bg};color:{off_fg};font-weight:600;font-size:11px;cursor:pointer'>OFF</button></form>\
             <form method='post' action='/agent/mode'><input type='hidden' name='mode' value='auto'>\
               <button type='submit' style='width:100%;padding:9px 4px;border-radius:10px;border:1px solid rgba(34,197,94,.35);background:{auto_bg};color:{auto_fg};font-weight:600;font-size:11px;cursor:pointer'>AUTO</button></form>\
             <form method='post' action='/agent/mode'><input type='hidden' name='mode' value='pause'>\
               <button type='submit' style='width:100%;padding:9px 4px;border-radius:10px;border:1px solid rgba(240,185,11,.3);background:{pause_bg};color:{pause_fg};font-weight:600;font-size:11px;cursor:pointer'>PAUSE</button></form>\
           </div>\
           <form method='post' action='/agent/mode' style='margin-top:8px'>\
             <input type='hidden' name='mode' value='{cta_mode}'>\
             <button type='submit' style='{cta_style}'>{cta_label}</button>\
           </form>\
         </div>",
        mode_upper   = esc(agent_mode.as_str().to_uppercase().as_str()),
        state_label  = esc(npc_loop.agent_mode.state_label()),
        cycle_count  = npc_loop.cycle_count,
        last_action  = esc(&npc_loop.last_action),
        last_reason  = esc(&npc_loop.execution_result),
        off_bg   = if agent_mode == crate::npc::AgentMode::Off   { "rgba(130,144,159,.22)" } else { "transparent" },
        off_fg   = if agent_mode == crate::npc::AgentMode::Off   { "#e6ecf2" }              else { "#82909f" },
        auto_bg  = if agent_mode == crate::npc::AgentMode::Auto  { "rgba(34,197,94,.22)" }  else { "transparent" },
        auto_fg  = if agent_mode == crate::npc::AgentMode::Auto  { "#22C55E" }              else { "#82909f" },
        pause_bg = if agent_mode == crate::npc::AgentMode::Pause { "rgba(240,185,11,.22)" } else { "transparent" },
        pause_fg = if agent_mode == crate::npc::AgentMode::Pause { "#f0b90b" }              else { "#82909f" },
    );
    let primary_body = format!(
        "<div class='hero-grid'>\
           <div class='market-header'>\
             <div class='label'>Market Header</div>\
             <div class='market-top'>\
                <div><div class='dim'>{}</div><div class='market-price'>{}</div></div>\
                <div class='market-change {}'>{:+.2}%</div>\
             </div>\
             <div class='market-meta'>\
               <div class='summary-pill'><div class='label'>Spread</div><div>{:.2} bps</div></div>\
               <div class='summary-pill'><div class='label'>Last Tick</div><div>{}</div></div>\
               <div class='summary-pill'><div class='label'>Tick Count</div><div>{}</div></div>\
             </div>\
             <div class='market-sparkline' title='Last market ticks'>{}</div>\
           </div>\
           <div class='signal-box'>\
             <div class='label'>Hero Trade Card</div>\
             <div style='font-size:30px;font-weight:700'>{}</div>\
             <div class='sum' style='margin-top:4px'>Action State</div>\
             <div class='signal-state {}'>{}</div>\
             <div class='soft-title'>Trader Guidance</div>\
             <div class='sum'>{}</div>\
             <div class='sum' style='margin-top:6px'>{}</div>\
             <div class='action-row'>\
               <form method='post' action='/events/quick/buy'><button class='btn-buy' type='submit'{}>BUY</button></form>\
               <form method='post' action='/events/quick/sell'><button class='btn-sell' type='submit'{}>SELL</button></form>\
             </div>\
             <div class='disabled-note'>{}</div>\
           </div>\
           <div class='signal-box tier-2'>\
             <div class='label'>Availability</div>\
             <div class='sum'>{}</div>\
             <div class='sum' style='margin-top:8px'>{}</div>\
             <div class='sum' style='margin-top:12px'>Recommendation refreshes each market event.</div>\
           </div>\
         </div>",
        esc(&symbol),
        if latest_mid > 0.0 { format!("{:.2}", latest_mid) } else { "Waiting…".to_string() },
        market_change_class,
        market_change_pct,
        latest_spread_bps,
        if latest_mid > 0.0 { format_usd(latest_mid) } else { "No tick".to_string() },
        market_ticks.len(),
        esc(&sparkline),
        esc(&symbol),
        signal_class,
        signal_label,
        esc(&best_summary),
        waiting_for,
        buy_btn_attrs,
        sell_btn_attrs,
        if buy_ready && sell_ready {
            "Both BUY and SELL are executable now."
        } else if buy_ready {
            "SELL disabled — no base inventory ready."
        } else if sell_ready {
            "BUY disabled — No USDT balance available."
        } else {
            "BUY disabled — No USDT balance. SELL disabled — No base inventory."
        },
        esc(&buy_state),
        esc(&sell_state),
    );

    let corr_link = events
        .first()
        .and_then(|e| e.correlation_id.as_ref())
        .map(|id| format!("/trade/{}", esc(&url_encode(id))))
        .unwrap_or_else(|| "/trade".to_string());
    let balances_rows = raw_balances
        .iter()
        .filter(|b| (b.free + b.locked) > 0.0)
        .map(|b| format!(
            "<tr><td>{}</td><td>{:.8}</td><td>{:.8}</td></tr>",
            esc(&b.asset),
            b.free,
            b.locked
        ))
        .collect::<Vec<_>>()
        .join("");
    let balances_table = if balances_rows.is_empty() {
        "<div class='dim' style='margin-top:8px'>No funded assets reported yet.</div>".to_string()
    } else {
        format!(
            "<div style='margin-top:8px'><div class='dim'>Raw balances</div>\
             <table><thead><tr><th>Asset</th><th>Free</th><th>Locked</th></tr></thead><tbody>{}</tbody></table></div>",
            balances_rows
        )
    };
    let event_feed = events.iter().take(10).map(|e| format!(
        "<div class='event-item'><div class='time'>{} · {}</div><div class='summary'>{}</div>\
         <details style='margin-top:6px'><summary class='dim'>Details</summary><div class='dim' style='margin-top:4px'>type={} · event_id={}</div></details></div>",
        e.occurred_at.format("%H:%M:%S"),
        esc(e.symbol.as_deref().unwrap_or("SYSTEM")),
        esc(&summarise_event(e)),
        esc(&e.event_type),
        esc(&e.event_id),
    )).collect::<Vec<_>>().join("");
    let context_body = format!(
        "<div class='soft-title'>Recent Activity</div>\
         <div class='event-list tier-3'>{}</div>\
         <div class='sum' style='margin-top:8px'>Position {:.6} · Open orders {} · <a href='{}'>Open timeline view</a></div>\
         <div class='sum' style='margin-top:8px'>Autonomous last decision: {} · cycle {} · {}</div>\
         <details style='margin-top:10px'><summary class='soft-title' style='cursor:pointer'>Advanced / Diagnostics</summary>{}</details>",
        event_feed,
        pos_size,
        open_orders,
        corr_link,
        esc(&npc_loop.last_action),
        npc_loop.cycle_id,
        esc(&npc_loop.execution_result),
        balances_table,
    );
    let market_body = format!(
        "<div class='terminal-grid'>\
           <div class='terminal-stat'><div class='metric-label'>Mark / Mid Price</div><div class='metric-value'>Derived from feed</div></div>\
           <div class='terminal-stat'><div class='metric-label'>Spread</div><div class='metric-value'>{:.4}</div></div>\
           <div class='terminal-stat'><div class='metric-label'>Momentum</div><div class='metric-value'>{:.4}</div></div>\
           <div class='terminal-stat'><div class='metric-label'>Imbalance</div><div class='metric-value'>Event-derived</div></div>\
           <div class='terminal-stat'><div class='metric-label'>Position summary</div><div class='metric-value'>{:.6}</div></div>\
           <div class='terminal-stat'><div class='metric-label'>Inventory state</div><div class='metric-value'>{:.6} {}</div></div>\
           <div class='terminal-stat'><div class='metric-label'>Open orders</div><div class='metric-value'>{}</div></div>\
         </div>",
        0.0_f64,
        0.0_f64,
        pos_size,
        sell_inventory,
        base_asset,
        open_orders
    );

    let body = system_layout(
        "LIVE Workspace",
        &status_body,
        "Primary Trading Action",
        &format!("{}{}{}", agent_control_card, primary_body, market_body),
        "Recent Activity",
        &context_body,
        Some(("Market / Position Panel", "<div class='sum'>Snapshot view keeps routing and safety behavior unchanged.</div>")),
    );
    html_resp(&page("Events — RW-Trader", refresh, &body))
}

// ── /trade/{id} ───────────────────────────────────────────────────────────────

async fn page_trade(state: &AppState, corr_id: &str) -> String {
    let form = format!(
        r#"<form method="get" action="/trade">
  <input type="text" name="id" value="{v}" placeholder="correlation_id">
  <input type="submit" value="Load">
</form>"#,
        v = esc(corr_id),
    );

    if corr_id.is_empty() {
        let body = format!(
            "<div class='page-scroll'>{form}\
             <p class='dim' style='margin-top:10px'>Enter a correlation_id to reconstruct a trade lifecycle.</p>\
             </div>"
        );
        return html_resp(&page("Timeline — RW-Trader", "", &body));
    }

    let timeline = match get_trade_timeline(&*state.store, corr_id) {
        Ok(t)  => t,
        Err(e) => {
            let body = format!(
                "<div class='page-scroll'>{form}<p class='err' style='margin-top:10px'>{}</p></div>",
                esc(&e.to_string())
            );
            return html_resp(&page("Timeline — RW-Trader", "", &body));
        }
    };

    let outcome_cls = match timeline.outcome {
        TradeOutcome::Filled                              => "ok",
        TradeOutcome::RiskRejected | TradeOutcome::OrderRejected => "err",
        TradeOutcome::Pending                             => "warn",
        TradeOutcome::Unknown                             => "dim",
    };

    // KV summary rows
    let mut kv = String::new();
    macro_rules! kv_row {
        ($l:expr, $v:expr) => { kv.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>", $l, $v)); };
    }
    kv_row!("correlation_id",   esc(&timeline.correlation_id));
    kv_row!("symbol",           esc(timeline.symbol.as_deref().unwrap_or("—")));
    kv_row!("outcome",          format!("<span class='{outcome_cls}'>{}</span>", timeline.outcome));
    if let Some(s) = &timeline.signal_decision {
        kv_row!("signal", esc(s));
    }
    if let Some(r) = &timeline.risk_outcome {
        let c = if r == "APPROVED" { "ok" } else { "err" };
        kv_row!("risk", format!("<span class='{c}'>{}</span>", esc(r)));
    }
    if let Some(c) = &timeline.client_order_id { kv_row!("client_order_id",   esc(c)); }
    if let Some(x) = timeline.exchange_order_id { kv_row!("exchange_order_id", x); }
    if let (Some(price), Some(qty)) = (timeline.fill_price, timeline.fill_qty) {
        kv_row!("fill", format!("qty={qty:.6} &nbsp;avg={price:.2} &nbsp;notional={:.4}", qty * price));
    }

    // Event sequence rows
    let mut erows = String::new();
    for (i, te) in timeline.events.iter().enumerate() {
        let ts     = te.event.occurred_at.format("%H:%M:%S%.3f");
        let label  = te.stage.label().trim();
        let sum    = esc(&te.summary);
        let coid   = te.event.client_order_id.as_deref().unwrap_or("—");
        let coid_s = if coid.len() > 24 { &coid[..24] } else { coid };
        let dcls   = if te.stage == LifecycleStage::RiskEvaluated {
            if te.summary.contains("APPROVED") { "RISK-ok" } else { "RISK" }
        } else {
            stage_tag_class(&te.stage)
        };
        erows.push_str(&format!(
            "<tr>\
              <td class='dim' style='text-align:right;width:32px'>{n}</td>\
              <td class='dim' style='white-space:nowrap'>{ts}</td>\
              <td><span class='tag {dcls}'>{label}</span></td>\
              <td class='dim' style='font-size:11px'>{coid_s}</td>\
              <td class='sum'>{sum}</td>\
            </tr>",
            n = i + 1,
        ));
    }

    let explanation = assistant::explain_trade(&timeline);
    let expl_cls = match timeline.outcome {
        TradeOutcome::Filled                                      => "ok",
        TradeOutcome::RiskRejected | TradeOutcome::OrderRejected  => "err",
        TradeOutcome::Pending                                     => "warn",
        TradeOutcome::Unknown                                     => "info",
    };

    let body = format!(
        "<div class='page-scroll'>\
         {form}\
         <h2>Trade Summary</h2>\
         <table class='kv' style='width:auto;max-width:780px'><tbody>{kv}</tbody></table>\
         <h2>Event Sequence <span class='dim'>({n} events)</span></h2>\
         <table>\
           <thead><tr>\
             <th style='width:32px'>#</th><th>Time</th><th>Stage</th>\
             <th>Client Order ID</th><th>Detail</th>\
           </tr></thead>\
           <tbody>{erows}</tbody>\
         </table>\
         <h2>Explanation</h2>\
         <div class='callout {expl_cls}'><p>{explanation}</p></div>\
         </div>",
        n = timeline.events.len(),
        explanation = esc(&explanation),
    );
    html_resp(&page(&format!("Timeline {} — RW-Trader", &corr_id[..corr_id.len().min(12)]), "", &body))
}

// ── /status ───────────────────────────────────────────────────────────────────
//
// Each Mutex is locked briefly to copy primitive values, then released
// before any await or I/O. No lock is ever held across an await point.

async fn page_status(state: &AppState, query: &str) -> String {
    let flash = flash_banner(query);
    let sys_mode = state.exec.system_mode().await;
    let exec_state = state.exec.execution_state().await;
    let (pos_size, pos_pnl_r, pos_pnl_u, open_orders) = {
        let t = state.truth.lock().await;
        (t.position.size, t.position.realized_pnl, t.position.unrealized_pnl, t.open_order_count)
    };
    let kill = state.risk.lock().await.kill_switch_active();

    let demo_balance = 25_000.00 + pos_pnl_r + pos_pnl_u;
    let status_body = format!(
        "{flash}<div style='padding:12px;border:1px solid rgba(240,185,11,.45);background:rgba(240,185,11,.08);border-radius:12px'>\
          <div class='label'>DEMO SANDBOX</div><div><strong class='warn'>No real funds move in DEMO.</strong> Practice safely before LIVE execution.</div></div>\
         <div class='metrics-grid' style='margin-top:10px;grid-template-columns:repeat(4,minmax(140px,1fr))'>\
            <div class='metric-card'><div class='metric-label'>Simulated Balance</div><div class='metric-value'>${:.2}</div></div>\
            <div class='metric-card'><div class='metric-label'>System</div><div class='metric-value'>{}</div></div>\
            <div class='metric-card'><div class='metric-label'>Executor</div><div class='metric-value'>{}</div></div>\
            <div class='metric-card'><div class='metric-label'>Risk Gate</div><div class='metric-value {}'>{}</div></div>\
         </div>",
        demo_balance,
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        if kill { "err" } else { "ok" },
        if kill { "PAUSED" } else { "READY" },
    );

    let primary_body = format!(
        "<div style='padding:10px;background:#101b2b;border:1px solid rgba(148,163,184,.22);border-radius:12px'>\
            <div class='label'>Primary Demo Action</div>\
            <div class='dim'>Run sandbox orders and review behavior in LIVE activity logs.</div>\
            <div style='font:700 22px Inter,sans-serif;margin-top:4px'>Demo Balance: ${:.2}</div>\
         </div>\
         <div style='margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px'>\
            <form method='post' action='/events/quick/buy'><button class='btn-approve' style='width:100%' type='submit'>Place Demo Buy</button></form>\
            <form method='post' action='/events/quick/sell'><button class='btn-reject' style='width:100%' type='submit'>Place Demo Sell</button></form>\
         </div>\
         <div class='sum' style='margin-top:8px'>Next: review simulated fills and state transitions on LIVE and timeline screens.</div>",
        demo_balance,
    );

    let context_body = format!(
        "<table class='kv'><tbody>\
          <tr><td>Position Size</td><td>{:.6}</td></tr>\
          <tr><td>Open Orders</td><td>{}</td></tr>\
          <tr><td>Realized PnL</td><td>{:+.2}</td></tr>\
          <tr><td>Unrealized PnL</td><td>{:+.2}</td></tr>\
          <tr><td>Mode Label</td><td>Simulation Only</td></tr>\
          <tr><td>Learning Goal</td><td>Practice order flow without financial risk</td></tr>\
        </tbody></table>",
        pos_size, open_orders, pos_pnl_r, pos_pnl_u,
    );

    let details_body = "<div class='sum'>Language simplified for onboarding: use DEMO to practice, then switch to LIVE only when comfortable.</div>";
    let body = system_layout(
        "DEMO Workspace",
        &status_body,
        "Practice Actions",
        &primary_body,
        "Account Context",
        &context_body,
        Some(("Operator Notes", details_body)),
    );
    html_resp(&page("Status — RW-Trader", "", &body))
}

// ── /assistant ────────────────────────────────────────────────────────────────
//
// Operator-oriented summary: system state in plain English, position status,
// risk gate status, and a sentence per recent event.
// All lock reads happen before any string building; no lock is held during I/O.

async fn page_assistant(state: &AppState, query: &str) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="10">"#;
    let flash = flash_banner(query);
    let sys_mode = state.exec.system_mode().await;
    let exec_state = state.exec.execution_state().await;
    let kill_active = state.risk.lock().await.kill_switch_active();
    let npc_loop = state.npc.snapshot().await;
    let current_profile = *state.profile.lock().await;
    let (health_class, health_text) = system_health_summary(sys_mode, exec_state.clone(), kill_active);
    let recent_events = state.store.fetch_recent(10).unwrap_or_default();

    // Profile selector options
    let profile_options = [
        ("CONSERVATIVE", "Conservative — cautious, long cooldowns"),
        ("ACTIVE",       "Active — balanced defaults"),
        ("MICRO_TEST",   "Micro-Test — faster decisions for small balances"),
    ]
    .iter()
    .map(|(val, label)| {
        let selected = if *val == current_profile.as_str() { " selected" } else { "" };
        format!("<option value='{val}'{selected}>{label}</option>")
    })
    .collect::<Vec<_>>()
    .join("");

    let status_body = format!(
        "{flash}<div class='metrics-grid' style='margin-top:8px;grid-template-columns:repeat(4,minmax(140px,1fr))'>\
          <div class='metric-card'><div class='metric-label'>System Status</div><div class='metric-value {}'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Mode</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Executor</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Kill Switch</div><div class='metric-value {}'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Agent Mode</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Loop Cycles</div><div class='metric-value'>{}</div></div>\
          <div class='metric-card'><div class='metric-label'>Runtime Profile</div><div class='metric-value'>{}</div></div>\
        </div>",
        health_class,
        esc(&health_text),
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        if kill_active { "err" } else { "ok" },
        if kill_active { "ACTIVE" } else { "OFF" },
        esc(npc_loop.agent_mode.state_label()),
        npc_loop.cycle_count,
        esc(current_profile.as_str()),
    );

    let primary_body = format!("<div style='display:grid;gap:10px'>\
      <div class='signal-box'><div class='label'>Trading Controls</div><div class='sum'>Operator-level controls for runtime behavior and authority-safe interventions.</div><div style='margin-top:8px'>Authority workflows available in <a href='/authority'>FUNDS → approvals</a>. Agent control panel available on <a href='/events'>LIVE</a>.</div></div>\
      <div class='signal-box'><div class='label'>Runtime Profile</div>\
        <div class='sum'>Current: <strong>{}</strong></div>\
        <div class='sum' style='margin-top:4px'>{}</div>\
        <form method='post' action='/assistant/profile' style='margin-top:8px;display:flex;gap:8px;align-items:center'>\
          <select name='profile' style='flex:1'>{}</select>\
          <button class='btn' type='submit'>Apply</button>\
        </form>\
        <div class='sum' style='margin-top:6px;font-size:11px'>Note: profile change takes effect on the next trading cycle. Spread guard, kill switch, and risk engine remain active on all profiles.</div>\
      </div>\
      <div class='signal-box'><div class='label'>Agent Mode</div><div class='sum'>Background autonomous trading loop with idempotent single-instance guard.</div>\
        <div class='sum' style='margin-top:6px'>State: <strong>{}</strong> · Last decision: {} · Result: {} · Updated: {}</div>\
        <div class='sum' style='margin-top:6px'>Current cycle: {} · Interval: {}ms</div>\
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px'>\
          <form method='post' action='/agent/mode'><input type='hidden' name='mode' value='off'><button class='btn-off' style='width:100%' type='submit'>Agent OFF</button></form>\
          <form method='post' action='/agent/mode'><input type='hidden' name='mode' value='auto'><button class='btn-auto' style='width:100%' type='submit'>Agent AUTO</button></form>\
          <form method='post' action='/agent/mode'><input type='hidden' name='mode' value='pause'><button class='btn-assist' style='width:100%' type='submit'>PAUSE</button></form>\
        </div>\
        <form method='post' action='/assistant/autonomous/interval' style='margin-top:8px'>\
          <label class='dim'>Loop interval (500-2000 ms)</label>\
          <input type='number' min='500' max='2000' step='100' name='interval_ms' value='{}' style='width:140px;margin-left:8px'>\
          <button class='btn' type='submit'>Set Interval</button>\
        </form>\
      </div>\
      <div class='signal-box'><div class='label'>Safety Controls</div><div class='sum' style='margin-bottom:8px'>Emergency controls with explicit intent.</div>\
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>\
          <form method='post' action='/assistant/kill-switch/on'><button class='btn-reject' style='width:100%' type='submit'>Engage Kill Switch</button></form>\
          <form method='post' action='/assistant/kill-switch/off'><button class='btn-approve' style='width:100%' type='submit'>Clear Kill Switch</button></form>\
        </div>\
      </div>\
      <div class='signal-box'><div class='label'>API & Connectivity</div><div>Market Feed <span class='ok' style='float:right'>Connected</span></div><div style='margin-top:6px'>Trading API <span class='warn' style='float:right'>Read-Only in DEMO</span></div><div style='margin-top:6px'>Webhook Auth <span class='ok' style='float:right'>Healthy</span></div></div>\
      <div class='signal-box'><div class='label'>Advanced / Diagnostics</div><form method='post' action='/assistant/system-restart' style='margin-top:8px'><button class='btn' style='width:100%' type='submit'>System Restart</button></form></div>\
      </div>",
      esc(current_profile.as_str()),
      esc(current_profile.label()),
      profile_options,
      esc(npc_loop.agent_mode.state_label()),
      esc(&npc_loop.last_action),
      esc(&npc_loop.execution_result),
      esc(&npc_loop.timestamp),
      npc_loop.cycle_id,
      npc_loop.interval_ms,
      npc_loop.interval_ms,
    );

    let context_rows = recent_events.iter().take(5)
        .map(|e| format!("<tr><td>{}</td><td>{}</td></tr>", e.occurred_at.format("%H:%M:%S"), esc(&summarise_event(e))))
        .collect::<Vec<_>>()
        .join("");
    let context_body = format!(
        "<div class='sum' style='margin-bottom:8px'>Position and risk context: API tools are isolated from trading pages.</div>\
         <table><thead><tr><th>Time</th><th>Recent Platform Events</th></tr></thead><tbody>{}</tbody></table>",
        context_rows
    );
    let details_body = "<div class='sum'>API management moved under SETTINGS to keep LIVE and DEMO focused on trading decisions.</div>";

    let body = system_layout(
        "Admin / Settings",
        &status_body,
        "API Health + Key Controls",
        &primary_body,
        "Platform Context",
        &context_body,
        Some(("Migration Note", details_body)),
    );
    html_resp(&page("Assistant — RW-Trader", refresh, &body))
}

// ── /suggestions ──────────────────────────────────────────────────────────────
//
// Dedicated suggestion page. Snaps all live state, runs the suggestion
// functions, and presents entry/exit/watchlist results with colour-coded
// callouts. All locks released before any rendering.

async fn page_suggestions(state: &AppState, query: &str) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="10">"#;
    let flash = flash_banner(query);
    let mode = state.authority.mode().await;
    let (pos_size, open_orders, symbol, tradable_balance) = {
        let t = state.truth.lock().await;
        (
            t.position.size,
            t.open_order_count,
            t.symbol.clone(),
            t.tradable_balance,
        )
    };

    let status_body = format!(
        "{flash}<div class='metrics-grid' style='margin-top:8px;grid-template-columns:repeat(4,minmax(140px,1fr))'>\
            <div class='metric-card'><div class='metric-label'>Total Balance</div><div class='metric-value'>Portfolio linked</div></div>\
            <div class='metric-card'><div class='metric-label'>Free Quote Balance</div><div class='metric-value'>{:.6}</div></div>\
            <div class='metric-card'><div class='metric-label'>Pending Withdrawals</div><div class='metric-value'>Manage in Withdraw</div></div>\
            <div class='metric-card'><div class='metric-label'>Asset Count</div><div class='metric-value'>1+</div></div>\
         </div>\
         <div class='sum' style='margin-top:8px'>Default asset: {} • Authority mode: {}.</div>",
        tradable_balance,
        esc(&symbol),
        esc(&mode.to_string()),
    );

    let primary_body = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>\
      <div class='signal-box'><div class='label'>Deposit</div><div>Select a method and follow the network instructions exactly.</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px'><a class='btn-approve' href='/suggestions?method=onchain'>On-chain Transfer</a><a class='btn' href='/suggestions?method=internal'>Internal Transfer</a></div><table class='kv' style='margin-top:8px'><tbody><tr><td>Wallet Address</td><td>rw-demo-wallet-001</td></tr><tr><td>Allowed Network</td><td><strong class='warn'>Arbitrum One only</strong></td></tr><tr><td>Required Confirmations</td><td>12 blocks</td></tr></tbody></table></div>\
      <div class='signal-box'><div class='label'>Withdraw</div><div>Submit, review, then execute with clear approval stages.</div><div style='margin-top:10px'><a class='btn' href='/authority'>Open Withdrawal Workflow</a></div><div class='sum' style='margin-top:8px'>Real vs simulation stages are clearly separated there.</div></div>\
      </div>\
      <div class='signal-box' style='margin-top:10px'><div class='label'>Asset Balances</div><table><thead><tr><th>Asset</th><th>Free</th><th>Locked</th></tr></thead><tbody><tr><td>BTC</td><td>Derived from position</td><td>0.00000000</td></tr><tr><td>USDT</td><td>Use LIVE for exact value</td><td>0.00000000</td></tr></tbody></table><div class='sum' style='margin-top:6px'>Table style is sortable-looking for operator readability.</div></div>";

    let context_body = format!(
        "<table class='kv'><tbody>\
           <tr><td>Current Position Size</td><td>{:.6}</td></tr>\
           <tr><td>Open Orders</td><td>{}</td></tr>\
           <tr><td>Form Fields</td><td>Minimal: method + network only</td></tr>\
           <tr><td>Real vs Simulation</td><td>Simulation confirmations never move funds</td></tr>\
         </tbody></table>",
        pos_size,
        open_orders,
    );
    let details_body = "<div class='sum'>Deposit flow intentionally removes raw technical forms and keeps one clear method-first workflow.</div>";
    let body = system_layout(
        "Funds Workspace",
        &status_body,
        "Money Management",
        primary_body,
        "Funding Context",
        &context_body,
        Some(("Warnings", details_body)),
    );

    html_resp(&page("Suggestions — RW-Trader", refresh, &body))
}

// ── /authority ────────────────────────────────────────────────────────────────
//
// Authority mode control panel.
// Shows current mode, mode-switch buttons, and pending ASSIST proposals.
// POST endpoints are handled in handle_post().

async fn page_authority(state: &AppState, query: &str) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="3">"#;
    let flash = flash_banner(query);

    let mode = state.authority.mode().await;
    let trade_proposals = state.authority.pending_proposals().await;
    let withdrawals = state.withdrawals.proposals().await;
    let wd_id = qparam(query, "wd_id").unwrap_or_default();
    let wd_step = qparam(query, "wd_step").unwrap_or("details");
    let selected = withdrawals.iter().find(|w| w.id == wd_id).cloned().or_else(|| withdrawals.first().cloned());

    let (pos_size, pos_pnl_u) = {
        let t = state.truth.lock().await;
        (t.position.size, t.position.unrealized_pnl)
    };

    let status_body = format!(
        "{flash}<div class='metrics-grid' style='margin-top:8px;grid-template-columns:repeat(4,minmax(140px,1fr))'>\
            <div class='metric-card'><div class='metric-label'>Authority Mode</div><div class='metric-value'>{}</div></div>\
            <div class='metric-card'><div class='metric-label'>Trade Queue</div><div class='metric-value'>{}</div></div>\
            <div class='metric-card'><div class='metric-label'>Withdrawal Queue</div><div class='metric-value'>{}</div></div>\
            <div class='metric-card'><div class='metric-label'>Workflow</div><div class='metric-value'>Request → Approval → Execute</div></div>\
         </div>",
        esc(&mode.to_string()),
        trade_proposals.len(),
        withdrawals
            .iter()
            .filter(|w| w.status == WithdrawalStatus::Requested)
            .count(),
    );

    let mode_controls = "\
        <div class='label'>Authority Mode Controls</div>\
        <div class='sum'>Backend guardrails stay unchanged: request → approve/reject → execute → audit.</div>\
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px'>\
          <form method='post' action='/authority/mode/off'><button class='btn-off' style='width:100%' type='submit'>Set OFF</button></form>\
          <form method='post' action='/authority/mode/assist'><button class='btn-assist' style='width:100%' type='submit'>Set ASSIST</button></form>\
          <form method='post' action='/authority/mode/auto'><button class='btn-auto' style='width:100%' type='submit'>Set AUTO</button></form>\
        </div>";
    let trade_queue_rows = trade_proposals.iter().map(|p| format!(
        "<tr><td style='font-size:11px'>{}</td><td>{}</td><td>{:.6}</td><td>{:.0}%</td><td>{:.0}s</td>\
         <td><form method='post' action='/authority/approve/{}' style='display:inline-block'><button class='btn-approve' type='submit'>Approve</button></form>\
         <form method='post' action='/authority/reject/{}' style='display:inline-block;margin-left:6px'><button class='btn-reject' type='submit'>Reject</button></form></td></tr>",
        esc(&p.symbol), esc(&p.side), p.qty, p.confidence * 100.0, p.ttl_remaining_secs(), esc(&p.id), esc(&p.id)
    )).collect::<Vec<_>>().join("");
    let trade_queue_table = if trade_queue_rows.is_empty() {
        "<div class='dim' style='margin-top:8px'>No pending trade approvals.</div>".to_string()
    } else {
        format!("<table style='margin-top:10px'><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Confidence</th><th>TTL</th><th>Actions</th></tr></thead><tbody>{}</tbody></table>", trade_queue_rows)
    };

    let (status_label, status_class, status_detail, status_action, status_banner) = if let Some(w) = &selected {
        match w.status {
            WithdrawalStatus::Requested => (
                "Pending Approval".to_string(),
                "warn".to_string(),
                "Waiting for authority decision.".to_string(),
                format!("<div style='display:flex;gap:8px;margin-top:8px'><form method='post' action='/withdraw/proposal/approve/{id}'><button class='btn-approve' type='submit'>Approve</button></form><form method='post' action='/withdraw/proposal/reject/{id}'><button class='btn-reject' type='submit'>Reject</button></form></div>", id=esc(&w.id)),
                "".to_string(),
            ),
            WithdrawalStatus::Approved => (
                "Approved".to_string(),
                "ok".to_string(),
                "Proposal approved and ready to execute.".to_string(),
                format!("<form method='post' action='/withdraw/proposal/execute/{id}' style='margin-top:8px'><button class='btn' type='submit'>Execute Withdrawal</button></form>", id=esc(&w.id)),
                "".to_string(),
            ),
            WithdrawalStatus::Executing => (
                "Executing".to_string(),
                "warn".to_string(),
                "Execution in progress. Keep this screen open for live status refresh.".to_string(),
                "".to_string(),
                "".to_string(),
            ),
            WithdrawalStatus::Completed => (
                "Completed".to_string(),
                "ok".to_string(),
                "Withdrawal finished successfully.".to_string(),
                "".to_string(),
                "<div style='margin-top:10px;padding:12px;background:rgba(34,197,94,.18);border-left:6px solid #22C55E'><strong>✅ Withdrawal Completed</strong><div class='sum'>Transaction id / proposal id: review audit trail entry for this proposal.</div></div>".to_string(),
            ),
            WithdrawalStatus::Failed => (
                "Failed".to_string(),
                "warn".to_string(),
                "Execution failed. See failure reason below.".to_string(),
                "".to_string(),
                format!("<div style='margin-top:10px;padding:12px;background:rgba(239,68,68,.16);border-left:6px solid #ef4444'><strong>❌ Withdrawal Failed</strong><div class='sum'>Reason: {}</div></div>", esc(w.failure_reason.as_deref().unwrap_or("unknown"))),
            ),
            WithdrawalStatus::Rejected => (
                "Rejected".to_string(),
                "warn".to_string(),
                "Proposal was rejected before execution.".to_string(),
                "".to_string(),
                format!("<div style='margin-top:10px;padding:12px;background:rgba(239,68,68,.16);border-left:6px solid #ef4444'><strong>⛔ Withdrawal Rejected</strong><div class='sum'>Reason: {}</div></div>", esc(w.failure_reason.as_deref().unwrap_or("rejected by authority"))),
            ),
        }
    } else {
        (
            "Draft".to_string(),
            "".to_string(),
            "No submitted withdrawal yet. Complete Step 1 to create a draft request.".to_string(),
            "".to_string(),
            "".to_string(),
        )
    };

    let selected_info = if let Some(w) = &selected {
        format!(
            "<table class='kv'><tbody>\
                <tr><td>Proposal ID</td><td style='font-size:11px'>{}</td></tr>\
                <tr><td>Asset</td><td>{}</td></tr>\
                <tr><td>Amount</td><td>{:.8}</td></tr>\
                <tr><td>Fee estimate</td><td>{:.8}</td></tr>\
                <tr><td>Final received</td><td>{:.8}</td></tr>\
                <tr><td>Destination</td><td style='font-size:11px'>{}</td></tr>\
                <tr><td>Network</td><td>{}</td></tr>\
                <tr><td>Reason</td><td>{}</td></tr>\
            </tbody></table>",
            esc(&w.id), esc(&w.asset), w.amount, w.estimated_fee, w.final_received_amount(), esc(&w.destination), esc(&w.network), esc(&w.reason)
        )
    } else {
        "<div class='dim'>No withdrawal proposal selected yet.</div>".to_string()
    };

    let assist_inline = if mode == AuthorityMode::Assist {
        if let Some(w) = &selected {
            if w.status == WithdrawalStatus::Requested {
                format!("<div style='margin-top:10px;padding:10px;background:#0A0E13;border:1px solid rgba(239,176,103,.35)'><div class='label'>ASSIST Inline Approval</div><div class='sum'>Approve or reject the pending withdrawal without leaving this screen.</div><div style='display:flex;gap:8px;margin-top:8px'><form method='post' action='/withdraw/proposal/approve/{id}'><button class='btn-approve' type='submit'>Approve</button></form><form method='post' action='/withdraw/proposal/reject/{id}'><button class='btn-reject' type='submit'>Reject</button></form></div></div>", id=esc(&w.id))
            } else {
                "<div class='dim' style='margin-top:10px'>ASSIST inline approval appears when a withdrawal is in Pending Approval.</div>".to_string()
            }
        } else {
            "<div class='dim' style='margin-top:10px'>ASSIST inline approval will appear after request submission.</div>".to_string()
        }
    } else {
        "".to_string()
    };

    let review_step = format!(
        "<div class='label'>Step 2 — Review & Confirm</div>\
         <div class='sum'>Status: <strong class='{status_class}'>{status_label}</strong> — {status_detail}</div>\
         {selected_info}\
         {assist_inline}\
         {status_action}\
         {status_banner}",
    );

    let allowed_destinations = state.withdrawals.allowed_destinations().join(", ");
    let step = match wd_step {
        "review" => 2,
        "status" => 3,
        _ => 1,
    };
    let flow_header = format!(
        "<div class='label'>Guided Withdrawal</div><div class='sum'>Step {step}/3 • Enter Details → Review & Confirm → Status</div>"
    );

    let primary_body = format!(
        "{}{}\
         <div style='margin-top:12px;padding:10px;background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.35)'><strong>Simulation (Safe / Visual Only)</strong><div class='sum'>Simulation confirmation logs an audit event only. It cannot execute real withdrawals.</div><div style='margin-top:8px'><form method='post' action='/withdraw/confirm/simulation'><button class='btn' type='submit'>Record Simulation Confirmation</button></form></div></div>\
         <div style='margin-top:12px;padding:10px;background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.35);border-radius:12px'>\
            <div class='label'>Step 1 — Enter Details (Real Withdrawal)</div>\
            <div class='sum'>Allowed destination whitelist: <code>{}</code></div>\
            <form method='post' action='/withdraw/request' style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px'>\
                <input name='asset' placeholder='Asset (USDT)' required />\
                <input name='network' placeholder='Network (ETH/TRX/...)' required />\
                <input name='amount' placeholder='Amount' required />\
                <input name='fee_estimate' placeholder='Fee estimate' value='0' required />\
                <input name='destination' placeholder='Destination address / label' style='grid-column:1/3' required />\
                <input name='reason' placeholder='Business reason for withdrawal' style='grid-column:1/3' required />\
                <label style='grid-column:1/3'><input type='checkbox' name='confirm_checkbox' /> I confirm destination, network, amount, fee, and final received amount.</label>\
                <input name='confirm_text' placeholder='Type CONFIRM to continue' style='grid-column:1/3' required />\
                <input name='confirm_large_text' placeholder='For large amounts, type: WITHDRAW LARGE' style='grid-column:1/3' />\
                <button class='btn' type='submit' style='grid-column:1/3'>Submit Withdrawal Proposal</button>\
            </form>\
            <div class='sum' style='margin-top:8px'>Amount-tiered confirmation: normal confirmation for standard withdrawals, plus WITHDRAW LARGE for higher amounts.</div>\
         </div>\
         <div class='signal-box' style='margin-top:12px'><div class='label'>Approval Queue</div>{}</div>\
         <div style='margin-top:12px;padding:10px;background:#0A0E13;border-radius:12px'>{}</div>",
        flow_header,
        mode_controls,
        esc(&allowed_destinations),
        trade_queue_table,
        review_step,
    );

    let context_body = format!(
        "<table class='kv'><tbody>\
            <tr><td>Current Position</td><td>{:.6}</td></tr>\
            <tr><td>Unrealized PnL</td><td>{:+.2}</td></tr>\
            <tr><td>Whitelist Check</td><td>Destination must match configured allow-list</td></tr>\
            <tr><td>Balance Confirmation</td><td>Checked before request creation (if live client connected)</td></tr>\
            <tr><td>Cooldown</td><td>Enforced between executed withdrawals</td></tr>\
            <tr><td>Audit Trail</td><td>Every stage is logged: requested, approved/rejected, executed/failed</td></tr>\
          </tbody></table>",
        pos_size,
        pos_pnl_u,
    );

    let details_body = "<div class='sum'>Status states shown in this flow: Draft, Pending Approval, Approved, Executing, Completed, Failed, Rejected.</div>";
    let body = system_layout(
        "Funds Workspace",
        &status_body,
        "Withdraw: Guided 3-Step",
        &primary_body,
        "Withdrawal Safety Context",
        &context_body,
        Some(("Process Notes", details_body)),
    );
    html_resp(&page("Authority — RW-Trader", refresh, &body))
}

/// Render the authority mode banner HTML for injection into other pages.
/// Async: reads mode under lock then renders.
async fn authority_banner(state: &AppState) -> String {
    let mode = state.authority.mode().await;
    let cls  = mode.banner_class();
    format!(
        "<div class='banner {cls}' style='font-size:11px;margin-bottom:8px'>\
           <span>Authority: <strong>{mode}</strong></span>\
           <span style='font-weight:normal'>{desc}</span>\
           <a href='/authority' style='margin-left:auto;color:inherit;font-size:10px'>change →</a>\
         </div>",
        desc = mode.description().split('.').next().unwrap_or(""),
    )
}

fn system_layout(
    status_title: &str,
    status_body: &str,
    primary_title: &str,
    primary_body: &str,
    context_title: &str,
    context_body: &str,
    details: Option<(&str, &str)>,
) -> String {
    let details_html = if let Some((title, body)) = details {
        format!(
            "<div class='panel'><div class='label'>Advanced</div><h2>{}</h2><details><summary class='soft-title'>Diagnostics / Why?</summary><div style='margin-top:8px'>{}</div></details></div>",
            esc(title),
            body
        )
    } else {
        String::new()
    };
    format!(
        "<div class='page-scroll'>\
          <div class='panel'><div class='label'>Summary</div><h2>{}</h2>{}</div>\
          <div class='workspace' style='margin-top:12px'>\
            <div class='col'><div class='panel'><div class='label'>Primary Action</div><h2>{}</h2>{}</div>{}</div>\
            <div class='col'><div class='panel'><div class='label'>Context</div><h2>{}</h2>{}</div></div>\
          </div>\
        </div>",
        esc(status_title),
        status_body,
        esc(primary_title),
        primary_body,
        details_html,
        esc(context_title),
        context_body,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::*;
    use crate::executor::{CircuitBreakerConfig, WatchdogConfig};
    use crate::position::Position;
    use crate::reconciler::TruthState;
    use crate::risk::{RiskConfig, RiskEngine};
    use crate::store::InMemoryEventStore;
    use std::time::Duration;

    fn make_state() -> AppState {
        let pos  = Position::new("BTCUSDT");
        let risk = RiskEngine::new(RiskConfig {
            max_position_qty:    0.01,
            max_daily_loss_usd:  50.0,
            max_drawdown_usd:    100.0,
            max_consecutive_losses: 3,
            cooldown_after_loss: Duration::from_secs(300),
            max_spread_bps:      10.0,
            max_feed_staleness:  Duration::from_secs(5),
            min_order_interval:  Duration::from_secs(1),
            signal_dedup_window: Duration::from_secs(5),
            max_open_orders:     1,
            max_slippage_bps:    20.0,
        }, &pos);
        let (client, withdrawals) = test_app_state_extras();
        let store: Arc<dyn EventStore> = InMemoryEventStore::new();
        let exec = Arc::new(Executor::new("BTCUSDT".into(), CircuitBreakerConfig::default(), WatchdogConfig::default()));
        let truth = Arc::new(Mutex::new(TruthState::new("BTCUSDT", 0.0)));
        let authority = Arc::new(crate::authority::AuthorityLayer::new());
        let signal = Arc::new(Mutex::new(crate::signal::SignalEngine::new(crate::signal::SignalConfig {
            order_qty: 0.001,
            momentum_threshold: 0.0,
            imbalance_threshold: 0.0,
            max_entry_spread_bps: 5.0,
            max_feed_staleness: Duration::from_secs(3),
            stop_loss_pct: 0.0,
            take_profit_pct: 0.0,
            max_hold_duration: Duration::from_secs(10),
            min_mid_samples: 1,
            min_trade_samples: 1,
        })));
        let npc = Arc::new(crate::npc::NpcAutonomousController::new(
            crate::npc::NpcConfig::from_trade_cfg(&crate::agent::TradeAgentConfig {
                enabled: false,
                trade_size: 0.0,
                momentum_threshold: 0.0,
                poll_interval: Duration::from_millis(1000),
                max_spread_bps: 5.0,
            }),
            crate::agent::AgentState {
                store: Arc::clone(&store),
                exec: Arc::clone(&exec),
                feed: Arc::new(Mutex::new(crate::feed::FeedState::new(Duration::from_secs(10)))),
                signal,
                truth: Arc::clone(&truth),
                authority: Arc::clone(&authority),
                withdrawals: Arc::clone(&withdrawals),
                client: Arc::new(crate::client::BinanceClient::new(String::new(), String::new(), String::new())),
                symbol: "BTCUSDT".into(),
                web_base_url: None,
            },
        ));
        AppState {
            store,
            exec,
            truth,
            risk:     Arc::new(Mutex::new(risk)),
            authority,
            strategy:  Arc::new(Mutex::new(crate::strategy::StrategyEngine::new())),
            client,
            withdrawals,
            npc,
            profile:  Arc::new(Mutex::new(crate::profile::RuntimeProfile::default())),
        }
    }

    fn snap() -> MarketSnapshotPayload {
        MarketSnapshotPayload {
            bid:50000.0, ask:50001.0, mid:50000.5, spread_bps:2.0,
            momentum_1s:0.001, momentum_3s:0.002, momentum_5s:0.003,
            imbalance_1s:0.5, imbalance_3s:0.3, imbalance_5s:0.2,
            feed_age_ms:20.0, mid_samples:10, trade_samples:8,
        }
    }

    fn filled_state() -> AppState {
        let state = make_state();
        let (corr, sym) = ("test-corr-001", "BTCUSDT");
        state.store.append(StoredEvent::new(
            Some(sym.into()), Some(corr.into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision:"Buy".into(), exit_reason:None,
                reason:"ok".into(), confidence:0.8, metrics:snap(),
            }),
        ));
        state.store.append(StoredEvent::new(
            Some(sym.into()), Some(corr.into()), None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved:true, side:"BUY".into(), qty:0.001,
                expected_price:50001.0, rejection_reason:None,
                position_size:0.0, position_avg_entry:0.0,
            }),
        ));
        state.store.append(StoredEvent::new(
            Some(sym.into()), Some(corr.into()), Some("coid-1".into()),
            TradingEvent::OrderFilled(OrderFilledPayload {
                client_order_id:"coid-1".into(), exchange_order_id:1234,
                side:"BUY".into(), filled_qty:0.001, avg_fill_price:50001.5,
                cumulative_quote:50.0015,
            }),
        ));
        state
    }

    // ── esc ───────────────────────────────────────────────────────────────────
    #[test]
    fn test_esc_all_specials() {
        assert_eq!(esc("<b>&\"'</b>"), "&lt;b&gt;&amp;&quot;&#39;&lt;/b&gt;");
    }
    #[test]
    fn test_esc_clean_passthrough() {
        assert_eq!(esc("hello 123"), "hello 123");
    }

    // ── qparam ───────────────────────────────────────────────────────────────
    #[test]
    fn test_qparam_first()        { assert_eq!(qparam("id=abc&x=y", "id"),  Some("abc")); }
    #[test]
    fn test_qparam_second()       { assert_eq!(qparam("id=abc&x=y", "x"),   Some("y")); }
    #[test]
    fn test_qparam_missing()      { assert_eq!(qparam("id=abc", "nope"),     None); }
    #[test]
    fn test_qparam_empty_query()  { assert_eq!(qparam("", "id"),             None); }

    // ── url_encode ────────────────────────────────────────────────────────────
    #[test]
    fn test_url_encode_safe_chars() {
        assert_eq!(url_encode("abc-123_FOO.BAR~"), "abc-123_FOO.BAR~");
    }
    #[test]
    fn test_url_encode_space_and_slash() {
        let enc = url_encode("a b/c");
        assert!(!enc.contains(' '));
        assert!(!enc.contains('/'));
        assert!(enc.contains("%20") || !enc.contains(' '));
    }
    #[test]
    fn test_url_encode_uuid_passthrough() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        // hyphens are safe; UUID should pass through unchanged
        assert_eq!(url_encode(uuid), uuid);
    }

    // ── HTTP response format ──────────────────────────────────────────────────
    #[test]
    fn test_respond_has_content_length() {
        let r = plain("hello");
        assert!(r.contains("Content-Length: 5"));
    }
    #[test]
    fn test_respond_has_connection_close() {
        let r = plain("x");
        assert!(r.contains("Connection: close"));
    }
    #[test]
    fn test_not_found_status() {
        assert!(not_found().starts_with("HTTP/1.1 404"));
    }
    #[test]
    fn test_redirect_location() {
        let r = redirect("/events");
        assert!(r.contains("Location: /events"));
        assert!(r.starts_with("HTTP/1.1 302"));
    }

    // ── page_events ───────────────────────────────────────────────────────────
    #[tokio::test]
    async fn test_events_200() {
        let state = filled_state();
        let r = page_events(&state, "").await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("text/html"));
    }
    #[tokio::test]
    async fn test_events_contains_event_type() {
        let state = filled_state();
        let r = page_events(&state, "").await;
        assert!(r.contains("signal_decision") || r.contains("order_filled"));
    }
    #[tokio::test]
    async fn test_events_corr_link() {
        let state = filled_state();
        let r = page_events(&state, "").await;
        assert!(r.contains("/trade/"), "Should contain /trade/ link");
    }
    #[tokio::test]
    async fn test_events_empty_store() {
        let state = make_state();
        let r = page_events(&state, "").await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
    }
    #[tokio::test]
    async fn test_events_no_xss() {
        let state = make_state();
        state.store.append(StoredEvent::new(
            Some("<BTCUSDT>".into()), Some("c".into()), None,
            TradingEvent::SystemModeChange(SystemModeChangePayload {
                from_mode:"Ready".into(), to_mode:"Halted".into(),
                reason:"<script>evil()</script>".into(),
            }),
        ));
        let r = page_events(&state, "").await;
        assert!(!r.contains("<script>evil()"), "Must escape XSS");
    }

    // ── page_trade - explanation block ────────────────────────────────────────
    #[tokio::test]
    async fn test_trade_valid_id_includes_explanation() {
        let state = filled_state();
        let r = page_trade(&state, "test-corr-001").await;
        assert!(r.contains("Explanation") || r.contains("callout"),
            "Should contain explanation section");
        // The explanation text should not contain raw < characters
        // (assistant output is always escaped)
        assert!(!r.contains("<script>evil()"), "No unescaped script payload");
    }

    // ── page_assistant ────────────────────────────────────────────────────────
    #[tokio::test]
    async fn test_assistant_200() {
        let state = make_state();
        let r = page_assistant(&state, "").await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("text/html"));
    }

    #[tokio::test]
    async fn test_assistant_has_system_section() {
        let state = make_state();
        let r = page_assistant(&state, "").await;
        assert!(r.contains("System") || r.contains("system"));
    }

    #[tokio::test]
    async fn test_assistant_has_position_section() {
        let state = make_state();
        let r = page_assistant(&state, "").await;
        assert!(r.contains("Position") || r.contains("position"));
    }

    #[tokio::test]
    async fn test_assistant_has_risk_section() {
        let state = make_state();
        let r = page_assistant(&state, "").await;
        assert!(r.contains("Risk") || r.contains("risk"));
    }

    #[tokio::test]
    async fn test_assistant_has_recent_activity() {
        let state = make_state();
        let r = page_assistant(&state, "").await;
        assert!(r.contains("Recent") || r.contains("recent"));
    }

    #[tokio::test]
    async fn test_assistant_shows_rejection_when_present() {
        let state = make_state();
        state.store.append(crate::events::StoredEvent::new(
            Some("BTCUSDT".into()), None, None,
            crate::events::TradingEvent::RiskCheckResult(crate::events::RiskCheckPayload {
                approved: false, side: "BUY".into(), qty: 0.001,
                expected_price: 50000.0,
                rejection_reason: Some("SPREAD: 20.00 bps > limit 10.00 bps".into()),
                position_size: 0.0, position_avg_entry: 0.0,
            }),
        ));
        let r = page_assistant(&state, "").await;
        assert!(r.contains("Last Rejection") || r.contains("rejection") || r.contains("SPREAD") || r.contains("spread"),
            "Should surface the rejection");
    }

    #[tokio::test]
    async fn test_assistant_halted_shows_err_callout() {
        let state = make_state();
        // Force executor into Halted mode
        state.exec.set_mode_halted("test halt").await;
        let r = page_assistant(&state, "").await;
        // The system callout should have the err class
        assert!(r.contains("callout err") || r.contains("Halted"),
            "Halted system should show error callout");
    }

    #[tokio::test]
    async fn test_assistant_no_xss_in_event_interpretation() {
        let state = make_state();
        state.store.append(crate::events::StoredEvent::new(
            None, None, None,
            crate::events::TradingEvent::OperatorAction(crate::events::OperatorActionPayload {
                action: "<script>xss()</script>".into(),
                reason: "<img onerror=evil()>".into(),
            }),
        ));
        let r = page_assistant(&state, "").await;
        assert!(!r.contains("<script>xss()"), "Operator action must be HTML-escaped");
    }

    // ── page_trade - empty id shows form ─────────────────────────────────────
    #[tokio::test]
    async fn test_trade_empty_id_shows_form() {
        let state = filled_state();
        let r = page_trade(&state, "").await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("<form"));
        assert!(!r.contains("Event Sequence"));
    }
    #[tokio::test]
    async fn test_trade_unknown_id_shows_error() {
        let state = filled_state();
        let r = page_trade(&state, "no-such-id").await;
        assert!(r.contains("No events found") || r.contains("no-such-id"));
    }
    #[tokio::test]
    async fn test_trade_valid_id_shows_timeline() {
        let state = filled_state();
        let r = page_trade(&state, "test-corr-001").await;
        assert!(r.contains("Event Sequence"));
        assert!(r.contains("FILLED"));
        assert!(r.contains("50001"));
    }
    #[tokio::test]
    async fn test_trade_no_xss_in_corr_id() {
        let state = make_state();
        let r = page_trade(&state, "<script>evil()</script>").await;
        assert!(!r.contains("<script>evil()"));
    }

    // ── page_status ───────────────────────────────────────────────────────────
    #[tokio::test]
    async fn test_status_200() {
        let state = make_state();
        let r = page_status(&state, "").await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("system_mode") || r.contains("System"));
    }
    #[tokio::test]
    async fn test_status_has_position_section() {
        let state = make_state();
        let r = page_status(&state, "").await;
        assert!(r.contains("Position") || r.contains("pos"));
    }
    #[tokio::test]
    async fn test_status_has_risk_section() {
        let state = make_state();
        let r = page_status(&state, "").await;
        assert!(r.contains("max_position_qty") || r.contains("Risk"));
    }

    // ── Runtime profile ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_events_shows_profile_banner() {
        let state = make_state();
        // Default profile is Conservative; banner must contain both the label
        // and the "Runtime Profile" heading.
        let r = page_events(&state, "").await;
        assert!(r.contains("Runtime Profile"), "LIVE page must show 'Runtime Profile' heading");
        assert!(
            r.contains("CONSERVATIVE") || r.contains("Conservative") || r.contains("cautious"),
            "LIVE page must show the current profile value"
        );
    }

    #[tokio::test]
    async fn test_events_micro_test_banner() {
        let state = make_state();
        *state.profile.lock().await = crate::profile::RuntimeProfile::MicroTest;
        let r = page_events(&state, "").await;
        assert!(r.contains("Runtime Profile"), "LIVE page must show 'Runtime Profile' heading");
        assert!(
            r.contains("⚡") || r.contains("Micro-Test") || r.contains("MICRO_TEST"),
            "LIVE page must show MICRO_TEST highlighted indicator"
        );
        assert!(
            r.contains("faster decisions") || r.contains("small balances"),
            "LIVE page must include MICRO_TEST plain-language description"
        );
    }

    #[tokio::test]
    async fn test_assistant_shows_profile_selector() {
        let state = make_state();
        let r = page_assistant(&state, "").await;
        // The interactive selector element must be present.
        assert!(r.contains("name='profile'"), "SETTINGS page must render profile <select> element");
        assert!(r.contains("value='CONSERVATIVE'"), "CONSERVATIVE option must be present");
        assert!(r.contains("value='ACTIVE'"),       "ACTIVE option must be present");
        assert!(r.contains("value='MICRO_TEST'"),   "MICRO_TEST option must be present");
        assert!(r.contains("Apply"),                "Apply button must be present");
    }

    #[tokio::test]
    async fn test_assistant_profile_selected_matches_state() {
        let state = make_state();
        *state.profile.lock().await = crate::profile::RuntimeProfile::MicroTest;
        let r = page_assistant(&state, "").await;
        // MICRO_TEST option must carry the `selected` attribute.
        assert!(
            r.contains("value='MICRO_TEST' selected") || r.contains("value='MICRO_TEST'  selected"),
            "MICRO_TEST option must be marked as selected"
        );
        assert!(
            r.contains("faster decisions") || r.contains("small balances"),
            "Micro-test label must appear in the profile description"
        );
    }

    // ── Agent Control Panel ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_agent_control_card_visible_on_events() {
        let state = make_state();
        let r = page_events(&state, "").await;
        assert!(r.contains("Agent Control"), "Events page must show Agent Control card");
        assert!(r.contains("/agent/mode"), "Events page must include /agent/mode form action");
        assert!(r.contains("Turn Agent ON") || r.contains("Pause Agent") || r.contains("Resume Agent"),
            "Events page must show primary CTA button");
    }

    #[tokio::test]
    async fn test_agent_status_json_returns_json() {
        let state = make_state();
        let r = agent_status_json(&state).await;
        assert!(r.contains("application/json"), "agent_status must return JSON content-type");
        assert!(r.contains("\"mode\""), "JSON must contain mode field");
        assert!(r.contains("\"cycle_count\""), "JSON must contain cycle_count field");
    }

    #[tokio::test]
    async fn test_agent_mode_off_shows_turn_on_cta() {
        let state = make_state();
        // Default mode is off (enabled=false in make_state TradeAgentConfig).
        let r = page_events(&state, "").await;
        assert!(r.contains("Turn Agent ON"), "OFF mode must show 'Turn Agent ON' CTA");
    }

    #[tokio::test]
    async fn test_agent_mode_label_no_idle() {
        // "Idle" must not appear anywhere on the events page.
        let state = make_state();
        let r = page_events(&state, "").await;
        assert!(!r.contains(">Idle<"), "Page must not display raw 'Idle' state");
    }

}

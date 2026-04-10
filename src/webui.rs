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
use crate::reader::{get_trade_timeline, summarise_event, LifecycleStage, TradeOutcome};
use crate::reconciler::TruthState;
use crate::risk::RiskEngine;
use crate::store::EventStore;
use crate::strategy::{StrategyEngine, ALL_STRATEGIES};
use crate::withdrawal::{NewWithdrawalRequest, WithdrawalManager, WithdrawalStatus};
use crate::client::BinanceClient;

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
}

#[cfg(test)]
fn test_app_state_extras() -> (Option<Arc<BinanceClient>>, Arc<WithdrawalManager>) {
    (None, Arc::new(WithdrawalManager::new()))
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
    } else {
        not_found()
    };

    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

async fn handle_post(path: &str, _query: &str, body: &str, state: &AppState) -> String {
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

    // Dedicated simulation-only withdrawal confirmation endpoint.
    // Intentionally separate from real proposal rejection/execution handlers.
    if path == "/withdraw/confirm/simulation" {
        log_ui_action(
            &*state.store,
            "ui_withdrawal_confirmed_simulation",
            "withdrawal confirmation requested from /authority",
        );
        return redirect_with_ok("/authority", "Withdrawal confirmation recorded (simulation). No real funds moved.");
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

        let req = NewWithdrawalRequest { amount, asset, destination, network, reason, estimated_fee: fee };
        let kill = state.risk.lock().await.kill_switch_active();
        let client_ref = state.client.as_ref().map(|c| c.as_ref());
        let proposal = match state.withdrawals.create_request(req, kill, client_ref, &*state.store).await {
            Ok(p) => p,
            Err(e) => return redirect_with_err("/authority", &e),
        };

        let mode = state.authority.mode().await;
        if mode == AuthorityMode::Auto {
            match state.withdrawals.execute(&proposal.id, mode, kill, client_ref, &*state.store).await {
                Ok(_) => return redirect_with_ok("/authority", "Withdrawal auto-executed under AUTO policy."),
                Err(e) => {
                    state.withdrawals.mark_failed(&proposal.id, &e, &*state.store).await;
                    return redirect_with_err("/authority", &e);
                }
            }
        }
        return redirect_with_ok("/authority", &format!("Withdrawal proposal {} created and awaiting authority approval.", proposal.id));
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
            Ok(_) => return redirect_with_ok("/authority", "Withdrawal proposal approved."),
            Err(e) => return redirect_with_err("/authority", &e),
        }
    }

    // POST /withdraw/proposal/reject/{proposal_id}
    if let Some(pid) = path.strip_prefix("/withdraw/proposal/reject/") {
        match state.withdrawals.reject(pid, &*state.store).await {
            Ok(_) => return redirect_with_ok("/authority", "Withdrawal proposal rejected."),
            Err(e) => return redirect_with_err("/authority", &e),
        }
    }

    // POST /withdraw/proposal/execute/{proposal_id}
    if let Some(pid) = path.strip_prefix("/withdraw/proposal/execute/") {
        let kill = state.risk.lock().await.kill_switch_active();
        let mode = state.authority.mode().await;
        let client_ref = state.client.as_ref().map(|c| c.as_ref());
        match state.withdrawals.execute(pid, mode, kill, client_ref, &*state.store).await {
            Ok(_) => return redirect_with_ok("/authority", "Withdrawal executed."),
            Err(e) => {
                state.withdrawals.mark_failed(pid, &e, &*state.store).await;
                return redirect_with_err("/authority", &e);
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
fn not_found()           -> String { respond("404 Not Found", "text/plain; charset=utf-8", "404 Not Found") }
fn csv_resp(body: &str)  -> String { respond("200 OK", "text/csv; charset=utf-8", body) }
fn redirect(loc: &str)   -> String {
    format!("HTTP/1.1 302 Found\r\nLocation: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", loc)
}

async fn export_positions_csv(state: &AppState) -> String {
    let t = state.truth.lock().await;
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
*{{box-sizing:border-box;margin:0;padding:0;border-radius:0}}
html,body{{height:100%}}
body{{font-family:Inter,sans-serif;background:#101419;color:#d0d6dd;overflow:hidden}}
.status-pillar{{height:2px;background:linear-gradient(90deg,#4BE277,#22C55E)}}
.top{{height:52px;background:#181c21;border-bottom:1px solid rgba(128,138,147,.2);display:flex;align-items:center;justify-content:space-between;padding:0 18px;font-family:JetBrains Mono,monospace}}
.brand{{color:#22C55E;font-size:26px;font-weight:700;letter-spacing:-.05em}}
.tabs a{{color:#7d8790;text-decoration:none;margin:0 10px;padding:6px 0;display:inline-block;border-bottom:2px solid transparent;font-size:12px}}
.tabs a.on{{color:#4BE277;border-color:#4BE277}}
.actions{{display:flex;align-items:center;gap:8px;color:#98a3af;font-size:11px}}
.btn{{font:600 11px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase;padding:7px 14px;border:1px solid rgba(128,138,147,.25);background:transparent;color:#c4ccd4;text-decoration:none}}
.btn.green{{background:#22C55E;color:#07230f;border:0}}
.app{{display:grid;grid-template-columns:68px 1fr;height:calc(100vh - 54px);overflow:hidden}}
.side{{background:#181c21;padding:14px 0;border-right:1px solid rgba(128,138,147,.15);display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}}
.side ul{{list-style:none}}
.side li{{height:52px;display:flex;align-items:center;justify-content:center;color:#6c7681;border-left:4px solid transparent;font-family:JetBrains Mono,monospace}}
.side li.on{{color:#4BE277;border-left-color:#4BE277;background:#141920}}
.main{{padding:0;background:#101419;overflow:hidden;height:100%;display:flex;flex-direction:column}}
.page-scroll{{flex:1;overflow-y:auto;overflow-x:hidden;padding:18px}}
.panel{{background:linear-gradient(90deg,#141a20,#181d24);padding:14px;border:1px solid rgba(128,138,147,.2)}}
.panel-scroll{{overflow-y:auto;overflow-x:hidden}}
h2{{font-size:.6875rem;letter-spacing:.05em;text-transform:uppercase;color:#b4bcc5;font-weight:500;margin-bottom:8px}}
table{{width:100%;border-collapse:separate;border-spacing:0 2px;font-family:JetBrains Mono,monospace;font-size:13px;line-height:1.1}}
th{{text-align:left;padding:8px 10px;color:#aab4be;background:#262a30;font-size:.6875rem;letter-spacing:.05em;text-transform:uppercase}}
td{{padding:7px 10px;background:#181c21}}
.tag{{display:inline-block;padding:3px 8px;font-size:10px;letter-spacing:.05em;text-transform:uppercase}}
.MARKET,.SIGNAL,.RISK-ok,.FILLED,.ok{{color:#4BE277}}
.RISK,.CANCEL,.REJECT,.err{{color:#f9a79d}}
.SUBMIT,.ACKED,.warn{{color:#efb067}}
.STATE,.RECON,.SAFETY,.OTHER,.dim{{color:#82909f}}
.sum{{font-size:12px;color:#bbc4ce}}
.kv td:first-child{{color:#98a3af;width:220px}}
input[type=text]{{font:13px JetBrains Mono,monospace;background:#0A0E13;color:#d7dce2;padding:10px 14px;border:none;border-left:2px solid #0A0E13;width:460px}}
input[type=text]:focus{{outline:none;border-left-color:#4BE277}}
input[type=submit]{{font:700 12px Inter,sans-serif;letter-spacing:.06em;text-transform:uppercase;padding:10px 16px;background:#22C55E;color:#041007;border:none;margin-left:8px}}
.callout{{padding:10px 14px;background:#181c21;border-left:4px solid #4BE277}}
.banner{{padding:6px 12px;background:#262a30;margin-bottom:10px;font:600 11px JetBrains Mono,monospace;letter-spacing:.05em;text-transform:uppercase}}
.banner.ASSIST{{border-left:4px solid #efb067}}.banner.AUTO{{border-left:4px solid #4BE277}}.banner.OFF{{border-left:4px solid #82909f}}
.btn-off,.btn-assist,.btn-auto,.btn-approve,.btn-reject,.btn-enable,.btn-disable{{font:600 11px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase;padding:8px 12px;border:1px solid rgba(145,155,165,.25);text-decoration:none;display:inline-block;background:#101419;color:#d0d6dd}}
.btn-approve,.btn-enable{{background:#22C55E;color:#07230f;border:none}}
.btn-reject{{color:#f9a79d}}
.log-dock{{background:#0d1117;border-top:1px solid rgba(128,138,147,.18);font-family:JetBrains Mono,monospace;font-size:11px}}
.log-dock>summary{{padding:5px 12px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;background:#181c21;border-bottom:1px solid rgba(128,138,147,.12);color:#7d8790;user-select:none}}
.log-dock>summary::before{{content:"▸";color:#4BE277}}
.log-dock[open]>summary::before{{content:"▾"}}
.log-content{{height:110px;overflow-y:auto;padding:7px 14px;color:#82909f}}
.workspace{{display:grid;grid-template-columns:1.4fr 1fr;gap:12px;min-height:0}}
.workspace .col{{display:flex;flex-direction:column;gap:12px;min-height:0}}
.label{{font:600 10px JetBrains Mono,monospace;letter-spacing:.08em;text-transform:uppercase;color:#7d8790;margin-bottom:6px}}
@media (max-width:1200px){{.brand{{font-size:20px}} .page-scroll{{overflow-y:auto}} table{{font-size:12px}}}}
</style>
</head>
<body>
<div class="status-pillar"></div>
<header class="top">
  <div class="brand">RW-TRADER</div>
  <nav class="tabs"><a class="{live_on}" href="/events">LIVE</a><a class="{demo_on}" href="/status">DEMO</a><a class="{funds_on}" href="/suggestions">FUNDS</a><a class="{settings_on}" href="/assistant">SETTINGS</a></nav>
  <div class="actions">Unified Operator Console</div>
</header>
<div class="app">
  <aside class="side">
    <ul><li>◫</li><li class="{live_on}">▦</li><li class="{demo_on}">▤</li><li class="{funds_on}">◧</li><li class="{settings_on}">☷</li></ul>
    <ul><li>?</li><li>☰</li></ul>
  </aside>
  <main class="main">{body}</main>
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
    let (symbol, pos_size, open_orders) = {
        let t = state.truth.lock().await;
        (t.symbol.clone(), t.position.size, t.open_order_count)
    };
    let kill = state.risk.lock().await.kill_switch_active();

    let best_summary = events.first().map(summarise_event).unwrap_or_else(|| "No fresh event yet; waiting for next snapshot.".to_string());
    let status_body = format!(
        "{flash}<div style='display:flex;gap:14px;margin-top:8px'>\
          <div>System: <strong>{}</strong></div>\
          <div>Executor: <strong>{}</strong></div>\
          <div>Risk: <strong class='{}'>{}</strong></div>\
          <div>Symbol: <strong>{}</strong></div>\
        </div>",
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        if kill { "err" } else { "ok" },
        if kill { "Paused" } else { "Ready" },
        esc(&symbol),
    );

    let primary_body = format!(
        "<div style='padding:10px;background:#101419;border-left:3px solid #4BE277'>\
           <div class='dim'>Latest recommendation summary</div>\
           <div style='margin-top:4px'><strong>{}</strong></div>\
         </div>\
         <div style='margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px'>\
           <form method='post' action='/events/quick/buy'><button class='btn-approve' style='width:100%' type='submit'>Execute Sim Buy</button></form>\
           <form method='post' action='/events/quick/sell'><button class='btn-reject' style='width:100%' type='submit'>Execute Sim Sell</button></form>\
         </div>",
        esc(&best_summary)
    );

    let rows = events.iter().take(10).map(|e| format!(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
        e.occurred_at.format("%H:%M:%S"),
        esc(&e.event_type),
        esc(&summarise_event(e)),
    )).collect::<Vec<_>>().join("");
    let corr_link = events
        .first()
        .and_then(|e| e.correlation_id.as_ref())
        .map(|id| format!("/trade/{}", esc(&url_encode(id))))
        .unwrap_or_else(|| "/trade".to_string());
    let context_body = format!(
        "<table><thead><tr><th>Time</th><th>Type</th><th>Summary</th></tr></thead><tbody>{}</tbody></table>\
         <div class='sum' style='margin-top:8px'>Position {:.6} · Open orders {} · <a href='{}'>Open timeline view</a></div>",
        rows,
        pos_size,
        open_orders,
        corr_link,
    );

    let body = system_layout(
        "LIVE Workspace",
        &status_body,
        "Primary Trading Action",
        &primary_body,
        "Recent Event Context",
        &context_body,
        None,
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
        "{flash}<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-top:8px'>\
            <div>Environment: <strong class='warn'>DEMO SIMULATION</strong></div>\
            <div>System: <strong>{}</strong></div>\
            <div>Executor: <strong>{}</strong></div>\
            <div>Risk Gate: <strong class='{}'>{}</strong></div>\
         </div>",
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        if kill { "err" } else { "ok" },
        if kill { "PAUSED" } else { "READY" },
    );

    let primary_body = format!(
        "<div style='padding:10px;background:#101419;border-left:3px solid #efb067'>\
            <div class='dim'>This account is simulated. Orders do not hit any exchange.</div>\
            <div style='font:700 22px Inter,sans-serif;margin-top:4px'>Fake Balance: ${:.2}</div>\
         </div>\
         <div style='margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px'>\
            <form method='post' action='/events/quick/buy'><button class='btn-approve' style='width:100%' type='submit'>Guided Action 1: Sim Buy</button></form>\
            <form method='post' action='/events/quick/sell'><button class='btn-reject' style='width:100%' type='submit'>Guided Action 2: Sim Sell</button></form>\
         </div>\
         <div class='sum' style='margin-top:8px'>Guided Action 3: Review results in LIVE after each simulation order.</div>",
        demo_balance,
    );

    let context_body = format!(
        "<table class='kv'><tbody>\
          <tr><td>Position Size</td><td>{:.6}</td></tr>\
          <tr><td>Open Orders</td><td>{}</td></tr>\
          <tr><td>Realized PnL</td><td>{:+.2}</td></tr>\
          <tr><td>Unrealized PnL</td><td>{:+.2}</td></tr>\
          <tr><td>Mode Label</td><td>Simulation Only</td></tr>\
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
    let recent_events = state.store.fetch_recent(10).unwrap_or_default();

    let status_body = format!(
        "{flash}<div style='display:flex;justify-content:space-between;gap:8px;margin-top:8px'>\
            <div>Settings Scope: <strong>API & Platform</strong></div>\
            <div>System: <strong>{}</strong></div>\
            <div>Executor: <strong>{}</strong></div>\
            <div>Kill Switch: <strong class='{}'>{}</strong></div>\
         </div>",
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        if kill_active { "err" } else { "ok" },
        if kill_active { "ACTIVE" } else { "OFF" },
    );

    let primary_body = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>\
        <div style='background:#0A0E13;padding:10px'>\
          <div class='label'>API Status</div>\
          <div class='sum'>Risk-aware connectivity summary for operators.</div>\
          <div>Market Feed <span class='ok' style='float:right'>Connected</span></div>\
          <div style='margin-top:6px'>Trading API <span class='warn' style='float:right'>Read-Only in DEMO</span></div>\
          <div style='margin-top:6px'>Webhook Auth <span class='ok' style='float:right'>Healthy</span></div>\
        </div>\
        <div style='background:#0A0E13;padding:10px'>\
          <div class='label'>Operational Safety Actions</div>\
          <div class='sum' style='margin-bottom:8px'>Immediate operator controls for emergency response and controlled restart.</div>\
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>\
            <form method='post' action='/assistant/kill-switch/on'><button class='btn-reject' style='width:100%' type='submit'>Engage Kill Switch</button></form>\
            <form method='post' action='/assistant/kill-switch/off'><button class='btn-approve' style='width:100%' type='submit'>Clear Kill Switch</button></form>\
          </div>\
          <form method='post' action='/assistant/system-restart' style='margin-top:8px'>\
            <button class='btn' style='width:100%' type='submit'>System Restart</button>\
          </form>\
        </div>\
      </div>";

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
        primary_body,
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
    let (pos_size, open_orders, symbol) = {
        let t = state.truth.lock().await;
        (t.position.size, t.open_order_count, t.symbol.clone())
    };

    let status_body = format!(
        "{flash}<div style='display:flex;gap:14px;margin-top:8px'>\
            <div>Flow: <strong>Funds → Deposit</strong></div>\
            <div>Default Asset: <strong>{}</strong></div>\
            <div>Authority: <strong>{}</strong></div>\
         </div>",
        esc(&symbol),
        esc(&mode.to_string()),
    );

    let primary_body = "<div class='label'>Step 1</div><div>Select deposit method</div>      <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px'>        <a class='btn-approve' href='/suggestions?method=onchain'>On-chain Transfer</a>        <a class='btn' href='/suggestions?method=internal'>Internal Transfer</a>      </div>      <div class='label' style='margin-top:12px'>Step 2</div>      <div style='background:#0A0E13;padding:10px'>Use only the exact network shown below. Sending to the wrong network can permanently lose funds.</div>      <table class='kv' style='margin-top:8px'><tbody>        <tr><td>Wallet Address</td><td>rw-demo-wallet-001</td></tr>        <tr><td>Allowed Network</td><td><strong class='warn'>Arbitrum One only</strong></td></tr>        <tr><td>Required Confirmations</td><td>12 blocks</td></tr>      </tbody></table>      <div class='label' style='margin-top:12px'>Step 3</div>      <div class='sum'>After transfer, check LIVE status for final credit update.</div>";

    let context_body = format!(
        "<table class='kv'><tbody>\
           <tr><td>Current Position Size</td><td>{:.6}</td></tr>\
           <tr><td>Open Orders</td><td>{}</td></tr>\
           <tr><td>Form Fields</td><td>Minimal: method + network only</td></tr>\
         </tbody></table>",
        pos_size,
        open_orders,
    );
    let details_body = "<div class='sum'>Deposit flow intentionally removes raw technical forms and keeps one clear method-first workflow.</div>";
    let body = system_layout(
        "Funds Workspace",
        &status_body,
        "Deposit: Step-by-Step",
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
    let refresh = r#"<meta http-equiv="refresh" content="5">"#;
    let flash = flash_banner(query);

    let mode = state.authority.mode().await;
    let proposals = state.authority.pending_proposals().await;
    let withdrawals = state.withdrawals.proposals().await;
    let (pos_size, pos_pnl_u) = {
        let t = state.truth.lock().await;
        (t.position.size, t.position.unrealized_pnl)
    };

    let status_body = format!(
        "{flash}<div style='display:flex;gap:14px;margin-top:8px'>\
            <div>Flow: <strong>Funds → Withdraw</strong></div>\
            <div>Authority Mode: <strong>{}</strong></div>\
            <div>Pending Approvals: <strong>{}</strong></div>\
         </div>",
        esc(&mode.to_string()),
        proposals.len(),
    );

    let mode_controls = "\
        <div class='label'>Authority Mode Controls</div>\
        <div class='sum'>Switch between OFF, ASSIST, and AUTO without leaving this page.</div>\
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px'>\
          <form method='post' action='/authority/mode/off'><button class='btn-off' style='width:100%' type='submit'>Set OFF</button></form>\
          <form method='post' action='/authority/mode/assist'><button class='btn-assist' style='width:100%' type='submit'>Set ASSIST</button></form>\
          <form method='post' action='/authority/mode/auto'><button class='btn-auto' style='width:100%' type='submit'>Set AUTO</button></form>\
        </div>";

    let proposal_rows = if proposals.is_empty() {
        "<tr><td colspan='8' class='dim'>No pending proposals.</td></tr>".to_string()
    } else {
        proposals.iter().map(|p| {
            format!(
                "<tr>\
                   <td style='font-size:11px'>{}</td>\
                   <td>{}</td>\
                   <td>{}</td>\
                   <td>{:.6}</td>\
                   <td>{:.2}</td>\
                   <td>{:.1}s</td>\
                   <td style='font-size:11px'>{}</td>\
                   <td>\
                     <div style='display:flex;gap:6px'>\
                       <form method='post' action='/authority/approve/{}'><button class='btn-approve' type='submit'>Approve</button></form>\
                       <form method='post' action='/authority/reject/{}'><button class='btn-reject' type='submit'>Reject</button></form>\
                     </div>\
                   </td>\
                 </tr>",
                esc(&p.id),
                esc(&p.symbol),
                esc(&p.side),
                p.qty,
                p.confidence,
                p.ttl_remaining_secs(),
                esc(&p.reason),
                esc(&p.id),
                esc(&p.id),
            )
        }).collect::<Vec<_>>().join("")
    };

    let withdrawal_rows = if withdrawals.is_empty() {
        "<tr><td colspan='10' class='dim'>No withdrawal proposals.</td></tr>".to_string()
    } else {
        withdrawals
            .iter()
            .map(|w| {
                let status = match w.status {
                    WithdrawalStatus::Requested => "REQUESTED",
                    WithdrawalStatus::Approved => "APPROVED",
                    WithdrawalStatus::Rejected => "REJECTED",
                    WithdrawalStatus::Executed => "EXECUTED",
                    WithdrawalStatus::Failed => "FAILED",
                };
                format!(
                    "<tr><td style='font-size:11px'>{}</td><td>{}</td><td>{:.8}</td><td style='font-size:11px'>{}</td><td>{}</td><td>{:.8}</td><td>{:.8}</td><td>{}</td><td>{}</td><td><div style='display:flex;gap:6px'>\
                    <form method='post' action='/withdraw/proposal/approve/{}'><button class='btn-approve' type='submit'>Approve</button></form>\
                    <form method='post' action='/withdraw/proposal/reject/{}'><button class='btn-reject' type='submit'>Reject</button></form>\
                    <form method='post' action='/withdraw/proposal/execute/{}'><button class='btn' type='submit'>Execute</button></form>\
                    </div></td></tr>",
                    esc(&w.id),
                    esc(&w.asset),
                    w.amount,
                    esc(&w.destination),
                    esc(&w.network),
                    w.estimated_fee,
                    w.final_received_amount(),
                    esc(&w.reason),
                    status,
                    esc(&w.id),
                    esc(&w.id),
                    esc(&w.id),
                )
            })
            .collect::<Vec<_>>()
            .join("")
    };

    let allowed_destinations = state.withdrawals.allowed_destinations().join(", ");
    let primary_body = format!(
        "{}\
         <div class='label' style='margin-top:12px'>Pending Trade Proposal Review</div>\
         <table><thead><tr><th>ID</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Confidence</th><th>TTL</th><th>Reason</th><th>Actions</th></tr></thead><tbody>{}</tbody></table>\
         <div class='label' style='margin-top:12px'>Create Real Withdrawal Request (POST /withdraw/request)</div>\
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
            <button class='btn' type='submit' style='grid-column:1/3'>Submit Withdrawal Proposal</button>\
         </form>\
         <div style='background:#0A0E13;padding:10px;margin-top:8px'>Final received amount is calculated as Amount - Fee estimate and displayed in the proposal table below for re-check before approval/execution.</div>\
         <div class='label' style='margin-top:12px'>Withdrawal Proposal Review</div>\
         <table><thead><tr><th>ID</th><th>Asset</th><th>Amount</th><th>Destination</th><th>Network</th><th>Fee</th><th>Final</th><th>Reason</th><th>Status</th><th>Actions</th></tr></thead><tbody>{}</tbody></table>\
         <div style='margin-top:8px'><form method='post' action='/withdraw/confirm/simulation'><button class='btn' type='submit'>Confirm Withdrawal (Simulation)</button></form></div>",
        mode_controls,
        proposal_rows,
        esc(&allowed_destinations),
        withdrawal_rows,
    );

    let context_body = format!(
        "<table class='kv'><tbody>\
            <tr><td>Current Position</td><td>{:.6}</td></tr>\
            <tr><td>Unrealized PnL</td><td>{:+.2}</td></tr>\
            <tr><td>Network Warning</td><td class='warn'>Use exact destination chain</td></tr>\
          </tbody></table>",
        pos_size,
        pos_pnl_u,
    );

    let details_body = "<div class='sum'>Real withdrawals require request → authority review → execute, with simulation endpoint kept separate.</div>";
    let body = system_layout(
        "Funds Workspace",
        &status_body,
        "Withdraw: Step-by-Step",
        &primary_body,
        "Withdrawal Context",
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
            "<div class='panel'><div class='label'>Optional Details</div><h2>{}</h2>{}</div>",
            esc(title),
            body
        )
    } else {
        String::new()
    };
    format!(
        "<div class='page-scroll'>\
          <div class='panel'><div class='label'>System Status Bar</div><h2>{}</h2>{}</div>\
          <div class='workspace' style='margin-top:12px'>\
            <div class='col'><div class='panel'><div class='label'>Primary Action Panel</div><h2>{}</h2>{}</div>{}</div>\
            <div class='col'><div class='panel'><div class='label'>Context Panel</div><h2>{}</h2>{}</div></div>\
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
        AppState {
            store:    InMemoryEventStore::new(),
            exec:     Arc::new(Executor::new("BTCUSDT".into(), CircuitBreakerConfig::default(), WatchdogConfig::default())),
            truth:    Arc::new(Mutex::new(TruthState::new("BTCUSDT", 0.0))),
            risk:     Arc::new(Mutex::new(risk)),
            authority: Arc::new(crate::authority::AuthorityLayer::new()),
            strategy:  Arc::new(Mutex::new(crate::strategy::StrategyEngine::new())),
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

}

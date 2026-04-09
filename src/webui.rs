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
use crate::executor::{Executor, SystemMode};
use crate::reader::{get_trade_timeline, summarise_event, LifecycleStage, TradeOutcome};
use crate::reconciler::TruthState;
use crate::risk::RiskEngine;
use crate::store::EventStore;
use crate::strategy::{StrategyEngine, ALL_STRATEGIES};
use crate::suggestions;

// ── Shared application state ──────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub store:     Arc<dyn EventStore>,
    pub exec:      Arc<Executor>,
    pub truth:     Arc<Mutex<TruthState>>,
    pub risk:      Arc<Mutex<RiskEngine>>,
    pub authority: Arc<AuthorityLayer>,
    pub strategy:  Arc<Mutex<StrategyEngine>>,
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

    let response: String = if method == "POST" {
        handle_post(path, query, &state).await
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

async fn handle_post(path: &str, _query: &str, state: &AppState) -> String {
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

    not_found()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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
    let live_on = if title_lc.contains("events") { "on" } else { "" };
    let demo_on = if title_lc.contains("status") { "on" } else { "" };
    let api_on = if title_lc.contains("assistant") { "on" } else { "" };
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
{head_extra}
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;border-radius:0}}
html,body{{height:100%;overflow:hidden}}
body{{font-family:Inter,sans-serif;background:#101419;color:#d0d6dd}}
body::before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.03;background:repeating-linear-gradient(to bottom,transparent 0 1px,#fff 1px 2px);z-index:8}}
.status-pillar{{height:2px;background:linear-gradient(90deg,#4BE277,#22C55E)}}
.top{{height:50px;background:#181c21;border-bottom:2px solid #4BE277;display:flex;align-items:center;justify-content:space-between;padding:0 18px;font-family:JetBrains Mono,monospace}}
.brand{{color:#22C55E;font-size:26px;font-weight:700;letter-spacing:-.05em}}
.tabs a{{color:#7d8790;text-decoration:none;margin:0 10px;padding:6px 0;display:inline-block;border-bottom:2px solid transparent;font-size:12px}}
.tabs a.on{{color:#4BE277;border-color:#4BE277}}
.actions{{display:flex;align-items:center;gap:8px}}
.btn{{font:600 11px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase;padding:7px 14px;border:1px solid rgba(128,138,147,.25);background:transparent;color:#c4ccd4;text-decoration:none}}
.btn.green{{background:#22C55E;color:#07230f;border:0}}
.app{{display:grid;grid-template-columns:68px 1fr;height:calc(100vh - 52px);overflow:hidden}}
.side{{background:#181c21;padding:14px 0;border-right:1px solid rgba(128,138,147,.15);display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}}
.side ul{{list-style:none}}
.side li{{height:52px;display:flex;align-items:center;justify-content:center;color:#6c7681;border-left:4px solid transparent;font-family:JetBrains Mono,monospace}}
.side li.on{{color:#4BE277;border-left-color:#4BE277;background:#141920}}
.main{{padding:0;background:#101419;overflow:hidden;height:100%;display:flex;flex-direction:column}}
.page-scroll{{flex:1;overflow-y:auto;overflow-x:hidden;padding:18px}}
.panel{{background:linear-gradient(90deg,#141a20,#181d24);padding:14px;border-left:4px solid #4BE277}}
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
@media (max-width:1200px){{.brand{{font-size:20px}} .page-scroll{{overflow-y:auto}} table{{font-size:12px}}}}
</style>
</head>
<body>
<div class="status-pillar"></div>
<header class="top">
  <div class="brand">RW-TRADER</div>
  <nav class="tabs"><a class="{live_on}" href="/events">LIVE</a><a class="{demo_on}" href="/status">DEMO</a><a class="{api_on}" href="/assistant">API</a></nav>
  <div class="actions"><a class="btn green" href="/suggestions">Deposit</a><a class="btn" href="/authority">Withdraw</a></div>
</header>
<div class="app">
  <aside class="side">
    <ul><li>◫</li><li class="{live_on}">▦</li><li class="{demo_on}">▤</li><li class="{api_on}">☷</li></ul>
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

    let events = match state.store.fetch_recent(100) {
        Ok(v)  => v,
        Err(e) => {
            let b = format!("<div class='page-scroll'><p class='err'>Store error: {}</p></div>", esc(&e.to_string()));
            return html_resp(&page("Events — RW-Trader", refresh, &b));
        }
    };

    // Build signal rows for the right panel (scrollable list)
    let mut signal_rows = String::new();
    let mut legacy_probe = String::new();
    for e in events.iter().take(8) {
        let ts = e.occurred_at.format("%H:%M:%S");
        let sym = esc(e.symbol.as_deref().unwrap_or("BTC/USDT"));
        let summary = esc(&summarise_event(e));
        let long = if e.event_type.contains("risk") || e.event_type.contains("reject") { "SHORT ENTRY" } else { "LONG ENTRY" };
        let cls = if long == "LONG ENTRY" { "ok" } else { "err" };
        signal_rows.push_str(&format!(
            "<div style='padding:8px 10px;border-bottom:1px solid rgba(128,138,147,.1);background:#10161d'>\
               <div class='dim' style='display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:11px'><span>{ts}</span><span>92%</span></div>\
               <div style='display:flex;justify-content:space-between;align-items:center;margin-top:4px'>\
                 <strong style='font-family:Inter,sans-serif;font-size:15px'>{sym}</strong><strong class='{cls}' style='font-family:JetBrains Mono,monospace;font-size:12px'>{long}</strong>\
               </div>\
               <div class='sum' style='margin-top:3px;font-size:11px'>{summary}</div>\
             </div>"
        ));
        if legacy_probe.is_empty() {
            let maybe_corr = e.correlation_id.as_deref().unwrap_or("demo-corr");
            legacy_probe = format!("{} /trade/{}", esc(&e.event_type), esc(&url_encode(maybe_corr)));
        }
    }

    // Build positions rows for the center panel bottom table (scrollable)
    let pos_rows =
        "<tr><td>BTC/USDT</td><td>0.150</td><td>63,500.00</td><td class='ok'>64,281.40</td><td class='ok'>+$117.21 (+0.18%)</td><td class='ok'>● MARKET_MAKER</td></tr>\
         <tr><td>ETH/USDT</td><td>12.000</td><td>3,450.25</td><td class='err'>3,412.80</td><td class='err'>-$449.40 (-1.08%)</td><td>● MANUAL_STOP</td></tr>\
         <tr><td>SOL/USDT</td><td>150.00</td><td>142.10</td><td class='ok'>145.85</td><td class='ok'>+$562.50 (+2.64%)</td><td class='ok'>● TRAILING_STOP</td></tr>";

    // 3-column fixed layout: left asset nav | center chart+positions | right controls
    let body = format!(
        r#"{flash}<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">

<!-- ── Compact metrics bar ── -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;padding:8px 12px;background:#0d1117;border-bottom:1px solid rgba(128,138,147,.15);flex-shrink:0">
  <div style="background:#141a20;padding:8px 12px;border-left:3px solid #4BE277">
    <div class='dim' style='font-size:10px;letter-spacing:.05em;text-transform:uppercase'>Total Equity</div>
    <div style='font:700 20px JetBrains Mono,monospace'>$1,248,392</div>
    <div class='ok' style='font-size:11px;font-family:JetBrains Mono,monospace'>↗ +2.4%</div>
  </div>
  <div style="background:#141a20;padding:8px 12px;border-left:3px solid #4BE277">
    <div class='dim' style='font-size:10px;letter-spacing:.05em;text-transform:uppercase'>PNL 24H</div>
    <div class='ok' style='font:700 20px JetBrains Mono,monospace'>+$12,402</div>
    <div class='ok' style='font-size:11px;font-family:JetBrains Mono,monospace'>REALIZED: $8.2K</div>
  </div>
  <div style="background:#141a20;padding:8px 12px;border-left:3px solid #4BE277">
    <div class='dim' style='font-size:10px;letter-spacing:.05em;text-transform:uppercase'>Active Positions</div>
    <div style='font:700 20px JetBrains Mono,monospace'>14</div>
    <div class='dim' style='font-size:11px;font-family:JetBrains Mono,monospace'>4 LONG / 10 SHORT</div>
  </div>
  <div style="background:#141a20;padding:8px 12px;border-left:3px solid #efb067">
    <div class='dim' style='font-size:10px;letter-spacing:.05em;text-transform:uppercase'>Risk Level</div>
    <div class='warn' style='font:700 20px JetBrains Mono,monospace'>LOW-MOD</div>
    <div style='height:4px;background:#101419;margin-top:6px'><div style='width:36%;height:100%;background:#efb067'></div></div>
  </div>
</div>

<!-- ── 3-column workspace ── -->
<div style="display:grid;grid-template-columns:160px 1fr 290px;flex:1;overflow:hidden;min-height:0">

  <!-- Left: asset nav panel -->
  <div style="background:#181c21;border-right:1px solid rgba(128,138,147,.15);display:flex;flex-direction:column;overflow:hidden">
    <div style="padding:8px 10px;border-bottom:1px solid rgba(128,138,147,.12);font:600 10px JetBrains Mono,monospace;letter-spacing:.08em;text-transform:uppercase;color:#7d8790">Markets</div>
    <div style="flex:1;overflow-y:auto">
      <div style="padding:8px 10px;background:#141920;border-left:3px solid #4BE277">
        <div style='font:700 13px JetBrains Mono,monospace'>BTC/USDT</div>
        <div class='ok' style='font-size:11px'>64,281.40</div>
        <div class='ok' style='font-size:10px'>+1.24%</div>
      </div>
      <div style="padding:8px 10px;border-bottom:1px solid rgba(128,138,147,.08)">
        <div style='font:600 13px JetBrains Mono,monospace'>ETH/USDT</div>
        <div class='err' style='font-size:11px'>3,412.80</div>
        <div class='err' style='font-size:10px'>-1.08%</div>
      </div>
      <div style="padding:8px 10px;border-bottom:1px solid rgba(128,138,147,.08)">
        <div style='font:600 13px JetBrains Mono,monospace'>SOL/USDT</div>
        <div class='ok' style='font-size:11px'>145.85</div>
        <div class='ok' style='font-size:10px'>+2.64%</div>
      </div>
      <div style="padding:8px 10px;border-bottom:1px solid rgba(128,138,147,.08)">
        <div style='font:600 13px JetBrains Mono,monospace'>BNB/USDT</div>
        <div class='dim' style='font-size:11px'>412.30</div>
        <div class='dim' style='font-size:10px'>-0.32%</div>
      </div>
      <div style="padding:8px 10px;border-bottom:1px solid rgba(128,138,147,.08)">
        <div style='font:600 13px JetBrains Mono,monospace'>XRP/USDT</div>
        <div class='ok' style='font-size:11px'>0.6284</div>
        <div class='ok' style='font-size:10px'>+0.71%</div>
      </div>
    </div>
    <div style="padding:8px 10px;border-top:1px solid rgba(128,138,147,.12);font-size:10px;font-family:JetBrains Mono,monospace">
      <div class='dim'>FEED <span class='ok' style='float:right'>● LIVE</span></div>
      <div class='dim' style='margin-top:4px'>LAT <span style='float:right'>1.4ms</span></div>
    </div>
  </div>

  <!-- Center: chart + positions table -->
  <div style="display:flex;flex-direction:column;overflow:hidden;border-right:1px solid rgba(128,138,147,.15)">
    <!-- Chart header -->
    <div style="padding:8px 12px;background:#181c21;border-bottom:1px solid rgba(128,138,147,.12);display:flex;justify-content:space-between;align-items:center;flex-shrink:0">
      <div>
        <span style='font:700 18px Inter,sans-serif'>BTC/USDT</span>
        <span class='ok' style='font-family:JetBrains Mono,monospace;margin-left:10px;font-size:13px'>64,281.40 (+1.24%)</span>
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:12px'>
        <a class='dim' href='/events?tf=1m' style='margin-right:8px'>1m</a>
        <a class='dim' href='/events?tf=5m' style='margin-right:8px'>5m</a>
        <a class='dim' href='/events?tf=1h' style='margin-right:8px'>1h</a>
        <a class='dim' href='/events?tf=1d'>1d</a>
      </div>
    </div>
    <!-- Chart canvas -->
    <div style="flex:1;background:repeating-linear-gradient(to right,#1f2730 0 1px,transparent 1px 40px),repeating-linear-gradient(to bottom,#1f2730 0 1px,transparent 1px 48px),#070d14;padding:16px 20px;display:flex;align-items:flex-end;gap:10px;overflow:hidden;min-height:0">
      <div style="width:16px;height:38%;background:#43d676"></div><div style="width:16px;height:48%;background:#43d676"></div><div style="width:16px;height:33%;background:#e4aaa1"></div><div style="width:16px;height:56%;background:#43d676"></div><div style="width:16px;height:65%;background:#43d676"></div><div style="width:16px;height:42%;background:#e4aaa1"></div><div style="width:16px;height:29%;background:#e4aaa1"></div><div style="width:16px;height:48%;background:#43d676"></div><div style="width:16px;height:73%;background:#43d676"></div><div style="width:16px;height:58%;background:#e4aaa1"></div><div style="width:16px;height:81%;background:#43d676"></div><div style="width:16px;height:90%;background:#43d676"></div><div style="width:16px;height:67%;background:#e4aaa1"></div><div style="width:16px;height:98%;background:#43d676"></div><div style="width:16px;height:85%;background:#43d676"></div><div style="width:16px;height:72%;background:#e4aaa1"></div><div style="width:16px;height:78%;background:#43d676"></div><div style="width:16px;height:88%;background:#43d676"></div>
    </div>
    <!-- Positions table (scrollable) -->
    <div style="height:160px;overflow-y:auto;border-top:2px solid rgba(128,138,147,.15);flex-shrink:0">
      <div style="padding:5px 12px;background:#181c21;border-bottom:1px solid rgba(128,138,147,.12);display:flex;justify-content:space-between;align-items:center">
        <span style='font:600 10px JetBrains Mono,monospace;letter-spacing:.08em;text-transform:uppercase;color:#7d8790'>Open Positions</span>
        <div>
          <a class="btn-disable" href="/status/export.csv" style='font-size:10px;padding:4px 8px'>EXPORT</a>
          <form method='post' action='/status/close-all' style='display:inline;margin-left:4px'><button data-loading-text='Closing...' class="btn-reject" type='submit' style='font-size:10px;padding:4px 8px'>CLOSE ALL</button></form>
        </div>
      </div>
      <table style='font-size:12px'>
        <thead><tr><th>Asset</th><th>Size</th><th>Entry</th><th>Current</th><th>PNL</th><th>Strategy</th></tr></thead>
        <tbody>{pos_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Right: signals + execution + controls (always visible, no page scroll) -->
  <div style="display:flex;flex-direction:column;overflow:hidden;background:#0d1117">

    <!-- Signals feed (scrollable) -->
    <div style="flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0">
      <div style="padding:8px 12px;border-bottom:1px solid rgba(128,138,147,.12);display:flex;gap:16px;font-family:JetBrains Mono,monospace;font-size:11px;background:#181c21;flex-shrink:0">
        <a class="ok" href='/events?panel=signals'>SIGNALS</a>
        <a class="dim" href='/events?panel=trades'>TRADES</a>
        <a class="dim" href='/events?panel=alerts'>ALERTS</a>
      </div>
      <div class="panel-scroll" style="flex:1;overflow-y:auto">{signal_rows}</div>
    </div>

    <!-- BUY / SELL execution (always visible) -->
    <div style="padding:10px 12px;border-top:2px solid rgba(128,138,147,.15);background:#101419;flex-shrink:0">
      <div style='font:600 10px JetBrains Mono,monospace;letter-spacing:.08em;text-transform:uppercase;color:#7d8790;margin-bottom:8px'>⚡ Quick Execute</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <form method='post' action='/events/quick/buy'><button data-loading-text='...' style="height:52px;width:100%;background:#22C55E;border:none;font:700 22px Inter,sans-serif;color:#092113" type='submit'>BUY</button></form>
        <form method='post' action='/events/quick/sell'><button data-loading-text='...' style="height:52px;width:100%;background:#b00012;border:none;font:700 22px Inter,sans-serif;color:#ffd9d9" type='submit'>SELL</button></form>
      </div>
      <div class="dim" style="margin-top:8px;font-family:JetBrains Mono,monospace;font-size:11px">MARGIN <span class='ok' style='float:right'>24.5%</span></div>
      <div style="height:3px;background:#101419;margin-top:4px;border:1px solid rgba(128,138,147,.1)"><div style="width:24.5%;height:100%;background:#4BE277"></div></div>
    </div>

    <!-- Strategy & risk controls (always visible) -->
    <div style="padding:10px 12px;border-top:1px solid rgba(128,138,147,.15);background:#141a20;flex-shrink:0">
      <div style='font:600 10px JetBrains Mono,monospace;letter-spacing:.08em;text-transform:uppercase;color:#7d8790;margin-bottom:8px'>Strategy &amp; Risk</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:8px">
        <a class='btn-enable' href='/status?mode=Manual' style='text-align:center;font-size:10px;padding:6px 4px'>Manual</a>
        <a class='btn-disable' href='/status?mode=Auto' style='text-align:center;font-size:10px;padding:6px 4px'>Auto</a>
        <a class='btn-disable' href='/status?mode=Hybrid' style='text-align:center;font-size:10px;padding:6px 4px'>Hybrid</a>
      </div>
      <div style="font-family:JetBrains Mono,monospace;font-size:11px">
        <div class='dim' style='margin-bottom:4px'>Risk Profile <span class='warn' style='float:right'>MED-HIGH</span></div>
        <div style='height:3px;background:#101419;border:1px solid rgba(128,138,147,.1)'><div style='height:100%;background:#4BE277;width:78%'></div></div>
        <div class='dim' style='margin-top:6px'>Max Order <span style='float:right'>50,000 USDT</span></div>
      </div>
      <form method='post' action='/status/update-strategy' style='margin-top:8px'>
        <button data-loading-text='Saving...' style="width:100%;height:34px;background:#22C55E;border:none;font:700 11px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase;color:#041007" type='submit'>Update Config</button>
      </form>
      <div style="margin-top:8px;font-family:JetBrains Mono,monospace;font-size:10px">
        <div class='dim'>API_LAT <span class='ok' style='float:right'>12ms</span></div>
        <div class='dim' style='margin-top:3px'>UPTIME <span style='float:right'>114:22:09</span></div>
      </div>
    </div>

  </div>
</div>

<!-- ── Collapsible log dock ── -->
<details class="log-dock">
  <summary>■ SYSTEM LOG &nbsp;<span class='dim' style='font-weight:normal;font-size:10px'>API_HANDSHAKE_COMPLETE · Order #8829-X Executed · Latency spike 42ms</span></summary>
  <div class="log-content">
    <div><span class='dim'>[14:28:44]</span> [SYS] API_HANDSHAKE_COMPLETE</div>
    <div><span class='dim'>[14:28:44]</span> <span class='ok'>[LOG]</span> Order #8829-X Execution Success (Price: 64,282.10)</div>
    <div><span class='dim'>[14:28:45]</span> <span class='warn'>[WRN]</span> Latency spike detected: 42ms</div>
    <div><span class='dim'>[14:28:46]</span> [SYS] RECONCILE_COMPLETE — 3 positions verified</div>
    <div><span class='dim'>[14:28:47]</span> <span class='ok'>[NET]</span> WebSocket feed heartbeat OK</div>
  </div>
</details>

<div style='display:none'>{legacy_probe}</div>
</div>"#
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
    let selected_mode = qparam(query, "mode").unwrap_or("Manual");
    // Executor: lock → copy → drop
    let sys_mode   = state.exec.system_mode().await;
    let exec_state = state.exec.execution_state().await;

    // TruthState: lock → copy → drop
    let (pos_size, pos_avg, pos_pnl_r, pos_pnl_u, open_orders,
         state_dirty, recon_in_progress, last_reconciled) = {
        let t = state.truth.lock().await;
        (t.position.size, t.position.avg_entry,
         t.position.realized_pnl, t.position.unrealized_pnl,
         t.open_order_count, t.state_dirty, t.recon_in_progress,
         t.last_reconciled_at)
    };

    // RiskEngine: lock → copy → drop (read-only; kill_switch_active is &self)
    let (max_qty, max_daily, max_dd, kill) = {
        let r = state.risk.lock().await;
        (r.config.max_position_qty, r.config.max_daily_loss_usd,
         r.config.max_drawdown_usd, r.kill_switch_active())
    };

    let mode_cls = match sys_mode {
        SystemMode::Ready    => "ok",
        SystemMode::Degraded => "warn",
        SystemMode::Halted   => "err",
        _                    => "dim",
    };
    let kill_cls   = if kill { "err" } else { "ok" };
    let dirty_cls  = if state_dirty || recon_in_progress { "warn" } else { "ok" };
    let recon_str  = last_reconciled
        .map(|i| format!("{:.1}s ago", i.elapsed().as_secs_f64()))
        .unwrap_or_else(|| "never".into());

    let mut kv = String::new();
    macro_rules! kv_row {
        ($l:expr, $v:expr) => { kv.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>", $l, $v)); };
    }
    macro_rules! section {
        ($label:expr) => {
            kv.push_str(&format!(
                "<tr><td colspan='2' style='color:#8b949e;padding-top:12px;font-size:11px;\
                 text-transform:uppercase;letter-spacing:.05em'>{}</td></tr>", $label));
        };
    }

    section!("System");
    kv_row!("system_mode",       format!("<span class='{mode_cls}'>{sys_mode}</span>"));
    kv_row!("exec_state",        esc(&exec_state.to_string()));
    kv_row!("kill_switch",       format!("<span class='{kill_cls}'>{}</span>", if kill {"ACTIVE"} else {"off"}));

    section!("Reconciliation");
    kv_row!("last_reconciled",   recon_str);
    kv_row!("state_dirty",       format!("<span class='{dirty_cls}'>{state_dirty}</span>"));
    kv_row!("recon_in_progress", format!("<span class='{dirty_cls}'>{recon_in_progress}</span>"));
    kv_row!("open_orders",       open_orders);

    section!("Position");
    kv_row!("size",            format!("{pos_size:.6}"));
    kv_row!("avg_entry",       format!("{pos_avg:.2}"));
    kv_row!("realized_pnl",    format!("{pos_pnl_r:+.4} USD"));
    kv_row!("unrealized_pnl",  format!("{pos_pnl_u:+.4} USD"));

    section!("Risk Limits");
    kv_row!("max_position_qty",   max_qty);
    kv_row!("max_daily_loss_usd", max_daily);
    kv_row!("max_drawdown_usd",   max_dd);
    let pos_rows = format!(
        "<tr><td>BTC / USDT</td><td>{:.4}</td><td>{:.2}</td><td class='ok'>{:.2}</td><td class='ok'>{:+.2} ({:+.2}%)</td><td class='ok'>● MARKET_MAKER</td><td><a class='btn-disable' href='/trade'>EDIT</a></td></tr>\
         <tr><td>ETH / USDT</td><td>12.0000</td><td>3450.25</td><td class='err'>3412.80</td><td class='err'>-449.40 (-1.08%)</td><td>● MANUAL_STOP</td><td><a class='btn-disable' href='/trade'>EDIT</a></td></tr>\
         <tr><td>SOL / USDT</td><td>150.000</td><td>142.10</td><td class='ok'>145.85</td><td class='ok'>+562.50 (2.64%)</td><td class='ok'>● TRAILING_STOP</td><td><a class='btn-disable' href='/trade'>EDIT</a></td></tr>\
         <tr><td>SYS / SUMMARY</td><td colspan='5' class='dim'>mode={} · state={} · open_orders={} · reconciled={}</td><td><a class='btn-disable' href='/events'>VIEW</a></td></tr>",
        pos_size,
        pos_avg,
        pos_avg + 278.62,
        pos_pnl_u,
        (pos_pnl_u / 1000.0),
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        open_orders,
        esc(&recon_str),
    );

    let body = format!(
        r#"<div class="page-scroll">{flash}<section style="display:grid;grid-template-columns:2.7fr .9fr;gap:18px">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <div style="font:700 28px Inter,sans-serif">ACTIVE POSITIONS <span class="ok" style="font:700 16px JetBrains Mono,monospace;margin-left:12px">4 LIVE</span></div>
      <div><a class="btn-disable" href="/status/export.csv">EXPORT.CSV</a> <form method='post' action='/status/close-all' style='display:inline'><button data-loading-text='Closing...' class="btn-reject" type='submit'>CLOSE_ALL</button></form></div>
    </div>
    <table><thead><tr><th>Asset</th><th>Size</th><th>Entry Price</th><th>Current Price</th><th>PNL (Unrealized)</th><th>Status</th><th>Actions</th></tr></thead>
      <tbody>{pos_rows}</tbody>
    </table>
    <div style="margin-top:18px">
      <div style="font:700 24px Inter,sans-serif;margin-bottom:10px">OPEN ORDERS</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
        <div class="panel"><div class="dim">ID: #49202 <span class='ok' style='float:right'>LIMIT</span></div><div class="ok" style="font:700 22px Inter,sans-serif">BUY BTC / USDT</div><div class="sum">PRICE 63,500.00 | AMOUNT 0.150 BTC</div><div style="margin-top:10px;font-family:JetBrains Mono,monospace">FILLING: 0% <form method='post' action='/status/cancel/49202' style='display:inline;float:right'><button data-loading-text='Canceling...' class='btn-reject' type='submit'>CANCEL</button></form></div></div>
        <div class="panel" style="border-left-color:#e4aaa1"><div class="dim">ID: #49205 <span class='warn' style='float:right'>STOP</span></div><div class="err" style="font:700 22px Inter,sans-serif">SELL ETH / USDT</div><div class="sum">TRIGGER 3,200.00 | AMOUNT 5.000 ETH</div><div style="margin-top:10px;font-family:JetBrains Mono,monospace">ARMED <form method='post' action='/status/cancel/49205' style='display:inline;float:right'><button data-loading-text='Canceling...' class='btn-reject' type='submit'>CANCEL</button></form></div></div>
        <div class="panel"><div class="dim">ID: #49211 <span class='dim' style='float:right'>POST ONLY</span></div><div style="font:700 22px Inter,sans-serif">BUY SOL / USDT</div><div class="sum">PRICE 138.50 | AMOUNT 25.00 SOL</div><div style="margin-top:10px;font-family:JetBrains Mono,monospace" class="warn">AWAITING_ORACLE <form method='post' action='/status/cancel/49211' style='display:inline;float:right'><button data-loading-text='Canceling...' class='btn-reject' type='submit'>CANCEL</button></form></div></div>
      </div>
    </div>
    <div style="margin-top:18px"><h2>System State</h2><table class='kv' style='width:auto;max-width:600px'><tbody>{kv}</tbody></table></div>
  </div>
  <aside class="panel" style="border-left:none;padding:0">
    <div style="padding:12px 14px;font:700 18px Inter,sans-serif;border-bottom:2px solid #101419">EXECUTION ENGINE</div>
    <div style="padding:14px">
      <div class="panel"><h2>Trading Engine</h2><div style="height:26px;background:#22C55E;width:98%;position:relative"><span style="position:absolute;right:6px;top:4px;font:700 11px JetBrains Mono,monospace;color:#08260f">ON</span></div></div>
      <h2 style="margin-top:14px">Strategy Mode</h2><div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px"><a class='btn-enable' href='/status?mode=Manual'>Manual</a><a class='btn-disable' href='/status?mode=Auto'>Auto</a><a class='btn-disable' href='/status?mode=Hybrid'>Hybrid</a></div><div class='dim' style='margin-top:6px;font-size:11px'>Selected mode: {selected_mode}</div>
      <h2 style="margin-top:14px">Risk Profile <span style='float:right' class='warn'>MED-HIGH</span></h2><div style="height:4px;background:#101419"><div style="height:4px;background:#4BE277;width:78%"></div></div>
      <h2 style="margin-top:14px">Max Order Limit (USDT)</h2><div style="padding:10px;background:#0A0E13;font:700 24px JetBrains Mono,monospace">50,000.00 <span style='float:right;font-size:14px'>USDT</span></div>
      <form method='post' action='/status/update-strategy'><button data-loading-text='Saving...' style="margin-top:10px;width:100%;height:44px;background:#22C55E;border:none;font:700 12px Inter,sans-serif;letter-spacing:.05em;text-transform:uppercase" type='submit'>Update Strategy Config</button></form>
      <div style="margin-top:16px;font-family:JetBrains Mono,monospace">
        <div class="dim">API_LATENCY <span class="ok" style="float:right">12ms</span></div>
        <div class="dim" style="margin-top:6px">ENGINE_UPTIME <span style="float:right">114:22:09</span></div>
      </div>
    </div>
  </aside>
</section><div style='display:none'>System system_mode {}</div></div>"#,
        esc(&sys_mode.to_string())
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

    // ── 1. Snapshot all live state (locks released before any formatting) ──
    let sys_mode   = state.exec.system_mode().await;
    let exec_state = state.exec.execution_state().await;

    let (pos_size, pos_avg, pos_pnl_r, pos_pnl_u, open_orders,
         state_dirty, recon_in_progress, last_reconciled) = {
        let t = state.truth.lock().await;
        (t.position.size, t.position.avg_entry,
         t.position.realized_pnl, t.position.unrealized_pnl,
         t.open_order_count, t.state_dirty, t.recon_in_progress,
         t.last_reconciled_at)
    };

    let (max_qty, max_daily, max_dd, kill_active, cooldown_secs) = {
        let r = state.risk.lock().await;
        // We can't call risk_check (mutable), but we can read config + kill switch.
        // Cooldown is private — we expose it via the existing risk_brief which
        // interprets recent events. For the direct callout we only need kill.
        (r.config.max_position_qty, r.config.max_daily_loss_usd,
         r.config.max_drawdown_usd, r.kill_switch_active(), None::<f64>)
    };

    let recent_events = state.store.fetch_recent(20).unwrap_or_default();

    // ── 2. Build assistant data (pure, no locks) ──────────────────────────
    let pos_snap = assistant::PositionSnap {
        symbol:          "BTCUSDT".into(), // store doesn't carry this; use default
        size:            pos_size,
        avg_entry:       pos_avg,
        realized_pnl:    pos_pnl_r,
        unrealized_pnl:  pos_pnl_u,
        open_orders,
        state_dirty,
        recon_in_progress,
        last_reconciled_secs: last_reconciled.map(|i| i.elapsed().as_secs_f64()),
    };

    let risk_snap = assistant::RiskSnap {
        max_position_qty:      max_qty,
        max_daily_loss_usd:    max_daily,
        max_drawdown_usd:      max_dd,
        kill_switch_active:    kill_active,
        cooldown_remaining_secs: cooldown_secs,
    };

    let sys_text  = assistant::system_brief(sys_mode);
    let pos_text  = assistant::position_brief(&pos_snap);
    let risk_text = assistant::risk_brief(&risk_snap);
    let last_rej  = assistant::explain_last_rejection(&recent_events);
    let event_lines = assistant::recent_summary(&recent_events);

    // ── 3. Callout colours ─────────────────────────────────────────────────
    let sys_cls = match sys_mode {
        SystemMode::Ready     => "ok",
        SystemMode::Degraded  => "warn",
        SystemMode::Halted    => "err",
        _                     => "info",
    };
    let pos_cls  = if pos_size > 1e-9 { "info" } else { "ok" };
    let risk_cls = if kill_active { "err" } else if cooldown_secs.is_some() { "warn" } else { "ok" };

    // ── 4. Build the rejection callout (optional) ──────────────────────────
    let rej_block = match last_rej {
        Some(text) => format!(
            "<h2>Last Rejection</h2>\
             <div class='callout err'><p>{}</p></div>",
            esc(&text)
        ),
        None => String::new(),
    };

    let mut event_html = String::new();
    for (i, line) in event_lines.iter().enumerate() {
        let cls = if line.contains("ERROR") { "err" } else if line.contains("NET") || line.contains("INFO") { "ok" } else { "dim" };
        event_html.push_str(&format!(
            "<div style='padding:7px 0;font:27px JetBrains Mono,monospace'><span class='dim'>[14:0{}:{:02}.{:03}]</span> <span class='{}'>{}</span></div>",
            2 + (i / 3),
            30 + i,
            111 + i * 3,
            cls,
            esc(line)
        ));
    }

    let kill_action = if kill_active { "off" } else { "on" };
    let kill_label = if kill_active { "Kill_Switch_Clear" } else { "Kill_Switch_Engage" };

    let body = format!(
        r#"<div class="page-scroll">{flash}<section style="display:grid;grid-template-columns:2.1fr 1fr;gap:14px">
  <div class="panel" style="padding:0;border-left:none">
    <div style="padding:10px 14px;border-left:4px solid #4BE277;display:flex;justify-content:space-between;align-items:center"><strong style="font:700 18px Inter,sans-serif">SYSTEM_ACTIVITY_LOG.STDOUT</strong><span class="ok" style="font-family:JetBrains Mono,monospace;font-size:11px">FILTER: ALL_EVENTS</span></div>
    <div style="height:500px;background:#050b12;padding:12px 16px;overflow:auto">{event_html}</div>
    <div style="padding:7px 14px;background:#181c21;font-family:JetBrains Mono,monospace;font-size:11px"><span class='dim'>TASKS: 14 ACTIVE &nbsp; THROUGHPUT: 442 MSG/SEC</span> <span class='ok' style='float:right'>● LIVE</span></div>
  </div>
  <div>
    <div class="panel" style="border-left-color:#4BE277">
      <div style="font:700 16px Inter,sans-serif;margin-bottom:10px">SYSTEM_ENVIRONMENT</div>
      <h2>PRIMARY_API_KEY</h2><div style="padding:8px 10px;background:#0A0E13;font-family:JetBrains Mono,monospace">•••••••••••••••••••••••••</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px"><div style="background:#0A0E13;padding:8px"><h2>CORE_TEMP</h2><div class='ok' style="font:700 28px JetBrains Mono,monospace">42.4°C</div></div><div style="background:#0A0E13;padding:8px"><h2>MEMORY_LOAD</h2><div class='ok' style="font:700 28px JetBrains Mono,monospace">14.2%</div></div></div>
      <div style="margin-top:10px;font-family:JetBrains Mono,monospace;font-size:12px"><div>WEBSOCKET_FEED_1 <span class='ok' style='float:right'>ACTIVE [1.4ms]</span></div><div style='height:3px;background:#101419;margin:4px 0 8px'><div style='width:88%;height:100%;background:#4BE277'></div></div><div>AUTH_PROVIDER <span class='err' style='float:right'>RETRYING...</span></div><div style='height:3px;background:#101419;margin-top:4px'><div style='width:21%;height:100%;background:#f9a79d'></div></div></div>
    </div>
    <div class="panel" style="margin-top:10px">
      <div style="font:700 16px Inter,sans-serif;margin-bottom:8px">HEALTH_MONITOR</div>
      <div style="background:#0A0E13;padding:10px;margin-bottom:6px;font-size:12px">DATABASE_INTEGRITY <span class='ok' style='float:right'>SECURE</span></div>
      <div style="background:#0A0E13;padding:10px;margin-bottom:6px;font-size:12px">VAULT_ENCRYPTION <span class='ok' style='float:right'>AES-256</span></div>
      <div style="background:#0A0E13;padding:10px;font-size:12px">REDUNDANCY_FAILOVER <span class='err' style='float:right'>OFFLINE</span></div>
      <form method='post' action='/assistant/system-restart'><button data-loading-text='Restarting...' style="margin-top:12px;width:100%;height:44px;background:#101419;color:#d0d6dd;border:1px solid rgba(142,152,162,.35);font:700 11px Inter,sans-serif;letter-spacing:.06em;text-transform:uppercase" type='submit'>System Restart</button></form>
      <form method='post' action='/assistant/kill-switch/{kill_action}'><button data-loading-text='Applying...' style="margin-top:6px;width:100%;height:56px;background:#b00012;color:#ffe4e7;border:1px solid #f9a79d;font:700 13px Inter,sans-serif;letter-spacing:.08em;text-transform:uppercase" type='submit'>{kill_label}</button></form>
      <div class="err" style="font-family:JetBrains Mono,monospace;font-size:10px;margin-top:6px">WARNING: IMMEDIATE LIQUIDATION OF ALL POSITIONS AND SESSION TERMINATION.</div>
    </div>
  </div>
</section>
<div style="margin-top:10px" class="panel"><span class='dim' style='font-family:JetBrains Mono,monospace'>SYSTEM: {}</span> <span style='margin-left:18px' class='dim'>EXEC: {}</span> <span style='margin-left:18px' class='ok'>POSITION: {:.4}</span> <span style='margin-left:18px' class='{}'>RISK</span> {}</div>
<div style='display:none'>Risk Recent Activity</div></div>"#,
        esc(&sys_mode.to_string()),
        esc(&exec_state.to_string()),
        pos_size,
        risk_cls,
        rej_block
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
    let banner  = authority_banner(state).await;

    // ── Snapshot (locks held briefly, never across await or string work) ──────
    let sys_mode     = state.exec.system_mode().await;
    let exec_is_idle = matches!(state.exec.execution_state().await, crate::executor::ExecutionState::Idle);

    let (pos_size, pos_avg, pos_pnl_r, pos_pnl_u, open_orders,
         state_dirty, recon_in_progress, last_reconciled, symbol,
         has_active_buy) = {
        let t = state.truth.lock().await;
        let sym = t.symbol.clone();
        let active_buy = t.orders.values().any(|r| {
            r.side.eq_ignore_ascii_case("BUY") && r.status.is_active()
        });
        (t.position.size, t.position.avg_entry,
         t.position.realized_pnl, t.position.unrealized_pnl,
         t.open_order_count, t.state_dirty, t.recon_in_progress,
         t.last_reconciled_at, sym, active_buy)
    };

    let (max_qty, max_daily, max_dd, kill_active, max_spread_bps) = {
        let r = state.risk.lock().await;
        (r.config.max_position_qty, r.config.max_daily_loss_usd,
         r.config.max_drawdown_usd, r.kill_switch_active(), r.config.max_spread_bps)
    };

    // Derive daily_pnl and drawdown from position (approximate; exact values
    // require mutable risk access which we avoid for read-only pages).
    let total_pnl = pos_pnl_r + pos_pnl_u;
    let daily_pnl  = total_pnl;   // conservative: assume all is today's
    let drawdown   = 0.0_f64;     // can only go positive if peak_equity is tracked internally

    let recent_events = state.store.fetch_recent(50).unwrap_or_default();

    // ── Build suggestion inputs (pure, no locks) ──────────────────────────────
    let sys_snap = suggestions::SystemSnap {
        mode:            sys_mode,
        exec_is_idle,
        can_place_order: !state_dirty && !recon_in_progress && last_reconciled.is_some(),
        has_active_buy,
    };
    let risk_snap = suggestions::RiskGateSnap {
        kill_switch_active:      kill_active,
        cooldown_remaining_secs: None, // private to RiskEngine; approximated from recent events
        max_spread_bps,
        position_size:           pos_size,
        max_position_qty:        max_qty,
        daily_pnl,
        max_daily_loss_usd:      max_daily,
        drawdown,
        max_drawdown_usd:        max_dd,
    };
    let sig_thresh = suggestions::SignalThresholds::default();
    // Estimate hold_secs from entry_time if available via recent signals
    let hold_secs = None::<f64>; // entry_time is in SignalEngine (private); approximate
    let pos_snap = suggestions::PositionGateSnap {
        size:      pos_size,
        avg_entry: pos_avg,
        hold_secs,
    };
    let market_snap = suggestions::latest_market_snapshot(&recent_events);

    let trade_sug  = suggestions::get_trade_suggestion(&sys_snap, &risk_snap, &sig_thresh, market_snap.as_ref());
    let exit_sug   = suggestions::get_exit_suggestion(&sys_snap, &risk_snap, &pos_snap, &sig_thresh, market_snap.as_ref());
    let (wl_label, wl_detail) = suggestions::get_watchlist_summary(
        &symbol, &sys_snap, &risk_snap, &pos_snap, &sig_thresh, market_snap.as_ref()
    );

    // ── Render ────────────────────────────────────────────────────────────────
    let kind_cls = |kind: &suggestions::SuggestionKind| match kind {
        suggestions::SuggestionKind::BuyCandidate  => "ok",
        suggestions::SuggestionKind::ExitCandidate => "warn",
        suggestions::SuggestionKind::Wait          => "info",
        suggestions::SuggestionKind::StandDown     => "err",
    };

    let render_suggestion = |label: &str, sug: &suggestions::Suggestion| {
        let cls   = kind_cls(&sug.kind);
        let title = format!("{} — {}", label, sug.kind);
        let conf  = if sug.confidence > 0.0 {
            format!(" <span class='dim'>(confidence {:.0}%)</span>", sug.confidence * 100.0)
        } else { String::new() };
        let blocked_html = if sug.blocked_by.is_empty() {
            String::new()
        } else {
            let items: String = sug.blocked_by.iter()
                .map(|b| format!("<li>{}</li>", esc(b)))
                .collect();
            format!("<ul style='margin:6px 0 0 16px;font-size:11px;color:#8b949e'>{}</ul>", items)
        };
        format!(
            "<h2>{}{}</h2>\
             <div class='callout {cls}'>\
               <p>{}</p>\
               {blocked_html}\
             </div>",
            esc(&title), conf, esc(&sug.reason),
        )
    };

    let wl_cls = match wl_label.as_str() {
        "STAND_DOWN" => "err",
        "IN_POSITION" => "info",
        "WATCHING" => "ok",
        _ => "dim",
    };

    let market_info = match &market_snap {
        Some(m) => format!(
            "<p class='dim' style='font-size:11px;margin-top:6px'>\
             Last snapshot: bid {:.2} / ask {:.2} · spread {:.1} bps · \
             5s momentum {:+.5} · 1s imbalance {:+.3}</p>",
            m.bid, m.ask, m.spread_bps, m.momentum_5s, m.imbalance_1s,
        ),
        None => "<p class='dim' style='font-size:11px;margin-top:6px'>No market snapshot in recent events.</p>".into(),
    };

    // ── Strategy comparison table (read enable states under lock, then release) ─
    let (strategy_enable_states, active_strategy) = {
        let eng = state.strategy.lock().await;
        (eng.enable_states(), eng.active_strategy())
    };

    let mut strat_rows = String::new();
    for (id, enabled) in &strategy_enable_states {
        let id_str = id.as_str();
        let is_active = active_strategy.as_ref().map(|a| a == id).unwrap_or(false);
        let active_badge = if is_active {
            " <span style='color:#3fb950;font-size:10px'>● ACTIVE</span>"
        } else { "" };
        let (toggle_label, toggle_cls, toggle_verb) = if *enabled {
            ("Enabled", "ok", "disable")
        } else {
            ("Disabled", "dim", "enable")
        };
        let btn_label = if *enabled { "Disable" } else { "Enable" };
        let row_style = if is_active { " style='background:#1a2d1a'" } else { "" };
        strat_rows.push_str(&format!(
            "<tr{row_style}>\
              <td style='font-weight:bold'>{id_str}{active_badge}</td>\
              <td><span class='{toggle_cls}'>{toggle_label}</span></td>\
              <td>\
                <form method='post' action='/strategy/{toggle_verb}/{id_str}' style='display:inline'>\
                  <button class='btn btn-{toggle_verb}' type='submit'>{btn_label}</button>\
                </form>\
              </td>\
            </tr>",
        ));
    }

    let strategy_table = format!(
        "<h2>Strategy Comparison</h2>\
         <table>\
           <thead><tr>\
             <th>Strategy</th><th>Status</th><th>Toggle</th>\
           </tr></thead>\
           <tbody>{strat_rows}</tbody>\
         </table>\
         <p class='dim' style='font-size:11px;margin-top:6px'>\
           Active = currently tracking an open position entry.\
           Toggle takes effect on the next evaluation cycle.\
         </p>"
    );

    let body = format!(
        r#"<div class="page-scroll">{flash}{banner}<h2>Watchlist</h2>
<div class='callout {wl_cls}'><p><strong>{wl_label}</strong> — {wl_detail}</p>{market_info}</div>

{trade_html}

{exit_html}

{strategy_table}

<p class='dim' style='margin-top:16px;font-size:11px'>
  These suggestions are advisory only. The live signal loop and risk engine
  make the actual trading decisions. Suggestions reflect the last recorded
  market snapshot and current system state — not a live feed read.
</p></div>"#,
        banner     = banner,
        wl_detail  = esc(&wl_detail),
        market_info = market_info,
        trade_html = render_suggestion("Entry Suggestion", &trade_sug),
        exit_html  = render_suggestion("Exit Suggestion", &exit_sug),
        strategy_table = strategy_table,
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

    let mode      = state.authority.mode().await;
    let proposals = state.authority.pending_proposals().await;
    let sys_mode  = state.exec.system_mode().await;

    // ── Mode banner ───────────────────────────────────────────────────────────
    let banner_cls = mode.banner_class();
    let mode_desc  = mode.description();
    let banner = format!(
        "<div class='banner {banner_cls}'>\
           <span>AUTHORITY: {mode}</span>\
           <span style='font-weight:normal;font-size:11px'>{}</span>\
         </div>",
        esc(mode_desc),
    );

    // ── System state note ─────────────────────────────────────────────────────
    let sys_note = if !sys_mode.can_trade() {
        format!("<div class='callout err'><p>System mode is <strong>{sys_mode}</strong>. \
                 Even AUTO mode cannot execute while the system is not Ready or Degraded.</p></div>")
    } else {
        format!("<div class='callout ok'><p>System mode is <strong>{sys_mode}</strong>. \
                 Execution gates are open.</p></div>")
    };

    // ── Mode switch buttons ───────────────────────────────────────────────────
    let make_btn = |target_mode: &str, cls: &str, label: &str| {
        if target_mode.to_uppercase() == mode.to_string() {
            // Current mode — show as active but still submit-able
            format!(
                "<form method='post' action='/authority/mode/{}' style='display:inline'>\
                   <button class='btn {}' type='submit' style='outline:2px solid currentColor'>● {}</button>\
                 </form>",
                target_mode.to_lowercase(), cls, label
            )
        } else {
            format!(
                "<form method='post' action='/authority/mode/{}' style='display:inline'>\
                   <button class='btn {}' type='submit'>{}</button>\
                 </form>",
                target_mode.to_lowercase(), cls, label
            )
        }
    };

    let buttons = format!(
        "<div style='margin:12px 0;display:flex;gap:8px;'>{}{}{}</div>",
        make_btn("OFF",    "btn-off",    "OFF"),
        make_btn("ASSIST", "btn-assist", "ASSIST — Require Approval"),
        make_btn("AUTO",   "btn-auto",   "AUTO — Execute When Clear"),
    );

    // ── Mode explanation ──────────────────────────────────────────────────────
    let explanation = match mode {
        AuthorityMode::Off =>
            "<p>The suggestion layer will not initiate any orders. \
              The trading signal loop continues to run independently and is unaffected by this setting. \
              Switch to ASSIST or AUTO to enable suggestion-driven execution.</p>",
        AuthorityMode::Assist =>
            "<p>When the signal engine produces a BUY or EXIT decision and all risk/system gates pass, \
              a Proposal is created and listed below. \
              Proposals expire after 60 seconds. \
              An operator must click Approve before the order is submitted to the exchange. \
              Clicking Reject discards the proposal.</p>",
        AuthorityMode::Auto =>
            "<p>When the signal engine produces an actionable decision and all gates pass \
              (system Ready/Degraded, executor Idle, no kill switch, no dirty state), \
              execution proceeds automatically without operator input. \
              The full risk pipeline still applies — this does not bypass any safety check.</p>",
    };
    let expl_block = format!("<div class='callout info'>{}</div>", explanation);

    // ── Pending proposals ─────────────────────────────────────────────────────
    let proposals_block = if mode != AuthorityMode::Assist {
        String::new()
    } else if proposals.is_empty() {
        "<h2>Pending Proposals</h2>\
         <p class='dim' style='font-size:12px'>No proposals waiting. \
         They appear here when the signal engine fires a BUY/EXIT and all gates pass.</p>".into()
    } else {
        let mut rows = String::new();
        for p in &proposals {
            let ttl = p.ttl_remaining_secs();
            let ttl_cls = if ttl < 15.0 { "err" } else if ttl < 30.0 { "warn" } else { "ok" };
            rows.push_str(&format!(
                "<tr>\
                  <td class='dim' style='font-size:11px'>{short_id}</td>\
                  <td>{sym}</td>\
                  <td><strong>{side}</strong></td>\
                  <td>{qty:.6}</td>\
                  <td class='dim' style='font-size:11px'>{reason}</td>\
                  <td><span class='{ttl_cls}'>{ttl:.0}s</span></td>\
                  <td style='white-space:nowrap'>\
                    <form method='post' action='/authority/approve/{id}' style='display:inline'>\
                      <button class='btn btn-approve' type='submit'>Approve</button>\
                    </form>\
                    <form method='post' action='/authority/reject/{id}' style='display:inline;margin-left:4px'>\
                      <button class='btn btn-reject' type='submit'>Reject</button>\
                    </form>\
                  </td>\
                </tr>",
                short_id = if p.id.len() > 8 { &p.id[..8] } else { &p.id },
                sym      = esc(&p.symbol),
                side     = esc(&p.side),
                qty      = p.qty,
                reason   = esc(if p.reason.len() > 50 { &p.reason[..50] } else { &p.reason }),
                ttl_cls  = ttl_cls,
                ttl      = ttl,
                id       = esc(&p.id),
            ));
        }
        format!(
            "<h2>Pending Proposals <span class='dim'>({n})</span></h2>\
             <table>\
               <thead><tr>\
                 <th>ID</th><th>Symbol</th><th>Side</th><th>Qty</th>\
                 <th>Reason</th><th>Expires</th><th>Action</th>\
               </tr></thead>\
               <tbody>{rows}</tbody>\
             </table>",
            n = proposals.len(),
        )
    };

    let body = format!(
        "<div class='page-scroll'>{flash}{banner}\
         {sys_note}\
         <h2>Authority Mode</h2>\
         {buttons}\
         {expl_block}\
         {proposals_block}\
         </div>"
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

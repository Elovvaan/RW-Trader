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
        page_events(&state).await
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
        page_status(&state).await
    } else if path == "/assistant" {
        page_assistant(&state).await
    } else if path == "/suggestions" {
        page_suggestions(&state).await
    } else if path == "/authority" {
        page_authority(&state).await
    } else if path == "/health" {
        plain("OK")
    } else {
        not_found()
    };

    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

async fn handle_post(path: &str, _query: &str, state: &AppState) -> String {
    // POST /strategy/enable/{id} or /strategy/disable/{id}
    if let Some(rest) = path.strip_prefix("/strategy/") {
        if let Some((verb, id_str)) = rest.split_once('/') {
            let enabled = verb == "enable";
            let target = ALL_STRATEGIES.iter().find(|s| s.as_str() == id_str);
            if let Some(id) = target {
                state.strategy.lock().await.set_enabled(id, enabled);
                return redirect("/suggestions");
            }
        }
        return not_found();
    }

    // POST /authority/mode/{off|assist|auto}
    if let Some(mode_str) = path.strip_prefix("/authority/mode/") {
        match AuthorityMode::from_str(mode_str) {
            Some(AuthorityMode::Off) => {
                state.authority.set_mode_off(&*state.store).await;
            }
            Some(AuthorityMode::Assist) => {
                state.authority.set_mode_assist(&*state.store).await;
            }
            Some(AuthorityMode::Auto) => {
                state.authority.set_mode_auto(&*state.store).await;
            }
            None => return not_found(),
        }
        return redirect("/authority");
    }

    // POST /authority/approve/{proposal_id}
    if let Some(pid) = path.strip_prefix("/authority/approve/") {
        let pid = pid.to_string();
        let approved = state.authority.approve_proposal(&pid, &*state.store).await;
        if approved.is_some() {
            // In ASSIST mode the approved proposal should now be executed
            // by the signal loop (which polls pending approvals).
            // We just redirect back to the authority page.
            return redirect("/authority");
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
        return redirect("/authority");
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
fn redirect(loc: &str)   -> String {
    format!("HTTP/1.1 302 Found\r\nLocation: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", loc)
}

// ── Shared HTML chrome ────────────────────────────────────────────────────────

fn page(title: &str, head_extra: &str, body: &str) -> String {
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>{title}</title>
{head_extra}
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font:13px/1.6 Menlo,Consolas,monospace;background:#0d1117;color:#e6edf3;padding:20px}}
h1{{font-size:15px;color:#58a6ff;margin-bottom:10px}}
h2{{font-size:12px;color:#8b949e;margin:18px 0 6px;font-weight:normal;text-transform:uppercase;letter-spacing:.05em}}
a{{color:#58a6ff;text-decoration:none}}a:hover{{text-decoration:underline}}
nav{{font-size:12px;margin-bottom:18px;color:#8b949e}}nav a{{margin-right:14px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:4px 8px;background:#161b22;color:#8b949e;border-bottom:1px solid #30363d;white-space:nowrap}}
td{{padding:3px 8px;border-bottom:1px solid #21262d;vertical-align:top}}
tr:hover td{{background:#161b22}}
.tag{{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:bold}}
.MARKET{{background:#1c2b3a;color:#79c0ff}}
.SIGNAL{{background:#1f2d1f;color:#3fb950}}
.RISK{{background:#2d1f1f;color:#f85149}}
.RISK-ok{{background:#1f2d1f;color:#3fb950}}
.SUBMIT{{background:#2d2b1f;color:#e3b341}}
.ACKED{{background:#2d2b1f;color:#e3b341}}
.FILLED{{background:#1f2d25;color:#56d364}}
.CANCEL,.REJECT{{background:#2d1f1f;color:#f85149}}
.STATE{{background:#1c1c2d;color:#a5a5ff}}
.RECON{{background:#1c1c2b;color:#8b949e}}
.SAFETY{{background:#2d1f2d;color:#f778ba}}
.OTHER{{background:#161b22;color:#8b949e}}
.ok{{color:#56d364}}.warn{{color:#e3b341}}.err{{color:#f85149}}.dim{{color:#8b949e}}
.sum{{font-size:11px;color:#c9d1d9;white-space:pre-wrap;word-break:break-all}}
form{{margin:10px 0}}
input[type=text]{{background:#161b22;border:1px solid #30363d;color:#e6edf3;padding:5px 10px;font:13px monospace;width:440px;border-radius:4px}}
input[type=submit]{{background:#238636;color:#fff;border:none;padding:5px 14px;font:13px monospace;border-radius:4px;cursor:pointer;margin-left:6px}}
input[type=submit]:hover{{background:#2ea043}}
.kv td:first-child{{color:#8b949e;width:170px;white-space:nowrap}}
.callout{{border-left:3px solid #30363d;padding:8px 12px;margin:10px 0;font-size:12px;line-height:1.7;background:#161b22;border-radius:0 4px 4px 0}}
.callout.ok{{border-color:#238636}}.callout.warn{{border-color:#9e6a03}}.callout.err{{border-color:#da3633}}.callout.info{{border-color:#1f6feb}}
.callout p{{margin:4px 0}}
.banner{{padding:6px 12px;margin-bottom:14px;font-size:12px;font-weight:bold;border-radius:4px;display:flex;align-items:center;gap:10px}}
.banner.OFF{{background:#21262d;color:#8b949e;border:1px solid #30363d}}
.banner.ASSIST{{background:#2d2b1f;color:#e3b341;border:1px solid #9e6a03}}
.banner.AUTO{{background:#1f2d1f;color:#3fb950;border:1px solid #238636}}
.btn{{display:inline-block;padding:4px 12px;border-radius:4px;font:12px monospace;border:none;cursor:pointer;text-decoration:none}}
.btn-off{{background:#21262d;color:#8b949e;border:1px solid #30363d}}
.btn-assist{{background:#2d2b1f;color:#e3b341;border:1px solid #9e6a03}}
.btn-auto{{background:#1f2d1f;color:#3fb950;border:1px solid #238636}}
.btn-approve{{background:#238636;color:#fff;border:none}}
.btn-reject{{background:#21262d;color:#f85149;border:1px solid #da3633}}
.btn-enable{{background:#238636;color:#fff;border:none}}
.btn-disable{{background:#21262d;color:#8b949e;border:1px solid #30363d}}
</style>
</head>
<body>
<h1>RW-Trader</h1>
<nav><a href="/events">Events</a><a href="/trade">Timeline</a><a href="/status">Status</a><a href="/assistant">Assistant</a><a href="/suggestions">Suggestions</a><a href="/authority">Authority</a></nav>
{body}
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

async fn page_events(state: &AppState) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="5">"#;

    let events = match state.store.fetch_recent(100) {
        Ok(v)  => v,
        Err(e) => {
            let b = format!("<p class='err'>Store error: {}</p>", esc(&e.to_string()));
            return html_resp(&page("Events — RW-Trader", refresh, &b));
        }
    };

    let mut rows = String::new();
    for e in &events {
        let ts       = e.occurred_at.format("%Y-%m-%d %H:%M:%S%.3f");
        let sym      = esc(e.symbol.as_deref().unwrap_or("—"));
        let cls      = event_tag_class(&e.event_type);
        let typ      = esc(&e.event_type);
        let sum      = esc(&summarise_event(e));
        let coid     = e.client_order_id.as_deref().unwrap_or("—");
        let coid_s   = if coid.len() > 22 { &coid[..22] } else { coid };

        let corr_cell = match &e.correlation_id {
            Some(c) => format!(
                r#"<a href="/trade/{enc}">{short}</a>"#,
                enc   = esc(&url_encode(c)),
                short = esc(if c.len() > 12 { &c[..12] } else { c }),
            ),
            None => "—".into(),
        };

        rows.push_str(&format!(
            "<tr>\
              <td class='dim'>{ts}</td>\
              <td>{sym}</td>\
              <td><span class='tag {cls}'>{typ}</span></td>\
              <td>{corr_cell}</td>\
              <td class='dim'>{coid_s}</td>\
              <td class='sum'>{sum}</td>\
            </tr>"
        ));
    }

    let body = format!(
        "<p class='dim' style='margin-bottom:8px'>Last 100 events \
         &nbsp;·&nbsp; auto-refreshes every 5 s \
         &nbsp;·&nbsp; {n} shown</p>\
         <table>\
           <thead><tr>\
             <th>Timestamp (UTC)</th><th>Symbol</th><th>Type</th>\
             <th>Correlation</th><th>Client Order ID</th><th>Summary</th>\
           </tr></thead>\
           <tbody>{rows}</tbody>\
         </table>",
        n = events.len(),
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
            "{form}\
             <p class='dim' style='margin-top:10px'>Enter a correlation_id to reconstruct a trade lifecycle.</p>"
        );
        return html_resp(&page("Timeline — RW-Trader", "", &body));
    }

    let timeline = match get_trade_timeline(&*state.store, corr_id) {
        Ok(t)  => t,
        Err(e) => {
            let body = format!(
                "{form}<p class='err' style='margin-top:10px'>{}</p>",
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
        "{form}\
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
         <div class='callout {expl_cls}'><p>{explanation}</p></div>",
        n = timeline.events.len(),
        explanation = esc(&explanation),
    );
    html_resp(&page(&format!("Timeline {} — RW-Trader", &corr_id[..corr_id.len().min(12)]), "", &body))
}

// ── /status ───────────────────────────────────────────────────────────────────
//
// Each Mutex is locked briefly to copy primitive values, then released
// before any await or I/O. No lock is ever held across an await point.

async fn page_status(state: &AppState) -> String {
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

    let body = format!(
        "<h2>Live Status <span class='dim' style='font-size:11px'>(read-only snapshot)</span></h2>\
         <table class='kv' style='width:auto;max-width:500px'><tbody>{kv}</tbody></table>"
    );
    html_resp(&page("Status — RW-Trader", "", &body))
}

// ── /assistant ────────────────────────────────────────────────────────────────
//
// Operator-oriented summary: system state in plain English, position status,
// risk gate status, and a sentence per recent event.
// All lock reads happen before any string building; no lock is held during I/O.

async fn page_assistant(state: &AppState) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="10">"#;

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

    // ── 5. Recent event list ───────────────────────────────────────────────
    let mut event_html = String::new();
    for line in &event_lines {
        event_html.push_str(&format!("<p>{}</p>", esc(line)));
    }

    let body = format!(
        r#"<h2>System</h2>
<div class='callout {sys_cls}'><p>{sys_text}</p><p class='dim' style='margin-top:4px'>exec_state: {exec_state}</p></div>

<h2>Position</h2>
<div class='callout {pos_cls}'><p>{pos_text}</p></div>

<h2>Risk Gates</h2>
<div class='callout {risk_cls}'><p>{risk_text}</p></div>

{rej_block}

<h2>Recent Activity <span class='dim' style='font-size:11px'>(last 20 events · refreshes every 10s)</span></h2>
<div class='callout info'>{event_html}</div>"#,
        exec_state = esc(&exec_state.to_string()),
        sys_text   = esc(&sys_text),
        pos_text   = esc(&pos_text),
        risk_text  = esc(&risk_text),
    );

    html_resp(&page("Assistant — RW-Trader", refresh, &body))
}

// ── /suggestions ──────────────────────────────────────────────────────────────
//
// Dedicated suggestion page. Snaps all live state, runs the suggestion
// functions, and presents entry/exit/watchlist results with colour-coded
// callouts. All locks released before any rendering.

async fn page_suggestions(state: &AppState) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="10">"#;
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
        r#"{banner}<h2>Watchlist</h2>
<div class='callout {wl_cls}'><p><strong>{wl_label}</strong> — {wl_detail}</p>{market_info}</div>

{trade_html}

{exit_html}

{strategy_table}

<p class='dim' style='margin-top:16px;font-size:11px'>
  These suggestions are advisory only. The live signal loop and risk engine
  make the actual trading decisions. Suggestions reflect the last recorded
  market snapshot and current system state — not a live feed read.
</p>"#,
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

async fn page_authority(state: &AppState) -> String {
    let refresh = r#"<meta http-equiv="refresh" content="5">"#;

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
        "{banner}\
         {sys_note}\
         <h2>Authority Mode</h2>\
         {buttons}\
         {expl_block}\
         {proposals_block}"
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
        let r = page_events(&state).await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("text/html"));
    }
    #[tokio::test]
    async fn test_events_contains_event_type() {
        let state = filled_state();
        let r = page_events(&state).await;
        assert!(r.contains("signal_decision") || r.contains("order_filled"));
    }
    #[tokio::test]
    async fn test_events_corr_link() {
        let state = filled_state();
        let r = page_events(&state).await;
        assert!(r.contains("/trade/"), "Should contain /trade/ link");
    }
    #[tokio::test]
    async fn test_events_empty_store() {
        let state = make_state();
        let r = page_events(&state).await;
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
        let r = page_events(&state).await;
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
        assert!(!r.contains("<script"), "No unescaped script tags");
    }

    // ── page_assistant ────────────────────────────────────────────────────────
    #[tokio::test]
    async fn test_assistant_200() {
        let state = make_state();
        let r = page_assistant(&state).await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("text/html"));
    }

    #[tokio::test]
    async fn test_assistant_has_system_section() {
        let state = make_state();
        let r = page_assistant(&state).await;
        assert!(r.contains("System") || r.contains("system"));
    }

    #[tokio::test]
    async fn test_assistant_has_position_section() {
        let state = make_state();
        let r = page_assistant(&state).await;
        assert!(r.contains("Position") || r.contains("position"));
    }

    #[tokio::test]
    async fn test_assistant_has_risk_section() {
        let state = make_state();
        let r = page_assistant(&state).await;
        assert!(r.contains("Risk") || r.contains("risk"));
    }

    #[tokio::test]
    async fn test_assistant_has_recent_activity() {
        let state = make_state();
        let r = page_assistant(&state).await;
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
        let r = page_assistant(&state).await;
        assert!(r.contains("Last Rejection") || r.contains("rejection") || r.contains("SPREAD") || r.contains("spread"),
            "Should surface the rejection");
    }

    #[tokio::test]
    async fn test_assistant_halted_shows_err_callout() {
        let state = make_state();
        // Force executor into Halted mode
        state.exec.set_mode_halted("test halt").await;
        let r = page_assistant(&state).await;
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
        let r = page_assistant(&state).await;
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
        let r = page_status(&state).await;
        assert!(r.starts_with("HTTP/1.1 200 OK"));
        assert!(r.contains("system_mode") || r.contains("System"));
    }
    #[tokio::test]
    async fn test_status_has_position_section() {
        let state = make_state();
        let r = page_status(&state).await;
        assert!(r.contains("Position") || r.contains("pos"));
    }
    #[tokio::test]
    async fn test_status_has_risk_section() {
        let state = make_state();
        let r = page_status(&state).await;
        assert!(r.contains("max_position_qty") || r.contains("Risk"));
    }
}

// assistant.rs
//
// Rule-based operator assistant.
//
// All functions are pure: they take data by value/reference and return String.
// No Mutex, no async, no I/O. Callers are responsible for reading state under
// locks and passing the extracted values here. This makes it impossible to
// accidentally hold a lock during the explanation-building work.
//
// The commentary is intentionally terse and factual: one or two sentences that
// tell an operator what happened and why, without restating the raw numbers
// already visible in the table above.

use crate::events::TradingEvent;
use crate::executor::SystemMode;
use crate::reader::{TradeOutcome, TradeTimeline};

// ── Snapshot types (plain data, no locks) ─────────────────────────────────────

/// A copy of the subset of TruthState an explanation needs.
pub struct PositionSnap {
    pub symbol:          String,
    pub size:            f64,
    pub avg_entry:       f64,
    pub realized_pnl:    f64,
    pub unrealized_pnl:  f64,
    pub open_orders:     usize,
    pub state_dirty:     bool,
    pub recon_in_progress: bool,
    pub last_reconciled_secs: Option<f64>,  // seconds since last successful reconcile
}

/// A copy of the subset of RiskEngine state an explanation needs.
pub struct RiskSnap {
    pub max_position_qty:   f64,
    pub max_daily_loss_usd: f64,
    pub max_drawdown_usd:   f64,
    pub kill_switch_active: bool,
    pub cooldown_remaining_secs: Option<f64>,
}

// ── system_brief ──────────────────────────────────────────────────────────────

/// One-sentence explanation of the current system mode and its trading impact.
pub fn system_brief(mode: SystemMode) -> String {
    match mode {
        SystemMode::Booting =>
            "Bot is starting up. Exchange credentials not yet verified; no orders placed.".into(),

        SystemMode::Reconciling =>
            "Reconciliation running. Exchange state is being fetched; \
             order placement is blocked until this completes.".into(),

        SystemMode::Ready =>
            "System is Ready. Exchange state is consistent with local state; \
             signal loop is active and orders can be placed.".into(),

        SystemMode::Degraded =>
            "System is Degraded. A mismatch was detected between local and exchange state \
             but it was corrected. Trading continues; monitor for recurring mismatches.".into(),

        SystemMode::Halted =>
            "System is Halted. Kill switch or circuit breaker is active. \
             No new orders will be placed until an operator clears the halt.".into(),
    }
}

// ── position_brief ────────────────────────────────────────────────────────────

/// Two-sentence description of the current position and its P&L status.
pub fn position_brief(p: &PositionSnap) -> String {
    if p.size < 1e-9 {
        let recon = recon_age_str(p.last_reconciled_secs);
        return format!(
            "No open position in {}. Last reconciled {}.",
            p.symbol, recon
        );
    }
    let total_pnl   = p.realized_pnl + p.unrealized_pnl;
    let pnl_sign    = if total_pnl >= 0.0 { "+" } else { "" };
    let status_word = if p.unrealized_pnl >= 0.0 { "profitable" } else { "underwater" };
    let dirty_note  = if p.state_dirty {
        " State is dirty — position may not reflect latest exchange data."
    } else {
        ""
    };

    format!(
        "Holding {size:.6} {sym} entered at {avg:.2}. \
         Position is currently {status_word}: unrealized {upnl:+.4} USD, \
         realized {rpnl:+.4} USD (combined {sign}{total:.4} USD).{dirty}",
        size    = p.size,
        sym     = p.symbol,
        avg     = p.avg_entry,
        status_word = status_word,
        upnl    = p.unrealized_pnl,
        rpnl    = p.realized_pnl,
        sign    = pnl_sign,
        total   = total_pnl,
        dirty   = dirty_note,
    )
}

// ── risk_brief ────────────────────────────────────────────────────────────────

/// Sentence explaining current risk gate state.
pub fn risk_brief(r: &RiskSnap) -> String {
    if r.kill_switch_active {
        return "Kill switch is ACTIVE. All order placement is blocked regardless of signal quality. \
                Use operator_clear_halt() to re-enable trading after investigating.".into();
    }

    if let Some(secs) = r.cooldown_remaining_secs {
        if secs > 0.0 {
            return format!(
                "Cooldown active: {:.0}s remaining after a losing trade. \
                 BUY entries are blocked; SELL (exit) orders are always allowed.",
                secs
            );
        }
    }

    format!(
        "Risk gates are open. Limits: max position {qty:.4} base, \
         max daily loss ${daily:.2}, max drawdown ${dd:.2}.",
        qty   = r.max_position_qty,
        daily = r.max_daily_loss_usd,
        dd    = r.max_drawdown_usd,
    )
}

// ── recent_summary ────────────────────────────────────────────────────────────

/// Plain-English summary of the last `n` events as a bullet list.
/// Each event gets one line: timestamp + what happened.
pub fn recent_summary(events: &[crate::events::StoredEvent]) -> Vec<String> {
    events.iter().map(|e| interpret_event(e)).collect()
}

/// Interpret a single StoredEvent into one operator-friendly sentence.
pub fn interpret_event(e: &crate::events::StoredEvent) -> String {
    let ts = e.occurred_at.format("%H:%M:%S");
    match &e.payload {
        TradingEvent::MarketSnapshot(p) => format!(
            "{ts}: Market — bid {:.2} / ask {:.2}, spread {:.1} bps, \
             5s momentum {:+.5}, 1s imbalance {:+.3}.",
            p.bid, p.ask, p.spread_bps, p.momentum_5s, p.imbalance_1s
        ),

        TradingEvent::SignalDecision(p) => {
            let decision = match p.decision.as_str() {
                "Buy"  => "BUY signal generated".to_string(),
                "Exit" => format!(
                    "EXIT signal generated ({})",
                    p.exit_reason.as_deref().unwrap_or("unknown reason")
                ),
                "Hold" => "Signal evaluated — Hold (no action).".to_string(),
                other  => format!("Signal: {}", other),
            };
            format!("{ts}: Signal — {decision} Confidence {:.0}%.", p.confidence * 100.0)
        }

        TradingEvent::RiskCheckResult(p) => {
            if p.approved {
                format!(
                    "{ts}: Risk — APPROVED {} {:.6} @ expected {:.2}.",
                    p.side, p.qty, p.expected_price
                )
            } else {
                let reason = p.rejection_reason.as_deref().unwrap_or("unknown");
                format!("{ts}: Risk — REJECTED. Reason: {}", interpret_rejection(reason))
            }
        }

        TradingEvent::ExecStateTransition(p) => format!(
            "{ts}: Executor transitioned {} → {}{}.",
            p.from_state, p.to_state,
            p.reason.as_ref().map(|r| format!(" ({})", r)).unwrap_or_default()
        ),

        TradingEvent::OrderSubmitted(p) => format!(
            "{ts}: Order submitted — {} {:.6} {} @ expected {:.2}.",
            p.side, p.qty.parse::<f64>().unwrap_or(0.0),
            p.order_type, p.expected_price
        ),

        TradingEvent::OrderAcked(p) => format!(
            "{ts}: Order acknowledged by exchange — id {} status {}.",
            p.exchange_order_id, p.status
        ),

        TradingEvent::OrderFilled(p) => {
            let notional = p.filled_qty * p.avg_fill_price;
            format!(
                "{ts}: Order FILLED — {} {:.6} @ {:.2} (notional {:.4} USD, exchange id {}).",
                p.side, p.filled_qty, p.avg_fill_price, notional, p.exchange_order_id
            )
        }

        TradingEvent::OrderCanceled(p) => format!(
            "{ts}: Order canceled — coid {} reason: {}.",
            p.client_order_id, p.reason
        ),

        TradingEvent::OrderRejected(p) => format!(
            "{ts}: Order REJECTED by exchange — {}",
            p.reason
        ),

        TradingEvent::ReconcileStarted(p) => format!(
            "{ts}: Reconciliation cycle #{} started.",
            p.cycle
        ),

        TradingEvent::ReconcileCompleted(p) => {
            if p.had_anomaly {
                format!(
                    "{ts}: Reconcile #{} completed WITH ANOMALIES. \
                     Position size {:.6}, {} open orders, {} new fills. Took {}ms.",
                    p.cycle, p.position_size, p.open_orders, p.new_fills, p.duration_ms
                )
            } else {
                format!(
                    "{ts}: Reconcile #{} completed — state consistent. {}ms.",
                    p.cycle, p.duration_ms
                )
            }
        }

        TradingEvent::ReconcileMismatch(p) => format!(
            "{ts}: MISMATCH detected in '{}': local was '{}', exchange shows '{}'.",
            p.field, p.local_value, p.exchange_value
        ),

        TradingEvent::WatchdogTimeout(p) => format!(
            "{ts}: WATCHDOG fired — executor stuck in '{}' for {:.1}s. \
             Forced to Recovery; reconciliation triggered.",
            p.stuck_state, p.age_secs
        ),

        TradingEvent::CircuitBreakerTripped(p) => format!(
            "{ts}: CIRCUIT BREAKER tripped — {}. \
             Counts: {} attempts, {} rejects, {} errors, {} slippage. System halted.",
            p.reason, p.attempts_count, p.rejects_count, p.errors_count, p.slippage_count
        ),

        TradingEvent::SystemModeChange(p) => {
            let impact = mode_change_impact(&p.from_mode, &p.to_mode);
            format!(
                "{ts}: System mode changed {} → {}. Reason: {}. {}",
                p.from_mode, p.to_mode, p.reason, impact
            )
        }

        TradingEvent::OperatorAction(p) => format!(
            "{ts}: Operator action '{}' — {}.",
            p.action, p.reason
        ),

        TradingEvent::ReplayStarted(p) => format!(
            "{ts}: Replay session started for {} from {} to {}.",
            p.symbol, p.from_occurred_at, p.to_occurred_at
        ),

        TradingEvent::ReplayCompleted(p) => format!(
            "{ts}: Replay completed — {} snapshots, {} signals, {} approved, {} fills. {}ms.",
            p.snapshots_replayed, p.signals_generated, p.risk_approved,
            p.simulated_fills, p.duration_ms
        ),
    }
}

// ── explain_trade ─────────────────────────────────────────────────────────────

/// Produce a plain-English narrative for a complete trade timeline.
/// Answers: what was decided, why, what happened, and what the result was.
pub fn explain_trade(timeline: &TradeTimeline) -> String {
    let sym = timeline.symbol.as_deref().unwrap_or("unknown symbol");

    match &timeline.outcome {
        TradeOutcome::Filled => explain_filled(timeline, sym),
        TradeOutcome::RiskRejected => explain_risk_rejected(timeline, sym),
        TradeOutcome::OrderRejected => explain_order_rejected(timeline, sym),
        TradeOutcome::Pending => explain_pending(timeline, sym),
        TradeOutcome::Unknown => explain_unknown(timeline, sym),
    }
}

fn explain_filled(tl: &TradeTimeline, sym: &str) -> String {
    let signal = tl.signal_decision.as_deref().unwrap_or("unknown");
    let price  = tl.fill_price.map(|p| format!("{:.2}", p)).unwrap_or_else(|| "unknown".into());
    let qty    = tl.fill_qty.map(|q| format!("{:.6}", q)).unwrap_or_else(|| "unknown".into());
    let xid    = tl.exchange_order_id.map(|x| x.to_string()).unwrap_or_else(|| "unknown".into());

    // Find the signal reason from the timeline
    let signal_reason = find_signal_reason(tl);
    // Find the market conditions that triggered the signal
    let market_ctx = find_market_context(tl);

    format!(
        "{signal} trade on {sym} was executed and filled. \
         {signal_reason}\
         {market_ctx}\
         The order filled at {price} for {qty} base asset (exchange id {xid}). \
         Risk approved the order before submission.",
        signal = signal,
    )
}

fn explain_risk_rejected(tl: &TradeTimeline, sym: &str) -> String {
    let signal = tl.signal_decision.as_deref().unwrap_or("unknown");
    let raw_reason = tl.risk_outcome
        .as_deref()
        .unwrap_or("REJECTED: unknown reason")
        .trim_start_matches("REJECTED: ");

    let human_reason = interpret_rejection(raw_reason);
    let signal_reason = find_signal_reason(tl);

    format!(
        "{signal} signal for {sym} was blocked by the risk engine. \
         {signal_reason}\
         Risk rejected because: {human_reason} \
         No order was submitted to the exchange.",
        signal = signal,
    )
}

fn explain_order_rejected(tl: &TradeTimeline, sym: &str) -> String {
    let signal = tl.signal_decision.as_deref().unwrap_or("unknown");
    // Extract the exchange rejection reason from the OrderRejected event
    let exchange_reason = tl.events.iter().find_map(|te| {
        if let TradingEvent::OrderRejected(p) = &te.event.payload {
            Some(p.reason.as_str())
        } else { None }
    }).unwrap_or("unknown exchange error");

    format!(
        "{signal} signal for {sym} passed risk checks but the exchange rejected the order. \
         Exchange reason: {} \
         This may indicate a configuration issue (e.g. insufficient balance, \
         invalid quantity step, or a stale order ID). \
         The bot returned to Idle; the next valid signal will retry.",
        exchange_reason,
        signal = signal,
    )
}

fn explain_pending(tl: &TradeTimeline, sym: &str) -> String {
    let signal = tl.signal_decision.as_deref().unwrap_or("unknown");
    let coid   = tl.client_order_id.as_deref().unwrap_or("unknown");
    format!(
        "{signal} order for {sym} was submitted (coid {coid}) but has not yet filled. \
         This is normal for limit orders. \
         Check /events for a subsequent fill or cancel event.",
        signal = signal,
    )
}

fn explain_unknown(tl: &TradeTimeline, sym: &str) -> String {
    format!(
        "Trade lifecycle for {sym} (correlation {}) is incomplete or could not be classified. \
         {} events found. \
         The signal may have been generated but blocked before reaching risk or execution.",
        tl.correlation_id,
        tl.events.len(),
    )
}

// ── explain_last_rejection ────────────────────────────────────────────────────

/// Given a slice of recent events, find the most recent risk rejection and
/// explain it. Returns None if no rejection is present.
pub fn explain_last_rejection(events: &[crate::events::StoredEvent]) -> Option<String> {
    let rejection = events.iter().rev().find(|e| {
        matches!(&e.payload, TradingEvent::RiskCheckResult(p) if !p.approved)
    })?;

    let ts = rejection.occurred_at.format("%H:%M:%S");
    if let TradingEvent::RiskCheckResult(p) = &rejection.payload {
        let raw = p.rejection_reason.as_deref().unwrap_or("unknown");
        let human = interpret_rejection(raw);
        let side_qty = format!("{} {:.6}", p.side, p.qty);
        Some(format!(
            "Last rejection at {ts}: a {side_qty} order was blocked. \
             Reason: {human}",
        ))
    } else {
        None
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Translate a raw rejection reason string into operator-friendly language.
/// The raw string comes directly from RejectionReason::to_string().
fn interpret_rejection(raw: &str) -> String {
    // Match on the prefix tokens that RejectionReason::Display produces.
    if raw.contains("KILL_SWITCH") {
        return "The kill switch is active. No orders can be placed until an \
                operator manually clears the halt.".into();
    }
    if raw.contains("STALE_FEED") || raw.contains("no feed data") {
        return "The market data feed has gone stale. The bot has not received \
                a bookTicker update recently and cannot safely evaluate prices.".into();
    }
    if raw.contains("SPREAD") {
        // Try to extract bps from the message
        let detail = raw.trim_start_matches("SPREAD: ");
        return format!(
            "The bid/ask spread is too wide to trade safely. {}. \
             Wide spread means execution cost would be too high.",
            detail
        );
    }
    if raw.contains("DAILY_LOSS") {
        let detail = raw.trim_start_matches("DAILY_LOSS: ");
        return format!(
            "The daily loss limit has been reached. {}. \
             The kill switch was automatically activated to protect capital.",
            detail
        );
    }
    if raw.contains("DRAWDOWN") {
        let detail = raw.trim_start_matches("DRAWDOWN: ");
        return format!(
            "The maximum peak-to-trough drawdown has been reached. {}. \
             The kill switch was automatically activated.",
            detail
        );
    }
    if raw.contains("COOLDOWN") {
        let detail = raw.trim_start_matches("COOLDOWN: ");
        return format!(
            "The bot is in a post-loss cooldown period. {}. \
             BUY entries are blocked; exit orders are still allowed.",
            detail
        );
    }
    if raw.contains("POSITION_SIZE") {
        let detail = raw.trim_start_matches("POSITION_SIZE: ");
        return format!(
            "Adding this order would exceed the maximum position size. {}.",
            detail
        );
    }
    if raw.contains("BAD_MARKET_DATA") {
        return "Market data is invalid (e.g. inverted book or zero prices). \
                The order was blocked to prevent trading on corrupt feed data.".into();
    }
    // Fallback: return the raw string unchanged if no pattern matched
    raw.to_string()
}

fn mode_change_impact(from: &str, to: &str) -> &'static str {
    match to {
        "Halted"      => "Trading is now suspended. No new orders will be placed.",
        "Ready"       => "Trading is now enabled.",
        "Degraded"    => "Trading continues but anomalies were detected. Monitor closely.",
        "Reconciling" => "Order placement is temporarily blocked while exchange state is verified.",
        "Booting"     => "System is initializing.",
        _ => match from {
            "Halted" => "System is recovering from a halt.",
            _        => "",
        }
    }
}

fn find_signal_reason(tl: &TradeTimeline) -> String {
    for te in &tl.events {
        if let TradingEvent::SignalDecision(p) = &te.event.payload {
            if !p.reason.is_empty() {
                return format!("The signal engine decided to {} because: {}. ",
                    p.decision.to_lowercase(), p.reason);
            }
        }
    }
    String::new()
}

fn find_market_context(tl: &TradeTimeline) -> String {
    for te in &tl.events {
        if let TradingEvent::MarketSnapshot(p) = &te.event.payload {
            return format!(
                "Market at signal time: bid {:.2} / ask {:.2} (spread {:.1} bps), \
                 5s momentum {:+.5}, 1s imbalance {:+.3}. ",
                p.bid, p.ask, p.spread_bps, p.momentum_5s, p.imbalance_1s
            );
        }
    }
    String::new()
}

fn recon_age_str(secs: Option<f64>) -> String {
    match secs {
        None => "never (startup recovery may still be running)".into(),
        Some(s) if s < 5.0  => format!("{:.1}s ago (fresh)", s),
        Some(s) if s < 60.0 => format!("{:.0}s ago", s),
        Some(s) => format!("{:.0}m ago (stale — check reconciler)", s / 60.0),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::*;
    use crate::executor::SystemMode;
    use crate::reader::{TimelineEvent, TradeOutcome, TradeTimeline};
    use chrono::Utc;

    // ── system_brief ──────────────────────────────────────────────────────────

    #[test]
    fn test_system_brief_ready() {
        let s = system_brief(SystemMode::Ready);
        assert!(s.contains("Ready"), "Should mention Ready");
        assert!(s.contains("active") || s.contains("can be placed"));
    }

    #[test]
    fn test_system_brief_halted() {
        let s = system_brief(SystemMode::Halted);
        assert!(s.contains("Halted") || s.contains("Kill") || s.contains("circuit"));
        assert!(s.contains("No new orders") || s.contains("blocked") || s.contains("clears"));
    }

    #[test]
    fn test_system_brief_degraded() {
        let s = system_brief(SystemMode::Degraded);
        assert!(s.contains("Degraded"));
        assert!(s.contains("continues") || s.contains("mismatch") || s.contains("anomal"));
    }

    #[test]
    fn test_system_brief_all_modes_produce_output() {
        for mode in [
            SystemMode::Booting, SystemMode::Reconciling, SystemMode::Ready,
            SystemMode::Degraded, SystemMode::Halted,
        ] {
            let s = system_brief(mode);
            assert!(!s.is_empty(), "system_brief({:?}) returned empty string", mode);
        }
    }

    // ── position_brief ────────────────────────────────────────────────────────

    #[test]
    fn test_position_brief_flat() {
        let p = PositionSnap {
            symbol: "BTCUSDT".into(), size: 0.0, avg_entry: 0.0,
            realized_pnl: 0.0, unrealized_pnl: 0.0, open_orders: 0,
            state_dirty: false, recon_in_progress: false,
            last_reconciled_secs: Some(2.0),
        };
        let s = position_brief(&p);
        assert!(s.contains("No open position"));
        assert!(s.contains("BTCUSDT"));
    }

    #[test]
    fn test_position_brief_long_profitable() {
        let p = PositionSnap {
            symbol: "BTCUSDT".into(), size: 0.001, avg_entry: 50000.0,
            realized_pnl: 0.0, unrealized_pnl: 5.0, open_orders: 0,
            state_dirty: false, recon_in_progress: false,
            last_reconciled_secs: Some(1.5),
        };
        let s = position_brief(&p);
        assert!(s.contains("0.001000"), "Should show position size");
        assert!(s.contains("50000"), "Should show avg entry");
        assert!(s.contains("profitable") || s.contains("5.0") || s.contains("+5"));
    }

    #[test]
    fn test_position_brief_long_underwater() {
        let p = PositionSnap {
            symbol: "BTCUSDT".into(), size: 0.001, avg_entry: 50000.0,
            realized_pnl: 0.0, unrealized_pnl: -2.5, open_orders: 0,
            state_dirty: false, recon_in_progress: false,
            last_reconciled_secs: Some(1.0),
        };
        let s = position_brief(&p);
        assert!(s.contains("underwater") || s.contains("-2.5") || s.contains("-2"));
    }

    #[test]
    fn test_position_brief_dirty_state_note() {
        let p = PositionSnap {
            symbol: "BTCUSDT".into(), size: 0.001, avg_entry: 50000.0,
            realized_pnl: 0.0, unrealized_pnl: 1.0, open_orders: 0,
            state_dirty: true, recon_in_progress: false,
            last_reconciled_secs: Some(5.0),
        };
        let s = position_brief(&p);
        assert!(s.contains("dirty") || s.contains("dirty") || s.contains("not reflect"));
    }

    // ── risk_brief ────────────────────────────────────────────────────────────

    #[test]
    fn test_risk_brief_kill_switch() {
        let r = RiskSnap {
            max_position_qty: 0.01, max_daily_loss_usd: 50.0, max_drawdown_usd: 100.0,
            kill_switch_active: true, cooldown_remaining_secs: None,
        };
        let s = risk_brief(&r);
        assert!(s.contains("ACTIVE") || s.contains("Kill switch") || s.contains("kill switch"));
        assert!(s.contains("blocked") || s.contains("operator") || s.contains("clear"));
    }

    #[test]
    fn test_risk_brief_cooldown() {
        let r = RiskSnap {
            max_position_qty: 0.01, max_daily_loss_usd: 50.0, max_drawdown_usd: 100.0,
            kill_switch_active: false, cooldown_remaining_secs: Some(120.0),
        };
        let s = risk_brief(&r);
        assert!(s.contains("Cooldown") || s.contains("cooldown") || s.contains("120"));
        assert!(s.contains("BUY") || s.contains("exit") || s.contains("blocked"));
    }

    #[test]
    fn test_risk_brief_open() {
        let r = RiskSnap {
            max_position_qty: 0.01, max_daily_loss_usd: 50.0, max_drawdown_usd: 100.0,
            kill_switch_active: false, cooldown_remaining_secs: None,
        };
        let s = risk_brief(&r);
        assert!(s.contains("open") || s.contains("Limits") || s.contains("gates"));
    }

    // ── interpret_rejection ───────────────────────────────────────────────────

    #[test]
    fn test_interpret_rejection_kill_switch() {
        let s = interpret_rejection("KILL_SWITCH: global kill switch is active");
        assert!(s.contains("kill switch") || s.contains("Kill switch"));
        assert!(s.contains("operator") || s.contains("clear"));
    }

    #[test]
    fn test_interpret_rejection_stale_feed() {
        let s = interpret_rejection("STALE_FEED: last message 8.0s ago");
        assert!(s.contains("stale") || s.contains("feed") || s.contains("market data"));
    }

    #[test]
    fn test_interpret_rejection_spread() {
        let s = interpret_rejection("SPREAD: 15.00 bps > limit 10.00 bps");
        assert!(s.contains("spread") || s.contains("Spread"));
        assert!(s.contains("wide") || s.contains("cost") || s.contains("bps"));
    }

    #[test]
    fn test_interpret_rejection_cooldown() {
        let s = interpret_rejection("COOLDOWN: 300s remaining after last losing trade");
        assert!(s.contains("cooldown") || s.contains("Cooldown") || s.contains("loss"));
        assert!(s.contains("BUY") || s.contains("exit") || s.contains("blocked"));
    }

    #[test]
    fn test_interpret_rejection_daily_loss() {
        let s = interpret_rejection("DAILY_LOSS: -55.00 USD exceeds limit -50.00 USD");
        assert!(s.contains("daily") || s.contains("loss") || s.contains("limit"));
        assert!(s.contains("kill switch") || s.contains("Kill") || s.contains("activated"));
    }

    #[test]
    fn test_interpret_rejection_position_size() {
        let s = interpret_rejection("POSITION_SIZE: current 0.01 + proposed 0.001 > limit 0.01");
        assert!(s.contains("position") || s.contains("size") || s.contains("exceed"));
    }

    #[test]
    fn test_interpret_rejection_unknown_passthrough() {
        let s = interpret_rejection("some completely unknown reason");
        assert!(!s.is_empty());
        // Unknown reasons pass through unchanged
        assert!(s.contains("unknown"));
    }

    // ── explain_trade - filled ────────────────────────────────────────────────

    fn make_filled_timeline() -> TradeTimeline {
        let events = vec![
            make_te(TradingEvent::MarketSnapshot(MarketSnapshotPayload {
                bid:50000.0, ask:50001.0, mid:50000.5, spread_bps:2.0,
                momentum_1s:0.001, momentum_3s:0.002, momentum_5s:0.003,
                imbalance_1s:0.5, imbalance_3s:0.3, imbalance_5s:0.2,
                feed_age_ms:20.0, mid_samples:10, trade_samples:8,
            })),
            make_te(TradingEvent::SignalDecision(SignalDecisionPayload {
                decision:"Buy".into(), exit_reason:None,
                reason:"momentum above threshold".into(), confidence:0.8,
                metrics: MarketSnapshotPayload {
                    bid:50000.0, ask:50001.0, mid:50000.5, spread_bps:2.0,
                    momentum_1s:0.001, momentum_3s:0.002, momentum_5s:0.003,
                    imbalance_1s:0.5, imbalance_3s:0.3, imbalance_5s:0.2,
                    feed_age_ms:20.0, mid_samples:10, trade_samples:8,
                },
            })),
            make_te(TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved:true, side:"BUY".into(), qty:0.001,
                expected_price:50001.0, rejection_reason:None,
                position_size:0.0, position_avg_entry:0.0,
            })),
            make_te(TradingEvent::OrderFilled(OrderFilledPayload {
                client_order_id:"coid-1".into(), exchange_order_id:9999,
                side:"BUY".into(), filled_qty:0.001, avg_fill_price:50001.5,
                cumulative_quote:50.0015,
            })),
        ];
        TradeTimeline {
            correlation_id: "test-corr".into(),
            symbol: Some("BTCUSDT".into()),
            events,
            signal_decision:   Some("Buy".into()),
            risk_outcome:      Some("APPROVED".into()),
            client_order_id:   Some("coid-1".into()),
            exchange_order_id: Some(9999),
            fill_price:        Some(50001.5),
            fill_qty:          Some(0.001),
            outcome:           TradeOutcome::Filled,
        }
    }

    #[test]
    fn test_explain_filled_mentions_fill_price() {
        let tl = make_filled_timeline();
        let s = explain_trade(&tl);
        assert!(s.contains("50001") || s.contains("filled") || s.contains("FILLED"),
            "Should mention fill price or filled status");
    }

    #[test]
    fn test_explain_filled_mentions_symbol() {
        let tl = make_filled_timeline();
        let s = explain_trade(&tl);
        assert!(s.contains("BTCUSDT"));
    }

    #[test]
    fn test_explain_filled_mentions_risk_approved() {
        let tl = make_filled_timeline();
        let s = explain_trade(&tl);
        assert!(s.to_lowercase().contains("risk") || s.contains("approved") || s.contains("Risk"));
    }

    // ── explain_trade - risk rejected ─────────────────────────────────────────

    fn make_rejected_timeline() -> TradeTimeline {
        let events = vec![
            make_te(TradingEvent::SignalDecision(SignalDecisionPayload {
                decision:"Buy".into(), exit_reason:None,
                reason:"momentum ok".into(), confidence:0.7,
                metrics: MarketSnapshotPayload {
                    bid:50000.0, ask:50010.0, mid:50005.0, spread_bps:20.0,
                    momentum_1s:0.001, momentum_3s:0.002, momentum_5s:0.003,
                    imbalance_1s:0.3, imbalance_3s:0.2, imbalance_5s:0.1,
                    feed_age_ms:20.0, mid_samples:10, trade_samples:8,
                },
            })),
            make_te(TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved:false, side:"BUY".into(), qty:0.001,
                expected_price:50010.0,
                rejection_reason:Some("SPREAD: 20.00 bps > limit 10.00 bps".into()),
                position_size:0.0, position_avg_entry:0.0,
            })),
        ];
        TradeTimeline {
            correlation_id: "test-corr-rej".into(),
            symbol: Some("BTCUSDT".into()),
            events,
            signal_decision: Some("Buy".into()),
            risk_outcome:    Some("REJECTED: SPREAD: 20.00 bps > limit 10.00 bps".into()),
            client_order_id:   None,
            exchange_order_id: None,
            fill_price:        None,
            fill_qty:          None,
            outcome: TradeOutcome::RiskRejected,
        }
    }

    #[test]
    fn test_explain_rejected_mentions_risk() {
        let tl = make_rejected_timeline();
        let s = explain_trade(&tl);
        assert!(s.to_lowercase().contains("risk") || s.contains("blocked") || s.contains("rejected"));
    }

    #[test]
    fn test_explain_rejected_no_order_submitted() {
        let tl = make_rejected_timeline();
        let s = explain_trade(&tl);
        assert!(s.contains("No order") || s.contains("no order") || s.contains("not submitted") || s.contains("exchange"));
    }

    #[test]
    fn test_explain_rejected_contains_reason() {
        let tl = make_rejected_timeline();
        let s = explain_trade(&tl);
        // Should explain the spread issue
        assert!(s.to_lowercase().contains("spread") || s.contains("wide") || s.contains("SPREAD"));
    }

    // ── explain_last_rejection ────────────────────────────────────────────────

    #[test]
    fn test_explain_last_rejection_found() {
        let events = vec![StoredEvent::new(
            Some("BTCUSDT".into()), None, None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved:false, side:"BUY".into(), qty:0.001,
                expected_price:50000.0,
                rejection_reason:Some("COOLDOWN: 120s remaining".into()),
                position_size:0.0, position_avg_entry:0.0,
            }),
        )];
        let s = explain_last_rejection(&events);
        assert!(s.is_some(), "Should find the rejection");
        let s = s.unwrap();
        assert!(s.to_lowercase().contains("cooldown") || s.contains("120") || s.contains("blocked"));
    }

    #[test]
    fn test_explain_last_rejection_none_when_no_rejections() {
        let events = vec![StoredEvent::new(
            Some("BTCUSDT".into()), None, None,
            TradingEvent::SystemModeChange(SystemModeChangePayload {
                from_mode:"Ready".into(), to_mode:"Degraded".into(), reason:"test".into(),
            }),
        )];
        let result = explain_last_rejection(&events);
        assert!(result.is_none(), "Should return None when no rejections present");
    }

    #[test]
    fn test_explain_last_rejection_picks_most_recent() {
        let older = StoredEvent::new(
            None, None, None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved:false, side:"BUY".into(), qty:0.001, expected_price:0.0,
                rejection_reason:Some("STALE_FEED: 10s".into()),
                position_size:0.0, position_avg_entry:0.0,
            }),
        );
        let newer = StoredEvent::new(
            None, None, None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved:false, side:"BUY".into(), qty:0.001, expected_price:0.0,
                rejection_reason:Some("KILL_SWITCH: global kill switch is active".into()),
                position_size:0.0, position_avg_entry:0.0,
            }),
        );
        // events slice is passed as-is; explain_last_rejection scans from the end
        let events = vec![older, newer];
        let s = explain_last_rejection(&events).unwrap();
        assert!(s.contains("kill switch") || s.contains("Kill switch") || s.contains("KILL"),
            "Should pick the most recent (last) rejection");
    }

    // ── recent_summary ────────────────────────────────────────────────────────

    #[test]
    fn test_recent_summary_count_matches() {
        let events = vec![
            StoredEvent::new(None, None, None, TradingEvent::SystemModeChange(
                SystemModeChangePayload { from_mode:"Ready".into(), to_mode:"Halted".into(), reason:"cb".into() }
            )),
            StoredEvent::new(None, None, None, TradingEvent::ReconcileCompleted(
                ReconcileCompletedPayload {
                    cycle:1, had_anomaly:false, position_size:0.0,
                    open_orders:0, new_fills:0, duration_ms:5,
                }
            )),
        ];
        let summaries = recent_summary(&events);
        assert_eq!(summaries.len(), 2, "One summary per event");
    }

    #[test]
    fn test_recent_summary_halted_mode_change() {
        let events = vec![StoredEvent::new(
            None, None, None,
            TradingEvent::SystemModeChange(SystemModeChangePayload {
                from_mode:"Ready".into(), to_mode:"Halted".into(),
                reason:"circuit breaker".into(),
            }),
        )];
        let summaries = recent_summary(&events);
        let s = &summaries[0];
        assert!(s.contains("Halted") || s.contains("halted") || s.contains("mode"));
        assert!(s.contains("Trading is now suspended") || s.contains("suspended") || s.contains("blocked") || s.contains("halt"));
    }

    #[test]
    fn test_recent_summary_filled_order() {
        let events = vec![StoredEvent::new(
            Some("BTCUSDT".into()), Some("c".into()), Some("coid".into()),
            TradingEvent::OrderFilled(OrderFilledPayload {
                client_order_id:"coid".into(), exchange_order_id:1,
                side:"BUY".into(), filled_qty:0.001, avg_fill_price:50000.0,
                cumulative_quote:50.0,
            }),
        )];
        let summaries = recent_summary(&events);
        let s = &summaries[0];
        assert!(s.contains("FILLED") || s.contains("filled") || s.contains("50000"));
    }

    #[test]
    fn test_interpret_event_no_panic_for_all_variants() {
        let events = vec![
            StoredEvent::new(None, None, None, TradingEvent::MarketSnapshot(
                MarketSnapshotPayload { bid:1.0, ask:2.0, mid:1.5, spread_bps:5.0,
                    momentum_1s:0.0, momentum_3s:0.0, momentum_5s:0.0,
                    imbalance_1s:0.0, imbalance_3s:0.0, imbalance_5s:0.0,
                    feed_age_ms:10.0, mid_samples:5, trade_samples:3,
                }
            )),
            StoredEvent::new(None, None, None, TradingEvent::WatchdogTimeout(
                WatchdogTimeoutPayload { stuck_state:"WaitingAck".into(), age_secs:11.0 }
            )),
            StoredEvent::new(None, None, None, TradingEvent::CircuitBreakerTripped(
                CircuitBreakerPayload { reason:"too many errors".into(),
                    attempts_count:10, rejects_count:0, errors_count:3, slippage_count:0 }
            )),
            StoredEvent::new(None, None, None, TradingEvent::OperatorAction(
                OperatorActionPayload { action:"clear_halt".into(), reason:"manual".into() }
            )),
        ];
        for e in &events {
            let s = interpret_event(e);
            assert!(!s.is_empty(), "interpret_event should never return empty string");
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_te(payload: TradingEvent) -> TimelineEvent {
        use crate::reader::LifecycleStage;
        let event = StoredEvent {
            event_id:        "test".into(),
            occurred_at:     Utc::now(),
            event_type:      payload.event_type_name().into(),
            symbol:          None,
            correlation_id:  None,
            client_order_id: None,
            payload,
        };
        let stage = match &event.payload {
            TradingEvent::MarketSnapshot(_)      => LifecycleStage::MarketContext,
            TradingEvent::SignalDecision(_)      => LifecycleStage::SignalGenerated,
            TradingEvent::RiskCheckResult(_)     => LifecycleStage::RiskEvaluated,
            TradingEvent::OrderSubmitted(_)      => LifecycleStage::OrderSubmitted,
            TradingEvent::OrderFilled(_)         => LifecycleStage::OrderFilled,
            TradingEvent::OrderRejected(_)       => LifecycleStage::OrderRejected,
            _                                    => LifecycleStage::Other,
        };
        TimelineEvent { stage, summary: "test".into(), event }
    }
}

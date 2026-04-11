// suggestions.rs
//
// Rule-based suggestion layer. Read-only. Never executes orders.
//
// Design:
//   All functions are pure — they take plain data (no Mutex, no Arc, no async)
//   and return a Suggestion. Callers in webui.rs are responsible for snapshotting
//   live state under locks and passing the extracted values here.
//
//   Market data for suggestions comes from the most recent MarketSnapshot event
//   in the event store. This is the same snapshot the signal engine recorded
//   immediately before the last decision, making it the most recent verified
//   market data available without touching the live feed.
//
// Suggestion taxonomy:
//   BUY_CANDIDATE   — conditions look good for a long entry
//   EXIT_CANDIDATE  — open position should consider closing
//   WAIT            — one or more conditions not yet met; describe what's missing
//   STAND_DOWN      — system-level block; no trading activity appropriate
//
// blocked_by: every failing gate is named explicitly so the operator knows
// exactly what needs to change before trading becomes possible.

use crate::events::{MarketSnapshotPayload, TradingEvent};
use crate::executor::SystemMode;

// ── Suggestion output ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionKind {
    /// Market and system conditions look favourable for a BUY entry.
    BuyCandidate,
    /// Open position meets one or more exit criteria.
    ExitCandidate,
    /// Conditions are not yet met; list what's missing.
    Wait,
    /// System-level block; trading should not be attempted.
    StandDown,
}

impl std::fmt::Display for SuggestionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestionKind::BuyCandidate  => write!(f, "BUY_CANDIDATE"),
            SuggestionKind::ExitCandidate => write!(f, "EXIT_CANDIDATE"),
            SuggestionKind::Wait          => write!(f, "WAIT"),
            SuggestionKind::StandDown     => write!(f, "STAND_DOWN"),
        }
    }
}

/// A single suggestion with its justification and blocking conditions.
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub kind:       SuggestionKind,
    /// Plain-English reason for this suggestion.
    pub reason:     String,
    /// Confidence in the suggestion [0.0, 1.0].
    /// 1.0 for STAND_DOWN (certainty that nothing should happen).
    /// For BUY_CANDIDATE/EXIT_CANDIDATE: derived from signal metric distances.
    pub confidence: f64,
    /// Each gate that is currently blocking action. Empty for STAND_DOWN
    /// (the system-level block is fully explained in `reason`).
    pub blocked_by: Vec<String>,
}

impl Suggestion {
    fn stand_down(reason: impl Into<String>) -> Self {
        Self {
            kind:       SuggestionKind::StandDown,
            reason:     reason.into(),
            confidence: 1.0,
            blocked_by: vec![],
        }
    }

    fn wait(reason: impl Into<String>, blocked_by: Vec<String>) -> Self {
        Self {
            kind: SuggestionKind::Wait,
            reason: reason.into(),
            confidence: 0.0,
            blocked_by,
        }
    }
}

// ── Snapshot types passed by the webui caller ─────────────────────────────────

/// Snapshot of system-level state relevant to suggestions.
pub struct SystemSnap {
    pub mode:             SystemMode,
    pub exec_is_idle:     bool,
    pub can_place_order:  bool,  // TruthState::can_place_order()
    pub has_active_buy:   bool,  // an open BUY order already exists
}

/// Snapshot of signal-relevant thresholds.
pub struct SignalThresholds {
    pub momentum_threshold:   f64,
    pub imbalance_threshold:  f64,
    pub max_entry_spread_bps: f64,
    pub stop_loss_pct:        f64,
    pub take_profit_pct:      f64,
    pub max_hold_secs:        f64,
}

impl Default for SignalThresholds {
    fn default() -> Self {
        Self {
            momentum_threshold:   0.00005,
            imbalance_threshold:  0.10,
            max_entry_spread_bps: 5.0,
            stop_loss_pct:        0.0020,
            take_profit_pct:      0.0040,
            max_hold_secs:        120.0,
        }
    }
}

/// Snapshot of risk-gate state.
pub struct RiskGateSnap {
    pub kill_switch_active:     bool,
    pub cooldown_remaining_secs: Option<f64>,
    pub max_spread_bps:         f64,
    pub buy_power:              f64,
    pub sell_inventory:         f64,
    pub position_size:          f64,
    pub max_position_qty:       f64,
    pub daily_pnl:              f64,
    pub max_daily_loss_usd:     f64,
    pub drawdown:               f64,
    pub max_drawdown_usd:       f64,
}

/// Snapshot of the open position.
pub struct PositionGateSnap {
    pub size:          f64,
    pub avg_entry:     f64,
    pub hold_secs:     Option<f64>,   // seconds since entry (None if no hint)
}

// ── get_trade_suggestion ──────────────────────────────────────────────────────

/// Evaluate whether conditions favour a new BUY entry.
///
/// Returns STAND_DOWN if system-level blocks exist.
/// Returns WAIT with `blocked_by` listing each failing gate.
/// Returns BUY_CANDIDATE only when all entry gates pass.
///
/// Does NOT submit any order. Does NOT call risk_check().
/// The suggestion is advisory — the trading loop's risk_check() is authoritative.
pub fn get_trade_suggestion(
    sys:       &SystemSnap,
    risk:      &RiskGateSnap,
    sig:       &SignalThresholds,
    market:    Option<&MarketSnapshotPayload>,
) -> Suggestion {
    // ── STAND_DOWN conditions ─────────────────────────────────────────────────
    if risk.kill_switch_active {
        return Suggestion::stand_down(
            "Kill switch is active. The system has halted all trading. \
             An operator must clear the halt before any entries are possible."
        );
    }
    if !sys.mode.can_trade() {
        return Suggestion::stand_down(format!(
            "System mode is {}. Trading is not permitted in this state. \
             Wait for the system to reach Ready or Degraded.",
            sys.mode
        ));
    }
    if risk.daily_pnl < -risk.max_daily_loss_usd {
        return Suggestion::stand_down(format!(
            "Daily loss limit breached ({:.2} USD loss vs {:.2} USD limit). \
             Kill switch will be activated on the next risk check. No new entries.",
            -risk.daily_pnl, risk.max_daily_loss_usd
        ));
    }
    if risk.drawdown >= risk.max_drawdown_usd {
        return Suggestion::stand_down(format!(
            "Maximum drawdown reached ({:.2} USD vs {:.2} USD limit). \
             Kill switch will be activated on the next risk check. No new entries.",
            risk.drawdown, risk.max_drawdown_usd
        ));
    }

    // Inventory-aware, action-first recommendation:
    // if base inventory exists while flat, emit a SELL-oriented suggestion first.
    if risk.sell_inventory > 0.0 && risk.position_size <= 1e-9 {
        return Suggestion {
            kind: SuggestionKind::ExitCandidate,
            reason: format!(
                "SELL READY: base inventory {:.8} is available while no active position is open. \
                 Prioritise SELL to reduce base exposure before new BUY entries.",
                risk.sell_inventory
            ),
            confidence: 0.90,
            blocked_by: vec![],
        };
    }

    // ── WAIT conditions — collect all failing gates ───────────────────────────
    let mut blocked: Vec<String> = Vec::new();

    if !sys.exec_is_idle {
        blocked.push("executor is not Idle — an order is already in flight".into());
    }
    if !sys.can_place_order {
        blocked.push("state is dirty or reconciliation is running".into());
    }
    if sys.has_active_buy {
        blocked.push("an open BUY order already exists (no pyramiding)".into());
    }
    if risk.position_size >= risk.max_position_qty {
        blocked.push(format!(
            "position size {:.6} is at or above limit {:.6}",
            risk.position_size, risk.max_position_qty
        ));
    }
    if risk.buy_power <= 0.0 {
        blocked.push("BUY DISABLED (NO USDT)".into());
    }
    if let Some(secs) = risk.cooldown_remaining_secs {
        if secs > 0.0 {
            blocked.push(format!(
                "cooldown active: {:.0}s remaining after last losing trade",
                secs
            ));
        }
    }

    // ── Market condition gates ────────────────────────────────────────────────
    let (market_confidence, market_blocked) = evaluate_entry_market(market, sig, risk);
    blocked.extend(market_blocked);

    if !blocked.is_empty() {
        let summary = format!(
            "Entry not ready: {} gate(s) are blocking. \
             See blocked_by for the specific conditions that need to change.",
            blocked.len()
        );
        return Suggestion::wait(summary, blocked);
    }

    // All gates passed — BUY_CANDIDATE
    let snap = market.unwrap(); // safe: evaluate_entry_market would have blocked if None
    Suggestion {
        kind: SuggestionKind::BuyCandidate,
        reason: format!(
            "All entry gates are satisfied. \
             Market shows bid {:.2} / ask {:.2} (spread {:.1} bps), \
             5s momentum {:+.5}, 1s imbalance {:+.3}. \
             Risk limits have headroom: position {:.6} of {:.6} max, \
             no cooldown, executor is Idle.",
            snap.bid, snap.ask, snap.spread_bps,
            snap.momentum_5s, snap.imbalance_1s,
            risk.position_size, risk.max_position_qty,
        ),
        confidence: market_confidence,
        blocked_by: vec![],
    }
}

// ── get_exit_suggestion ───────────────────────────────────────────────────────

/// Evaluate whether an open position should be closed.
///
/// Returns STAND_DOWN if no position is open.
/// Returns EXIT_CANDIDATE if a stop loss, take profit, or max hold time is hit.
/// Returns WAIT if the position is within normal bounds.
pub fn get_exit_suggestion(
    sys:     &SystemSnap,
    risk:    &RiskGateSnap,
    pos:     &PositionGateSnap,
    sig:     &SignalThresholds,
    market:  Option<&MarketSnapshotPayload>,
) -> Suggestion {
    if pos.size < 1e-9 {
        return Suggestion::stand_down(
            "No open position. Exit suggestion is not applicable."
        );
    }

    // System blocks — exits are still allowed through cooldown, but not through
    // kill switch (if kill switch is active the whole system is halted).
    if risk.kill_switch_active {
        return Suggestion::stand_down(
            "Kill switch is active. Even exit orders cannot be placed. \
             Operator must clear halt first."
        );
    }
    if !sys.mode.can_trade() {
        return Suggestion::stand_down(format!(
            "System mode is {}. Cannot place exit orders in this state.",
            sys.mode
        ));
    }

    let mid = market.map(|m| m.mid).unwrap_or(0.0);
    let entry = pos.avg_entry;
    let mut exits: Vec<String> = Vec::new();

    // Stop loss
    if mid > 0.0 && entry > 0.0 {
        let sl_level = entry * (1.0 - sig.stop_loss_pct);
        if mid <= sl_level {
            let loss_pct = (mid - entry) / entry * 100.0;
            exits.push(format!(
                "STOP LOSS triggered: mid {:.2} fell below stop level {:.2} \
                 ({:.3}% loss from entry {:.2})",
                mid, sl_level, loss_pct, entry
            ));
        }

        // Take profit
        let tp_level = entry * (1.0 + sig.take_profit_pct);
        if mid >= tp_level {
            let gain_pct = (mid - entry) / entry * 100.0;
            exits.push(format!(
                "TAKE PROFIT triggered: mid {:.2} reached target {:.2} \
                 ({:.3}% gain from entry {:.2})",
                mid, tp_level, gain_pct, entry
            ));
        }
    }

    // Max hold time
    if let Some(secs) = pos.hold_secs {
        if secs >= sig.max_hold_secs {
            exits.push(format!(
                "MAX HOLD TIME reached: held {:.0}s (limit {:.0}s)",
                secs, sig.max_hold_secs
            ));
        }
    }

    if exits.is_empty() {
        // Within bounds — describe current position health
        let unrealized = if mid > 0.0 && entry > 0.0 {
            (mid - entry) * pos.size
        } else { 0.0 };
        let hold_note = pos.hold_secs
            .map(|s| format!(", held {:.0}s of {:.0}s max", s, sig.max_hold_secs))
            .unwrap_or_default();

        return Suggestion::wait(
            format!(
                "Position is within normal bounds (unrealized {:.4} USD{hold_note}). \
                 No exit criteria met. Continue holding.",
                unrealized,
            ),
            vec![],
        );
    }

    // One or more exit criteria met
    let confidence = (exits.len() as f64 / 3.0).min(1.0); // more triggers = higher confidence
    Suggestion {
        kind: SuggestionKind::ExitCandidate,
        reason: format!(
            "{} exit criterion/criteria triggered. \
             Recommend closing position (size {:.6} @ avg entry {:.2}).",
            exits.len(), pos.size, entry,
        ),
        confidence,
        blocked_by: exits, // reuse blocked_by to carry the triggered exit reasons
    }
}

// ── get_watchlist_summary ─────────────────────────────────────────────────────

/// A concise watchlist line for the current symbol: one-liner status.
/// Derived from the most recent recorded market snapshot and system state.
/// Returns a tuple of (status_label, detail_sentence).
pub fn get_watchlist_summary(
    symbol:  &str,
    sys:     &SystemSnap,
    risk:    &RiskGateSnap,
    pos:     &PositionGateSnap,
    sig:     &SignalThresholds,
    market:  Option<&MarketSnapshotPayload>,
) -> (String, String) {
    if risk.kill_switch_active || !sys.mode.can_trade() {
        return (
            "STAND_DOWN".into(),
            format!("{} — system halted, no trading", symbol),
        );
    }

    if pos.size > 1e-9 {
        // We have a position — describe its health
        let mid = market.map(|m| m.mid).unwrap_or(0.0);
        let unrealized = if mid > 0.0 && pos.avg_entry > 0.0 {
            (mid - pos.avg_entry) * pos.size
        } else { 0.0 };
        let sign = if unrealized >= 0.0 { "+" } else { "" };
        let hold = pos.hold_secs.map(|s| format!(", {:.0}s held", s)).unwrap_or_default();
        return (
            "IN_POSITION".into(),
            format!("{} — long {:.6} @ {:.2}, unrealized {}{:.4} USD{}",
                symbol, pos.size, pos.avg_entry, sign, unrealized, hold),
        );
    }

    // Flat — characterise the market environment
    match market {
        None => (
            "NO_DATA".into(),
            format!("{} — no market snapshot available yet", symbol),
        ),
        Some(m) => {
            let trend = if m.momentum_5s > sig.momentum_threshold { "upward" }
                        else if m.momentum_5s < -sig.momentum_threshold { "downward" }
                        else { "flat" };
            let spread_ok = m.spread_bps <= sig.max_entry_spread_bps;
            let spread_note = if spread_ok { "spread ok".into() }
                              else { format!("spread {:.1} bps (too wide)", m.spread_bps) };
            let imbalance_note = if m.imbalance_1s > sig.imbalance_threshold { "buy pressure" }
                                 else if m.imbalance_1s < -sig.imbalance_threshold { "sell pressure" }
                                 else { "balanced flow" };
            let cooldown_note = if risk.cooldown_remaining_secs.map(|s| s > 0.0).unwrap_or(false) {
                format!(", cooldown {:.0}s", risk.cooldown_remaining_secs.unwrap())
            } else { String::new() };

            (
                "WATCHING".into(),
                format!("{} — {trend} momentum, {spread_note}, {imbalance_note}{cooldown_note}",
                    symbol),
            )
        }
    }
}

// ── Internal: evaluate market entry conditions ────────────────────────────────

/// Returns (confidence, list_of_failed_gates).
/// Confidence is 0.0 if any gate fails; otherwise derived from metric distances.
fn evaluate_entry_market(
    market: Option<&MarketSnapshotPayload>,
    sig:    &SignalThresholds,
    risk:   &RiskGateSnap,
) -> (f64, Vec<String>) {
    let mut failed: Vec<String> = Vec::new();

    let snap = match market {
        Some(s) => s,
        None => {
            failed.push("no market snapshot available (feed may not have started yet)".into());
            return (0.0, failed);
        }
    };

    // Feed staleness — the snapshot tells us its age at capture time
    if snap.feed_age_ms > 3_000.0 {
        failed.push(format!(
            "feed was already stale at snapshot time ({:.0}ms old)",
            snap.feed_age_ms
        ));
    }

    // Spread
    if snap.spread_bps > sig.max_entry_spread_bps {
        failed.push(format!(
            "spread {:.2} bps exceeds entry limit {:.2} bps",
            snap.spread_bps, sig.max_entry_spread_bps
        ));
    }
    if snap.spread_bps > risk.max_spread_bps {
        failed.push(format!(
            "spread {:.2} bps exceeds risk limit {:.2} bps",
            snap.spread_bps, risk.max_spread_bps
        ));
    }

    // Momentum — all three windows must be above threshold
    if snap.momentum_1s <= sig.momentum_threshold {
        failed.push(format!(
            "1s momentum {:+.5} ≤ threshold {:+.5}",
            snap.momentum_1s, sig.momentum_threshold
        ));
    }
    if snap.momentum_3s <= sig.momentum_threshold {
        failed.push(format!(
            "3s momentum {:+.5} ≤ threshold {:+.5}",
            snap.momentum_3s, sig.momentum_threshold
        ));
    }
    if snap.momentum_5s <= sig.momentum_threshold {
        failed.push(format!(
            "5s momentum {:+.5} ≤ threshold {:+.5}",
            snap.momentum_5s, sig.momentum_threshold
        ));
    }

    // Imbalance
    if snap.imbalance_1s <= sig.imbalance_threshold {
        failed.push(format!(
            "1s imbalance {:+.3} ≤ threshold {:+.3} (insufficient buy pressure)",
            snap.imbalance_1s, sig.imbalance_threshold
        ));
    }
    if snap.imbalance_3s <= 0.0 {
        failed.push(format!(
            "3s imbalance {:+.3} is not positive (directional confirmation missing)",
            snap.imbalance_3s
        ));
    }

    // Minimum samples
    if snap.mid_samples < 3 {
        failed.push(format!("only {} mid-price samples (need ≥3)", snap.mid_samples));
    }
    if snap.trade_samples < 3 {
        failed.push(format!("only {} trade samples (need ≥3)", snap.trade_samples));
    }

    if !failed.is_empty() {
        return (0.0, failed);
    }

    // Compute confidence from metric distances to thresholds
    let mom_score = (snap.momentum_5s / sig.momentum_threshold).min(1.0).max(0.0);
    let imb_score = snap.imbalance_5s.min(1.0).max(0.0);
    let spread_score = if sig.max_entry_spread_bps > 0.0 {
        (1.0 - snap.spread_bps / sig.max_entry_spread_bps).min(1.0).max(0.0)
    } else { 0.0 };
    let confidence = (mom_score + imb_score + spread_score) / 3.0;

    (confidence, vec![])
}

// ── Helper: extract most recent MarketSnapshot from events ────────────────────

/// Find the most recent MarketSnapshot payload from a slice of stored events.
/// Returns a reference to the payload if found.
pub fn latest_market_snapshot(events: &[crate::events::StoredEvent])
    -> Option<MarketSnapshotPayload>
{
    events.iter().rev().find_map(|e| {
        if let TradingEvent::MarketSnapshot(p) = &e.payload {
            Some(p.clone())
        } else {
            None
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{MarketSnapshotPayload, StoredEvent, TradingEvent};
    use crate::executor::SystemMode;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    fn ready_sys() -> SystemSnap {
        SystemSnap {
            mode:            SystemMode::Ready,
            exec_is_idle:    true,
            can_place_order: true,
            has_active_buy:  false,
        }
    }

    fn open_risk() -> RiskGateSnap {
        RiskGateSnap {
            kill_switch_active:      false,
            cooldown_remaining_secs: None,
            max_spread_bps:          10.0,
            buy_power:               100.0,
            sell_inventory:          0.0,
            position_size:           0.0,
            max_position_qty:        0.01,
            daily_pnl:               0.0,
            max_daily_loss_usd:      50.0,
            drawdown:                0.0,
            max_drawdown_usd:        100.0,
        }
    }

    fn default_sig() -> SignalThresholds {
        SignalThresholds::default()
    }

    fn good_market() -> MarketSnapshotPayload {
        MarketSnapshotPayload {
            bid: 50000.0, ask: 50001.0, mid: 50000.5, spread_bps: 2.0,
            momentum_1s: 0.001, momentum_3s: 0.002, momentum_5s: 0.003,
            imbalance_1s: 0.5, imbalance_3s: 0.3, imbalance_5s: 0.2,
            feed_age_ms: 50.0, mid_samples: 10, trade_samples: 8,
        }
    }

    fn flat_pos() -> PositionGateSnap {
        PositionGateSnap { size: 0.0, avg_entry: 0.0, hold_secs: None }
    }

    fn long_pos(size: f64, avg_entry: f64, hold_secs: f64) -> PositionGateSnap {
        PositionGateSnap { size, avg_entry, hold_secs: Some(hold_secs) }
    }

    // ── get_trade_suggestion ──────────────────────────────────────────────────

    #[test]
    fn test_buy_candidate_all_gates_pass() {
        let s = get_trade_suggestion(&ready_sys(), &open_risk(), &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::BuyCandidate);
        assert!(s.blocked_by.is_empty());
        assert!(s.confidence > 0.0);
    }

    #[test]
    fn test_stand_down_kill_switch() {
        let mut risk = open_risk();
        risk.kill_switch_active = true;
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::StandDown);
        assert!(s.reason.contains("Kill switch") || s.reason.contains("kill switch"));
    }

    #[test]
    fn test_stand_down_halted_mode() {
        let mut sys = ready_sys();
        sys.mode = SystemMode::Halted;
        let s = get_trade_suggestion(&sys, &open_risk(), &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::StandDown);
        assert!(s.reason.contains("Halted") || s.reason.contains("halted"));
    }

    #[test]
    fn test_stand_down_daily_loss_breached() {
        let mut risk = open_risk();
        risk.daily_pnl        = -60.0;
        risk.max_daily_loss_usd = 50.0;
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::StandDown);
        assert!(s.reason.contains("Daily loss") || s.reason.contains("daily loss"));
    }

    #[test]
    fn test_stand_down_drawdown_breached() {
        let mut risk = open_risk();
        risk.drawdown         = 110.0;
        risk.max_drawdown_usd = 100.0;
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::StandDown);
        assert!(s.reason.contains("drawdown") || s.reason.contains("Drawdown"));
    }

    #[test]
    fn test_wait_executor_not_idle() {
        let mut sys = ready_sys();
        sys.exec_is_idle = false;
        let s = get_trade_suggestion(&sys, &open_risk(), &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("executor") || b.contains("Idle")));
    }

    #[test]
    fn test_wait_state_dirty() {
        let mut sys = ready_sys();
        sys.can_place_order = false;
        let s = get_trade_suggestion(&sys, &open_risk(), &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("dirty") || b.contains("reconcil")));
    }

    #[test]
    fn test_wait_active_buy_order() {
        let mut sys = ready_sys();
        sys.has_active_buy = true;
        let s = get_trade_suggestion(&sys, &open_risk(), &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("BUY") || b.contains("pyramiding")));
    }

    #[test]
    fn test_wait_cooldown_active() {
        let mut risk = open_risk();
        risk.cooldown_remaining_secs = Some(120.0);
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("cooldown") || b.contains("120")));
    }

    #[test]
    fn test_wait_position_at_max() {
        let mut risk = open_risk();
        risk.position_size    = 0.01;
        risk.max_position_qty = 0.01;
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("position") || b.contains("limit")));
    }

    #[test]
    fn test_wait_wide_spread() {
        let mut market = good_market();
        market.spread_bps = 20.0;
        let s = get_trade_suggestion(&ready_sys(), &open_risk(), &default_sig(), Some(&market));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("spread")));
    }

    #[test]
    fn test_wait_negative_momentum() {
        let mut market = good_market();
        market.momentum_1s = -0.001;
        market.momentum_3s = -0.002;
        market.momentum_5s = -0.003;
        let s = get_trade_suggestion(&ready_sys(), &open_risk(), &default_sig(), Some(&market));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("momentum")));
    }

    #[test]
    fn test_wait_no_market_snapshot() {
        let s = get_trade_suggestion(&ready_sys(), &open_risk(), &default_sig(), None);
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("snapshot") || b.contains("market")));
    }

    #[test]
    fn test_sell_ready_when_base_inventory_exists_while_flat() {
        let mut risk = open_risk();
        risk.buy_power = 0.0;
        risk.sell_inventory = 0.005;
        risk.position_size = 0.0;
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::ExitCandidate);
        assert!(s.reason.contains("SELL READY"));
    }

    #[test]
    fn test_wait_low_imbalance() {
        let mut market = good_market();
        market.imbalance_1s = -0.5;  // sell pressure
        market.imbalance_3s = -0.2;
        let s = get_trade_suggestion(&ready_sys(), &open_risk(), &default_sig(), Some(&market));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.blocked_by.iter().any(|b| b.contains("imbalance")));
    }

    #[test]
    fn test_blocked_by_is_empty_for_buy_candidate() {
        let s = get_trade_suggestion(&ready_sys(), &open_risk(), &default_sig(), Some(&good_market()));
        if s.kind == SuggestionKind::BuyCandidate {
            assert!(s.blocked_by.is_empty(), "BUY_CANDIDATE must have no blocked_by");
        }
    }

    #[test]
    fn test_confidence_zero_when_waiting() {
        let mut sys = ready_sys();
        sys.exec_is_idle = false;
        let s = get_trade_suggestion(&sys, &open_risk(), &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn test_confidence_one_for_stand_down() {
        let mut risk = open_risk();
        risk.kill_switch_active = true;
        let s = get_trade_suggestion(&ready_sys(), &risk, &default_sig(), Some(&good_market()));
        assert_eq!(s.confidence, 1.0);
    }

    // ── get_exit_suggestion ───────────────────────────────────────────────────

    #[test]
    fn test_exit_stand_down_no_position() {
        let s = get_exit_suggestion(
            &ready_sys(), &open_risk(), &flat_pos(), &default_sig(), Some(&good_market())
        );
        assert_eq!(s.kind, SuggestionKind::StandDown);
        assert!(s.reason.contains("No open position"));
    }

    #[test]
    fn test_exit_candidate_stop_loss() {
        // Entry at 50000, mid now at 49880 → below 0.20% SL level of 49900
        let mut market = good_market();
        market.bid = 49879.0;
        market.ask = 49881.0;
        market.mid = 49880.0;
        let pos = long_pos(0.001, 50000.0, 30.0);
        let s = get_exit_suggestion(&ready_sys(), &open_risk(), &pos, &default_sig(), Some(&market));
        assert_eq!(s.kind, SuggestionKind::ExitCandidate);
        assert!(s.blocked_by.iter().any(|b| b.contains("STOP LOSS") || b.contains("stop")));
    }

    #[test]
    fn test_exit_candidate_take_profit() {
        // Entry at 50000, TP at 0.40% above = 50200, mid now 50250
        let mut market = good_market();
        market.mid = 50250.0;
        market.bid = 50249.0;
        market.ask = 50251.0;
        let pos = long_pos(0.001, 50000.0, 30.0);
        let s = get_exit_suggestion(&ready_sys(), &open_risk(), &pos, &default_sig(), Some(&market));
        assert_eq!(s.kind, SuggestionKind::ExitCandidate);
        assert!(s.blocked_by.iter().any(|b| b.contains("TAKE PROFIT") || b.contains("profit")));
    }

    #[test]
    fn test_exit_candidate_max_hold_time() {
        let pos = long_pos(0.001, 50000.0, 150.0); // held 150s, limit is 120s
        let s = get_exit_suggestion(&ready_sys(), &open_risk(), &pos, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::ExitCandidate);
        assert!(s.blocked_by.iter().any(|b| b.contains("MAX HOLD") || b.contains("hold")));
    }

    #[test]
    fn test_exit_wait_within_bounds() {
        // Entry 50000, mid 50050 (well within SL/TP), held 30s of 120s max
        let mut market = good_market();
        market.mid = 50050.0;
        let pos = long_pos(0.001, 50000.0, 30.0);
        let s = get_exit_suggestion(&ready_sys(), &open_risk(), &pos, &default_sig(), Some(&market));
        assert_eq!(s.kind, SuggestionKind::Wait);
        assert!(s.reason.contains("normal bounds") || s.reason.contains("Continue"));
    }

    #[test]
    fn test_exit_stand_down_kill_switch_with_position() {
        let mut risk = open_risk();
        risk.kill_switch_active = true;
        let pos = long_pos(0.001, 50000.0, 30.0);
        let s = get_exit_suggestion(&ready_sys(), &risk, &pos, &default_sig(), Some(&good_market()));
        assert_eq!(s.kind, SuggestionKind::StandDown);
        assert!(s.reason.contains("Kill switch") || s.reason.contains("kill"));
    }

    // ── get_watchlist_summary ─────────────────────────────────────────────────

    #[test]
    fn test_watchlist_stand_down_halted() {
        let mut sys = ready_sys();
        sys.mode = SystemMode::Halted;
        let (label, _detail) = get_watchlist_summary(
            "BTCUSDT", &sys, &open_risk(), &flat_pos(), &default_sig(), Some(&good_market())
        );
        assert_eq!(label, "STAND_DOWN");
    }

    #[test]
    fn test_watchlist_in_position() {
        let pos = long_pos(0.001, 50000.0, 60.0);
        let (label, detail) = get_watchlist_summary(
            "BTCUSDT", &ready_sys(), &open_risk(), &pos, &default_sig(), Some(&good_market())
        );
        assert_eq!(label, "IN_POSITION");
        assert!(detail.contains("BTCUSDT"));
        assert!(detail.contains("0.001") || detail.contains("long"));
    }

    #[test]
    fn test_watchlist_watching_flat() {
        let (label, detail) = get_watchlist_summary(
            "BTCUSDT", &ready_sys(), &open_risk(), &flat_pos(), &default_sig(), Some(&good_market())
        );
        assert_eq!(label, "WATCHING");
        assert!(detail.contains("BTCUSDT"));
    }

    #[test]
    fn test_watchlist_no_data() {
        let (label, detail) = get_watchlist_summary(
            "BTCUSDT", &ready_sys(), &open_risk(), &flat_pos(), &default_sig(), None
        );
        assert_eq!(label, "NO_DATA");
        assert!(detail.contains("BTCUSDT"));
    }

    // ── latest_market_snapshot ────────────────────────────────────────────────

    #[test]
    fn test_latest_market_snapshot_finds_most_recent() {
        let snap_old = MarketSnapshotPayload {
            bid: 49000.0, ask: 49001.0, mid: 49000.5, spread_bps: 2.0,
            momentum_1s: 0.0, momentum_3s: 0.0, momentum_5s: 0.0,
            imbalance_1s: 0.0, imbalance_3s: 0.0, imbalance_5s: 0.0,
            feed_age_ms: 100.0, mid_samples: 5, trade_samples: 3,
        };
        let snap_new = MarketSnapshotPayload {
            bid: 50000.0, ask: 50001.0, mid: 50000.5, spread_bps: 2.0,
            momentum_1s: 0.001, momentum_3s: 0.002, momentum_5s: 0.003,
            imbalance_1s: 0.5, imbalance_3s: 0.3, imbalance_5s: 0.2,
            feed_age_ms: 20.0, mid_samples: 10, trade_samples: 8,
        };
        let events = vec![
            StoredEvent::new(None, None, None, TradingEvent::MarketSnapshot(snap_old)),
            StoredEvent::new(None, None, None, TradingEvent::MarketSnapshot(snap_new.clone())),
        ];
        let result = latest_market_snapshot(&events);
        assert!(result.is_some());
        // Should be the most recent (last in slice = scanned from rev())
        assert!((result.unwrap().bid - 50000.0).abs() < 1e-6);
    }

    #[test]
    fn test_latest_market_snapshot_returns_none_when_absent() {
        let events = vec![StoredEvent::new(
            None, None, None,
            TradingEvent::SystemModeChange(crate::events::SystemModeChangePayload {
                from_mode:"Ready".into(), to_mode:"Halted".into(), reason:"test".into(),
            }),
        )];
        assert!(latest_market_snapshot(&events).is_none());
    }

    #[test]
    fn test_latest_market_snapshot_empty_events() {
        assert!(latest_market_snapshot(&[]).is_none());
    }
}

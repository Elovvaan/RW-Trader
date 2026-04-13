// events.rs
//
// Strongly typed event taxonomy for the append-only audit store.
// Every important system event is represented here.
//
// Design principles:
//   - Enums are closed: adding a new event type is a compile error everywhere
//     that matches on TradingEvent, forcing conscious updates.
//   - Each variant carries exactly the fields needed — no ad-hoc JSON blobs.
//   - StoredEvent wraps any TradingEvent with mandatory envelope fields.
//   - correlation_id ties signal → risk → execution → fill into one queryable chain.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Envelope ──────────────────────────────────────────────────────────────────

/// An event as it exists in the store: envelope + typed payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvent {
    /// Globally unique ID for this event row.
    pub event_id: String,
    /// Wall-clock UTC timestamp when the event occurred.
    pub occurred_at: DateTime<Utc>,
    /// Discriminant string matching the TradingEvent variant name.
    pub event_type: String,
    /// Symbol this event relates to (None for system-wide events).
    pub symbol: Option<String>,
    /// Ties a signal decision, risk check, order submission, and fill together.
    /// Generated once when a non-Hold signal fires; propagated to all downstream events.
    pub correlation_id: Option<String>,
    /// The clientOrderId of the order this event relates to, if applicable.
    pub client_order_id: Option<String>,
    /// Full typed payload, serialized to JSON for storage.
    pub payload: TradingEvent,
}

impl StoredEvent {
    /// Construct a new StoredEvent, generating a fresh event_id and timestamp.
    pub fn new(
        symbol: Option<String>,
        correlation_id: Option<String>,
        client_order_id: Option<String>,
        payload: TradingEvent,
    ) -> Self {
        let event_type = payload.event_type_name().to_string();
        Self {
            event_id: Uuid::new_v4().to_string(),
            occurred_at: Utc::now(),
            event_type,
            symbol,
            correlation_id,
            client_order_id,
            payload,
        }
    }
}

// ── Event payload taxonomy ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TradingEvent {
    // ── Market data ───────────────────────────────────────────────────────────
    /// Snapshot of market state captured immediately before signal evaluation.
    /// Stored for every non-Hold signal to enable deterministic replay.
    MarketSnapshot(MarketSnapshotPayload),

    // ── Signal ────────────────────────────────────────────────────────────────
    /// A SignalDecision emitted by the signal engine (Buy, Hold, or Exit).
    SignalDecision(SignalDecisionPayload),

    // ── Risk ──────────────────────────────────────────────────────────────────
    /// Result of a risk_check() call, approved or rejected with reason.
    RiskCheckResult(RiskCheckPayload),

    // ── Execution state ───────────────────────────────────────────────────────
    /// The Executor transitioned from one ExecutionState to another.
    ExecStateTransition(ExecStateTransitionPayload),

    // ── Order lifecycle ───────────────────────────────────────────────────────
    /// A market order was submitted to the exchange.
    OrderSubmitted(OrderSubmittedPayload),
    /// The exchange acknowledged the order (returned any non-error status).
    OrderAcked(OrderAckedPayload),
    /// The order reached FILLED status (partial or full).
    OrderFilled(OrderFilledPayload),
    /// The order was cancelled (by bot or exchange).
    OrderCanceled(OrderCanceledPayload),
    /// The order was rejected by the exchange.
    OrderRejected(OrderRejectedPayload),

    // ── Reconciliation ────────────────────────────────────────────────────────
    /// Reconciliation cycle started.
    ReconcileStarted(ReconcileStartedPayload),
    /// Reconciliation cycle completed.
    ReconcileCompleted(ReconcileCompletedPayload),
    /// A mismatch was detected between local and exchange state.
    ReconcileMismatch(ReconcileMismatchPayload),

    // ── Safety mechanisms ─────────────────────────────────────────────────────
    /// The watchdog detected a stuck execution state and fired.
    WatchdogTimeout(WatchdogTimeoutPayload),
    /// The circuit breaker threshold was exceeded.
    CircuitBreakerTripped(CircuitBreakerPayload),

    // ── System mode ───────────────────────────────────────────────────────────
    /// SystemMode changed (e.g. Booting → Ready, Ready → Halted).
    SystemModeChange(SystemModeChangePayload),

    // ── Operator ─────────────────────────────────────────────────────────────
    /// A human operator performed an action (kill switch, halt clear, etc.).
    OperatorAction(OperatorActionPayload),

    // ── Reconcile lifecycle ───────────────────────────────────────────────────
    /// Reconciliation applied new fills to position. Emitted when new fill IDs
    /// are discovered and the position is updated from exchange trade history.
    ReconcileApplied(ReconcileAppliedPayload),
    /// Account balances changed during a reconciliation cycle.
    /// Emitted whenever buy_power or sell_inventory differ from the previous cycle.
    BalanceUpdated(BalanceUpdatedPayload),

    // ── Replay ───────────────────────────────────────────────────────────────
    /// A replay session started.
    ReplayStarted(ReplayStartedPayload),
    /// A replay session completed.
    ReplayCompleted(ReplayCompletedPayload),
}

impl TradingEvent {
    pub fn event_type_name(&self) -> &'static str {
        match self {
            TradingEvent::MarketSnapshot(_)       => "market_snapshot",
            TradingEvent::SignalDecision(_)        => "signal_decision",
            TradingEvent::RiskCheckResult(_)       => "risk_check_result",
            TradingEvent::ExecStateTransition(_)   => "exec_state_transition",
            TradingEvent::OrderSubmitted(_)        => "order_submitted",
            TradingEvent::OrderAcked(_)            => "order_acked",
            TradingEvent::OrderFilled(_)           => "order_filled",
            TradingEvent::OrderCanceled(_)         => "order_canceled",
            TradingEvent::OrderRejected(_)         => "order_rejected",
            TradingEvent::ReconcileStarted(_)      => "reconcile_started",
            TradingEvent::ReconcileCompleted(_)    => "reconcile_completed",
            TradingEvent::ReconcileMismatch(_)     => "reconcile_mismatch",
            TradingEvent::ReconcileApplied(_)      => "reconcile_applied",
            TradingEvent::BalanceUpdated(_)        => "balance_updated",
            TradingEvent::WatchdogTimeout(_)       => "watchdog_timeout",
            TradingEvent::CircuitBreakerTripped(_) => "circuit_breaker_tripped",
            TradingEvent::SystemModeChange(_)      => "system_mode_change",
            TradingEvent::OperatorAction(_)        => "operator_action",
            TradingEvent::ReplayStarted(_)         => "replay_started",
            TradingEvent::ReplayCompleted(_)       => "replay_completed",
        }
    }
}

// ── Payload structs ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotPayload {
    pub bid:           f64,
    pub ask:           f64,
    pub mid:           f64,
    pub spread_bps:    f64,
    pub momentum_1s:   f64,
    pub momentum_3s:   f64,
    pub momentum_5s:   f64,
    pub imbalance_1s:  f64,
    pub imbalance_3s:  f64,
    pub imbalance_5s:  f64,
    pub feed_age_ms:   f64,
    pub mid_samples:   usize,
    pub trade_samples: usize,
}

impl From<&crate::signal::SignalMetrics> for MarketSnapshotPayload {
    fn from(m: &crate::signal::SignalMetrics) -> Self {
        Self {
            bid:           m.bid,
            ask:           m.ask,
            mid:           m.mid,
            spread_bps:    m.spread_bps,
            momentum_1s:   m.momentum_1s,
            momentum_3s:   m.momentum_3s,
            momentum_5s:   m.momentum_5s,
            imbalance_1s:  m.imbalance_1s,
            imbalance_3s:  m.imbalance_3s,
            imbalance_5s:  m.imbalance_5s,
            feed_age_ms:   m.feed_age_ms,
            mid_samples:   m.mid_samples,
            trade_samples: m.trade_samples,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDecisionPayload {
    /// "Buy", "Hold", or "Exit"
    pub decision:   String,
    /// For Exit: "StopLoss", "TakeProfit", or "MaxHoldTime"
    pub exit_reason: Option<String>,
    pub reason:     String,
    pub confidence: f64,
    pub metrics:    MarketSnapshotPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckPayload {
    pub approved:           bool,
    pub side:               String,
    pub qty:                f64,
    pub expected_price:     f64,
    pub rejection_reason:   Option<String>,
    /// Position size at time of check, for replay verification.
    pub position_size:      f64,
    pub position_avg_entry: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecStateTransitionPayload {
    pub from_state:       String,
    pub to_state:         String,
    pub client_order_id:  Option<String>,
    pub exchange_order_id: Option<i64>,
    pub reason:           Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSubmittedPayload {
    pub client_order_id: String,
    pub side:            String,
    pub qty:             String,
    pub order_type:      String,   // "MARKET", "LIMIT"
    pub expected_price:  f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAckedPayload {
    pub client_order_id:   String,
    pub exchange_order_id: i64,
    pub status:            String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFilledPayload {
    pub client_order_id:   String,
    pub exchange_order_id: i64,
    pub side:              String,
    pub filled_qty:        f64,
    pub avg_fill_price:    f64,
    pub cumulative_quote:  f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderCanceledPayload {
    pub client_order_id:   String,
    pub exchange_order_id: Option<i64>,
    pub reason:            String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRejectedPayload {
    pub client_order_id: String,
    pub reason:          String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileStartedPayload {
    pub cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileCompletedPayload {
    pub cycle:             u64,
    pub had_anomaly:       bool,
    pub position_size:     f64,
    pub open_orders:       usize,
    pub new_fills:         usize,
    pub duration_ms:       u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileMismatchPayload {
    pub field:          String,
    pub local_value:    String,
    pub exchange_value: String,
}

/// Summary of a single fill discovered during reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillDetail {
    pub fill_id: i64,
    pub side:    String,
    pub qty:     f64,
    pub price:   f64,
}

/// Emitted when one or more new fills were applied to the position during reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileAppliedPayload {
    /// Reconciliation cycle number.
    pub cycle:       u64,
    /// Number of new fills processed.
    pub fills_count: usize,
    /// Per-fill detail (side, qty, price) for each new fill discovered.
    pub fills:       Vec<FillDetail>,
}

/// Emitted when account balances change during a reconciliation cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceUpdatedPayload {
    /// Estimated total account value in USD.
    pub total_balance_usd: f64,
    /// Free quote-asset available for BUY orders.
    pub buy_power:         f64,
    /// Free base-asset available for SELL orders.
    pub sell_inventory:    f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogTimeoutPayload {
    pub stuck_state: String,
    pub age_secs:    f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerPayload {
    pub reason:          String,
    pub attempts_count:  u32,
    pub rejects_count:   u32,
    pub errors_count:    u32,
    pub slippage_count:  u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemModeChangePayload {
    pub from_mode: String,
    pub to_mode:   String,
    pub reason:    String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorActionPayload {
    pub action: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStartedPayload {
    pub from_occurred_at: String,
    pub to_occurred_at:   String,
    pub symbol:           String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayCompletedPayload {
    pub snapshots_replayed: usize,
    pub signals_generated:  usize,   // non-Hold decisions
    pub risk_approved:      usize,
    pub simulated_fills:    usize,
    pub duration_ms:        u64,
}

// ── Builder helpers ───────────────────────────────────────────────────────────

/// Build a SignalDecision event from a SignalResult.
pub fn signal_event(
    result: &crate::signal::SignalResult,
    symbol: &str,
    correlation_id: &str,
) -> StoredEvent {
    let (decision_str, exit_reason) = match &result.decision {
        crate::signal::SignalDecision::Buy         => ("Buy".into(),  None),
        crate::signal::SignalDecision::Hold        => ("Hold".into(), None),
        crate::signal::SignalDecision::Exit { reason } =>
            ("Exit".into(), Some(reason.to_string())),
    };

    StoredEvent::new(
        Some(symbol.to_string()),
        Some(correlation_id.to_string()),
        None,
        TradingEvent::SignalDecision(SignalDecisionPayload {
            decision:    decision_str,
            exit_reason,
            reason:      result.reason.clone(),
            confidence:  result.confidence,
            metrics:     MarketSnapshotPayload::from(&result.metrics),
        }),
    )
}

/// Build a MarketSnapshot event from SignalMetrics.
pub fn market_snapshot_event(
    metrics: &crate::signal::SignalMetrics,
    symbol: &str,
    correlation_id: &str,
) -> StoredEvent {
    StoredEvent::new(
        Some(symbol.to_string()),
        Some(correlation_id.to_string()),
        None,
        TradingEvent::MarketSnapshot(MarketSnapshotPayload::from(metrics)),
    )
}

/// Build a RiskCheckResult event.
pub fn risk_event(
    verdict: &crate::risk::RiskVerdict,
    proposed: &crate::risk::ProposedOrder,
    position: &crate::position::Position,
    symbol: &str,
    correlation_id: &str,
) -> StoredEvent {
    let (approved, rejection_reason) = match verdict {
        crate::risk::RiskVerdict::Approved => (true, None),
        crate::risk::RiskVerdict::Rejected(r) => (false, Some(r.to_string())),
    };
    let side_str = match proposed.side {
        crate::risk::OrderSide::Buy  => "BUY",
        crate::risk::OrderSide::Sell => "SELL",
    };
    StoredEvent::new(
        Some(symbol.to_string()),
        Some(correlation_id.to_string()),
        None,
        TradingEvent::RiskCheckResult(RiskCheckPayload {
            approved,
            side:               side_str.to_string(),
            qty:                proposed.qty,
            expected_price:     proposed.expected_price,
            rejection_reason,
            position_size:      position.size,
            position_avg_entry: position.avg_entry,
        }),
    )
}

/// Build an OrderSubmitted event.
pub fn order_submitted_event(
    coid: &str,
    side: &str,
    qty: &str,
    expected_price: f64,
    symbol: &str,
    correlation_id: &str,
) -> StoredEvent {
    StoredEvent::new(
        Some(symbol.to_string()),
        Some(correlation_id.to_string()),
        Some(coid.to_string()),
        TradingEvent::OrderSubmitted(OrderSubmittedPayload {
            client_order_id: coid.to_string(),
            side:            side.to_string(),
            qty:             qty.to_string(),
            order_type:      "MARKET".to_string(),
            expected_price,
        }),
    )
}

/// Build an OrderFilled event from an exchange response.
pub fn order_filled_event(
    resp: &crate::client::OrderResponse,
    symbol: &str,
    correlation_id: &str,
) -> StoredEvent {
    let filled_qty: f64 = resp.executed_qty.parse().unwrap_or(0.0);
    let cumulative: f64 = resp.cumulative_quote_qty.parse().unwrap_or(0.0);
    let avg_price = if filled_qty > 0.0 { cumulative / filled_qty } else { 0.0 };

    StoredEvent::new(
        Some(symbol.to_string()),
        Some(correlation_id.to_string()),
        Some(resp.client_order_id.clone()),
        TradingEvent::OrderFilled(OrderFilledPayload {
            client_order_id:   resp.client_order_id.clone(),
            exchange_order_id: resp.order_id,
            side:              resp.side.clone(),
            filled_qty,
            avg_fill_price:    avg_price,
            cumulative_quote:  cumulative,
        }),
    )
}

/// Build a SystemModeChange event.
pub fn mode_change_event(from: &str, to: &str, reason: &str) -> StoredEvent {
    StoredEvent::new(
        None,
        None,
        None,
        TradingEvent::SystemModeChange(SystemModeChangePayload {
            from_mode: from.to_string(),
            to_mode:   to.to_string(),
            reason:    reason.to_string(),
        }),
    )
}

/// Build a WatchdogTimeout event.
pub fn watchdog_event(stuck_state: &str, age_secs: f64) -> StoredEvent {
    StoredEvent::new(
        None,
        None,
        None,
        TradingEvent::WatchdogTimeout(WatchdogTimeoutPayload {
            stuck_state: stuck_state.to_string(),
            age_secs,
        }),
    )
}

/// Build a CircuitBreakerTripped event.
pub fn circuit_breaker_event(
    reason: &str,
    attempts: u32,
    rejects: u32,
    errors: u32,
    slippage: u32,
) -> StoredEvent {
    StoredEvent::new(
        None,
        None,
        None,
        TradingEvent::CircuitBreakerTripped(CircuitBreakerPayload {
            reason:         reason.to_string(),
            attempts_count: attempts,
            rejects_count:  rejects,
            errors_count:   errors,
            slippage_count: slippage,
        }),
    )
}

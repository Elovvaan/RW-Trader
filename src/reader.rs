// reader.rs
//
// Audit reader: reconstructs a single trade lifecycle from the event store.
//
// get_trade_timeline(store, correlation_id) fetches all events sharing a
// correlation_id, sorts them by insertion order (database id), classifies
// each into a lifecycle stage, and returns a TradeTimeline.
//
// print_trade_timeline() renders the timeline to stdout — no framework,
// no dependencies, just formatted text the operator can read.

use anyhow::{bail, Result};

use crate::events::{StoredEvent, TradingEvent};
use crate::store::EventStore;

// ── Stage classification ───────────────────────────────────────────────────────

/// Where in the trade lifecycle this event falls.
/// Events are sorted by `occurred_at` then classified into stages so the
/// printed output reads as a causal chain, not a flat list.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum LifecycleStage {
    MarketContext    = 0, // market snapshot captured at signal time
    SignalGenerated  = 1, // signal engine decision
    RiskEvaluated    = 2, // risk_check approved or rejected
    OrderSubmitted   = 3, // order sent to exchange
    OrderAcked       = 4, // exchange acknowledged
    OrderFilled      = 5, // order filled
    OrderCanceled    = 6, // order canceled
    OrderRejected    = 7, // order rejected
    StateTransition  = 8, // executor state machine transition
    ReconcileEvent   = 9, // reconcile cycle
    SafetyEvent      = 10,// watchdog / circuit breaker
    Other            = 99,
}

impl LifecycleStage {
    pub fn label(&self) -> &'static str {
        match self {
            LifecycleStage::MarketContext   => "MARKET",
            LifecycleStage::SignalGenerated => "SIGNAL",
            LifecycleStage::RiskEvaluated   => "RISK  ",
            LifecycleStage::OrderSubmitted  => "SUBMIT",
            LifecycleStage::OrderAcked      => "ACKED ",
            LifecycleStage::OrderFilled     => "FILLED",
            LifecycleStage::OrderCanceled   => "CANCEL",
            LifecycleStage::OrderRejected   => "REJECT",
            LifecycleStage::StateTransition => "STATE ",
            LifecycleStage::ReconcileEvent  => "RECON ",
            LifecycleStage::SafetyEvent     => "SAFETY",
            LifecycleStage::Other           => "OTHER ",
        }
    }
}

fn classify(event: &StoredEvent) -> LifecycleStage {
    match &event.payload {
        TradingEvent::MarketSnapshot(_)      => LifecycleStage::MarketContext,
        TradingEvent::SignalDecision(_)      => LifecycleStage::SignalGenerated,
        TradingEvent::RiskCheckResult(_)     => LifecycleStage::RiskEvaluated,
        TradingEvent::OrderSubmitted(_)      => LifecycleStage::OrderSubmitted,
        TradingEvent::OrderAcked(_)          => LifecycleStage::OrderAcked,
        TradingEvent::OrderFilled(_)         => LifecycleStage::OrderFilled,
        TradingEvent::OrderCanceled(_)       => LifecycleStage::OrderCanceled,
        TradingEvent::OrderRejected(_)       => LifecycleStage::OrderRejected,
        TradingEvent::ExecStateTransition(_) => LifecycleStage::StateTransition,
        TradingEvent::ReconcileStarted(_)
        | TradingEvent::ReconcileCompleted(_)
        | TradingEvent::ReconcileMismatch(_) => LifecycleStage::ReconcileEvent,
        TradingEvent::WatchdogTimeout(_)
        | TradingEvent::CircuitBreakerTripped(_) => LifecycleStage::SafetyEvent,
        _ => LifecycleStage::Other,
    }
}

// ── Timeline event ─────────────────────────────────────────────────────────────

/// One event in the trade timeline, enriched with its lifecycle stage.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    pub stage:    LifecycleStage,
    pub event:    StoredEvent,
    /// Human-readable one-line summary extracted from the payload.
    pub summary:  String,
}

impl TimelineEvent {
    fn from_stored(event: StoredEvent) -> Self {
        let stage   = classify(&event);
        let summary = summarise(&event);
        Self { stage, event, summary }
    }
}

fn summarise(e: &StoredEvent) -> String {
    summarise_event(e)
}

/// Public alias for summarise — used by the --recent CLI command in main.rs.
pub fn summarise_event(e: &StoredEvent) -> String {
    match &e.payload {
        TradingEvent::MarketSnapshot(p) =>
            format!(
                "bid={:.2} ask={:.2} spread={:.2}bps  \
                 mom1s={:+.5} mom3s={:+.5} mom5s={:+.5}  \
                 imb1s={:+.3} imb3s={:+.3}",
                p.bid, p.ask, p.spread_bps,
                p.momentum_1s, p.momentum_3s, p.momentum_5s,
                p.imbalance_1s, p.imbalance_3s,
            ),

        TradingEvent::SignalDecision(p) => {
            let mut s = format!("decision={}", p.decision);
            if let Some(r) = &p.exit_reason { s.push_str(&format!("({})", r)); }
            s.push_str(&format!("  confidence={:.2}  reason=\"{}\"", p.confidence, p.reason));
            s
        }

        TradingEvent::RiskCheckResult(p) => {
            if p.approved {
                format!(
                    "APPROVED  side={}  qty={:.6}  price={:.2}  \
                     pos_size={:.6}  pos_avg={:.2}",
                    p.side, p.qty, p.expected_price,
                    p.position_size, p.position_avg_entry,
                )
            } else {
                format!(
                    "REJECTED  reason=\"{}\"",
                    p.rejection_reason.as_deref().unwrap_or("unknown"),
                )
            }
        }

        TradingEvent::OrderSubmitted(p) =>
            format!(
                "coid={}  side={}  qty={}  type={}  expected_price={:.2}",
                p.client_order_id, p.side, p.qty, p.order_type, p.expected_price,
            ),

        TradingEvent::OrderAcked(p) =>
            format!(
                "coid={}  exchange_id={}  status={}",
                p.client_order_id, p.exchange_order_id, p.status,
            ),

        TradingEvent::OrderFilled(p) =>
            format!(
                "coid={}  exchange_id={}  side={}  qty={:.6}  avg_price={:.2}  \
                 notional={:.4}",
                p.client_order_id, p.exchange_order_id, p.side,
                p.filled_qty, p.avg_fill_price, p.cumulative_quote,
            ),

        TradingEvent::OrderCanceled(p) =>
            format!(
                "coid={}  reason=\"{}\"",
                p.client_order_id, p.reason,
            ),

        TradingEvent::OrderRejected(p) =>
            format!(
                "coid={}  reason=\"{}\"",
                p.client_order_id, p.reason,
            ),

        TradingEvent::ExecStateTransition(p) => {
            let mut s = format!("{} → {}", p.from_state, p.to_state);
            if let Some(c) = &p.client_order_id { s.push_str(&format!("  coid={}", c)); }
            if let Some(r) = &p.reason          { s.push_str(&format!("  reason=\"{}\"", r)); }
            s
        }

        TradingEvent::ReconcileStarted(p) =>
            format!("cycle={}", p.cycle),

        TradingEvent::ReconcileCompleted(p) =>
            format!(
                "cycle={}  anomaly={}  pos_size={:.6}  open_orders={}  \
                 new_fills={}  duration={}ms",
                p.cycle, p.had_anomaly, p.position_size,
                p.open_orders, p.new_fills, p.duration_ms,
            ),

        TradingEvent::ReconcileMismatch(p) =>
            format!(
                "field={}  local=\"{}\"  exchange=\"{}\"",
                p.field, p.local_value, p.exchange_value,
            ),

        TradingEvent::WatchdogTimeout(p) =>
            format!("stuck_state={}  age={:.1}s", p.stuck_state, p.age_secs),

        TradingEvent::CircuitBreakerTripped(p) =>
            format!(
                "reason=\"{}\"  attempts={}  rejects={}  errors={}  slippage={}",
                p.reason, p.attempts_count, p.rejects_count,
                p.errors_count, p.slippage_count,
            ),

        TradingEvent::SystemModeChange(p) =>
            format!("{} → {}  reason=\"{}\"", p.from_mode, p.to_mode, p.reason),

        TradingEvent::OperatorAction(p) =>
            format!("action={}  reason=\"{}\"", p.action, p.reason),

        TradingEvent::ReplayStarted(p) =>
            format!("symbol={}  from={}  to={}", p.symbol, p.from_occurred_at, p.to_occurred_at),

        TradingEvent::ReplayCompleted(p) =>
            format!(
                "snapshots={}  signals={}  approved={}  fills={}  duration={}ms",
                p.snapshots_replayed, p.signals_generated,
                p.risk_approved, p.simulated_fills, p.duration_ms,
            ),
    }
}

// ── TradeTimeline ─────────────────────────────────────────────────────────────

/// The fully reconstructed lifecycle of one trade decision.
#[derive(Debug)]
pub struct TradeTimeline {
    pub correlation_id: String,
    pub symbol:         Option<String>,
    pub events:         Vec<TimelineEvent>,

    // Derived summary fields, computed once from the event list.
    pub signal_decision:  Option<String>,   // "Buy" / "Exit(StopLoss)" etc.
    pub risk_outcome:     Option<String>,   // "APPROVED" / "REJECTED: <reason>"
    pub client_order_id:  Option<String>,
    pub exchange_order_id: Option<i64>,
    pub fill_price:       Option<f64>,
    pub fill_qty:         Option<f64>,
    pub outcome:          TradeOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TradeOutcome {
    /// Signal emitted and order was filled.
    Filled,
    /// Signal emitted but risk rejected it.
    RiskRejected,
    /// Signal emitted, risk approved, but order was rejected by exchange.
    OrderRejected,
    /// Signal emitted, risk approved, order submitted, but not yet filled.
    Pending,
    /// Could not determine outcome from stored events.
    Unknown,
}

impl std::fmt::Display for TradeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeOutcome::Filled        => write!(f, "FILLED"),
            TradeOutcome::RiskRejected  => write!(f, "RISK_REJECTED"),
            TradeOutcome::OrderRejected => write!(f, "ORDER_REJECTED"),
            TradeOutcome::Pending       => write!(f, "PENDING"),
            TradeOutcome::Unknown       => write!(f, "UNKNOWN"),
        }
    }
}

// ── Core function ─────────────────────────────────────────────────────────────

/// Reconstruct the complete trade timeline for `correlation_id`.
///
/// Fetches all events sharing the correlation_id from the store, sorts them
/// by occurrence time (database insertion order = causal order because events
/// are appended synchronously within each pipeline tick), and derives summary
/// fields for quick inspection.
///
/// Returns an error if no events are found for the given correlation_id.
pub fn get_trade_timeline(
    store: &dyn EventStore,
    correlation_id: &str,
) -> Result<TradeTimeline> {
    // fetch_trade_lifecycle returns events ordered by database id ASC,
    // which equals causal order: signal → risk → submit → ack → fill.
    let raw = store.fetch_trade_lifecycle(correlation_id)?;

    if raw.is_empty() {
        bail!(
            "No events found for correlation_id '{}'. \
             Verify the ID is correct and the event store is the same database \
             used during live trading.",
            correlation_id
        );
    }

    // Classify and build timeline events.
    let events: Vec<TimelineEvent> = raw
        .into_iter()
        .map(TimelineEvent::from_stored)
        .collect();

    // Derive symbol from the first event that carries one.
    let symbol = events.iter()
        .find_map(|te| te.event.symbol.clone());

    // Extract summary fields by scanning the event list once.
    let mut signal_decision   = None;
    let mut risk_outcome      = None;
    let mut client_order_id   = None;
    let mut exchange_order_id = None;
    let mut fill_price        = None;
    let mut fill_qty          = None;

    for te in &events {
        match &te.event.payload {
            TradingEvent::SignalDecision(p) => {
                let mut s = p.decision.clone();
                if let Some(r) = &p.exit_reason { s.push_str(&format!("({})", r)); }
                signal_decision = Some(s);
            }
            TradingEvent::RiskCheckResult(p) => {
                risk_outcome = Some(if p.approved {
                    "APPROVED".to_string()
                } else {
                    format!(
                        "REJECTED: {}",
                        p.rejection_reason.as_deref().unwrap_or("unknown")
                    )
                });
            }
            TradingEvent::OrderSubmitted(p) => {
                client_order_id = Some(p.client_order_id.clone());
            }
            TradingEvent::OrderAcked(p) => {
                exchange_order_id = Some(p.exchange_order_id);
            }
            TradingEvent::OrderFilled(p) => {
                fill_price = Some(p.avg_fill_price);
                fill_qty   = Some(p.filled_qty);
                if exchange_order_id.is_none() {
                    exchange_order_id = Some(p.exchange_order_id);
                }
            }
            _ => {}
        }
    }

    // Determine overall outcome from events present.
    let outcome = determine_outcome(&events);

    Ok(TradeTimeline {
        correlation_id: correlation_id.to_string(),
        symbol,
        events,
        signal_decision,
        risk_outcome,
        client_order_id,
        exchange_order_id,
        fill_price,
        fill_qty,
        outcome,
    })
}

fn determine_outcome(events: &[TimelineEvent]) -> TradeOutcome {
    let has_fill     = events.iter().any(|te| matches!(te.event.payload, TradingEvent::OrderFilled(_)));
    let has_rejected = events.iter().any(|te| matches!(te.event.payload, TradingEvent::OrderRejected(_)));
    let has_submit   = events.iter().any(|te| matches!(te.event.payload, TradingEvent::OrderSubmitted(_)));
    let risk_rejected = events.iter().any(|te| {
        matches!(&te.event.payload, TradingEvent::RiskCheckResult(p) if !p.approved)
    });

    if has_fill            { TradeOutcome::Filled }
    else if risk_rejected  { TradeOutcome::RiskRejected }
    else if has_rejected   { TradeOutcome::OrderRejected }
    else if has_submit     { TradeOutcome::Pending }
    else                   { TradeOutcome::Unknown }
}

// ── CLI printer ───────────────────────────────────────────────────────────────

/// Print a TradeTimeline to stdout in a readable format.
/// No external dependencies — plain text, works in any terminal.
pub fn print_trade_timeline(timeline: &TradeTimeline) {
    let sep  = "─".repeat(80);
    let sep2 = "═".repeat(80);

    println!();
    println!("{}", sep2);
    println!("  TRADE TIMELINE");
    println!("{}", sep2);
    println!("  correlation_id : {}", timeline.correlation_id);
    println!("  symbol         : {}", timeline.symbol.as_deref().unwrap_or("—"));
    println!("  outcome        : {}", timeline.outcome);

    if let Some(s) = &timeline.signal_decision {
        println!("  signal         : {}", s);
    }
    if let Some(r) = &timeline.risk_outcome {
        println!("  risk           : {}", r);
    }
    if let Some(c) = &timeline.client_order_id {
        println!("  client_order_id: {}", c);
    }
    if let Some(x) = timeline.exchange_order_id {
        println!("  exchange_id    : {}", x);
    }
    if let (Some(price), Some(qty)) = (timeline.fill_price, timeline.fill_qty) {
        println!("  fill           : qty={:.6}  avg_price={:.2}  notional={:.4}",
            qty, price, qty * price);
    }

    println!("{}", sep);
    println!(
        "  {:>3}  {:26}  {:<6}  {}",
        "#", "timestamp (UTC)", "stage", "detail"
    );
    println!("{}", sep);

    for (i, te) in timeline.events.iter().enumerate() {
        let ts = te.event.occurred_at.format("%Y-%m-%d %H:%M:%S%.3f");
        println!(
            "  {:>3}  {}  {}  {}",
            i + 1,
            ts,
            te.stage.label(),
            te.summary,
        );
        // For market snapshot, print a second line with momentum/imbalance context
        // so operators can see why the signal fired without scrolling.
        if let TradingEvent::MarketSnapshot(p) = &te.event.payload {
            println!(
                "       {}                        mid={:.2}  samples={}t/{}m",
                " ".repeat(26),
                p.mid, p.trade_samples, p.mid_samples,
            );
        }
    }

    println!("{}", sep2);
    println!("  {} event(s) in timeline", timeline.events.len());
    println!("{}", sep2);
    println!();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::*;
    use crate::store::InMemoryEventStore;
    use std::sync::Arc;

    // ── helpers ───────────────────────────────────────────────────────────────

    const CORR: &str = "test-corr-001";
    const SYMBOL: &str = "BTCUSDT";

    fn snap() -> MarketSnapshotPayload {
        MarketSnapshotPayload {
            bid: 50000.0, ask: 50001.0, mid: 50000.5, spread_bps: 2.0,
            momentum_1s: 0.001, momentum_3s: 0.002, momentum_5s: 0.003,
            imbalance_1s: 0.5, imbalance_3s: 0.3, imbalance_5s: 0.2,
            feed_age_ms: 20.0, mid_samples: 10, trade_samples: 8,
        }
    }

    fn append_full_trade_lifecycle(store: &Arc<InMemoryEventStore>) {
        // 1. Market snapshot
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::MarketSnapshot(snap()),
        ));
        // 2. Signal decision
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Buy".into(), exit_reason: None,
                reason: "all gates passed".into(), confidence: 0.75,
                metrics: snap(),
            }),
        ));
        // 3. Risk approved
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: true, side: "BUY".into(), qty: 0.001,
                expected_price: 50001.0, rejection_reason: None,
                position_size: 0.0, position_avg_entry: 0.0,
            }),
        ));
        // 4. Order submitted
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), Some("coid-abc".into()),
            TradingEvent::OrderSubmitted(OrderSubmittedPayload {
                client_order_id: "coid-abc".into(), side: "BUY".into(),
                qty: "0.00100".into(), order_type: "MARKET".into(),
                expected_price: 50001.0,
            }),
        ));
        // 5. Order acked
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), Some("coid-abc".into()),
            TradingEvent::OrderAcked(OrderAckedPayload {
                client_order_id: "coid-abc".into(),
                exchange_order_id: 99999,
                status: "FILLED".into(),
            }),
        ));
        // 6. Order filled
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), Some("coid-abc".into()),
            TradingEvent::OrderFilled(OrderFilledPayload {
                client_order_id: "coid-abc".into(), exchange_order_id: 99999,
                side: "BUY".into(), filled_qty: 0.001, avg_fill_price: 50001.5,
                cumulative_quote: 50.0015,
            }),
        ));
    }

    // ── get_trade_timeline ────────────────────────────────────────────────────

    #[test]
    fn test_returns_error_for_unknown_correlation_id() {
        let store = InMemoryEventStore::new();
        let result = get_trade_timeline(&*store, "nonexistent-correlation-id");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No events found"));
    }

    #[test]
    fn test_full_lifecycle_reconstructed() {
        let store = InMemoryEventStore::new();
        append_full_trade_lifecycle(&store);

        let timeline = get_trade_timeline(&*store, CORR).unwrap();

        assert_eq!(timeline.correlation_id, CORR);
        assert_eq!(timeline.symbol.as_deref(), Some(SYMBOL));
        assert_eq!(timeline.events.len(), 6);
    }

    #[test]
    fn test_events_in_causal_order() {
        let store = InMemoryEventStore::new();
        append_full_trade_lifecycle(&store);

        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        let stages: Vec<&LifecycleStage> = timeline.events.iter().map(|e| &e.stage).collect();

        // Must follow causal order
        assert_eq!(stages[0], &LifecycleStage::MarketContext);
        assert_eq!(stages[1], &LifecycleStage::SignalGenerated);
        assert_eq!(stages[2], &LifecycleStage::RiskEvaluated);
        assert_eq!(stages[3], &LifecycleStage::OrderSubmitted);
        assert_eq!(stages[4], &LifecycleStage::OrderAcked);
        assert_eq!(stages[5], &LifecycleStage::OrderFilled);
    }

    #[test]
    fn test_outcome_filled() {
        let store = InMemoryEventStore::new();
        append_full_trade_lifecycle(&store);
        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        assert_eq!(timeline.outcome, TradeOutcome::Filled);
    }

    #[test]
    fn test_outcome_risk_rejected() {
        let store = InMemoryEventStore::new();
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Buy".into(), exit_reason: None,
                reason: "signal fired".into(), confidence: 0.6,
                metrics: snap(),
            }),
        ));
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: false, side: "BUY".into(), qty: 0.001,
                expected_price: 50001.0,
                rejection_reason: Some("COOLDOWN: 120s remaining".into()),
                position_size: 0.0, position_avg_entry: 0.0,
            }),
        ));

        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        assert_eq!(timeline.outcome, TradeOutcome::RiskRejected);
    }

    #[test]
    fn test_outcome_order_rejected() {
        let store = InMemoryEventStore::new();
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Buy".into(), exit_reason: None,
                reason: "ok".into(), confidence: 0.8, metrics: snap(),
            }),
        ));
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: true, side: "BUY".into(), qty: 0.001,
                expected_price: 50001.0, rejection_reason: None,
                position_size: 0.0, position_avg_entry: 0.0,
            }),
        ));
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), Some("coid-x".into()),
            TradingEvent::OrderSubmitted(OrderSubmittedPayload {
                client_order_id: "coid-x".into(), side: "BUY".into(),
                qty: "0.001".into(), order_type: "MARKET".into(), expected_price: 50001.0,
            }),
        ));
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), Some("coid-x".into()),
            TradingEvent::OrderRejected(OrderRejectedPayload {
                client_order_id: "coid-x".into(),
                reason: "Binance error -2010: insufficient balance".into(),
            }),
        ));

        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        assert_eq!(timeline.outcome, TradeOutcome::OrderRejected);
    }

    #[test]
    fn test_outcome_pending_when_submitted_but_not_filled() {
        let store = InMemoryEventStore::new();
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: true, side: "BUY".into(), qty: 0.001,
                expected_price: 50001.0, rejection_reason: None,
                position_size: 0.0, position_avg_entry: 0.0,
            }),
        ));
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), Some("coid-pending".into()),
            TradingEvent::OrderSubmitted(OrderSubmittedPayload {
                client_order_id: "coid-pending".into(), side: "BUY".into(),
                qty: "0.001".into(), order_type: "LIMIT".into(), expected_price: 50000.0,
            }),
        ));

        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        assert_eq!(timeline.outcome, TradeOutcome::Pending);
    }

    #[test]
    fn test_derived_fields_extracted() {
        let store = InMemoryEventStore::new();
        append_full_trade_lifecycle(&store);
        let timeline = get_trade_timeline(&*store, CORR).unwrap();

        assert_eq!(timeline.signal_decision.as_deref(), Some("Buy"));
        assert_eq!(timeline.risk_outcome.as_deref(), Some("APPROVED"));
        assert_eq!(timeline.client_order_id.as_deref(), Some("coid-abc"));
        assert_eq!(timeline.exchange_order_id, Some(99999));
        assert!((timeline.fill_price.unwrap() - 50001.5).abs() < 1e-6);
        assert!((timeline.fill_qty.unwrap()   - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_exit_signal_includes_reason() {
        let store = InMemoryEventStore::new();
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some(CORR.into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Exit".into(), exit_reason: Some("StopLoss".into()),
                reason: "price below stop".into(), confidence: 1.0, metrics: snap(),
            }),
        ));

        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        // Derived signal_decision should include the exit reason
        assert_eq!(timeline.signal_decision.as_deref(), Some("Exit(StopLoss)"));
    }

    #[test]
    fn test_only_events_for_this_correlation_returned() {
        let store = InMemoryEventStore::new();
        append_full_trade_lifecycle(&store);

        // Add events with a different correlation_id
        store.append(StoredEvent::new(
            Some(SYMBOL.into()), Some("other-corr".into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Buy".into(), exit_reason: None,
                reason: "different trade".into(), confidence: 0.5, metrics: snap(),
            }),
        ));

        let timeline = get_trade_timeline(&*store, CORR).unwrap();
        // Still 6 — the other correlation's events must not appear
        assert_eq!(timeline.events.len(), 6);

        for te in &timeline.events {
            assert_eq!(
                te.event.correlation_id.as_deref(),
                Some(CORR),
                "All events must belong to the queried correlation_id"
            );
        }
    }

    // ── summarise ─────────────────────────────────────────────────────────────

    #[test]
    fn test_summarise_does_not_panic_for_any_variant() {
        // Run summarise over every event type to ensure no panics or missing arms.
        let events = vec![
            StoredEvent::new(None, None, None, TradingEvent::MarketSnapshot(snap())),
            StoredEvent::new(None, None, None, TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Buy".into(), exit_reason: None,
                reason: "".into(), confidence: 0.0, metrics: snap(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: true, side: "BUY".into(), qty: 0.001,
                expected_price: 0.0, rejection_reason: None,
                position_size: 0.0, position_avg_entry: 0.0,
            })),
            StoredEvent::new(None, None, None, TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: false, side: "BUY".into(), qty: 0.001,
                expected_price: 0.0,
                rejection_reason: Some("cooldown".into()),
                position_size: 0.0, position_avg_entry: 0.0,
            })),
            StoredEvent::new(None, None, None, TradingEvent::ExecStateTransition(ExecStateTransitionPayload {
                from_state: "Idle".into(), to_state: "Submitting".into(),
                client_order_id: Some("c".into()), exchange_order_id: None, reason: None,
            })),
            StoredEvent::new(None, None, None, TradingEvent::OrderSubmitted(OrderSubmittedPayload {
                client_order_id: "c".into(), side: "BUY".into(),
                qty: "0.001".into(), order_type: "MARKET".into(), expected_price: 50000.0,
            })),
            StoredEvent::new(None, None, None, TradingEvent::OrderAcked(OrderAckedPayload {
                client_order_id: "c".into(), exchange_order_id: 1, status: "FILLED".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::OrderFilled(OrderFilledPayload {
                client_order_id: "c".into(), exchange_order_id: 1,
                side: "BUY".into(), filled_qty: 0.001, avg_fill_price: 50000.0,
                cumulative_quote: 50.0,
            })),
            StoredEvent::new(None, None, None, TradingEvent::OrderCanceled(OrderCanceledPayload {
                client_order_id: "c".into(), exchange_order_id: None,
                reason: "test".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::OrderRejected(OrderRejectedPayload {
                client_order_id: "c".into(), reason: "test".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::ReconcileStarted(ReconcileStartedPayload { cycle: 1 })),
            StoredEvent::new(None, None, None, TradingEvent::ReconcileCompleted(ReconcileCompletedPayload {
                cycle: 1, had_anomaly: false, position_size: 0.0,
                open_orders: 0, new_fills: 0, duration_ms: 50,
            })),
            StoredEvent::new(None, None, None, TradingEvent::ReconcileMismatch(ReconcileMismatchPayload {
                field: "position".into(), local_value: "0.1".into(), exchange_value: "0.0".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::WatchdogTimeout(WatchdogTimeoutPayload {
                stuck_state: "WaitingAck".into(), age_secs: 11.2,
            })),
            StoredEvent::new(None, None, None, TradingEvent::CircuitBreakerTripped(CircuitBreakerPayload {
                reason: "too many rejects".into(),
                attempts_count: 10, rejects_count: 3, errors_count: 0, slippage_count: 0,
            })),
            StoredEvent::new(None, None, None, TradingEvent::SystemModeChange(SystemModeChangePayload {
                from_mode: "Ready".into(), to_mode: "Halted".into(), reason: "cb".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::OperatorAction(OperatorActionPayload {
                action: "clear_halt".into(), reason: "manual".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::ReplayStarted(ReplayStartedPayload {
                from_occurred_at: "2024-01-01".into(), to_occurred_at: "2024-01-02".into(),
                symbol: "BTCUSDT".into(),
            })),
            StoredEvent::new(None, None, None, TradingEvent::ReplayCompleted(ReplayCompletedPayload {
                snapshots_replayed: 100, signals_generated: 5,
                risk_approved: 3, simulated_fills: 3, duration_ms: 250,
            })),
        ];
        for e in events {
            let _ = summarise(&e); // must not panic
        }
    }

    // ── classify ─────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_all_stages_covered() {
        let e = |payload| StoredEvent::new(None, None, None, payload);
        assert_eq!(classify(&e(TradingEvent::MarketSnapshot(snap()))), LifecycleStage::MarketContext);
        assert_eq!(classify(&e(TradingEvent::OrderFilled(OrderFilledPayload {
            client_order_id: "c".into(), exchange_order_id: 1,
            side: "BUY".into(), filled_qty: 0.001, avg_fill_price: 50000.0, cumulative_quote: 50.0,
        }))), LifecycleStage::OrderFilled);
        assert_eq!(classify(&e(TradingEvent::WatchdogTimeout(WatchdogTimeoutPayload {
            stuck_state: "WaitingAck".into(), age_secs: 10.0,
        }))), LifecycleStage::SafetyEvent);
    }
}

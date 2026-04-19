// executor.rs
//
// Hardened execution layer. Every exchange order goes through here.
// Nothing bypasses it — not signal logic, not reconciliation.
//
// Guarantees:
//   1. Only one order slot is active at any time.
//   2. New orders are only accepted in Idle.
//   3. The same client_order_id is reused on retries.
//   4. Duplicate coids are rejected before touching the exchange.
//   5. A circuit breaker halts trading if submission error rates spike.
//   6. A watchdog detects stuck states and forces reconciliation.
//   7. Reconciliation authority can reset any bad state.
//   8. Trading is only allowed in SystemMode::Ready or Degraded.
//   9. The Mutex is never held across an async exchange call.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::client::{BinanceClient, OpenOrder, OrderResponse};
use crate::orders::{submit_market_with_retry, RetryPolicy};
use crate::reconciler::{OrderRecord, OrderStatus, TruthState};
use crate::risk::RiskEngine;
use crate::store::EventStore;

// ── ExecutionState ────────────────────────────────────────────────────────────

/// State of the single active order slot.
/// Only `Idle` allows new order actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionState {
    /// No active order. New orders accepted.
    Idle,
    /// HTTP request in flight. `client_order_id` locked in.
    Submitting { client_order_id: String },
    /// Request accepted by exchange; awaiting status confirmation.
    WaitingAck { client_order_id: String, since: Instant },
    /// Order is live on the exchange.
    Open { exchange_order_id: i64, client_order_id: String, since: Instant },
    /// Cancel request in flight.
    Canceling { exchange_order_id: i64, since: Instant },
    /// Cancel confirmed; building replacement order.
    Replacing { since: Instant },
    /// State is inconsistent. Reconciliation required before any order.
    Recovery { reason: String, since: Instant },
}

impl ExecutionState {
    /// Only `Idle` allows new order submission.
    pub fn can_accept_order(&self) -> bool {
        matches!(self, ExecutionState::Idle)
    }

    pub fn name(&self) -> &'static str {
        match self {
            ExecutionState::Idle             => "Monitoring for trigger",
            ExecutionState::Submitting { .. } => "Submitting Order",
            ExecutionState::WaitingAck { .. } => "Order Working",
            ExecutionState::Open { .. }       => "Order Working",
            ExecutionState::Canceling { .. }  => "Canceling",
            ExecutionState::Replacing { .. }  => "Replacing",
            ExecutionState::Recovery { .. }   => "Recovery",
        }
    }

    /// How long we've been in the current state (if time-stamped).
    pub fn age(&self) -> Option<Duration> {
        let since = match self {
            ExecutionState::WaitingAck { since, .. } => *since,
            ExecutionState::Open { since, .. }       => *since,
            ExecutionState::Canceling { since, .. }  => *since,
            ExecutionState::Replacing { since }      => *since,
            ExecutionState::Recovery { since, .. }   => *since,
            _ => return None,
        };
        Some(since.elapsed())
    }

    pub fn client_order_id(&self) -> Option<&str> {
        match self {
            ExecutionState::Submitting { client_order_id } => Some(client_order_id),
            ExecutionState::WaitingAck { client_order_id, .. } => Some(client_order_id),
            ExecutionState::Open { client_order_id, .. } => Some(client_order_id),
            _ => None,
        }
    }
}

impl std::fmt::Display for ExecutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionState::Idle => write!(f, "Monitoring for trigger"),
            ExecutionState::Submitting { client_order_id } =>
                write!(f, "Submitting Order({})", client_order_id),
            ExecutionState::WaitingAck { client_order_id, .. } =>
                write!(f, "Order Working({})", client_order_id),
            ExecutionState::Open { exchange_order_id, .. } =>
                write!(f, "Order Working({})", exchange_order_id),
            ExecutionState::Canceling { exchange_order_id, .. } =>
                write!(f, "Canceling({})", exchange_order_id),
            ExecutionState::Replacing { .. } => write!(f, "Replacing"),
            ExecutionState::Recovery { reason, .. } =>
                write!(f, "Recovery({})", reason),
        }
    }
}

// ── SystemMode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
    /// Process starting up.
    Booting,
    /// Exchange reconciliation is running.
    Reconciling,
    /// State is clean and consistent. Trading allowed.
    Ready,
    /// Anomalies detected but not halted. Trading continues with extra logging.
    Degraded,
    /// Kill switch active or circuit breaker tripped. No trading.
    Halted,
}

impl SystemMode {
    pub fn can_trade(self) -> bool {
        matches!(self, SystemMode::Ready | SystemMode::Degraded)
    }
}

impl std::fmt::Display for SystemMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemMode::Booting     => write!(f, "Booting"),
            SystemMode::Reconciling => write!(f, "Reconciling"),
            SystemMode::Ready       => write!(f, "Ready"),
            SystemMode::Degraded    => write!(f, "Degraded"),
            SystemMode::Halted      => write!(f, "Halted"),
        }
    }
}

// ── Circuit breaker ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Max order attempts per minute before tripping.
    pub max_attempts_per_min: u32,
    /// Max Binance-side rejects per minute (e.g. -1111, -2010).
    pub max_rejects_per_min: u32,
    /// Max network/transport errors per minute.
    pub max_errors_per_min: u32,
    /// Max slippage breaches per minute.
    pub max_slippage_per_min: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            max_attempts_per_min:  10,
            max_rejects_per_min:   3,
            max_errors_per_min:    3,
            max_slippage_per_min:  2,
        }
    }
}

/// Per-minute sliding-window counters.
#[derive(Debug, Default)]
struct CircuitBreaker {
    config:    CircuitBreakerConfig,
    attempts:  VecDeque<Instant>,
    rejects:   VecDeque<Instant>,
    errors:    VecDeque<Instant>,
    slippages: VecDeque<Instant>,
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self { config, ..Default::default() }
    }

    fn prune(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(60);
        Self::prune_queue(&mut self.attempts, cutoff);
        Self::prune_queue(&mut self.rejects, cutoff);
        Self::prune_queue(&mut self.errors, cutoff);
        Self::prune_queue(&mut self.slippages, cutoff);
    }

    fn prune_queue(q: &mut VecDeque<Instant>, cutoff: Instant) {
        while q.front().map_or(false, |&t| t < cutoff) {
            q.pop_front();
        }
    }

    fn record_attempt(&mut self) { self.attempts.push_back(Instant::now()); }
    fn record_reject(&mut self)  { self.rejects.push_back(Instant::now()); }
    fn record_error(&mut self)   { self.errors.push_back(Instant::now()); }
    fn record_slippage(&mut self){ self.slippages.push_back(Instant::now()); }

    /// Returns Some(reason) if any threshold is breached.
    fn check_thresholds(&mut self) -> Option<String> {
        self.prune();
        let attempts  = self.attempts.len() as u32;
        let rejects   = self.rejects.len() as u32;
        let errors    = self.errors.len() as u32;
        let slippages = self.slippages.len() as u32;

        if attempts >= self.config.max_attempts_per_min {
            return Some(format!("Too many attempts: {} in 60s (limit {})",
                attempts, self.config.max_attempts_per_min));
        }
        if rejects >= self.config.max_rejects_per_min {
            return Some(format!("Too many exchange rejects: {} in 60s (limit {})",
                rejects, self.config.max_rejects_per_min));
        }
        if errors >= self.config.max_errors_per_min {
            return Some(format!("Too many submission errors: {} in 60s (limit {})",
                errors, self.config.max_errors_per_min));
        }
        if slippages >= self.config.max_slippage_per_min {
            return Some(format!("Too many slippage breaches: {} in 60s (limit {})",
                slippages, self.config.max_slippage_per_min));
        }
        None
    }

    fn counts(&self) -> (u32, u32, u32, u32) {
        (
            self.attempts.len() as u32,
            self.rejects.len() as u32,
            self.errors.len() as u32,
            self.slippages.len() as u32,
        )
    }
}

// ── Watchdog timeouts ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    pub waiting_ack_timeout: Duration,
    pub canceling_timeout:   Duration,
    pub replacing_timeout:   Duration,
    pub check_interval:      Duration,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            waiting_ack_timeout: Duration::from_secs(10),
            canceling_timeout:   Duration::from_secs(5),
            replacing_timeout:   Duration::from_secs(15),
            check_interval:      Duration::from_millis(500),
        }
    }
}

// ── Executor inner state (held under Mutex) ───────────────────────────────────

struct ExecutorInner {
    exec_state:      ExecutionState,
    system_mode:     SystemMode,
    circuit_breaker: CircuitBreaker,
    /// All coids submitted this session (for duplicate detection).
    submitted_coids: std::collections::HashSet<String>,
    watchdog_cfg:    WatchdogConfig,
    symbol:          String,
    /// Event store — append is sync/non-blocking (channel send).
    /// None = no-op (used in tests that don't need audit trail).
    event_store:     Option<Arc<dyn EventStore>>,
}

impl ExecutorInner {
    fn transition(&mut self, next: ExecutionState) {
        let prev_name = self.exec_state.name();
        let next_name = next.name();
        let coid = next.client_order_id()
            .or_else(|| self.exec_state.client_order_id())
            .map(str::to_string);
        let oid = match &next {
            ExecutionState::Open { exchange_order_id, .. }     => Some(*exchange_order_id),
            ExecutionState::Canceling { exchange_order_id, .. }=> Some(*exchange_order_id),
            _ => None,
        };
        info!(
            from  = prev_name,
            to    = next_name,
            state = %next,
            "[EXEC] State transition"
        );
        if let Some(store) = &self.event_store {
            store.append(crate::events::StoredEvent::new(
                Some(self.symbol.clone()),
                coid.clone(),
                coid,
                crate::events::TradingEvent::ExecStateTransition(
                    crate::events::ExecStateTransitionPayload {
                        from_state:        prev_name.to_string(),
                        to_state:          next_name.to_string(),
                        client_order_id:   next.client_order_id()
                            .or_else(|| self.exec_state.client_order_id())
                            .map(str::to_string),
                        exchange_order_id: oid,
                        reason:            None,
                    },
                ),
            ));
        }
        self.exec_state = next;
    }

    fn transition_to_recovery(&mut self, reason: &str) {
        let prev = self.exec_state.name();
        warn!(from = prev, reason, "[EXEC] → Recovery");
        if let Some(store) = &self.event_store {
            store.append(crate::events::StoredEvent::new(
                Some(self.symbol.clone()),
                None,
                None,
                crate::events::TradingEvent::ExecStateTransition(
                    crate::events::ExecStateTransitionPayload {
                        from_state:        prev.to_string(),
                        to_state:          "Recovery".to_string(),
                        client_order_id:   None,
                        exchange_order_id: None,
                        reason:            Some(reason.to_string()),
                    },
                ),
            ));
        }
        self.exec_state = ExecutionState::Recovery {
            reason: reason.to_string(),
            since:  Instant::now(),
        };
    }

    fn trip_circuit_breaker(&mut self, reason: &str) {
        let (att, rej, err, slip) = self.circuit_breaker.counts();
        error!(reason, "[EXEC] Circuit breaker tripped → Halted");
        if let Some(store) = &self.event_store {
            store.append(crate::events::circuit_breaker_event(reason, att, rej, err, slip));
            store.append(crate::events::mode_change_event(
                &self.system_mode.to_string(), "Halted", reason,
            ));
        }
        self.system_mode = SystemMode::Halted;
    }

    fn check_circuit_breaker(&mut self) -> Option<String> {
        self.circuit_breaker.check_thresholds()
    }
}

// ── Public Executor ───────────────────────────────────────────────────────────

/// The single entry point for all exchange order actions.
/// All paths go through here; nothing bypasses it.
pub struct Executor {
    inner: Arc<Mutex<ExecutorInner>>,
}

impl Executor {
    pub fn new(
        symbol: String,
        cb_config: CircuitBreakerConfig,
        wd_config: WatchdogConfig,
    ) -> Self {
        Self::with_store(symbol, cb_config, wd_config, None)
    }

    pub fn with_store(
        symbol: String,
        cb_config: CircuitBreakerConfig,
        wd_config: WatchdogConfig,
        event_store: Option<Arc<dyn EventStore>>,
    ) -> Self {
        let inner = Arc::new(Mutex::new(ExecutorInner {
            exec_state:      ExecutionState::Idle,
            system_mode:     SystemMode::Booting,
            circuit_breaker: CircuitBreaker::new(cb_config),
            submitted_coids: std::collections::HashSet::new(),
            watchdog_cfg:    wd_config,
            symbol,
            event_store,
        }));
        Self { inner }
    }

    // ── State observers ───────────────────────────────────────────────────────

    pub async fn execution_state(&self) -> ExecutionState {
        self.inner.lock().await.exec_state.clone()
    }

    pub async fn system_mode(&self) -> SystemMode {
        self.inner.lock().await.system_mode
    }

    pub async fn can_trade(&self) -> bool {
        let g = self.inner.lock().await;
        g.system_mode.can_trade() && g.exec_state.can_accept_order()
    }

    // ── System mode transitions (called by reconciler/startup) ────────────────

    pub async fn set_mode_booting(&self) {
        let mut g = self.inner.lock().await;
        let prev = g.system_mode.to_string();
        g.system_mode = SystemMode::Booting;
        info!("[EXEC] SystemMode → Booting");
        if let Some(s) = &g.event_store {
            s.append(crate::events::mode_change_event(&prev, "Booting", "set_mode_booting"));
        }
    }

    pub async fn set_mode_reconciling(&self) {
        let mut g = self.inner.lock().await;
        let prev = g.system_mode.to_string();
        g.system_mode = SystemMode::Reconciling;
        info!("[EXEC] SystemMode → Reconciling");
        if let Some(s) = &g.event_store {
            s.append(crate::events::mode_change_event(&prev, "Reconciling", "set_mode_reconciling"));
        }
    }

    pub async fn set_mode_ready(&self) {
        let mut g = self.inner.lock().await;
        if g.system_mode == SystemMode::Halted {
            warn!("[EXEC] Cannot transition Halted → Ready without operator action");
            return;
        }
        let prev = g.system_mode.to_string();
        g.system_mode = SystemMode::Ready;
        info!(from = %prev, "[EXEC] SystemMode → Ready");
        if let Some(s) = &g.event_store {
            s.append(crate::events::mode_change_event(&prev, "Ready", "set_mode_ready"));
        }
    }

    pub async fn set_mode_degraded(&self) {
        let mut g = self.inner.lock().await;
        if g.system_mode == SystemMode::Halted { return; }
        let prev = g.system_mode.to_string();
        g.system_mode = SystemMode::Degraded;
        warn!("[EXEC] SystemMode → Degraded");
        if let Some(s) = &g.event_store {
            s.append(crate::events::mode_change_event(&prev, "Degraded", "set_mode_degraded"));
        }
    }

    pub async fn set_mode_halted(&self, reason: &str) {
        let mut g = self.inner.lock().await;
        let prev = g.system_mode.to_string();
        error!(reason, "[EXEC] SystemMode → Halted");
        if let Some(s) = &g.event_store {
            s.append(crate::events::mode_change_event(&prev, "Halted", reason));
        }
        g.system_mode = SystemMode::Halted;
    }

    /// Operator-only: clear halt and return to Reconciling.
    /// The reconciler must then run a clean cycle before Ready.
    pub async fn operator_clear_halt(&self) {
        let mut g = self.inner.lock().await;
        if g.system_mode != SystemMode::Halted {
            warn!("[EXEC] operator_clear_halt: not currently Halted");
            return;
        }
        g.system_mode = SystemMode::Reconciling;
        info!("[EXEC] Halt cleared by operator → Reconciling");
        if let Some(s) = &g.event_store {
            s.append(crate::events::StoredEvent::new(
                None, None, None,
                crate::events::TradingEvent::OperatorAction(
                    crate::events::OperatorActionPayload {
                        action: "clear_halt".to_string(),
                        reason: "operator request".to_string(),
                    },
                ),
            ));
            s.append(crate::events::mode_change_event("Halted", "Reconciling", "operator_clear_halt"));
        }
    }

    // ── Core: submit a market order ───────────────────────────────────────────
    //
    // This is the ONLY function that sends orders to the exchange.
    // It enforces all in-flight protections before touching the network.
    // The Mutex is NOT held across the async exchange call.

    pub async fn submit_market_order(
        &self,
        symbol: &str,
        side: &str,          // "BUY" or "SELL"
        qty_str: &str,
        client_order_id: &str,
        client: &BinanceClient,
        truth: &Arc<Mutex<TruthState>>,
        risk: &Arc<Mutex<RiskEngine>>,
        retry: &RetryPolicy,
    ) -> Result<OrderResponse> {

        // ── Step 1: Pre-flight checks (lock held briefly) ─────────────────────
        {
            let mut g = self.inner.lock().await;

            // System mode check
            if !g.system_mode.can_trade() {
                bail!("[EXEC] Cannot submit: SystemMode={}", g.system_mode);
            }

            // Execution state check
            if !g.exec_state.can_accept_order() {
                bail!(
                    "[EXEC] Cannot submit: ExecutionState={}. Wait for Idle.",
                    g.exec_state
                );
            }

            // Duplicate coid detection
            if g.submitted_coids.contains(client_order_id) {
                bail!(
                    "[EXEC] Duplicate client_order_id blocked: {}",
                    client_order_id
                );
            }

            // Circuit breaker pre-check (before recording attempt)
            if let Some(reason) = g.check_circuit_breaker() {
                g.trip_circuit_breaker(&reason);
                bail!("[EXEC] Circuit breaker: {}", reason);
            }

            // Record attempt and transition to Submitting
            g.circuit_breaker.record_attempt();
            g.submitted_coids.insert(client_order_id.to_string());
            g.transition(ExecutionState::Submitting {
                client_order_id: client_order_id.to_string(),
            });
        }
        // ── Mutex released ────────────────────────────────────────────────────

        // ── Step 2: Exchange call (no lock held) ──────────────────────────────
        let qty = qty_str.parse::<f64>().unwrap_or(0.0);
        let notional = {
            let t = truth.lock().await;
            let px = t.position.mark_price.max(0.0);
            if px > 0.0 { qty * px } else { 0.0 }
        };
        info!(
            "DISPATCH_ATTEMPT {{ side: {}, qty: {:.8}, notional: {:.8}, symbol: {}, execution_mode: LIVE_SPOT }}",
            side, qty, notional, symbol
        );
        info!(
            coid  = client_order_id,
            side  = side,
            qty   = qty_str,
            "[EXEC] Submitting market order"
        );

        let result = submit_market_with_retry(client, symbol, side, qty_str, client_order_id, retry).await;

        // ── Step 3: Post-submission state update (lock held briefly) ──────────
        match &result {
            Ok(resp) => {
                info!(
                    "DISPATCH_RESULT {{ ORDER_SENT_TO_BINANCE: true, exchange_response: \"status={} order_id={} executed_qty={} cumulative_quote_qty={}\", reject_reason: \"\" }}",
                    resp.status,
                    resp.order_id,
                    resp.executed_qty,
                    resp.cumulative_quote_qty
                );
                let status     = OrderStatus::from_str(&resp.status);
                let exchange_id = resp.order_id;
                let filled_qty: f64 = resp.executed_qty.parse().unwrap_or(0.0);
                let orig_qty:   f64 = resp.orig_qty.parse().unwrap_or(0.0);
                let avg_price:  f64 = if filled_qty > 0.0 {
                    resp.cumulative_quote_qty.parse::<f64>().unwrap_or(0.0) / filled_qty
                } else { 0.0 };

                info!(
                    exchange_id,
                    coid   = %resp.client_order_id,
                    status = %resp.status,
                    filled_qty,
                    avg_price,
                    "[EXEC] Order accepted"
                );

                // Update TruthState
                {
                    let mut t = truth.lock().await;
                    t.record_order_submitted(OrderRecord {
                        client_order_id:   resp.client_order_id.clone(),
                        exchange_order_id: exchange_id,
                        symbol:            resp.symbol.clone(),
                        side:              resp.side.clone(),
                        order_type:        resp.order_type.clone(),
                        orig_qty,
                        filled_qty,
                        remaining_qty:     (orig_qty - filled_qty).max(0.0),
                        avg_fill_price:    avg_price,
                        status,
                        last_seen:         Instant::now(),
                    });
                    if status != OrderStatus::Filled {
                        t.open_order_count += 1;
                    }
                }

                // Notify risk engine of fill (cooldown tracking)
                {
                    let t = truth.lock().await;
                    let mut r = risk.lock().await;
                    r.notify_fill(&t.position);
                }

                // Transition ExecState
                {
                    let mut g = self.inner.lock().await;
                    if status == OrderStatus::Filled {
                        // Market order filled synchronously — back to Idle
                        g.transition(ExecutionState::Idle);
                    } else {
                        g.transition(ExecutionState::WaitingAck {
                            client_order_id: resp.client_order_id.clone(),
                            since:           Instant::now(),
                        });
                    }
                }
            }

            Err(e) => {
                let msg = e.to_string();
                info!(
                    "DISPATCH_RESULT {{ ORDER_SENT_TO_BINANCE: false, exchange_response: \"\", reject_reason: \"{}\" }}",
                    msg
                );
                warn!(error = %msg, coid = client_order_id, "[EXEC] Submission failed");

                let mut g = self.inner.lock().await;

                // Classify and record
                if is_exchange_reject(&msg) {
                    g.circuit_breaker.record_reject();
                    warn!("[EXEC] Exchange reject recorded");
                } else {
                    g.circuit_breaker.record_error();
                }

                // Check circuit breaker after recording
                if let Some(cb_reason) = g.check_circuit_breaker() {
                    g.trip_circuit_breaker(&cb_reason);
                    g.transition_to_recovery("circuit breaker tripped");
                } else {
                    // Return to Idle so the signal loop can retry on next tick
                    g.transition(ExecutionState::Idle);
                }
            }
        }

        result
    }

    // ── Reconciler authority ──────────────────────────────────────────────────
    //
    // Called by the reconciliation loop after every successful exchange fetch.
    // The reconciler tells us what the exchange actually shows.
    // We use this to reset bad or stale execution state.

    pub async fn on_reconcile(
        &self,
        exchange_open_orders: &[OpenOrder],
        had_anomaly: bool,
    ) {
        let mut g = self.inner.lock().await;

        // Update system mode based on reconcile health
        match g.system_mode {
            SystemMode::Reconciling | SystemMode::Booting => {
                if had_anomaly {
                    g.system_mode = SystemMode::Degraded;
                    warn!("[EXEC] SystemMode → Degraded (post-reconcile anomalies)");
                } else {
                    g.system_mode = SystemMode::Ready;
                    info!("[EXEC] SystemMode → Ready");
                }
            }
            SystemMode::Ready => {
                if had_anomaly {
                    g.system_mode = SystemMode::Degraded;
                    warn!("[EXEC] SystemMode → Degraded (mid-session anomaly)");
                }
            }
            SystemMode::Degraded => {
                if !had_anomaly {
                    g.system_mode = SystemMode::Ready;
                    info!("[EXEC] SystemMode → Ready (anomaly resolved)");
                }
            }
            SystemMode::Halted => {
                // Halt is only cleared by operator action, not by reconcile.
            }
        }

        // Check execution state against exchange truth
        let exchange_coids: std::collections::HashSet<&str> = exchange_open_orders
            .iter()
            .map(|o| o.client_order_id.as_str())
            .collect();

        match &g.exec_state.clone() {
            ExecutionState::WaitingAck { client_order_id, .. } => {
                if !exchange_coids.contains(client_order_id.as_str()) {
                    // Order not visible on exchange → filled or rejected already
                    // Reconcile will have updated TruthState; just clear exec state
                    info!(
                        coid = %client_order_id,
                        "[EXEC] WaitingAck order gone from exchange → Idle"
                    );
                    g.exec_state = ExecutionState::Idle;
                }
            }
            ExecutionState::Open { exchange_order_id, client_order_id, .. } => {
                let oid = *exchange_order_id;
                let still_open = exchange_open_orders.iter().any(|o| o.order_id == oid);
                if !still_open {
                    info!(
                        oid = oid,
                        coid = %client_order_id,
                        "[EXEC] Open order gone from exchange → Idle"
                    );
                    g.exec_state = ExecutionState::Idle;
                }
            }
            ExecutionState::Canceling { exchange_order_id, .. } => {
                let oid = *exchange_order_id;
                let still_open = exchange_open_orders.iter().any(|o| o.order_id == oid);
                if !still_open {
                    info!(oid, "[EXEC] Canceling order gone from exchange → Idle");
                    g.exec_state = ExecutionState::Idle;
                }
            }
            ExecutionState::Recovery { .. } => {
                // Recovery is cleared after a clean reconcile with no anomaly
                if !had_anomaly {
                    info!("[EXEC] Recovery cleared by clean reconcile → Idle");
                    g.exec_state = ExecutionState::Idle;
                }
            }
            ExecutionState::Idle | ExecutionState::Submitting { .. } | ExecutionState::Replacing { .. } => {
                // Nothing to reconcile for these states
            }
        }

        let (attempts, rejects, errors, slips) = g.circuit_breaker.counts();
        info!(
            mode      = %g.system_mode,
            exec      = %g.exec_state,
            cb_att    = attempts,
            cb_rej    = rejects,
            cb_err    = errors,
            cb_slip   = slips,
            "[EXEC] Post-reconcile status"
        );
    }

    /// Force transition to Recovery. Called by watchdog or external error handler.
    pub async fn force_recovery(&self, reason: &str) {
        let mut g = self.inner.lock().await;
        g.transition_to_recovery(reason);
    }

    /// Force transition to Idle. Only call after confirmed reconcile shows no open orders.
    pub async fn force_idle(&self) {
        let mut g = self.inner.lock().await;
        let prev = g.exec_state.name();
        g.exec_state = ExecutionState::Idle;
        info!(from = prev, "[EXEC] Forced to Idle");
    }

    /// Record a slippage breach for circuit breaker tracking.
    pub async fn record_slippage_breach(&self) {
        let mut g = self.inner.lock().await;
        g.circuit_breaker.record_slippage();
        let (_, _, _, slips) = g.circuit_breaker.counts();
        warn!(slippage_count = slips, "[EXEC] Slippage breach recorded");

        if let Some(reason) = g.check_circuit_breaker() {
            g.trip_circuit_breaker(&reason);
            g.transition_to_recovery("circuit breaker: slippage");
        }
    }

    // ── Timeout watchdog ──────────────────────────────────────────────────────

    /// Spawn a background task that detects stuck execution states.
    /// On timeout: transitions to Recovery and triggers a reconciliation.
    pub fn spawn_watchdog(
        &self,
        client: Arc<BinanceClient>,
        truth: Arc<Mutex<TruthState>>,
    ) -> tokio::task::JoinHandle<()> {
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            loop {
                let check_interval = inner.lock().await.watchdog_cfg.check_interval;
                tokio::time::sleep(check_interval).await;

                let (timed_out, reason, stuck_state, age_secs) = {
                    let g = inner.lock().await;
                    let wd = &g.watchdog_cfg;
                    let (timed_out, age) = match &g.exec_state {
                        ExecutionState::WaitingAck { since, .. } =>
                            (since.elapsed() > wd.waiting_ack_timeout, since.elapsed()),
                        ExecutionState::Canceling { since, .. } =>
                            (since.elapsed() > wd.canceling_timeout, since.elapsed()),
                        ExecutionState::Replacing { since } =>
                            (since.elapsed() > wd.replacing_timeout, since.elapsed()),
                        _ => (false, Duration::ZERO),
                    };
                    let stuck_state = g.exec_state.name().to_string();
                    let reason = if timed_out {
                        format!("Timeout in state={} age={:.1}s", stuck_state, age.as_secs_f64())
                    } else {
                        String::new()
                    };
                    (timed_out, reason, stuck_state, age.as_secs_f64())
                };
                // Lock released before doing any async work

                if timed_out {
                    warn!(reason = %reason, "[EXEC] Watchdog: timeout detected");

                    // Emit WatchdogTimeout event before transitioning
                    {
                        let g = inner.lock().await;
                        if let Some(store) = &g.event_store {
                            store.append(crate::events::watchdog_event(&stuck_state, age_secs));
                        }
                    }

                    {
                        let mut g = inner.lock().await;
                        g.transition_to_recovery(&reason);
                    }
                    // Trigger reconciliation to restore truth
                    match crate::reconciler::run_reconciliation(&client, Arc::clone(&truth)).await {
                        Ok(result) => {
                            info!(
                                had_anomaly = result.has_anomaly(),
                                "[EXEC] Watchdog reconcile complete"
                            );
                        }
                        Err(e) => {
                            error!(error = %e, "[EXEC] Watchdog reconcile failed");
                        }
                    }
                }
            }
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// True when the error is a definitive exchange rejection (not a transient network issue).
/// These count against the circuit breaker reject bucket.
fn is_exchange_reject(msg: &str) -> bool {
    // Binance client errors: -1xxx (bad request format, bad symbol, etc.)
    for code in &["-1100", "-1102", "-1111", "-1121", "-2010", "-2015"] {
        if msg.contains(code) {
            return true;
        }
    }
    false
}

/// Generate a deterministic client_order_id for a logical order.
/// `seq` is a monotonic counter incremented per signal, NOT per retry.
/// Retrying the same logical order reuses the same coid.
pub fn make_client_order_id(seq: u64) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format: "rwt-{seq:08x}{ts:08x}" — fixed 22 chars, within Binance's 36-char limit
    format!("rwt-{:08x}{:08x}", seq, ts & 0xFFFF_FFFF)
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// All unit tests. No network calls. Exchange calls are not invoked.
// We test the state machine, circuit breaker, deduplication, and mode transitions.

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_executor() -> Executor {
        Executor::new(
            "BTCUSDT".into(),
            CircuitBreakerConfig::default(),
            WatchdogConfig::default(),
        )
    }

    // ── ExecutionState ────────────────────────────────────────────────────────

    #[test]
    fn test_idle_can_accept_order() {
        assert!(ExecutionState::Idle.can_accept_order());
    }

    #[test]
    fn test_non_idle_cannot_accept() {
        let states = [
            ExecutionState::Submitting { client_order_id: "x".into() },
            ExecutionState::WaitingAck { client_order_id: "x".into(), since: Instant::now() },
            ExecutionState::Open { exchange_order_id: 1, client_order_id: "x".into(), since: Instant::now() },
            ExecutionState::Canceling { exchange_order_id: 1, since: Instant::now() },
            ExecutionState::Replacing { since: Instant::now() },
            ExecutionState::Recovery { reason: "test".into(), since: Instant::now() },
        ];
        for state in states {
            assert!(!state.can_accept_order(), "Expected false for {:?}", state);
        }
    }

    // ── SystemMode ────────────────────────────────────────────────────────────

    #[test]
    fn test_ready_and_degraded_can_trade() {
        assert!(SystemMode::Ready.can_trade());
        assert!(SystemMode::Degraded.can_trade());
    }

    #[test]
    fn test_other_modes_cannot_trade() {
        assert!(!SystemMode::Booting.can_trade());
        assert!(!SystemMode::Reconciling.can_trade());
        assert!(!SystemMode::Halted.can_trade());
    }

    // ── Executor: mode transitions ────────────────────────────────────────────

    #[tokio::test]
    async fn test_starts_in_booting() {
        let ex = make_executor();
        assert_eq!(ex.system_mode().await, SystemMode::Booting);
    }

    #[tokio::test]
    async fn test_set_mode_ready() {
        let ex = make_executor();
        ex.set_mode_reconciling().await;
        ex.set_mode_ready().await;
        assert_eq!(ex.system_mode().await, SystemMode::Ready);
    }

    #[tokio::test]
    async fn test_halted_cannot_become_ready_directly() {
        let ex = make_executor();
        ex.set_mode_halted("test halt").await;
        ex.set_mode_ready().await; // should be ignored
        assert_eq!(ex.system_mode().await, SystemMode::Halted);
    }

    #[tokio::test]
    async fn test_operator_clear_halt_goes_to_reconciling() {
        let ex = make_executor();
        ex.set_mode_halted("test").await;
        ex.operator_clear_halt().await;
        assert_eq!(ex.system_mode().await, SystemMode::Reconciling);
    }

    #[tokio::test]
    async fn test_cannot_trade_when_booting() {
        let ex = make_executor();
        assert!(!ex.can_trade().await);
    }

    #[tokio::test]
    async fn test_can_trade_when_ready() {
        let ex = make_executor();
        ex.set_mode_reconciling().await;
        ex.set_mode_ready().await;
        assert!(ex.can_trade().await);
    }

    #[tokio::test]
    async fn test_cannot_trade_when_not_idle() {
        let ex = make_executor();
        ex.set_mode_reconciling().await;
        ex.set_mode_ready().await;
        // Force non-idle state
        ex.force_recovery("test").await;
        assert!(!ex.can_trade().await);
    }

    // ── Circuit breaker ───────────────────────────────────────────────────────

    #[test]
    fn test_circuit_breaker_no_trip_on_first_event() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        cb.record_attempt();
        assert!(cb.check_thresholds().is_none());
    }

    #[test]
    fn test_circuit_breaker_trips_on_reject_limit() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            max_rejects_per_min: 2,
            ..Default::default()
        });
        cb.record_reject();
        cb.record_reject();
        assert!(cb.check_thresholds().is_some(), "Should trip on 2 rejects");
    }

    #[test]
    fn test_circuit_breaker_trips_on_error_limit() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            max_errors_per_min: 2,
            ..Default::default()
        });
        cb.record_error();
        cb.record_error();
        assert!(cb.check_thresholds().is_some());
    }

    #[test]
    fn test_circuit_breaker_trips_on_attempt_limit() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            max_attempts_per_min: 3,
            ..Default::default()
        });
        cb.record_attempt();
        cb.record_attempt();
        cb.record_attempt();
        assert!(cb.check_thresholds().is_some());
    }

    #[test]
    fn test_circuit_breaker_trips_on_slippage_limit() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            max_slippage_per_min: 2,
            ..Default::default()
        });
        cb.record_slippage();
        cb.record_slippage();
        assert!(cb.check_thresholds().is_some());
    }

    #[test]
    fn test_circuit_breaker_prunes_old_events() {
        // Can't actually simulate time passing in a unit test without sleeping,
        // so we verify the prune logic is called and doesn't panic.
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            max_rejects_per_min: 1,
            ..Default::default()
        });
        cb.record_reject();
        cb.prune(); // should not panic
        // Still tripped since event is fresh
        assert!(cb.check_thresholds().is_some());
    }

    // ── Duplicate coid detection ──────────────────────────────────────────────

    #[test]
    fn test_duplicate_coid_detection() {
        let mut inner = ExecutorInner {
            exec_state:      ExecutionState::Idle,
            system_mode:     SystemMode::Ready,
            circuit_breaker: CircuitBreaker::new(CircuitBreakerConfig::default()),
            submitted_coids: std::collections::HashSet::new(),
            watchdog_cfg:    WatchdogConfig::default(),
            symbol:          "BTCUSDT".into(),
            event_store:     None,
        };
        inner.submitted_coids.insert("rwt-00000001deadbeef".into());

        // Simulate what submit_market_order checks:
        let is_dup = inner.submitted_coids.contains("rwt-00000001deadbeef");
        assert!(is_dup, "Should detect duplicate");

        let is_new = inner.submitted_coids.contains("rwt-00000002deadbeef");
        assert!(!is_new, "Should not detect new coid as duplicate");
    }

    // ── make_client_order_id ──────────────────────────────────────────────────

    #[test]
    fn test_coid_format_and_length() {
        let coid = make_client_order_id(1);
        assert!(coid.starts_with("rwt-"), "coid must start with rwt-");
        assert!(coid.len() <= 36, "coid must be ≤ 36 chars (Binance limit), got {}", coid.len());
        // Verify it's fixed length: "rwt-" + 8 + 8 = 20 chars
        assert_eq!(coid.len(), 20, "Expected 20 chars, got {}", coid.len());
    }

    #[test]
    fn test_coid_different_seq_different_id() {
        // Different seq → different coid (same second, so ts portion same)
        let c1 = make_client_order_id(1);
        let c2 = make_client_order_id(2);
        assert_ne!(c1, c2);
    }

    // ── Reconciler authority on execution state ───────────────────────────────

    #[tokio::test]
    async fn test_reconcile_clears_waiting_ack_if_order_gone() {
        let ex = make_executor();
        // Manually put executor in WaitingAck
        {
            let mut g = ex.inner.lock().await;
            g.exec_state = ExecutionState::WaitingAck {
                client_order_id: "rwt-abc".into(),
                since: Instant::now(),
            };
            g.system_mode = SystemMode::Reconciling;
        }

        // Reconcile with empty open orders (order gone from exchange)
        ex.on_reconcile(&[], false).await;

        // Should have transitioned to Idle
        let state = ex.execution_state().await;
        assert_eq!(state, ExecutionState::Idle, "Expected Idle, got {:?}", state);
    }

    #[tokio::test]
    async fn test_reconcile_does_not_clear_if_order_still_open() {
        let ex = make_executor();
        {
            let mut g = ex.inner.lock().await;
            g.exec_state = ExecutionState::WaitingAck {
                client_order_id: "rwt-abc".into(),
                since: Instant::now(),
            };
            g.system_mode = SystemMode::Reconciling;
        }

        // Build a fake OpenOrder that matches our coid
        let open_order = crate::client::OpenOrder {
            symbol:            "BTCUSDT".into(),
            order_id:          12345,
            client_order_id:   "rwt-abc".into(),
            price:             "50000".into(),
            orig_qty:          "0.001".into(),
            executed_qty:      "0.0".into(),
            status:            "NEW".into(),
            time_in_force:     "GTC".into(),
            order_type:        "LIMIT".into(),
            side:              "BUY".into(),
        };

        ex.on_reconcile(&[open_order], false).await;

        // Should still be WaitingAck (order is on exchange)
        let state = ex.execution_state().await;
        assert!(
            matches!(state, ExecutionState::WaitingAck { .. }),
            "Expected WaitingAck, got {:?}", state
        );
    }

    #[tokio::test]
    async fn test_reconcile_clears_recovery_on_clean_reconcile() {
        let ex = make_executor();
        ex.force_recovery("test").await;
        // Verify in Recovery
        assert!(matches!(ex.execution_state().await, ExecutionState::Recovery { .. }));

        // Clean reconcile with no anomaly
        ex.on_reconcile(&[], false).await;

        // Recovery should be cleared
        assert_eq!(ex.execution_state().await, ExecutionState::Idle);
    }

    #[tokio::test]
    async fn test_recovery_not_cleared_if_anomaly() {
        let ex = make_executor();
        ex.force_recovery("test").await;

        // Reconcile with anomaly still present
        ex.on_reconcile(&[], true).await; // had_anomaly = true

        // Recovery should persist
        assert!(matches!(ex.execution_state().await, ExecutionState::Recovery { .. }));
    }

    // ── Mode guards on submission (integration-style) ─────────────────────────

    #[tokio::test]
    async fn test_submit_blocked_when_halted() {
        // We test the guard path directly by checking can_trade()
        let ex = make_executor();
        ex.set_mode_halted("test").await;
        assert!(!ex.can_trade().await);
    }

    #[tokio::test]
    async fn test_submit_blocked_when_not_idle() {
        let ex = make_executor();
        ex.set_mode_reconciling().await;
        ex.set_mode_ready().await;
        ex.force_recovery("test").await;
        // Mode is Ready but state is Recovery
        assert!(!ex.can_trade().await);
    }

    // ── Reconcile mode transitions ────────────────────────────────────────────

    #[tokio::test]
    async fn test_reconcile_booting_clean_goes_ready() {
        let ex = make_executor();
        // Still in Booting
        ex.on_reconcile(&[], false).await;
        assert_eq!(ex.system_mode().await, SystemMode::Ready);
    }

    #[tokio::test]
    async fn test_reconcile_booting_anomaly_goes_degraded() {
        let ex = make_executor();
        ex.on_reconcile(&[], true).await;
        assert_eq!(ex.system_mode().await, SystemMode::Degraded);
    }

    #[tokio::test]
    async fn test_reconcile_degraded_clean_goes_ready() {
        let ex = make_executor();
        ex.set_mode_degraded().await;
        ex.on_reconcile(&[], false).await;
        assert_eq!(ex.system_mode().await, SystemMode::Ready);
    }

    #[tokio::test]
    async fn test_reconcile_does_not_clear_halt() {
        let ex = make_executor();
        ex.set_mode_halted("test").await;
        ex.on_reconcile(&[], false).await;
        // Halt must remain — only operator can clear it
        assert_eq!(ex.system_mode().await, SystemMode::Halted);
    }

    // ── is_exchange_reject ────────────────────────────────────────────────────

    #[test]
    fn test_reject_codes_detected() {
        assert!(is_exchange_reject("Binance error -1121: Invalid symbol"));
        assert!(is_exchange_reject("Binance error -2010: insufficient balance"));
        assert!(is_exchange_reject("-1111 bad qty"));
    }

    #[test]
    fn test_network_errors_not_reject() {
        assert!(!is_exchange_reject("connection timeout"));
        assert!(!is_exchange_reject("HTTP 503 service unavailable"));
        assert!(!is_exchange_reject("-1021 timestamp"));
    }
}

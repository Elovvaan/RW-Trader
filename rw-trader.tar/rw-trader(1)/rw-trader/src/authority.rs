// authority.rs
//
// Execution Authority Layer.
//
// AuthorityMode governs whether suggestions can become real orders:
//
//   OFF    — signals evaluated, logged; no execution ever initiated from the
//             suggestion/authority path. The normal signal loop is unaffected.
//
//   ASSIST — when conditions are met the authority layer produces a Proposal
//             and logs it. Execution only proceeds after an explicit operator
//             approval via the web UI or API. Proposals expire after a TTL.
//
//   AUTO   — when ALL system/risk/execution gates pass the authority layer
//             allows execution to proceed. Never bypasses existing checks.
//
// Invariants:
//   • Mode changes require an explicit operator call (set_mode_*).
//   • Every mode change is recorded in the EventStore.
//   • AUTO cannot execute while: Halted, Degraded (opt-in only), dirty state,
//     kill switch active, or executor not Idle.
//   • ASSIST proposals are keyed by a UUID, expire after proposal_ttl_secs,
//     and are consumed exactly once on approval.
//   • The Mutex is never held across any await point.
//   • No exchange calls here. Execution still goes through Executor.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use crate::events::{OperatorActionPayload, StoredEvent, TradingEvent};
use crate::executor::SystemMode;
use crate::store::EventStore;

// ── AuthorityMode ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityMode {
    /// No suggestion-driven execution. Default.
    Off,
    /// Proposals require explicit operator approval before executing.
    Assist,
    /// Execute automatically when all gates pass.
    Auto,
}

impl std::fmt::Display for AuthorityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthorityMode::Off    => write!(f, "OFF"),
            AuthorityMode::Assist => write!(f, "ASSIST"),
            AuthorityMode::Auto   => write!(f, "AUTO"),
        }
    }
}

impl AuthorityMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "OFF"    => Some(AuthorityMode::Off),
            "ASSIST" => Some(AuthorityMode::Assist),
            "AUTO"   => Some(AuthorityMode::Auto),
            _        => None,
        }
    }

    /// CSS class for the mode banner in the web UI.
    pub fn banner_class(self) -> &'static str {
        match self {
            AuthorityMode::Off    => "dim",
            AuthorityMode::Assist => "warn",
            AuthorityMode::Auto   => "ok",
        }
    }

    /// Short human-readable description for the UI banner.
    pub fn description(self) -> &'static str {
        match self {
            AuthorityMode::Off =>
                "OFF — suggestions are advisory only. No execution will occur from this layer.",
            AuthorityMode::Assist =>
                "ASSIST — proposals are generated and await operator approval before executing.",
            AuthorityMode::Auto =>
                "AUTO — executes automatically when all system, risk, and execution gates pass.",
        }
    }
}

// ── AutoBlockReason ───────────────────────────────────────────────────────────

/// Why AUTO mode cannot execute right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoBlockReason {
    ModeIsOff,
    ModeIsAssist,
    SystemNotReady(String),     // system mode name
    KillSwitchActive,
    StateDirty,
    ReconInProgress,
    ExecutorNotIdle,
    NeverReconciled,
    RiskRejected(String),       // rejection reason string
}

impl std::fmt::Display for AutoBlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoBlockReason::ModeIsOff          => write!(f, "authority mode is OFF"),
            AutoBlockReason::ModeIsAssist       => write!(f, "authority mode is ASSIST (requires approval)"),
            AutoBlockReason::SystemNotReady(m)  => write!(f, "system mode is {} (not Ready/Degraded)", m),
            AutoBlockReason::KillSwitchActive   => write!(f, "kill switch is active"),
            AutoBlockReason::StateDirty         => write!(f, "state is dirty"),
            AutoBlockReason::ReconInProgress    => write!(f, "reconciliation is in progress"),
            AutoBlockReason::ExecutorNotIdle    => write!(f, "executor is not Idle"),
            AutoBlockReason::NeverReconciled    => write!(f, "system has never reconciled"),
            AutoBlockReason::RiskRejected(r)    => write!(f, "risk rejected: {}", r),
        }
    }
}

// ── Proposal (ASSIST mode) ────────────────────────────────────────────────────

/// An executable proposal waiting for operator approval.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub id:         String,
    pub created_at: Instant,
    pub expires_at: Instant,
    pub symbol:     String,
    pub side:       String,         // "BUY" or "SELL"
    pub qty:        f64,
    pub reason:     String,         // why this was proposed
    pub confidence: f64,
}

impl Proposal {
    pub fn new(
        symbol: &str,
        side: &str,
        qty: f64,
        reason: &str,
        confidence: f64,
        ttl: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            id:         Uuid::new_v4().to_string(),
            created_at: now,
            expires_at: now + ttl,
            symbol:     symbol.to_string(),
            side:       side.to_string(),
            qty,
            reason:     reason.to_string(),
            confidence,
        }
    }

    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    pub fn age_secs(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }

    pub fn ttl_remaining_secs(&self) -> f64 {
        let now = Instant::now();
        if self.expires_at > now {
            (self.expires_at - now).as_secs_f64()
        } else {
            0.0
        }
    }
}

// ── AuthorityResult ───────────────────────────────────────────────────────────

/// Outcome of checking whether authority permits execution.
#[derive(Debug, Clone)]
pub enum AuthorityResult {
    /// AUTO mode + all gates clear → proceed with execution.
    Proceed,
    /// ASSIST mode + all gates clear → proposal created, waiting for approval.
    ProposalCreated(Proposal),
    /// A gate blocked execution. Includes the reason.
    Blocked(AutoBlockReason),
}

// ── AuthorityInner ────────────────────────────────────────────────────────────

struct AuthorityInner {
    mode:             AuthorityMode,
    /// Pending proposals (ASSIST mode). Keyed by proposal.id.
    proposals:        HashMap<String, Proposal>,
    /// How long a proposal lives before expiring.
    proposal_ttl:     Duration,
}

impl AuthorityInner {
    fn new(proposal_ttl: Duration) -> Self {
        Self {
            mode:         AuthorityMode::Off,
            proposals:    HashMap::new(),
            proposal_ttl,
        }
    }

    /// Prune expired proposals.
    fn prune_expired(&mut self) {
        self.proposals.retain(|_, p| !p.is_expired());
    }
}

// ── Public AuthorityLayer ─────────────────────────────────────────────────────

pub struct AuthorityLayer {
    inner: Arc<Mutex<AuthorityInner>>,
}

impl AuthorityLayer {
    /// Create with default proposal TTL of 60 seconds.
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(60))
    }

    pub fn with_ttl(proposal_ttl: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AuthorityInner::new(proposal_ttl))),
        }
    }

    // ── Mode queries ──────────────────────────────────────────────────────────

    pub async fn mode(&self) -> AuthorityMode {
        self.inner.lock().await.mode
    }

    pub async fn pending_proposals(&self) -> Vec<Proposal> {
        let mut g = self.inner.lock().await;
        g.prune_expired();
        g.proposals.values().cloned().collect()
    }

    // ── Mode transitions ──────────────────────────────────────────────────────
    // Each takes an EventStore reference so the change is immediately persisted.

    pub async fn set_mode_off(&self, store: &dyn EventStore) {
        let prev = {
            let mut g = self.inner.lock().await;
            let prev = g.mode;
            g.mode = AuthorityMode::Off;
            // Clear any pending proposals — they can't be approved in OFF mode
            let count = g.proposals.len();
            g.proposals.clear();
            if count > 0 {
                warn!("[AUTHORITY] Cleared {} pending proposals on mode→OFF", count);
            }
            prev
        };
        log_authority_mode_change(store, prev, AuthorityMode::Off, "operator set OFF");
        info!("[AUTHORITY] Mode {} → OFF", prev);
    }

    pub async fn set_mode_assist(&self, store: &dyn EventStore) {
        let prev = {
            let mut g = self.inner.lock().await;
            let prev = g.mode;
            g.mode = AuthorityMode::Assist;
            prev
        };
        log_authority_mode_change(store, prev, AuthorityMode::Assist, "operator set ASSIST");
        info!("[AUTHORITY] Mode {} → ASSIST", prev);
    }

    pub async fn set_mode_auto(&self, store: &dyn EventStore) {
        let prev = {
            let mut g = self.inner.lock().await;
            let prev = g.mode;
            g.mode = AuthorityMode::Auto;
            prev
        };
        log_authority_mode_change(store, prev, AuthorityMode::Auto, "operator set AUTO");
        info!("[AUTHORITY] Mode {} → AUTO", prev);
    }

    // ── Core gate check ───────────────────────────────────────────────────────

    /// Check whether the authority layer allows execution of a trade action.
    ///
    /// Called by the signal loop after risk approval.
    /// Does NOT execute. Does NOT touch the exchange.
    ///
    /// `sys_mode`:      current SystemMode from Executor
    /// `exec_is_idle`:  true if ExecutionState == Idle
    /// `kill_active`:   true if RiskEngine kill switch is set
    /// `can_place`:     TruthState::can_place_order()
    ///
    /// Returns:
    ///   Proceed          → AUTO gate cleared; caller should execute.
    ///   ProposalCreated  → ASSIST gate cleared; store proposal, wait for approval.
    ///   Blocked(reason)  → gate failed; caller should log and skip.
    pub async fn check(
        &self,
        symbol:       &str,
        side:         &str,
        qty:          f64,
        reason:       &str,
        confidence:   f64,
        sys_mode:     SystemMode,
        exec_is_idle: bool,
        kill_active:  bool,
        can_place:    bool,
        store:        &dyn EventStore,
    ) -> AuthorityResult {
        // ── Read mode under lock, then release ────────────────────────────────
        let (mode, ttl) = {
            let g = self.inner.lock().await;
            (g.mode, g.proposal_ttl)
        };

        // ── OFF: block everything ─────────────────────────────────────────────
        if mode == AuthorityMode::Off {
            return AuthorityResult::Blocked(AutoBlockReason::ModeIsOff);
        }

        // ── System-level gates (same for both ASSIST and AUTO) ────────────────
        if !sys_mode.can_trade() {
            return AuthorityResult::Blocked(AutoBlockReason::SystemNotReady(sys_mode.to_string()));
        }
        if kill_active {
            return AuthorityResult::Blocked(AutoBlockReason::KillSwitchActive);
        }
        if !can_place {
            // can_place = !dirty && !recon_in_progress && reconciled_at.is_some()
            // We distinguish the three sub-cases for better logging but the
            // boolean doesn't carry which one; report the composite.
            return AuthorityResult::Blocked(AutoBlockReason::StateDirty);
        }
        if !exec_is_idle {
            return AuthorityResult::Blocked(AutoBlockReason::ExecutorNotIdle);
        }

        // ── ASSIST: all gates passed → create proposal ────────────────────────
        if mode == AuthorityMode::Assist {
            let proposal = Proposal::new(symbol, side, qty, reason, confidence, ttl);
            let proposal_id = proposal.id.clone();

            // Store under lock (brief)
            {
                let mut g = self.inner.lock().await;
                g.prune_expired();
                g.proposals.insert(proposal_id.clone(), proposal.clone());
            }

            log_proposal_created(store, &proposal);
            info!(
                id = %proposal_id,
                symbol, side, qty,
                ttl_secs = ttl.as_secs(),
                "[AUTHORITY] ASSIST proposal created"
            );
            return AuthorityResult::ProposalCreated(proposal);
        }

        // ── AUTO: all gates passed → proceed ─────────────────────────────────
        log_auto_execution(store, symbol, side, qty, reason);
        info!(
            symbol, side, qty, reason,
            "[AUTHORITY] AUTO gate cleared → proceeding to execution"
        );
        AuthorityResult::Proceed
    }

    // ── Proposal approval (ASSIST) ────────────────────────────────────────────

    /// Attempt to approve a proposal by ID.
    ///
    /// Returns the Proposal if found and not expired (consuming it).
    /// Returns None if the proposal doesn't exist or has expired.
    /// Logs the approval event to the store.
    pub async fn approve_proposal(
        &self,
        proposal_id: &str,
        store: &dyn EventStore,
    ) -> Option<Proposal> {
        let mut g = self.inner.lock().await;
        g.prune_expired();

        let proposal = g.proposals.remove(proposal_id)?;
        if proposal.is_expired() {
            warn!("[AUTHORITY] Proposal {} expired, cannot approve", proposal_id);
            return None;
        }

        // Log the approval
        drop(g); // release lock before store write
        log_proposal_approved(store, &proposal);
        info!(
            id = %proposal.id,
            symbol = %proposal.symbol,
            side   = %proposal.side,
            qty    = proposal.qty,
            age_secs = proposal.age_secs(),
            "[AUTHORITY] ASSIST proposal approved by operator"
        );

        Some(proposal)
    }

    /// Reject/discard a pending proposal (operator chose not to approve).
    pub async fn reject_proposal(
        &self,
        proposal_id: &str,
        store: &dyn EventStore,
    ) -> bool {
        let proposal = {
            let mut g = self.inner.lock().await;
            g.proposals.remove(proposal_id)
        };

        match proposal {
            Some(p) => {
                log_proposal_rejected(store, &p);
                info!(id = %p.id, "[AUTHORITY] Proposal rejected by operator");
                true
            }
            None => false,
        }
    }

    /// Number of live (non-expired) proposals.
    pub async fn pending_count(&self) -> usize {
        let mut g = self.inner.lock().await;
        g.prune_expired();
        g.proposals.len()
    }
}

// ── Event logging helpers ─────────────────────────────────────────────────────

fn log_authority_mode_change(
    store: &dyn EventStore,
    from:   AuthorityMode,
    to:     AuthorityMode,
    reason: &str,
) {
    store.append(StoredEvent::new(
        None, None, None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("authority_mode_change:{} → {}", from, to),
            reason: reason.to_string(),
        }),
    ));
}

fn log_proposal_created(store: &dyn EventStore, p: &Proposal) {
    store.append(StoredEvent::new(
        Some(p.symbol.clone()), None, None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("assist_proposal_created:{}", p.id),
            reason: format!(
                "side={} qty={:.6} confidence={:.2} ttl={:.0}s reason={}",
                p.side, p.qty, p.confidence, p.ttl_remaining_secs(), p.reason
            ),
        }),
    ));
}

fn log_proposal_approved(store: &dyn EventStore, p: &Proposal) {
    store.append(StoredEvent::new(
        Some(p.symbol.clone()), None, None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("assist_proposal_approved:{}", p.id),
            reason: format!(
                "side={} qty={:.6} age={:.1}s",
                p.side, p.qty, p.age_secs()
            ),
        }),
    ));
}

fn log_proposal_rejected(store: &dyn EventStore, p: &Proposal) {
    store.append(StoredEvent::new(
        Some(p.symbol.clone()), None, None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("assist_proposal_rejected:{}", p.id),
            reason: format!("side={} qty={:.6}", p.side, p.qty),
        }),
    ));
}

fn log_auto_execution(store: &dyn EventStore, symbol: &str, side: &str, qty: f64, reason: &str) {
    store.append(StoredEvent::new(
        Some(symbol.to_string()), None, None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: "auto_execution_permitted".to_string(),
            reason: format!("side={} qty={:.6} reason={}", side, qty, reason),
        }),
    ));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::SystemMode;
    use crate::store::InMemoryEventStore;
    use std::sync::Arc;

    fn ready_args() -> (SystemMode, bool, bool, bool) {
        // sys_mode, exec_is_idle, kill_active, can_place
        (SystemMode::Ready, true, false, true)
    }

    fn make_store() -> Arc<InMemoryEventStore> {
        InMemoryEventStore::new()
    }

    async fn check_with_defaults(auth: &AuthorityLayer, store: &dyn EventStore) -> AuthorityResult {
        let (sys, idle, kill, can) = ready_args();
        auth.check("BTCUSDT", "BUY", 0.001, "test reason", 0.7,
            sys, idle, kill, can, store).await
    }

    // ── Mode transitions ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_default_mode_is_off() {
        let auth = AuthorityLayer::new();
        assert_eq!(auth.mode().await, AuthorityMode::Off);
    }

    #[tokio::test]
    async fn test_set_mode_assist() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        assert_eq!(auth.mode().await, AuthorityMode::Assist);
    }

    #[tokio::test]
    async fn test_set_mode_auto() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        assert_eq!(auth.mode().await, AuthorityMode::Auto);
    }

    #[tokio::test]
    async fn test_set_mode_off_from_auto() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        auth.set_mode_off(&*store).await;
        assert_eq!(auth.mode().await, AuthorityMode::Off);
    }

    // ── Mode change event logging ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_mode_change_logged_to_store() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        auth.set_mode_auto(&*store).await;
        auth.set_mode_off(&*store).await;

        let events = store.all_events();
        // Each mode change should produce an operator_action event
        let authority_events: Vec<_> = events.iter()
            .filter(|e| e.event_type == "operator_action")
            .collect();
        assert!(authority_events.len() >= 3, "Expected ≥3 mode change events, got {}", authority_events.len());
    }

    #[tokio::test]
    async fn test_mode_change_event_contains_from_to() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;

        let events = store.all_events();
        let ev = events.iter().find(|e| e.event_type == "operator_action").unwrap();
        if let TradingEvent::OperatorAction(p) = &ev.payload {
            assert!(p.action.contains("authority_mode_change"), "Action should describe mode change");
            assert!(p.action.contains("ASSIST"), "Should mention new mode");
        } else {
            panic!("Expected OperatorAction payload");
        }
    }

    // ── OFF mode ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_off_blocks_all_execution() {
        let auth  = AuthorityLayer::new(); // starts OFF
        let store = make_store();
        let result = check_with_defaults(&auth, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::ModeIsOff)));
    }

    #[tokio::test]
    async fn test_off_blocks_even_when_all_gates_pass() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        // Explicitly set OFF (it's the default, but be explicit)
        auth.set_mode_off(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::ModeIsOff)));
    }

    // ── ASSIST mode ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_assist_creates_proposal_when_gates_pass() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        assert!(matches!(result, AuthorityResult::ProposalCreated(_)),
            "ASSIST + all gates pass should create a proposal");
    }

    #[tokio::test]
    async fn test_assist_proposal_has_correct_fields() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let (sys, idle, kill, can) = ready_args();
        let result = auth.check("BTCUSDT", "BUY", 0.001, "good momentum", 0.8,
            sys, idle, kill, can, &*store).await;
        if let AuthorityResult::ProposalCreated(p) = result {
            assert_eq!(p.symbol, "BTCUSDT");
            assert_eq!(p.side, "BUY");
            assert!((p.qty - 0.001).abs() < 1e-9);
            assert!((p.confidence - 0.8).abs() < 1e-9);
            assert!(!p.id.is_empty());
            assert!(!p.is_expired());
        } else {
            panic!("Expected ProposalCreated, got {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_assist_requires_approval_not_auto_execute() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        // Must NOT be Proceed — ASSIST never auto-proceeds
        assert!(!matches!(result, AuthorityResult::Proceed),
            "ASSIST mode must never return Proceed");
    }

    #[tokio::test]
    async fn test_assist_proposal_stored_and_retrievable() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        check_with_defaults(&auth, &*store).await;
        let proposals = auth.pending_proposals().await;
        assert_eq!(proposals.len(), 1);
    }

    #[tokio::test]
    async fn test_assist_approval_consumes_proposal() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        let proposal_id = if let AuthorityResult::ProposalCreated(p) = result {
            p.id
        } else { panic!("Expected proposal") };

        let approved = auth.approve_proposal(&proposal_id, &*store).await;
        assert!(approved.is_some(), "Approval should return the proposal");
        // Proposal is consumed — second approval returns None
        let second = auth.approve_proposal(&proposal_id, &*store).await;
        assert!(second.is_none(), "Second approval of same id must fail");
    }

    #[tokio::test]
    async fn test_assist_reject_proposal() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        let pid = if let AuthorityResult::ProposalCreated(p) = result { p.id }
                  else { panic!("Expected proposal") };

        let rejected = auth.reject_proposal(&pid, &*store).await;
        assert!(rejected, "Reject should return true for existing proposal");
        // After rejection, count should be 0
        assert_eq!(auth.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_assist_approval_logged() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        let pid = if let AuthorityResult::ProposalCreated(p) = result { p.id }
                  else { panic!("Expected proposal") };
        auth.approve_proposal(&pid, &*store).await;

        let events = store.all_events();
        let approval_event = events.iter().find(|e| {
            if let TradingEvent::OperatorAction(p) = &e.payload {
                p.action.contains("approved")
            } else { false }
        });
        assert!(approval_event.is_some(), "Approval must be logged");
    }

    #[tokio::test]
    async fn test_assist_proposal_expires() {
        let auth  = AuthorityLayer::with_ttl(Duration::from_millis(1));
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        check_with_defaults(&auth, &*store).await;
        // Sleep past the TTL
        tokio::time::sleep(Duration::from_millis(10)).await;
        // Pruning happens on next pending_proposals() call
        let proposals = auth.pending_proposals().await;
        assert_eq!(proposals.len(), 0, "Expired proposals should be pruned");
    }

    #[tokio::test]
    async fn test_assist_expired_proposal_cannot_be_approved() {
        let auth  = AuthorityLayer::with_ttl(Duration::from_millis(1));
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        let pid = if let AuthorityResult::ProposalCreated(p) = result { p.id }
                  else { panic!("Expected proposal") };
        tokio::time::sleep(Duration::from_millis(10)).await;
        let approved = auth.approve_proposal(&pid, &*store).await;
        assert!(approved.is_none(), "Expired proposal must not be approved");
    }

    #[tokio::test]
    async fn test_off_clears_pending_proposals() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_assist(&*store).await;
        check_with_defaults(&auth, &*store).await;
        assert_eq!(auth.pending_count().await, 1);
        // Switch to OFF — should clear proposals
        auth.set_mode_off(&*store).await;
        assert_eq!(auth.pending_count().await, 0, "OFF must clear pending proposals");
    }

    // ── AUTO mode ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_auto_proceeds_when_all_gates_pass() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        let result = check_with_defaults(&auth, &*store).await;
        assert!(matches!(result, AuthorityResult::Proceed),
            "AUTO + all gates pass must return Proceed");
    }

    #[tokio::test]
    async fn test_auto_blocked_by_halted_mode() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        let result = auth.check("BTCUSDT", "BUY", 0.001, "r", 0.7,
            SystemMode::Halted, true, false, true, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::SystemNotReady(_))),
            "AUTO blocked by Halted system");
    }

    #[tokio::test]
    async fn test_auto_blocked_by_degraded_mode() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        // Degraded still can_trade() → should proceed (Degraded is allowed)
        let result = auth.check("BTCUSDT", "BUY", 0.001, "r", 0.7,
            SystemMode::Degraded, true, false, true, &*store).await;
        // Degraded CAN trade — only Halted/Booting/Reconciling cannot
        assert!(matches!(result, AuthorityResult::Proceed),
            "Degraded mode can still trade in AUTO");
    }

    #[tokio::test]
    async fn test_auto_blocked_by_kill_switch() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        let result = auth.check("BTCUSDT", "BUY", 0.001, "r", 0.7,
            SystemMode::Ready, true, true /* kill */, true, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::KillSwitchActive)));
    }

    #[tokio::test]
    async fn test_auto_blocked_by_dirty_state() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        let result = auth.check("BTCUSDT", "BUY", 0.001, "r", 0.7,
            SystemMode::Ready, true, false, false /* can_place = false */, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::StateDirty)));
    }

    #[tokio::test]
    async fn test_auto_blocked_by_executor_not_idle() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        let result = auth.check("BTCUSDT", "BUY", 0.001, "r", 0.7,
            SystemMode::Ready, false /* not idle */, false, true, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::ExecutorNotIdle)));
    }

    #[tokio::test]
    async fn test_auto_blocked_by_reconciling_mode() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        let result = auth.check("BTCUSDT", "BUY", 0.001, "r", 0.7,
            SystemMode::Reconciling, true, false, true, &*store).await;
        assert!(matches!(result, AuthorityResult::Blocked(AutoBlockReason::SystemNotReady(_))));
    }

    #[tokio::test]
    async fn test_auto_execution_logged() {
        let auth  = AuthorityLayer::new();
        let store = make_store();
        auth.set_mode_auto(&*store).await;
        check_with_defaults(&auth, &*store).await;
        let events = store.all_events();
        let exec_event = events.iter().find(|e| {
            if let TradingEvent::OperatorAction(p) = &e.payload {
                p.action.contains("auto_execution_permitted")
            } else { false }
        });
        assert!(exec_event.is_some(), "AUTO execution must be logged");
    }

    // ── AuthorityMode helpers ─────────────────────────────────────────────────

    #[test]
    fn test_mode_display() {
        assert_eq!(AuthorityMode::Off.to_string(),    "OFF");
        assert_eq!(AuthorityMode::Assist.to_string(), "ASSIST");
        assert_eq!(AuthorityMode::Auto.to_string(),   "AUTO");
    }

    #[test]
    fn test_mode_from_str() {
        assert_eq!(AuthorityMode::from_str("off"),    Some(AuthorityMode::Off));
        assert_eq!(AuthorityMode::from_str("ASSIST"), Some(AuthorityMode::Assist));
        assert_eq!(AuthorityMode::from_str("AUTO"),   Some(AuthorityMode::Auto));
        assert_eq!(AuthorityMode::from_str("UNKNOWN"),None);
    }

    #[test]
    fn test_auto_block_reason_display() {
        let r = AutoBlockReason::SystemNotReady("Halted".into());
        assert!(r.to_string().contains("Halted"));
        let r2 = AutoBlockReason::RiskRejected("SPREAD: too wide".into());
        assert!(r2.to_string().contains("risk rejected"));
    }

    #[test]
    fn test_proposal_expiry() {
        let p = Proposal::new("BTC", "BUY", 0.001, "test", 0.8, Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert!(p.is_expired());
    }

    #[test]
    fn test_proposal_not_expired_when_fresh() {
        let p = Proposal::new("BTC", "BUY", 0.001, "test", 0.8, Duration::from_secs(60));
        assert!(!p.is_expired());
        assert!(p.ttl_remaining_secs() > 50.0);
    }
}

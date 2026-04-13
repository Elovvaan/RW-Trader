// strategy.rs
//
// Strategy layer. Multiple named strategies evaluate the same live feed
// snapshot and return a decision. The StrategyEngine selects the best
// eligible (enabled, highest confidence) and returns it for the rest of
// the pipeline.
//
// ── Strategy modes ────────────────────────────────────────────────────────────
//
// PRIMARY (Phase 1): Phase1SpotDaySwing
//   - Spot Day+Swing Long-Only engine.
//   - Timeframe stack: 4H trend bias, 1H setup validation, 15M execution trigger.
//   - Entries only when regime = TREND_UP; all shorts blocked.
//   - Setup types: PullbackLong, BreakoutLong.
//   - Sizing from stop distance and fixed dollar risk — no vibe sizing.
//   - Full position management: SL, TP, break-even, trailing stop, time exit.
//
// LEGACY / EXPERIMENTAL: MomentumMicro, MeanReversionMicro, BreakoutMicro
//   - Original microstructure-momentum strategies.
//   - Retained for comparison and fallback experimentation.
//   - Can be toggled independently from Phase 1.
//
// Design principles:
//   - Strategy trait is pure: evaluate() takes plain data, returns StrategyDecision.
//   - No I/O, no locks, no async inside strategies themselves.
//   - StrategyEngine is the only thing that holds state (enable flags, entry hints).
//   - All strategies share the same risk/authority/execution pipeline.
//   - Priority order (for tie-breaking) is defined by the order in ALL_STRATEGIES.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use tracing::{debug, info};

use crate::phase1::{Phase1Engine, Phase1Result, Phase1Status};
use crate::reconciler::TruthState;
use crate::signal::SignalMetrics;

// ── Strategy ID ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StrategyId {
    /// PRIMARY — Phase 1 Spot Day+Swing Long-Only engine.
    Phase1SpotDaySwing,
    /// LEGACY — Original microstructure momentum strategy.
    MomentumMicro,
    /// LEGACY — Original mean-reversion micro strategy.
    MeanReversionMicro,
    /// LEGACY — Original breakout micro strategy.
    BreakoutMicro,
}

impl StrategyId {
    pub fn as_str(&self) -> &'static str {
        match self {
            StrategyId::Phase1SpotDaySwing => "Phase1SpotDaySwing",
            StrategyId::MomentumMicro      => "MomentumMicro",
            StrategyId::MeanReversionMicro => "MeanReversionMicro",
            StrategyId::BreakoutMicro      => "BreakoutMicro",
        }
    }

    /// Deterministic priority for tie-breaking (lower = higher priority).
    /// Phase1SpotDaySwing has highest priority (0).
    pub fn priority(&self) -> u8 {
        match self {
            StrategyId::Phase1SpotDaySwing => 0,
            StrategyId::MomentumMicro      => 1,
            StrategyId::MeanReversionMicro => 2,
            StrategyId::BreakoutMicro      => 3,
        }
    }
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// All strategies in deterministic priority order.
/// Phase1SpotDaySwing is listed first (primary path).
/// Legacy micro strategies follow.
pub const ALL_STRATEGIES: &[StrategyId] = &[
    StrategyId::Phase1SpotDaySwing,
    StrategyId::MomentumMicro,
    StrategyId::MeanReversionMicro,
    StrategyId::BreakoutMicro,
];

// ── Strategy action ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum StrategyAction {
    BuyCandidate,
    ExitCandidate { reason: String },
    Wait,
    StandDown,
}

impl StrategyAction {
    pub fn is_actionable(&self) -> bool {
        matches!(self, StrategyAction::BuyCandidate | StrategyAction::ExitCandidate { .. })
    }
}

impl std::fmt::Display for StrategyAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StrategyAction::BuyCandidate          => write!(f, "BUY_CANDIDATE"),
            StrategyAction::ExitCandidate { reason } => write!(f, "EXIT_CANDIDATE({})", reason),
            StrategyAction::Wait                  => write!(f, "WAIT"),
            StrategyAction::StandDown             => write!(f, "STAND_DOWN"),
        }
    }
}

// ── Strategy decision ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StrategyDecision {
    pub strategy_id: StrategyId,
    pub action:      StrategyAction,
    pub confidence:  f64,
    pub reason:      String,
}

impl StrategyDecision {
    fn wait(id: StrategyId, reason: impl Into<String>) -> Self {
        Self { strategy_id: id, action: StrategyAction::Wait, confidence: 0.0, reason: reason.into() }
    }
    fn stand_down(id: StrategyId, reason: impl Into<String>) -> Self {
        Self { strategy_id: id, action: StrategyAction::StandDown, confidence: 1.0, reason: reason.into() }
    }
}

// ── Strategy trait ────────────────────────────────────────────────────────────

pub trait Strategy: Send + Sync {
    fn id(&self) -> &StrategyId;

    /// Evaluate the current market snapshot.
    /// Pure function: no I/O, no locks, no side effects.
    ///
    /// `metrics` is pre-computed from FeedState by the engine caller.
    /// `truth` provides position size, open orders, and state guards.
    /// `entry_price_hint` / `entry_time` are held by StrategyEngine per-strategy.
    fn evaluate(
        &self,
        metrics:          &SignalMetrics,
        truth:            &TruthState,
        entry_price_hint: Option<f64>,
        entry_time:       Option<Instant>,
    ) -> StrategyDecision;
}

// ── Common guard helpers ──────────────────────────────────────────────────────

fn common_guards(
    id: &StrategyId,
    metrics: &SignalMetrics,
    truth: &TruthState,
    max_spread_bps: f64,
    max_feed_age_ms: f64,
) -> Option<StrategyDecision> {
    if truth.state_dirty {
        return Some(StrategyDecision::wait(id.clone(), "state_dirty"));
    }
    if truth.recon_in_progress {
        return Some(StrategyDecision::wait(id.clone(), "recon_in_progress"));
    }
    if metrics.feed_age_ms > max_feed_age_ms {
        return Some(StrategyDecision::wait(id.clone(), format!("feed stale: {:.0}ms", metrics.feed_age_ms)));
    }
    if metrics.bid <= 0.0 || metrics.ask <= 0.0 {
        return Some(StrategyDecision::wait(id.clone(), "no market data"));
    }
    if metrics.spread_bps > max_spread_bps {
        return Some(StrategyDecision::wait(
            id.clone(),
            format!(
                "entry_rejected stage=strategy.common_guards spread_bps={:.2} threshold_bps={:.2}",
                metrics.spread_bps, max_spread_bps
            ),
        ));
    }
    None
}

fn has_open_buy(truth: &TruthState) -> bool {
    truth.orders.values().any(|r| r.side.eq_ignore_ascii_case("BUY") && r.status.is_active())
}

/// Standard exit rules shared by all strategies (stop-loss, take-profit, max-hold).
fn evaluate_exit(
    id:               &StrategyId,
    metrics:          &SignalMetrics,
    position_size:    f64,
    entry_price_hint: Option<f64>,
    entry_time:       Option<Instant>,
    stop_loss_pct:    f64,
    take_profit_pct:  f64,
    max_hold:         Duration,
) -> Option<StrategyDecision> {
    let entry = entry_price_hint?;
    let _time = entry_time?;
    let mid   = metrics.mid;

    let sl = entry * (1.0 - stop_loss_pct);
    if mid <= sl {
        return Some(StrategyDecision {
            strategy_id: id.clone(),
            action:      StrategyAction::ExitCandidate { reason: "STOP_LOSS".into() },
            confidence:  1.0,
            reason:      format!("mid {:.2} ≤ stop {:.2} (entry {:.2})", mid, sl, entry),
        });
    }
    let tp = entry * (1.0 + take_profit_pct);
    if mid >= tp {
        return Some(StrategyDecision {
            strategy_id: id.clone(),
            action:      StrategyAction::ExitCandidate { reason: "TAKE_PROFIT".into() },
            confidence:  1.0,
            reason:      format!("mid {:.2} ≥ target {:.2} (entry {:.2})", mid, tp, entry),
        });
    }
    if let Some(et) = entry_time {
        if et.elapsed() >= max_hold {
            return Some(StrategyDecision {
                strategy_id: id.clone(),
                action:      StrategyAction::ExitCandidate { reason: "MAX_HOLD_TIME".into() },
                confidence:  0.8,
                reason:      format!("held {:.0}s ≥ limit {:.0}s", et.elapsed().as_secs_f64(), max_hold.as_secs_f64()),
            });
        }
    }
    None
}

// ── Strategy 1: MomentumMicro (LEGACY / EXPERIMENTAL) ────────────────────────
//
// Entry: all three momentum windows positive + 1s imbalance above threshold.
// The existing signal engine logic, now as a named legacy strategy.

pub struct MomentumMicro {
    pub momentum_threshold:   f64,
    pub imbalance_threshold:  f64,
    pub max_entry_spread_bps: f64,
    pub stop_loss_pct:        f64,
    pub take_profit_pct:      f64,
    pub max_hold:             Duration,
}

impl Default for MomentumMicro {
    fn default() -> Self {
        Self {
            momentum_threshold:   0.00005,
            imbalance_threshold:  0.10,
            max_entry_spread_bps: 5.0,
            stop_loss_pct:        0.0020,
            take_profit_pct:      0.0040,
            max_hold:             Duration::from_secs(120),
        }
    }
}

impl Strategy for MomentumMicro {
    fn id(&self) -> &StrategyId { &StrategyId::MomentumMicro }

    fn evaluate(
        &self,
        metrics: &SignalMetrics,
        truth: &TruthState,
        entry_price_hint: Option<f64>,
        entry_time: Option<Instant>,
    ) -> StrategyDecision {
        if let Some(g) = common_guards(&StrategyId::MomentumMicro, metrics, truth,
                                        self.max_entry_spread_bps, 3_000.0) {
            return g;
        }

        // Exit check (always first when in position)
        if !truth.position.is_flat() {
            if let Some(exit) = evaluate_exit(
                &StrategyId::MomentumMicro, metrics, truth.position.size,
                entry_price_hint, entry_time,
                self.stop_loss_pct, self.take_profit_pct, self.max_hold,
            ) {
                return exit;
            }
            return StrategyDecision::wait(StrategyId::MomentumMicro, "in position, no exit criteria");
        }

        if has_open_buy(truth) {
            return StrategyDecision::wait(StrategyId::MomentumMicro, "open BUY order exists");
        }

        let m1 = metrics.momentum_1s;
        let m3 = metrics.momentum_3s;
        let m5 = metrics.momentum_5s;
        let i1 = metrics.imbalance_1s;
        let i3 = metrics.imbalance_3s;
        let thr = self.momentum_threshold;
        let i_thr = self.imbalance_threshold;

        if m1 <= thr { return StrategyDecision::wait(StrategyId::MomentumMicro, format!("1s mom {:+.5} ≤ {:.5}", m1, thr)); }
        if m3 <= thr { return StrategyDecision::wait(StrategyId::MomentumMicro, format!("3s mom {:+.5} ≤ {:.5}", m3, thr)); }
        if m5 <= thr { return StrategyDecision::wait(StrategyId::MomentumMicro, format!("5s mom {:+.5} ≤ {:.5}", m5, thr)); }
        if i1 <= i_thr { return StrategyDecision::wait(StrategyId::MomentumMicro, format!("1s imb {:+.3} ≤ {:.3}", i1, i_thr)); }
        if i3 <= 0.0 { return StrategyDecision::wait(StrategyId::MomentumMicro, format!("3s imb {:+.3} not positive", i3)); }

        // Confidence: mean of normalised metric distances from threshold
        let mom_score = (m5 / thr).min(1.0).max(0.0);
        let imb_score = i1.min(1.0).max(0.0);
        let spr_score = (1.0 - metrics.spread_bps / self.max_entry_spread_bps).min(1.0).max(0.0);
        let confidence = (mom_score + imb_score + spr_score) / 3.0;

        StrategyDecision {
            strategy_id: StrategyId::MomentumMicro,
            action:      StrategyAction::BuyCandidate,
            confidence,
            reason:      format!("MomentumMicro: m1={:+.5} m3={:+.5} m5={:+.5} i1={:+.3} i3={:+.3} conf={:.2}", m1, m3, m5, i1, i3, confidence),
        }
    }
}

// ── Strategy 2: MeanReversionMicro (LEGACY / EXPERIMENTAL) ──────────────────
//
// Entry: short-window momentum (1s) is negative while the longer window (5s)
// is positive → price dipped within an upward trend (mean-reversion entry).
// Additionally requires buy imbalance recovering (1s imb > 0 even if small).
//
// This is conservative: requires the dip to be mild (1s below zero but not
// catastrophically negative) so we're not buying into a trend reversal.

pub struct MeanReversionMicro {
    pub max_entry_spread_bps: f64,
    pub min_5s_momentum:      f64,   // long-window must still be positive
    pub max_1s_momentum:      f64,   // short-window must be below this (mild dip)
    pub min_1s_momentum:      f64,   // short-window must not be too negative
    pub min_1s_imbalance:     f64,   // some buy pressure must be returning
    pub stop_loss_pct:        f64,
    pub take_profit_pct:      f64,
    pub max_hold:             Duration,
}

impl Default for MeanReversionMicro {
    fn default() -> Self {
        Self {
            max_entry_spread_bps: 4.0,
            min_5s_momentum:      0.00003,  // 5s must be upward
            max_1s_momentum:      0.0,      // 1s must be below zero (the dip)
            min_1s_momentum:      -0.0010,  // dip must not be too severe
            min_1s_imbalance:     0.05,     // some buy pressure returning
            stop_loss_pct:        0.0015,
            take_profit_pct:      0.0030,
            max_hold:             Duration::from_secs(90),
        }
    }
}

impl Strategy for MeanReversionMicro {
    fn id(&self) -> &StrategyId { &StrategyId::MeanReversionMicro }

    fn evaluate(
        &self,
        metrics: &SignalMetrics,
        truth: &TruthState,
        entry_price_hint: Option<f64>,
        entry_time: Option<Instant>,
    ) -> StrategyDecision {
        if let Some(g) = common_guards(&StrategyId::MeanReversionMicro, metrics, truth,
                                        self.max_entry_spread_bps, 3_000.0) {
            return g;
        }

        if !truth.position.is_flat() {
            if let Some(exit) = evaluate_exit(
                &StrategyId::MeanReversionMicro, metrics, truth.position.size,
                entry_price_hint, entry_time,
                self.stop_loss_pct, self.take_profit_pct, self.max_hold,
            ) {
                return exit;
            }
            return StrategyDecision::wait(StrategyId::MeanReversionMicro, "in position");
        }

        if has_open_buy(truth) {
            return StrategyDecision::wait(StrategyId::MeanReversionMicro, "open BUY exists");
        }

        let m1 = metrics.momentum_1s;
        let m5 = metrics.momentum_5s;
        let i1 = metrics.imbalance_1s;

        if m5 < self.min_5s_momentum {
            return StrategyDecision::wait(StrategyId::MeanReversionMicro,
                format!("5s mom {:+.5} < {:.5} (no uptrend)", m5, self.min_5s_momentum));
        }
        if m1 >= self.max_1s_momentum {
            return StrategyDecision::wait(StrategyId::MeanReversionMicro,
                format!("1s mom {:+.5} not in dip (<{:.5})", m1, self.max_1s_momentum));
        }
        if m1 < self.min_1s_momentum {
            return StrategyDecision::wait(StrategyId::MeanReversionMicro,
                format!("1s mom {:+.5} too negative (<{:.5})", m1, self.min_1s_momentum));
        }
        if i1 < self.min_1s_imbalance {
            return StrategyDecision::wait(StrategyId::MeanReversionMicro,
                format!("1s imb {:+.3} < {:.3} (no buy return)", i1, self.min_1s_imbalance));
        }

        // Confidence: how cleanly is the pattern present?
        let trend_score  = (m5 / self.min_5s_momentum).min(1.0).max(0.0);
        let depth_score  = ((-m1) / (-self.min_1s_momentum)).min(1.0).max(0.0); // depth of dip
        let return_score = (i1 / 0.5).min(1.0).max(0.0);  // how much buy pressure is returning
        let confidence   = (trend_score + depth_score + return_score) / 3.0;

        StrategyDecision {
            strategy_id: StrategyId::MeanReversionMicro,
            action:      StrategyAction::BuyCandidate,
            confidence,
            reason:      format!("MeanReversionMicro: dip m1={:+.5} in uptrend m5={:+.5} buy-return i1={:+.3} conf={:.2}", m1, m5, i1, confidence),
        }
    }
}

// ── Strategy 3: BreakoutMicro (LEGACY / EXPERIMENTAL) ────────────────────────
//
// Entry: 1s momentum significantly exceeds 5s momentum, indicating price
// just accelerated above its recent range. Buy on the breakout candle.
// Also requires strong buy imbalance to confirm the move is real volume.

pub struct BreakoutMicro {
    pub max_entry_spread_bps:  f64,
    pub min_5s_momentum:       f64,   // base trend must be positive
    pub breakout_ratio:        f64,   // m1 must be this multiple of m5
    pub min_imbalance_1s:      f64,
    pub stop_loss_pct:         f64,
    pub take_profit_pct:       f64,
    pub max_hold:              Duration,
}

impl Default for BreakoutMicro {
    fn default() -> Self {
        Self {
            max_entry_spread_bps: 5.0,
            min_5s_momentum:      0.00002,
            breakout_ratio:       2.5,   // 1s momentum ≥ 2.5× the 5s momentum
            min_imbalance_1s:     0.3,   // strong buy pressure required
            stop_loss_pct:        0.0025,
            take_profit_pct:      0.0060,
            max_hold:             Duration::from_secs(60),
        }
    }
}

impl Strategy for BreakoutMicro {
    fn id(&self) -> &StrategyId { &StrategyId::BreakoutMicro }

    fn evaluate(
        &self,
        metrics: &SignalMetrics,
        truth: &TruthState,
        entry_price_hint: Option<f64>,
        entry_time: Option<Instant>,
    ) -> StrategyDecision {
        if let Some(g) = common_guards(&StrategyId::BreakoutMicro, metrics, truth,
                                        self.max_entry_spread_bps, 3_000.0) {
            return g;
        }

        if !truth.position.is_flat() {
            if let Some(exit) = evaluate_exit(
                &StrategyId::BreakoutMicro, metrics, truth.position.size,
                entry_price_hint, entry_time,
                self.stop_loss_pct, self.take_profit_pct, self.max_hold,
            ) {
                return exit;
            }
            return StrategyDecision::wait(StrategyId::BreakoutMicro, "in position");
        }

        if has_open_buy(truth) {
            return StrategyDecision::wait(StrategyId::BreakoutMicro, "open BUY exists");
        }

        let m1 = metrics.momentum_1s;
        let m5 = metrics.momentum_5s;
        let i1 = metrics.imbalance_1s;

        if m5 < self.min_5s_momentum {
            return StrategyDecision::wait(StrategyId::BreakoutMicro,
                format!("5s mom {:+.5} < {:.5}", m5, self.min_5s_momentum));
        }
        // Breakout condition: 1s momentum is significantly larger than 5s
        let ratio = if m5 > 0.0 { m1 / m5 } else { 0.0 };
        if ratio < self.breakout_ratio {
            return StrategyDecision::wait(StrategyId::BreakoutMicro,
                format!("breakout ratio {:.2} < {:.2} (m1/m5)", ratio, self.breakout_ratio));
        }
        if i1 < self.min_imbalance_1s {
            return StrategyDecision::wait(StrategyId::BreakoutMicro,
                format!("1s imb {:+.3} < {:.3} (weak buy vol)", i1, self.min_imbalance_1s));
        }

        let accel_score = (ratio / (self.breakout_ratio * 2.0)).min(1.0);
        let imb_score   = (i1 / 0.8).min(1.0).max(0.0);
        let spr_score   = (1.0 - metrics.spread_bps / self.max_entry_spread_bps).min(1.0).max(0.0);
        let confidence  = (accel_score + imb_score + spr_score) / 3.0;

        StrategyDecision {
            strategy_id: StrategyId::BreakoutMicro,
            action:      StrategyAction::BuyCandidate,
            confidence,
            reason:      format!("BreakoutMicro: m1/m5 ratio={:.2} m1={:+.5} m5={:+.5} i1={:+.3} conf={:.2}", ratio, m1, m5, i1, confidence),
        }
    }
}

// ── Per-strategy entry tracking ───────────────────────────────────────────────

#[derive(Default)]
struct StrategyState {
    entry_price_hint: Option<f64>,
    entry_time:       Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RegimeBucket {
    Aligned,
    Divergent,
}

impl RegimeBucket {
    fn as_str(&self) -> &'static str {
        match self {
            RegimeBucket::Aligned => "aligned",
            RegimeBucket::Divergent => "divergent",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StrategyPerf {
    wins: u32,
    losses: u32,
}

impl StrategyPerf {
    fn win_rate(self) -> f64 {
        let total = self.wins + self.losses;
        if total == 0 { 0.5 } else { self.wins as f64 / total as f64 }
    }
}

#[derive(Debug, Clone, Copy)]
struct RegimeView {
    short_fast: f64,
    medium_stable: f64,
    aligned: bool,
    weight: f64,
    bucket: RegimeBucket,
}

// ── StrategyEngine ────────────────────────────────────────────────────────────

/// Holds all strategies, their enable flags, and per-strategy entry state.
/// Wrapped in Arc<Mutex<>> so the web UI can toggle enables at runtime.
pub struct StrategyEngine {
    // ── Phase 1 (primary strategy path) ──────────────────────────────────────
    /// Phase 1 Spot Day+Swing Long-Only engine.
    /// Maintains its own rolling mid-price history for multi-timeframe analysis.
    phase1:         Phase1Engine,
    /// Whether Phase 1 is enabled as the primary strategy path.
    phase1_enabled: bool,
    /// Last Phase 1 setup details (for entry tracking when Phase1 fires).
    phase1_last_setup: Option<crate::phase1::Phase1Setup>,

    // ── Legacy strategies (micro/agent system) ────────────────────────────────
    strategies:  Vec<Box<dyn Strategy>>,
    enabled:     HashMap<StrategyId, bool>,
    entry_state: HashMap<StrategyId, StrategyState>,
    base_confidence_threshold: f64,
    no_trade_lowering_after: Duration,
    no_trade_lowering_step: f64,
    loss_streak_threshold_step: f64,
    rapid_edge_decay_per_sec: f64,
    min_trade_samples: usize,
    min_abs_imbalance_1s: f64,
    perf: HashMap<(StrategyId, RegimeBucket), StrategyPerf>,
    active_strategy_hint: Option<StrategyId>,
    active_regime_bucket: RegimeBucket,
    last_trade_at: Option<Instant>,
    loss_streak: u32,
    last_edge_confidence: Option<(f64, Instant)>,
    recent_edge: HashMap<StrategyId, VecDeque<f64>>,
    paused_until: HashMap<StrategyId, Instant>,
    pause_duration: Duration,
    edge_decay_pause_threshold: f64,
}

impl StrategyEngine {
    /// Create with Phase1SpotDaySwing (primary) and all three legacy strategies enabled.
    pub fn new() -> Self {
        let strategies: Vec<Box<dyn Strategy>> = vec![
            Box::new(MomentumMicro::default()),
            Box::new(MeanReversionMicro::default()),
            Box::new(BreakoutMicro::default()),
        ];
        let mut enabled = HashMap::new();
        let mut entry_state = HashMap::new();
        for s in &strategies {
            enabled.insert(s.id().clone(), true);
            entry_state.insert(s.id().clone(), StrategyState::default());
        }
        Self {
            phase1:            Phase1Engine::default(),
            phase1_enabled:    true,
            phase1_last_setup: None,
            strategies,
            enabled,
            entry_state,
            base_confidence_threshold: 0.72,
            no_trade_lowering_after: Duration::from_secs(120),
            no_trade_lowering_step: 0.04,
            loss_streak_threshold_step: 0.04,
            rapid_edge_decay_per_sec: 0.35,
            min_trade_samples: 3,
            min_abs_imbalance_1s: 0.06,
            perf: HashMap::new(),
            active_strategy_hint: None,
            active_regime_bucket: RegimeBucket::Aligned,
            last_trade_at: None,
            loss_streak: 0,
            last_edge_confidence: None,
            recent_edge: HashMap::new(),
            paused_until: HashMap::new(),
            pause_duration: Duration::from_secs(300),
            edge_decay_pause_threshold: 0.08,
        }
    }

    // ── Enable/disable ────────────────────────────────────────────────────────

    pub fn set_enabled(&mut self, id: &StrategyId, enabled: bool) {
        if *id == StrategyId::Phase1SpotDaySwing {
            self.phase1_enabled = enabled;
            info!(strategy = %id, enabled, "[STRATEGY] Enable toggled");
            return;
        }
        if let Some(e) = self.enabled.get_mut(id) {
            *e = enabled;
            info!(strategy = %id, enabled, "[STRATEGY] Enable toggled");
        }
    }

    pub fn is_enabled(&self, id: &StrategyId) -> bool {
        if *id == StrategyId::Phase1SpotDaySwing {
            return self.phase1_enabled;
        }
        *self.enabled.get(id).unwrap_or(&false)
    }

    pub fn enable_states(&self) -> Vec<(StrategyId, bool)> {
        ALL_STRATEGIES.iter()
            .map(|id| (id.clone(), self.is_enabled(id)))
            .collect()
    }

    // ── Entry tracking ────────────────────────────────────────────────────────

    /// Called when a BUY is confirmed submitted for a specific strategy.
    pub fn on_entry_submitted(&mut self, id: &StrategyId, price: f64) {
        if *id == StrategyId::Phase1SpotDaySwing {
            if let Some(setup) = self.phase1_last_setup.take() {
                self.phase1.on_entry_confirmed(&setup);
            }
            self.last_trade_at = Some(Instant::now());
            self.active_strategy_hint = Some(id.clone());
            return;
        }
        if let Some(state) = self.entry_state.get_mut(id) {
            state.entry_price_hint = Some(price);
            state.entry_time       = Some(Instant::now());
        }
        self.last_trade_at = Some(Instant::now());
        self.active_strategy_hint = Some(id.clone());
    }

    /// Called when position is closed (sell executed).
    pub fn on_exit_submitted(&mut self, id: &StrategyId, exit_price: f64) {
        if *id == StrategyId::Phase1SpotDaySwing {
            self.phase1.on_exit_confirmed();
            self.last_trade_at = Some(Instant::now());
            self.active_strategy_hint = None;
            return;
        }
        let mut was_win = None;
        let mut realized_edge = None;
        if let Some(state) = self.entry_state.get_mut(id) {
            if let Some(entry) = state.entry_price_hint {
                was_win = Some(exit_price > entry);
                if entry > 0.0 {
                    realized_edge = Some((exit_price - entry) / entry);
                }
            }
            state.entry_price_hint = None;
            state.entry_time       = None;
        }
        if let Some(win) = was_win {
            let key = (id.clone(), self.active_regime_bucket);
            let row = self.perf.entry(key).or_insert(StrategyPerf { wins: 0, losses: 0 });
            if win {
                row.wins += 1;
                self.loss_streak = 0;
            } else {
                row.losses += 1;
                self.loss_streak = self.loss_streak.saturating_add(1);
            }
        }
        if let Some(edge) = realized_edge {
            self.record_realized_edge(id, edge);
            self.auto_adjust_strategy(id);
        }
        self.last_trade_at = Some(Instant::now());
        self.active_strategy_hint = None;
    }

    fn record_realized_edge(&mut self, id: &StrategyId, edge: f64) {
        let row = self.recent_edge.entry(id.clone()).or_insert_with(|| VecDeque::with_capacity(30));
        row.push_back(edge);
        while row.len() > 30 {
            row.pop_front();
        }
    }

    fn auto_adjust_strategy(&mut self, id: &StrategyId) {
        let Some(row) = self.recent_edge.get(id) else { return; };
        if row.len() < 6 {
            return;
        }
        let recent = row.iter().rev().take(6).sum::<f64>() / 6.0;
        let long = row.iter().sum::<f64>() / row.len() as f64;
        if (long - recent) >= self.edge_decay_pause_threshold {
            self.paused_until.insert(id.clone(), Instant::now() + self.pause_duration);
            info!(
                strategy = %id,
                recent_edge = recent,
                long_edge = long,
                pause_secs = self.pause_duration.as_secs(),
                "[STRATEGY] Auto-paused due to edge decay"
            );
        }
    }

    fn prune_pauses(&mut self) {
        let now = Instant::now();
        self.paused_until.retain(|_, until| *until > now);
    }

    /// Clear entry state for all strategies (e.g. on position reconcile).
    pub fn clear_all_entry_state(&mut self) {
        for state in self.entry_state.values_mut() {
            state.entry_price_hint = None;
            state.entry_time       = None;
        }
        self.phase1.on_exit_confirmed();
        self.phase1_last_setup = None;
        self.active_strategy_hint = None;
    }

    /// Which strategy ID is currently "active" (has an open entry hint).
    pub fn active_strategy(&self) -> Option<StrategyId> {
        // Check Phase1 first (primary strategy)
        if self.phase1.open_position.is_some() {
            return Some(StrategyId::Phase1SpotDaySwing);
        }
        self.entry_state.iter()
            .find(|(_, s)| s.entry_price_hint.is_some())
            .map(|(id, _)| id.clone())
    }

    pub fn entry_age_secs(&self, id: &StrategyId) -> Option<f64> {
        self.entry_state
            .get(id)
            .and_then(|s| s.entry_time)
            .map(|t| t.elapsed().as_secs_f64())
    }

    // ── Core evaluation ───────────────────────────────────────────────────────

    /// Evaluate all enabled strategies and select the best actionable decision.
    ///
    /// Phase 1 (primary) is evaluated first. Legacy micro strategies follow.
    ///
    /// Selection rules:
    ///   1. Only enabled strategies are considered.
    ///   2. Only strategies returning BuyCandidate or ExitCandidate qualify.
    ///   3. Among qualifying, pick the highest confidence.
    ///   4. Ties broken by StrategyId::priority() (deterministic).
    ///   5. If none qualify, return the first enabled strategy's Wait/Hold.
    ///
    /// Returns (selected: StrategyDecision, all: Vec<StrategyDecision>).
    /// `all` is the full per-strategy breakdown for the comparison view.
    pub fn evaluate(
        &mut self,
        metrics: &SignalMetrics,
        truth:   &TruthState,
    ) -> (StrategyDecision, Vec<StrategyDecision>) {
        self.prune_pauses();
        let mut all_decisions: Vec<StrategyDecision> = Vec::new();
        let mut candidates: Vec<StrategyDecision>    = Vec::new();
        let regime = self.regime_view(metrics);
        let dynamic_threshold = self.dynamic_confidence_threshold();
        let participation_ok = self.participation_ok(metrics);

        // ── Phase 1 evaluation (primary strategy path) ────────────────────────
        // Phase1 runs before legacy strategies and has highest priority.
        // It maintains its own rolling history so we must call it mutably here.
        let phase1_decision = if self.phase1_enabled {
            let result = self.phase1.evaluate(metrics, truth);
            let decision = Self::phase1_result_to_decision(result, metrics);
            // Cache the setup so on_entry_submitted can confirm it later.
            if let StrategyAction::BuyCandidate = &decision.action {
                self.phase1_last_setup = self.phase1.last_setup.clone();
            }
            Some(decision)
        } else {
            // Record disabled placeholder so the UI can show it.
            Some(StrategyDecision {
                strategy_id: StrategyId::Phase1SpotDaySwing,
                action:      StrategyAction::StandDown,
                confidence:  0.0,
                reason:      "disabled".into(),
            })
        };

        if let Some(d) = phase1_decision {
            debug!(
                strategy = %d.strategy_id,
                action   = %d.action,
                conf     = d.confidence,
                reason   = %d.reason,
                "[STRATEGY] Phase1 decision"
            );
            if d.action.is_actionable() {
                candidates.push(d.clone());
            }
            all_decisions.push(d);
        }

        // ── Legacy strategies (micro/agent) ───────────────────────────────────
        for strategy in &self.strategies {
            if !self.is_enabled(strategy.id()) {
                // Record a placeholder so the UI can show disabled state
                all_decisions.push(StrategyDecision {
                    strategy_id: strategy.id().clone(),
                    action:      StrategyAction::StandDown,
                    confidence:  0.0,
                    reason:      "disabled".into(),
                });
                continue;
            }
            if let Some(until) = self.paused_until.get(strategy.id()) {
                all_decisions.push(StrategyDecision {
                    strategy_id: strategy.id().clone(),
                    action:      StrategyAction::StandDown,
                    confidence:  0.0,
                    reason:      format!(
                        "paused_for_edge_decay {}s",
                        until.saturating_duration_since(Instant::now()).as_secs()
                    ),
                });
                continue;
            }

            let state   = self.entry_state.get(strategy.id())
                              .map(|s| (s.entry_price_hint, s.entry_time))
                              .unwrap_or((None, None));
            let mut decision = strategy.evaluate(metrics, truth, state.0, state.1);
            decision.confidence *= regime.weight;
            let perf_weight = self.strategy_performance_weight(&decision.strategy_id, regime.bucket);
            decision.confidence *= perf_weight;
            decision.confidence = decision.confidence.clamp(0.0, 1.0);

            if decision.action.is_actionable() && !participation_ok {
                decision.action = StrategyAction::Wait;
                decision.reason = format!(
                    "{} | blocked: low participation (trades={} imb1={:+.3})",
                    decision.reason, metrics.trade_samples, metrics.imbalance_1s
                );
                decision.confidence = 0.0;
            } else if decision.action.is_actionable() && decision.confidence < dynamic_threshold {
                decision.action = StrategyAction::Wait;
                decision.reason = format!(
                    "{} | below dynamic threshold {:.2}",
                    decision.reason, dynamic_threshold
                );
            } else if decision.action.is_actionable() && !regime.aligned {
                decision.reason = format!(
                    "{} | regimes diverged fast={:+.4} medium={:+.4} weight={:.2}",
                    decision.reason, regime.short_fast, regime.medium_stable, regime.weight
                );
            }

            debug!(
                strategy = %decision.strategy_id,
                action   = %decision.action,
                conf     = decision.confidence,
                reason   = %decision.reason,
                "[STRATEGY] Decision"
            );

            if decision.action.is_actionable() {
                candidates.push(decision.clone());
            }
            all_decisions.push(decision);
        }

        // Select: highest confidence, tie-broken by priority
        let selected = candidates.into_iter().reduce(|best, next| {
            let best_pri = best.strategy_id.priority();
            let next_pri = next.strategy_id.priority();
            if next.confidence > best.confidence
                || (next.confidence == best.confidence && next_pri < best_pri)
            {
                next
            } else {
                best
            }
        });

        let mut chosen = match selected {
            Some(d) => {
                info!(
                    strategy   = %d.strategy_id,
                    action     = %d.action,
                    confidence = d.confidence,
                    "[STRATEGY] Selected"
                );
                d
            }
            None => {
                // Return the first enabled strategy's non-actionable decision
                all_decisions.iter()
                    .find(|d| d.reason != "disabled")
                    .cloned()
                    .unwrap_or_else(|| StrategyDecision::stand_down(StrategyId::Phase1SpotDaySwing, "no strategies enabled"))
            }
        };

        if self.should_force_exit_for_edge_decay(metrics, truth, &chosen) {
            chosen = StrategyDecision {
                strategy_id: self.active_strategy_hint.clone().unwrap_or(StrategyId::Phase1SpotDaySwing),
                action: StrategyAction::ExitCandidate { reason: "EDGE_DECAY".into() },
                confidence: 1.0,
                reason: "Edge velocity decayed rapidly; forcing immediate exit".into(),
            };
        }

        (chosen, all_decisions)
    }

    /// Convert a Phase1Result into a StrategyDecision.
    ///
    /// This is called during evaluate() to integrate Phase1 into the
    /// standard strategy selection pipeline.
    fn phase1_result_to_decision(result: Phase1Result, metrics: &SignalMetrics) -> StrategyDecision {
        match result {
            Phase1Result::Setup(setup) => {
                // Confidence is the setup quality (0..1).
                let confidence = setup.setup_quality;
                let reason = format!(
                    "phase1:{} mode={} entry={:.2} stop={:.2} target={:.2} quality={:.2}",
                    setup.setup_type.as_str(),
                    setup.mode.as_str(),
                    setup.entry,
                    setup.stop,
                    setup.target,
                    setup.setup_quality,
                );
                StrategyDecision {
                    strategy_id: StrategyId::Phase1SpotDaySwing,
                    action:      StrategyAction::BuyCandidate,
                    confidence,
                    reason,
                }
            }
            Phase1Result::Exit { reason } => StrategyDecision {
                strategy_id: StrategyId::Phase1SpotDaySwing,
                action:      StrategyAction::ExitCandidate { reason: reason.clone() },
                confidence:  1.0,
                reason:      format!("phase1:exit reason={}", reason),
            },
            Phase1Result::HoldPosition { regime, position_note } => StrategyDecision {
                strategy_id: StrategyId::Phase1SpotDaySwing,
                action:      StrategyAction::Wait,
                confidence:  0.0,
                reason:      format!("phase1:hold regime={} {}", regime.as_str(), position_note),
            },
            Phase1Result::NoTrade { regime, block_reason } => {
                let _ = metrics; // metrics available for future use
                StrategyDecision {
                    strategy_id: StrategyId::Phase1SpotDaySwing,
                    action:      StrategyAction::Wait,
                    confidence:  0.0,
                    reason:      format!("phase1:no_trade regime={} block={}", regime.as_str(), block_reason),
                }
            }
        }
    }

    /// Return a snapshot of Phase1Engine state for UI display.
    pub fn phase1_status(&self) -> Phase1Status {
        let pos = self.phase1.open_position.as_ref();
        let setup = self.phase1.last_setup.as_ref();
        Phase1Status {
            enabled:         self.phase1_enabled,
            regime:          self.phase1.last_regime,
            block_reason:    self.phase1.last_block_reason.clone(),
            last_setup_type: setup.map(|s| s.setup_type),
            last_setup_mode: setup.map(|s| s.mode),
            last_entry:      setup.map(|s| s.entry),
            last_stop:       setup.map(|s| s.stop),
            last_target:     setup.map(|s| s.target),
            last_quality:    setup.map(|s| s.setup_quality),
            in_position:     pos.is_some(),
            position_stop:   pos.map(|p| p.stop_price),
            position_target: pos.map(|p| p.target_price),
            break_even:      pos.map(|p| p.break_even_triggered).unwrap_or(false),
            high_water:      pos.map(|p| p.high_water_mark),
        }
    }

    /// Compute Phase1 position size for a given setup.
    pub fn phase1_size_from_risk(&self, entry: f64, stop: f64) -> f64 {
        self.phase1.size_from_risk(entry, stop)
    }

    pub fn compute_position_size(
        &self,
        base_qty: f64,
        confidence: f64,
        regime_weight: f64,
        metrics: &SignalMetrics,
    ) -> f64 {
        let confidence_weight = confidence.clamp(0.0, 1.0);
        let volatility = metrics.momentum_1s.abs() + metrics.momentum_3s.abs() + metrics.momentum_5s.abs();
        let volatility_adjustment = (1.0 / (1.0 + (volatility * 200.0))).clamp(0.5, 1.2);
        (base_qty * confidence_weight * regime_weight * volatility_adjustment).max(0.0)
    }

    pub fn regime_weight(&self, metrics: &SignalMetrics) -> f64 {
        self.regime_view(metrics).weight
    }

    fn participation_ok(&self, metrics: &SignalMetrics) -> bool {
        metrics.trade_samples >= self.min_trade_samples
            && metrics.imbalance_1s.abs() >= self.min_abs_imbalance_1s
    }

    fn dynamic_confidence_threshold(&self) -> f64 {
        let mut t = self.base_confidence_threshold;
        if let Some(last) = self.last_trade_at {
            if last.elapsed() >= self.no_trade_lowering_after {
                t -= self.no_trade_lowering_step;
            }
        }
        t += self.loss_streak as f64 * self.loss_streak_threshold_step;
        t.clamp(0.55, 0.90)
    }

    fn strategy_performance_weight(&self, id: &StrategyId, bucket: RegimeBucket) -> f64 {
        self.perf
            .get(&(id.clone(), bucket))
            .map(|p| (0.7 + (p.win_rate() * 0.6)).clamp(0.7, 1.3))
            .unwrap_or(1.0)
    }

    fn regime_view(&self, metrics: &SignalMetrics) -> RegimeView {
        let short_fast = (metrics.momentum_1s * 0.7) + (metrics.imbalance_1s * 0.3);
        let medium_stable = (metrics.momentum_5s * 0.8) + (metrics.imbalance_5s * 0.2);
        let aligned = short_fast.signum() == medium_stable.signum();
        let weight = if aligned { 1.0 } else { 0.72 };
        let bucket = if aligned { RegimeBucket::Aligned } else { RegimeBucket::Divergent };
        RegimeView { short_fast, medium_stable, aligned, weight, bucket }
    }

    fn should_force_exit_for_edge_decay(
        &mut self,
        metrics: &SignalMetrics,
        truth: &TruthState,
        chosen: &StrategyDecision,
    ) -> bool {
        let edge_confidence = self.compute_edge_confidence(metrics);
        let now = Instant::now();
        let mut force_exit = false;
        if !truth.position.is_flat() {
            if let Some((prev, ts)) = self.last_edge_confidence {
                let dt = now.duration_since(ts).as_secs_f64().max(0.001);
                let velocity = (edge_confidence - prev) / dt;
                if velocity <= -self.rapid_edge_decay_per_sec
                    && !matches!(chosen.action, StrategyAction::ExitCandidate { .. })
                {
                    force_exit = true;
                }
            }
        }
        self.last_edge_confidence = Some((edge_confidence, now));
        self.active_regime_bucket = self.regime_view(metrics).bucket;
        force_exit
    }

    fn compute_edge_confidence(&self, metrics: &SignalMetrics) -> f64 {
        let momentum = ((metrics.momentum_1s + metrics.momentum_3s + metrics.momentum_5s) / 3.0).max(0.0);
        let imbalance = ((metrics.imbalance_1s + metrics.imbalance_3s + metrics.imbalance_5s) / 3.0).max(0.0);
        let spread = (1.0 - (metrics.spread_bps / 8.0)).clamp(0.0, 1.0);
        ((momentum * 2000.0).clamp(0.0, 1.0) * 0.45 + imbalance.clamp(0.0, 1.0) * 0.35 + spread * 0.20)
            .clamp(0.0, 1.0)
    }
}

impl Default for StrategyEngine {
    fn default() -> Self { Self::new() }
}

// ── Convert StrategyAction → signal::SignalDecision ───────────────────────────

impl StrategyDecision {
    /// Convert to the SignalDecision enum the existing pipeline expects.
    pub fn to_signal_decision(&self) -> crate::signal::SignalDecision {
        match &self.action {
            StrategyAction::BuyCandidate => crate::signal::SignalDecision::Buy,
            StrategyAction::ExitCandidate { reason } => {
                let exit_reason = match reason.as_str() {
                    "STOP_LOSS"
                    | "BREAK_EVEN_STOP"
                    | "TRAILING_STOP"  => crate::signal::ExitReason::StopLoss,
                    "TAKE_PROFIT"      => crate::signal::ExitReason::TakeProfit,
                    "MAX_HOLD_TIME"
                    | "TIME_EXIT"      => crate::signal::ExitReason::MaxHoldTime,
                    _                  => crate::signal::ExitReason::MaxHoldTime, // safe fallback
                };
                crate::signal::SignalDecision::Exit { reason: exit_reason }
            }
            StrategyAction::Wait | StrategyAction::StandDown => {
                crate::signal::SignalDecision::Hold
            }
        }
    }

    /// Build a SignalResult compatible with the rest of the pipeline from this decision.
    pub fn to_signal_result(&self, metrics: &SignalMetrics) -> crate::signal::SignalResult {
        crate::signal::SignalResult {
            decision:   self.to_signal_decision(),
            reason:     format!("[{}] {}", self.strategy_id, self.reason),
            confidence: self.confidence,
            metrics:    metrics.clone(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use crate::reconciler::TruthState;
    use crate::signal::SignalMetrics;
    use std::time::Duration;

    // ── Test fixtures ─────────────────────────────────────────────────────────

    fn clean_truth() -> TruthState {
        let mut t = TruthState::new("BTCUSDT", 0.0);
        t.state_dirty       = false;
        t.recon_in_progress = false;
        t.last_reconciled_at = Some(std::time::Instant::now());
        t
    }

    fn base_metrics() -> SignalMetrics {
        SignalMetrics {
            bid: 50000.0, ask: 50001.0, mid: 50000.5, spread_bps: 2.0,
            momentum_1s: 0.001, momentum_3s: 0.002, momentum_5s: 0.003,
            imbalance_1s: 0.5, imbalance_3s: 0.3, imbalance_5s: 0.2,
            feed_age_ms: 100.0, mid_samples: 10, trade_samples: 8,
        }
    }

    // ── StrategyId ────────────────────────────────────────────────────────────

    #[test]
    fn test_strategy_id_priority_order() {
        // Phase1SpotDaySwing is highest priority (lowest number)
        assert!(StrategyId::Phase1SpotDaySwing.priority() < StrategyId::MomentumMicro.priority());
        assert!(StrategyId::MomentumMicro.priority() < StrategyId::MeanReversionMicro.priority());
        assert!(StrategyId::MeanReversionMicro.priority() < StrategyId::BreakoutMicro.priority());
    }

    #[test]
    fn test_strategy_id_display() {
        assert_eq!(StrategyId::Phase1SpotDaySwing.to_string(), "Phase1SpotDaySwing");
        assert_eq!(StrategyId::MomentumMicro.to_string(), "MomentumMicro");
        assert_eq!(StrategyId::MeanReversionMicro.to_string(), "MeanReversionMicro");
        assert_eq!(StrategyId::BreakoutMicro.to_string(), "BreakoutMicro");
    }

    // ── MomentumMicro ─────────────────────────────────────────────────────────

    #[test]
    fn test_momentum_micro_buy_on_good_conditions() {
        let s = MomentumMicro::default();
        let m = base_metrics();
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::BuyCandidate), "Expected BuyCandidate, got {:?}", d.action);
    }

    #[test]
    fn test_momentum_micro_wait_on_negative_momentum() {
        let s = MomentumMicro::default();
        let mut m = base_metrics();
        m.momentum_1s = -0.001;
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    #[test]
    fn test_momentum_micro_wait_on_low_imbalance() {
        let s = MomentumMicro::default();
        let mut m = base_metrics();
        m.imbalance_1s = -0.5;
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    #[test]
    fn test_momentum_micro_exit_stop_loss() {
        let s = MomentumMicro::default();
        let mut m = base_metrics();
        m.mid = 49890.0; // below 0.20% of 50000
        let mut t = clean_truth();
        t.position.size      = 0.001;
        t.position.avg_entry = 50000.0;
        let d = s.evaluate(&m, &t, Some(50000.0), Some(Instant::now()));
        assert!(matches!(d.action, StrategyAction::ExitCandidate { .. }));
    }

    #[test]
    fn test_momentum_micro_wait_on_dirty_state() {
        let s = MomentumMicro::default();
        let m = base_metrics();
        let mut t = clean_truth();
        t.state_dirty = true;
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    // ── MeanReversionMicro ────────────────────────────────────────────────────

    #[test]
    fn test_mean_reversion_buy_on_dip_in_uptrend() {
        let s = MeanReversionMicro::default();
        let mut m = base_metrics();
        m.momentum_1s = -0.0001; // dip (below 0, above min)
        m.momentum_5s =  0.0005; // uptrend intact
        m.imbalance_1s = 0.1;    // buy pressure returning
        m.spread_bps   = 2.0;
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::BuyCandidate),
            "Expected BuyCandidate, got {:?}: {}", d.action, d.reason);
    }

    #[test]
    fn test_mean_reversion_wait_no_dip() {
        let s = MeanReversionMicro::default();
        let mut m = base_metrics();
        m.momentum_1s =  0.001; // not a dip — 1s is positive
        m.momentum_5s =  0.003;
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    #[test]
    fn test_mean_reversion_wait_downtrend() {
        let s = MeanReversionMicro::default();
        let mut m = base_metrics();
        m.momentum_1s = -0.001;
        m.momentum_5s = -0.002; // 5s also negative → not an uptrend
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    #[test]
    fn test_mean_reversion_wait_too_severe_dip() {
        let s = MeanReversionMicro::default();
        let mut m = base_metrics();
        m.momentum_1s = -0.005; // beyond min_1s_momentum (-0.001)
        m.momentum_5s =  0.003;
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    // ── BreakoutMicro ─────────────────────────────────────────────────────────

    #[test]
    fn test_breakout_buy_on_acceleration() {
        let s = BreakoutMicro::default();
        let mut m = base_metrics();
        m.momentum_1s  = 0.010;   // large 1s spike
        m.momentum_5s  = 0.003;   // ratio = 3.33 > 2.5
        m.imbalance_1s = 0.5;     // strong buy volume
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::BuyCandidate),
            "Expected BuyCandidate, got {:?}: {}", d.action, d.reason);
    }

    #[test]
    fn test_breakout_wait_ratio_too_low() {
        let s = BreakoutMicro::default();
        let mut m = base_metrics();
        m.momentum_1s  = 0.004;   // ratio = 4/3 = 1.33 < 2.5
        m.momentum_5s  = 0.003;
        m.imbalance_1s = 0.5;
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    #[test]
    fn test_breakout_wait_weak_imbalance() {
        let s = BreakoutMicro::default();
        let mut m = base_metrics();
        m.momentum_1s  = 0.010;
        m.momentum_5s  = 0.003;
        m.imbalance_1s = 0.1;  // below 0.3 threshold
        let t = clean_truth();
        let d = s.evaluate(&m, &t, None, None);
        assert!(matches!(d.action, StrategyAction::Wait));
    }

    // ── StrategyEngine ────────────────────────────────────────────────────────

    #[test]
    fn test_engine_all_enabled_by_default() {
        let engine = StrategyEngine::new();
        for id in ALL_STRATEGIES {
            assert!(engine.is_enabled(id), "{} should be enabled by default", id);
        }
    }

    #[test]
    fn test_engine_disabled_strategy_never_selected() {
        let mut engine = StrategyEngine::new();
        // Disable MomentumMicro
        engine.set_enabled(&StrategyId::MomentumMicro, false);
        // Good momentum conditions → MomentumMicro would normally fire
        let m = base_metrics();
        let t = clean_truth();
        let (selected, _all) = engine.evaluate(&m, &t);
        assert_ne!(selected.strategy_id, StrategyId::MomentumMicro,
            "Disabled strategy must not be selected");
    }

    #[test]
    fn test_engine_disabled_strategy_shows_in_all_decisions() {
        let mut engine = StrategyEngine::new();
        engine.set_enabled(&StrategyId::BreakoutMicro, false);
        let m = base_metrics();
        let t = clean_truth();
        let (_selected, all) = engine.evaluate(&m, &t);
        let breakout_entry = all.iter().find(|d| d.strategy_id == StrategyId::BreakoutMicro);
        assert!(breakout_entry.is_some(), "Disabled strategy should appear in all_decisions");
        assert_eq!(breakout_entry.unwrap().reason, "disabled");
    }

    #[test]
    fn test_engine_highest_confidence_wins() {
        let mut engine = StrategyEngine::new();
        // Conditions that strongly favour BreakoutMicro over MomentumMicro
        let mut m = base_metrics();
        m.momentum_1s  = 0.010;   // high acceleration
        m.momentum_5s  = 0.003;   // ratio = 3.33
        m.imbalance_1s = 0.8;     // very strong
        m.momentum_3s  = 0.005;
        m.spread_bps   = 1.0;
        let t = clean_truth();
        let (selected, all) = engine.evaluate(&m, &t);
        // All three strategies see these conditions; BreakoutMicro should have high confidence
        // We verify the engine picks exactly one and it's actionable
        assert!(selected.action.is_actionable(),
            "Engine should select an actionable strategy: {:?}", selected.action);
        // Confidence of selected must be ≥ all others that are actionable
        for d in &all {
            if d.action.is_actionable() && d.strategy_id != selected.strategy_id {
                assert!(selected.confidence >= d.confidence,
                    "Selected {} (conf={:.3}) must dominate {} (conf={:.3})",
                    selected.strategy_id, selected.confidence,
                    d.strategy_id, d.confidence);
            }
        }
    }

    #[test]
    fn test_engine_tie_uses_priority_order() {
        // Build two strategies with exactly equal confidence output.
        // We can simulate this by checking priority tiebreaking logic directly.
        // MomentumMicro priority (1) < MeanReversionMicro (2) < BreakoutMicro (3)
        // Phase1SpotDaySwing priority (0) beats all.
        let a = StrategyDecision {
            strategy_id: StrategyId::MeanReversionMicro,
            action:      StrategyAction::BuyCandidate,
            confidence:  0.5,
            reason:      "test".into(),
        };
        let b = StrategyDecision {
            strategy_id: StrategyId::MomentumMicro,
            action:      StrategyAction::BuyCandidate,
            confidence:  0.5,
            reason:      "test".into(),
        };
        // Simulate the engine's reduce logic:
        let winner = [a, b].into_iter().reduce(|best, next| {
            let bp = best.strategy_id.priority();
            let np = next.strategy_id.priority();
            if next.confidence > best.confidence || (next.confidence == best.confidence && np < bp) {
                next
            } else { best }
        }).unwrap();
        assert_eq!(winner.strategy_id, StrategyId::MomentumMicro,
            "MomentumMicro (priority 1) must win tie vs MeanReversionMicro (priority 2)");
    }

    #[test]
    fn test_phase1_has_highest_priority() {
        assert_eq!(StrategyId::Phase1SpotDaySwing.priority(), 0,
            "Phase1SpotDaySwing must have priority 0 (highest)");
        assert!(StrategyId::Phase1SpotDaySwing.priority() < StrategyId::MomentumMicro.priority());
    }

    #[test]
    fn test_engine_returns_wait_when_no_candidates() {
        let mut engine = StrategyEngine::new();
        let mut m = base_metrics();
        // Conditions where nothing should fire
        m.momentum_1s  = -0.005;
        m.momentum_3s  = -0.005;
        m.momentum_5s  = -0.005;
        m.imbalance_1s = -0.8;
        let t = clean_truth();
        let (selected, _) = engine.evaluate(&m, &t);
        assert!(!selected.action.is_actionable(),
            "No candidates should mean Wait/StandDown: {:?}", selected.action);
    }

    #[test]
    fn test_engine_no_strategies_enabled_returns_stand_down() {
        let mut engine = StrategyEngine::new();
        for id in ALL_STRATEGIES {
            engine.set_enabled(id, false);
        }
        let m = base_metrics();
        let t = clean_truth();
        let (selected, _) = engine.evaluate(&m, &t);
        // Should not be actionable — all disabled
        assert!(!selected.action.is_actionable());
    }

    #[test]
    fn test_engine_entry_state_tracking() {
        let mut engine = StrategyEngine::new();
        assert!(engine.active_strategy().is_none());
        engine.on_entry_submitted(&StrategyId::MomentumMicro, 50000.0);
        assert_eq!(engine.active_strategy(), Some(StrategyId::MomentumMicro));
        engine.on_exit_submitted(&StrategyId::MomentumMicro, 50010.0);
        assert!(engine.active_strategy().is_none());
    }

    #[test]
    fn test_engine_clear_all_entry_state() {
        let mut engine = StrategyEngine::new();
        engine.on_entry_submitted(&StrategyId::MomentumMicro, 50000.0);
        engine.on_entry_submitted(&StrategyId::BreakoutMicro, 50001.0);
        engine.clear_all_entry_state();
        assert!(engine.active_strategy().is_none());
    }

    // ── to_signal_result ─────────────────────────────────────────────────────

    #[test]
    fn test_to_signal_result_buy() {
        let d = StrategyDecision {
            strategy_id: StrategyId::MomentumMicro,
            action:      StrategyAction::BuyCandidate,
            confidence:  0.7,
            reason:      "test".into(),
        };
        let sr = d.to_signal_result(&base_metrics());
        assert!(matches!(sr.decision, crate::signal::SignalDecision::Buy));
        assert!(sr.reason.contains("MomentumMicro"));
    }

    #[test]
    fn test_to_signal_result_exit_stop_loss() {
        let d = StrategyDecision {
            strategy_id: StrategyId::BreakoutMicro,
            action:      StrategyAction::ExitCandidate { reason: "STOP_LOSS".into() },
            confidence:  1.0,
            reason:      "test".into(),
        };
        let sr = d.to_signal_result(&base_metrics());
        assert!(matches!(sr.decision, crate::signal::SignalDecision::Exit { reason: crate::signal::ExitReason::StopLoss }));
    }

    #[test]
    fn test_to_signal_result_wait() {
        let d = StrategyDecision {
            strategy_id: StrategyId::MomentumMicro,
            action:      StrategyAction::Wait,
            confidence:  0.0,
            reason:      "waiting".into(),
        };
        let sr = d.to_signal_result(&base_metrics());
        assert!(matches!(sr.decision, crate::signal::SignalDecision::Hold));
    }
}

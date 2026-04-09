// strategy.rs
//
// Strategy layer. Multiple named strategies evaluate the same live feed
// snapshot and return a decision. The StrategyEngine selects the best
// eligible (enabled, highest confidence) and returns it for the rest of
// the pipeline.
//
// Design principles:
//   - Strategy trait is pure: evaluate() takes plain data, returns StrategyDecision.
//   - No I/O, no locks, no async inside strategies themselves.
//   - StrategyEngine is the only thing that holds state (enable flags, entry hints).
//   - All strategies share the same risk/authority/execution pipeline.
//   - Priority order (for tie-breaking) is defined by the order in ALL_STRATEGIES.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::{debug, info};

use crate::reconciler::TruthState;
use crate::signal::SignalMetrics;

// ── Strategy ID ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StrategyId {
    MomentumMicro,
    MeanReversionMicro,
    BreakoutMicro,
}

impl StrategyId {
    pub fn as_str(&self) -> &'static str {
        match self {
            StrategyId::MomentumMicro      => "MomentumMicro",
            StrategyId::MeanReversionMicro => "MeanReversionMicro",
            StrategyId::BreakoutMicro      => "BreakoutMicro",
        }
    }

    /// Deterministic priority for tie-breaking (lower = higher priority).
    pub fn priority(&self) -> u8 {
        match self {
            StrategyId::MomentumMicro      => 0,
            StrategyId::MeanReversionMicro => 1,
            StrategyId::BreakoutMicro      => 2,
        }
    }
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Trending,
    Ranging,
    HighVolatility,
    LowVolatility,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Trending => write!(f, "Trending"),
            MarketRegime::Ranging => write!(f, "Ranging"),
            MarketRegime::HighVolatility => write!(f, "HighVolatility"),
            MarketRegime::LowVolatility => write!(f, "LowVolatility"),
        }
    }
}

fn detect_regime(metrics: &SignalMetrics) -> MarketRegime {
    let trend_strength = metrics.momentum_5s.abs();
    let short_impulse = metrics.momentum_1s.abs();
    let vol_proxy = (short_impulse + metrics.spread_bps / 10_000.0).max(0.0);

    if vol_proxy >= 0.0009 {
        MarketRegime::HighVolatility
    } else if vol_proxy <= 0.00015 {
        MarketRegime::LowVolatility
    } else if trend_strength >= 0.00018 {
        MarketRegime::Trending
    } else {
        MarketRegime::Ranging
    }
}

fn strategy_regime_fit(strategy: &StrategyId, regime: MarketRegime) -> bool {
    match strategy {
        StrategyId::MomentumMicro => matches!(regime, MarketRegime::Trending | MarketRegime::HighVolatility),
        StrategyId::MeanReversionMicro => matches!(regime, MarketRegime::Ranging | MarketRegime::LowVolatility),
        StrategyId::BreakoutMicro => matches!(regime, MarketRegime::Trending | MarketRegime::HighVolatility),
    }
}

fn confidence_score(metrics: &SignalMetrics, raw_signal: f64, regime_fit: bool) -> f64 {
    let momentum = (metrics.momentum_5s.abs() / 0.0015).clamp(0.0, 1.0);
    let imbalance = ((metrics.imbalance_1s + metrics.imbalance_3s + metrics.imbalance_5s) / 1.8).clamp(0.0, 1.0);
    let spread_quality = (1.0 - metrics.spread_bps / 8.0).clamp(0.0, 1.0);
    let volatility_quality = (1.0 - ((metrics.momentum_1s.abs() - 0.0007).max(0.0) / 0.0015)).clamp(0.0, 1.0);
    let recency = (1.0 - metrics.feed_age_ms / 3_000.0).clamp(0.0, 1.0);
    let regime = if regime_fit { 1.0 } else { 0.0 };
    ((raw_signal + momentum + imbalance + spread_quality + volatility_quality + recency + regime) / 7.0).clamp(0.0, 1.0)
}

/// All strategies in deterministic priority order.
pub const ALL_STRATEGIES: &[StrategyId] = &[
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
    pub regime:      MarketRegime,
}

impl StrategyDecision {
    fn wait(id: StrategyId, reason: impl Into<String>) -> Self {
        Self { strategy_id: id, action: StrategyAction::Wait, confidence: 0.0, reason: reason.into(), regime: MarketRegime::Ranging }
    }
    fn stand_down(id: StrategyId, reason: impl Into<String>) -> Self {
        Self { strategy_id: id, action: StrategyAction::StandDown, confidence: 1.0, reason: reason.into(), regime: MarketRegime::Ranging }
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
        return Some(StrategyDecision::wait(id.clone(), format!("spread {:.1}bps > {:.1}", metrics.spread_bps, max_spread_bps)));
    }
    if metrics.mid_samples < 6 {
        return Some(StrategyDecision::wait(id.clone(), format!("insufficient mid samples: {}", metrics.mid_samples)));
    }
    if metrics.trade_samples < 4 {
        return Some(StrategyDecision::wait(id.clone(), format!("insufficient trade samples: {}", metrics.trade_samples)));
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
    let vol_boost = (metrics.momentum_1s.abs() * 1_500.0).clamp(0.0, 1.0);
    let dynamic_stop_pct = stop_loss_pct * (1.0 + vol_boost);
    let dynamic_tp_pct = take_profit_pct * (1.0 + (1.0 - vol_boost) * 0.4);

    let sl = entry * (1.0 - dynamic_stop_pct);
    if mid <= sl {
        return Some(StrategyDecision {
            strategy_id: id.clone(),
            action:      StrategyAction::ExitCandidate { reason: "STOP_LOSS".into() },
            confidence:  1.0,
            reason:      format!("mid {:.2} ≤ dynamic stop {:.2} (entry {:.2})", mid, sl, entry),
            regime:      detect_regime(metrics),
        });
    }
    let tp = entry * (1.0 + dynamic_tp_pct);
    if mid >= tp {
        return Some(StrategyDecision {
            strategy_id: id.clone(),
            action:      StrategyAction::ExitCandidate { reason: "TAKE_PROFIT".into() },
            confidence:  1.0,
            reason:      format!("mid {:.2} ≥ dynamic target {:.2} (entry {:.2})", mid, tp, entry),
            regime:      detect_regime(metrics),
        });
    }
    // Trailing and break-even protection only when in profit.
    if mid > entry {
        let trailing_floor = entry * (1.0 + dynamic_tp_pct * 0.30);
        if mid <= trailing_floor {
            return Some(StrategyDecision {
                strategy_id: id.clone(),
                action:      StrategyAction::ExitCandidate { reason: "TRAILING_STOP".into() },
                confidence:  0.9,
                reason:      format!("in-profit trailing floor hit: mid {:.2} ≤ {:.2}", mid, trailing_floor),
                regime:      detect_regime(metrics),
            });
        }
        if mid >= entry * (1.0 + dynamic_tp_pct * 0.5) && mid <= entry * 1.0002 {
            return Some(StrategyDecision {
                strategy_id: id.clone(),
                action:      StrategyAction::ExitCandidate { reason: "BREAK_EVEN_PROTECT".into() },
                confidence:  0.85,
                reason:      format!("break-even protection triggered near entry {:.2}", entry),
                regime:      detect_regime(metrics),
            });
        }
    }
    if metrics.momentum_1s < -0.0002 && metrics.imbalance_1s < -0.15 {
        return Some(StrategyDecision {
            strategy_id: id.clone(),
            action:      StrategyAction::ExitCandidate { reason: "EDGE_DECAY".into() },
            confidence:  0.8,
            reason:      format!("edge decay detected: m1={:+.5}, i1={:+.3}", metrics.momentum_1s, metrics.imbalance_1s),
            regime:      detect_regime(metrics),
        });
    }
    if let Some(et) = entry_time {
        if et.elapsed() >= max_hold {
            return Some(StrategyDecision {
                strategy_id: id.clone(),
                action:      StrategyAction::ExitCandidate { reason: "MAX_HOLD_TIME".into() },
                confidence:  0.8,
                reason:      format!("held {:.0}s ≥ limit {:.0}s", et.elapsed().as_secs_f64(), max_hold.as_secs_f64()),
                regime:      detect_regime(metrics),
            });
        }
    }
    None
}

// ── Strategy 1: MomentumMicro ─────────────────────────────────────────────────
//
// Entry: all three momentum windows positive + 1s imbalance above threshold.
// The existing signal engine logic, now as a named strategy.

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
        let regime = detect_regime(metrics);
        if !strategy_regime_fit(&StrategyId::MomentumMicro, regime) {
            return StrategyDecision::wait(StrategyId::MomentumMicro, format!("regime {} not suitable", regime));
        }
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
        let confidence = confidence_score(metrics, (mom_score + imb_score + spr_score) / 3.0, true);

        StrategyDecision {
            strategy_id: StrategyId::MomentumMicro,
            action:      StrategyAction::BuyCandidate,
            confidence,
            reason:      format!("MomentumMicro: m1={:+.5} m3={:+.5} m5={:+.5} i1={:+.3} i3={:+.3} conf={:.2}", m1, m3, m5, i1, i3, confidence),
            regime,
        }
    }
}

// ── Strategy 2: MeanReversionMicro ───────────────────────────────────────────
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
        let regime = detect_regime(metrics);
        if !strategy_regime_fit(&StrategyId::MeanReversionMicro, regime) {
            return StrategyDecision::wait(StrategyId::MeanReversionMicro, format!("regime {} not suitable", regime));
        }
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
        let confidence   = confidence_score(metrics, (trend_score + depth_score + return_score) / 3.0, true);

        StrategyDecision {
            strategy_id: StrategyId::MeanReversionMicro,
            action:      StrategyAction::BuyCandidate,
            confidence,
            reason:      format!("MeanReversionMicro: dip m1={:+.5} in uptrend m5={:+.5} buy-return i1={:+.3} conf={:.2}", m1, m5, i1, confidence),
            regime,
        }
    }
}

// ── Strategy 3: BreakoutMicro ─────────────────────────────────────────────────
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
        let regime = detect_regime(metrics);
        if !strategy_regime_fit(&StrategyId::BreakoutMicro, regime) {
            return StrategyDecision::wait(StrategyId::BreakoutMicro, format!("regime {} not suitable", regime));
        }

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
        let confidence  = confidence_score(metrics, (accel_score + imb_score + spr_score) / 3.0, true);

        StrategyDecision {
            strategy_id: StrategyId::BreakoutMicro,
            action:      StrategyAction::BuyCandidate,
            confidence,
            reason:      format!("BreakoutMicro: m1/m5 ratio={:.2} m1={:+.5} m5={:+.5} i1={:+.3} conf={:.2}", ratio, m1, m5, i1, confidence),
            regime,
        }
    }
}

// ── Per-strategy entry tracking ───────────────────────────────────────────────

#[derive(Default)]
struct StrategyState {
    entry_price_hint: Option<f64>,
    entry_time:       Option<Instant>,
}

// ── StrategyEngine ────────────────────────────────────────────────────────────

/// Holds all strategies, their enable flags, and per-strategy entry state.
/// Wrapped in Arc<Mutex<>> so the web UI can toggle enables at runtime.
pub struct StrategyEngine {
    strategies:  Vec<Box<dyn Strategy>>,
    enabled:     HashMap<StrategyId, bool>,
    entry_state: HashMap<StrategyId, StrategyState>,
    min_confidence: f64,
}

impl StrategyEngine {
    /// Create with all three strategies enabled.
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
        Self { strategies, enabled, entry_state, min_confidence: 0.72 }
    }

    // ── Enable/disable ────────────────────────────────────────────────────────

    pub fn set_enabled(&mut self, id: &StrategyId, enabled: bool) {
        if let Some(e) = self.enabled.get_mut(id) {
            *e = enabled;
            info!(strategy = %id, enabled, "[STRATEGY] Enable toggled");
        }
    }

    pub fn is_enabled(&self, id: &StrategyId) -> bool {
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
        if let Some(state) = self.entry_state.get_mut(id) {
            state.entry_price_hint = Some(price);
            state.entry_time       = Some(Instant::now());
        }
    }

    /// Called when position is closed (sell executed).
    pub fn on_exit_submitted(&mut self, id: &StrategyId) {
        if let Some(state) = self.entry_state.get_mut(id) {
            state.entry_price_hint = None;
            state.entry_time       = None;
        }
    }

    /// Clear entry state for all strategies (e.g. on position reconcile).
    pub fn clear_all_entry_state(&mut self) {
        for state in self.entry_state.values_mut() {
            state.entry_price_hint = None;
            state.entry_time       = None;
        }
    }

    /// Which strategy ID is currently "active" (has an open entry hint).
    pub fn active_strategy(&self) -> Option<StrategyId> {
        self.entry_state.iter()
            .find(|(_, s)| s.entry_price_hint.is_some())
            .map(|(id, _)| id.clone())
    }

    // ── Core evaluation ───────────────────────────────────────────────────────

    /// Evaluate all enabled strategies and select the best actionable decision.
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
        &self,
        metrics: &SignalMetrics,
        truth:   &TruthState,
    ) -> (StrategyDecision, Vec<StrategyDecision>) {
        let mut all_decisions: Vec<StrategyDecision> = Vec::new();
        let mut candidates: Vec<StrategyDecision>    = Vec::new();

        for strategy in &self.strategies {
            if !self.is_enabled(strategy.id()) {
                // Record a placeholder so the UI can show disabled state
                all_decisions.push(StrategyDecision {
                    strategy_id: strategy.id().clone(),
                    action:      StrategyAction::StandDown,
                    confidence:  0.0,
                    reason:      "disabled".into(),
                    regime:      detect_regime(metrics),
                });
                continue;
            }

            let state   = self.entry_state.get(strategy.id())
                              .map(|s| (s.entry_price_hint, s.entry_time))
                              .unwrap_or((None, None));
            let decision = strategy.evaluate(metrics, truth, state.0, state.1);

            debug!(
                strategy = %decision.strategy_id,
                action   = %decision.action,
                conf     = decision.confidence,
                reason   = %decision.reason,
                "[STRATEGY] Decision"
            );

            if decision.action.is_actionable() && decision.confidence >= self.min_confidence {
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

        let chosen = match selected {
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
                    .unwrap_or_else(|| StrategyDecision::stand_down(StrategyId::MomentumMicro, "no strategies enabled"))
            }
        };

        (chosen, all_decisions)
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
                    "STOP_LOSS"    => crate::signal::ExitReason::StopLoss,
                    "TAKE_PROFIT"  => crate::signal::ExitReason::TakeProfit,
                    "MAX_HOLD_TIME"=> crate::signal::ExitReason::MaxHoldTime,
                    "TRAILING_STOP" => crate::signal::ExitReason::TrailingStop,
                    "BREAK_EVEN_PROTECT" => crate::signal::ExitReason::BreakEvenProtect,
                    "EDGE_DECAY" => crate::signal::ExitReason::EdgeDecay,
                    _              => crate::signal::ExitReason::MaxHoldTime, // safe fallback
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
            reason:     format!("[{}|regime={}] {}", self.strategy_id, self.regime, self.reason),
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
        assert!(StrategyId::MomentumMicro.priority() < StrategyId::MeanReversionMicro.priority());
        assert!(StrategyId::MeanReversionMicro.priority() < StrategyId::BreakoutMicro.priority());
    }

    #[test]
    fn test_strategy_id_display() {
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
        // MomentumMicro priority (0) < MeanReversionMicro (1) < BreakoutMicro (2)
        let a = StrategyDecision {
            strategy_id: StrategyId::MeanReversionMicro,
            action:      StrategyAction::BuyCandidate,
            confidence:  0.5,
            reason:      "test".into(),
            regime:      MarketRegime::Trending,
        };
        let b = StrategyDecision {
            strategy_id: StrategyId::MomentumMicro,
            action:      StrategyAction::BuyCandidate,
            confidence:  0.5,
            reason:      "test".into(),
            regime:      MarketRegime::Trending,
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
            "MomentumMicro (priority 0) must win tie");
    }

    #[test]
    fn test_engine_returns_wait_when_no_candidates() {
        let engine = StrategyEngine::new();
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
        engine.on_exit_submitted(&StrategyId::MomentumMicro);
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
            regime:      MarketRegime::Trending,
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
            regime:      MarketRegime::Trending,
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
            regime:      MarketRegime::Ranging,
        };
        let sr = d.to_signal_result(&base_metrics());
        assert!(matches!(sr.decision, crate::signal::SignalDecision::Hold));
    }
}

// phase1.rs
//
// Phase 1: Spot Day+Swing Long-Only Strategy Engine.
//
// This is the primary strategy path, layered on top of the existing
// micro/agent system (which is preserved as legacy/experimental mode).
//
// Design:
//   - Timeframe stack: 4H trend bias, 1H setup validation, 15M execution trigger.
//     (Approximated from rolling duration windows on live tick data.)
//   - Regime states: TREND_UP, TREND_DOWN, RANGE, NO_TRADE.
//   - Entries only when regime = TREND_UP; all shorts are blocked.
//   - Two setup types: PullbackLong and BreakoutLong.
//   - Each setup outputs: side, mode (DAY or SWING), entry, stop, target, setup_quality.
//   - Sizing from stop distance and fixed dollar risk — no vibe sizing.
//   - Position management: stop loss, take profit, break-even, trailing stop, time exit.
//   - Block reason is always explicit; no generic HOLD.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use tracing::debug;

use crate::reconciler::TruthState;
use crate::signal::SignalMetrics;

// ── Regime ────────────────────────────────────────────────────────────────────

/// Phase 1 market regime states.
///
/// Only TREND_UP allows entries. All other regimes block new longs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaySwingRegime {
    TrendUp,
    TrendDown,
    Range,
    NoTrade,
}

impl DaySwingRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TrendUp   => "TREND_UP",
            Self::TrendDown => "TREND_DOWN",
            Self::Range     => "RANGE",
            Self::NoTrade   => "NO_TRADE",
        }
    }
}

impl std::fmt::Display for DaySwingRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Setup type ────────────────────────────────────────────────────────────────

/// Setup types supported by Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupType {
    PullbackLong,
    BreakoutLong,
}

impl SetupType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PullbackLong => "PULLBACK_LONG",
            Self::BreakoutLong => "BREAKOUT_LONG",
        }
    }
}

impl std::fmt::Display for SetupType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Trade mode ────────────────────────────────────────────────────────────────

/// Trade mode: intraday (DAY) or multi-session swing (SWING).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeMode {
    Day,
    Swing,
}

impl TradeMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Day   => "DAY",
            Self::Swing => "SWING",
        }
    }
}

impl std::fmt::Display for TradeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Setup output ──────────────────────────────────────────────────────────────

/// A Phase 1 trade setup with all fields required by the execution path.
#[derive(Debug, Clone)]
pub struct Phase1Setup {
    pub setup_type:    SetupType,
    /// Always "BUY" in Phase 1 (long-only).
    pub side:          &'static str,
    pub mode:          TradeMode,
    /// Suggested entry price (ask price at signal time).
    pub entry:         f64,
    /// Hard stop loss level.
    pub stop:          f64,
    /// Take-profit target.
    pub target:        f64,
    /// Setup quality score in [0.0, 1.0].
    pub setup_quality: f64,
    /// Populated when the setup is blocked or unavailable.
    pub block_reason:  Option<String>,
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the Phase 1 engine.
#[derive(Debug, Clone)]
pub struct Phase1Config {
    /// Fixed dollar risk per trade (e.g. $25).
    /// All positions are sized so that hitting the stop costs exactly this amount.
    pub dollar_risk_per_trade: f64,

    /// Longer-term trend bias window in seconds.
    ///
    /// Acts as the "4H bias" filter in the multi-timeframe stack.
    /// NOTE: The live feed only retains a rolling window of recent ticks
    /// (max_window ≈ a few minutes). This value is therefore a *relative*
    /// lookback within the available history — not 4 literal hours.
    /// Use 240.0 (default) for the widest lookback the feed can sustain.
    pub trend_window_secs: f64,

    /// Setup validation window in seconds.
    ///
    /// Acts as the "1H setup" filter — a medium-term trend confirmation.
    /// Default 60.0 seconds; relative to the available tick history.
    pub setup_window_secs: f64,

    /// Execution trigger window in seconds.
    ///
    /// Acts as the "15M execution" trigger — short-term momentum confirmation.
    /// Default 15.0 seconds; relative to the available tick history.
    pub trigger_window_secs: f64,

    /// Minimum net drift (fraction) over trend_window to declare TREND_UP/TREND_DOWN.
    /// E.g. 0.0008 = 0.08% over the trend window.
    pub min_trend_drift: f64,

    /// Maximum spread in basis points allowed at entry.
    pub max_spread_bps: f64,

    /// Pullback setup: stop loss distance below entry (fraction).
    pub pullback_stop_pct: f64,

    /// Pullback setup: take-profit distance above entry (fraction).
    pub pullback_tp_pct: f64,

    /// Breakout setup: stop loss distance below entry (fraction).
    pub breakout_stop_pct: f64,

    /// Breakout setup: take-profit distance above entry (fraction).
    pub breakout_tp_pct: f64,

    /// DAY mode: maximum hold duration before time-based exit.
    pub day_max_hold: Duration,

    /// SWING mode: maximum hold duration before time-based exit.
    pub swing_max_hold: Duration,

    /// Break-even trigger: if price rises this fraction above entry, move stop to entry.
    pub break_even_trigger_pct: f64,

    /// Trailing stop: trail this fraction below the position's high-water mark.
    pub trailing_stop_pct: f64,

    /// Minimum mid-price samples required before computing regime or setups.
    pub min_samples: usize,

    /// Pullback: maximum allowed flush depth (fraction below trigger-window entry).
    /// Pullbacks deeper than this are ignored (too risky / might be a trend break).
    pub pullback_max_flush_pct: f64,

    /// Pullback: threshold below which a pullback becomes SWING (vs DAY) mode.
    /// Pullback depth < this → DAY; depth >= this → SWING.
    pub pullback_swing_depth_pct: f64,

    /// Breakout: minimum 1s momentum required to confirm acceleration.
    /// Prevents entering on drifting price action with no real volume.
    pub breakout_min_momentum_1s: f64,

    /// For Range regime: both 4H and 1H must be below (min_trend_drift * this factor).
    /// Lower factor = stricter range qualification.
    pub range_setup_factor: f64,

    /// Breakout: minimum buy imbalance required to confirm the move.
    ///
    /// Default 0.20 (20% net buy aggression). MICRO_ACTIVE lowers this to 0.12
    /// to allow earlier entry on smaller-balance moves without removing the
    /// imbalance gate entirely.
    pub breakout_min_imbalance: f64,
}

impl Default for Phase1Config {
    fn default() -> Self {
        Self {
            dollar_risk_per_trade:  25.0,
            trend_window_secs:     240.0,
            setup_window_secs:      60.0,
            trigger_window_secs:    15.0,
            min_trend_drift:        0.0008, // 0.08% drift to confirm trend direction
            max_spread_bps:         8.0,
            pullback_stop_pct:      0.0030, // 0.30% below entry
            pullback_tp_pct:        0.0090, // 0.90% above entry (3:1 R:R)
            breakout_stop_pct:      0.0025, // 0.25% below entry
            breakout_tp_pct:        0.0120, // 1.20% above entry (4.8:1 R:R)
            day_max_hold:           Duration::from_secs(3_600),   // 1 hour
            swing_max_hold:         Duration::from_secs(86_400),  // 24 hours
            break_even_trigger_pct: 0.0050, // 0.50% rise triggers break-even
            trailing_stop_pct:      0.0040, // trail 0.40% below high-water mark
            min_samples:            10,
            pullback_max_flush_pct:  0.012, // ignore pullbacks > 1.2% (possible trend break)
            pullback_swing_depth_pct: 0.003, // pullbacks < 0.3% are DAY; deeper are SWING
            breakout_min_momentum_1s: 0.0001, // minimum 1s momentum to confirm breakout
            range_setup_factor:      0.5,   // 1H must be < min_trend_drift * 0.5 for RANGE
            breakout_min_imbalance:  0.20,  // 20% net buy aggression to confirm breakout
        }
    }
}

impl Phase1Config {
    /// MICRO_ACTIVE preset: lower thresholds for high-frequency small-balance trading.
    ///
    /// Changes from default (behavior-only; all safety mechanics preserved):
    ///   trigger_window_secs      15.0  → 8.0    faster trigger acceptance
    ///   min_trend_drift          0.0008 → 0.0004 earlier momentum confirmation
    ///   min_samples                10  → 5      shorter confirmation window
    ///   breakout_min_momentum_1s 0.0001→ 0.00005 fewer "wait" holds
    ///   breakout_min_imbalance    0.20 → 0.12   more permissive breakout entry
    pub fn micro_active() -> Self {
        Self {
            trigger_window_secs:     8.0,
            min_trend_drift:         0.0004,
            min_samples:             5,
            breakout_min_momentum_1s: 0.00005,
            breakout_min_imbalance:  0.12,
            ..Self::default()
        }
    }
}

// ── Position state ────────────────────────────────────────────────────────────

/// Runtime state for an open Phase 1 position.
#[derive(Debug, Clone)]
pub struct Phase1PositionState {
    pub entry_price:          f64,
    pub stop_price:           f64,
    pub target_price:         f64,
    pub mode:                 TradeMode,
    pub setup_type:           SetupType,
    pub opened_at:            Instant,
    /// Highest mid price seen since entry (for trailing stop).
    pub high_water_mark:      f64,
    /// True once the stop has been moved to break-even.
    pub break_even_triggered: bool,
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result returned by Phase1Engine::evaluate().
#[derive(Debug)]
pub enum Phase1Result {
    /// A new entry setup is ready; includes full setup details for sizing and submission.
    Setup(Phase1Setup),
    /// Currently in a position; no new entry. Position is being managed.
    HoldPosition {
        regime:         DaySwingRegime,
        position_note:  String,
    },
    /// No trade this cycle. Exact block reason is always populated.
    NoTrade {
        regime:       DaySwingRegime,
        block_reason: String,
    },
    /// Exit signal: position management triggered an exit. Reason is explicit.
    Exit { reason: String },
}

impl Phase1Result {
    fn blocked(reason: impl Into<String>) -> Self {
        Self::NoTrade {
            regime:       DaySwingRegime::NoTrade,
            block_reason: reason.into(),
        }
    }
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Phase 1 Spot Day+Swing Long-Only engine.
///
/// Maintains its own rolling mid-price buffer for multi-timeframe analysis.
/// Called from StrategyEngine which owns it as a direct field.
pub struct Phase1Engine {
    pub cfg: Phase1Config,
    /// Rolling mid-price samples: (timestamp, mid).
    mid_history: VecDeque<(Instant, f64)>,
    /// Current open position tracked by this engine (None when flat).
    pub open_position: Option<Phase1PositionState>,
    /// Last regime computed (exposed for UI).
    pub last_regime: DaySwingRegime,
    /// Last block reason (exposed for UI, always explicit).
    pub last_block_reason: String,
    /// Last setup detected (None when no setup was available).
    pub last_setup: Option<Phase1Setup>,
}

impl Phase1Engine {
    pub fn new(cfg: Phase1Config) -> Self {
        Self {
            cfg,
            mid_history:       VecDeque::with_capacity(2048),
            open_position:     None,
            last_regime:       DaySwingRegime::NoTrade,
            last_block_reason: "engine_starting".into(),
            last_setup:        None,
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Ingest a new mid-price sample and prune stale history.
    fn update(&mut self, mid: f64) {
        let now = Instant::now();
        self.mid_history.push_back((now, mid));
        // Keep data up to the longest window + a generous buffer.
        let max_age = Duration::from_secs_f64(self.cfg.trend_window_secs + 120.0);
        while self.mid_history.front().map(|(t, _)| t.elapsed() > max_age).unwrap_or(false) {
            self.mid_history.pop_front();
        }
    }

    /// Return the mid price that was current approximately `seconds` ago.
    /// Falls back to the oldest available sample if history is shorter.
    fn mid_ago(&self, seconds: f64) -> Option<f64> {
        let target_duration = Duration::from_secs_f64(seconds);
        let now = Instant::now();
        // Walk from newest to oldest; find the first sample that is at least
        // `target_duration` old (i.e. the one just inside our window).
        self.mid_history
            .iter()
            .rev()
            .find(|(t, _)| now.duration_since(*t) >= target_duration)
            .or_else(|| self.mid_history.front())
            .map(|(_, mid)| *mid)
    }

    /// Net drift over `window_secs` as a fraction: (now - past) / past.
    fn drift(&self, window_secs: f64, current_mid: f64) -> Option<f64> {
        let past = self.mid_ago(window_secs)?;
        if past <= 0.0 { return None; }
        Some((current_mid - past) / past)
    }

    // ── Regime detection ─────────────────────────────────────────────────────

    /// Detect market regime from the multi-timeframe stack.
    ///
    /// Rules:
    ///   TREND_UP   : 4H window net positive AND 1H window net positive.
    ///   TREND_DOWN : 4H window net negative AND 1H window net negative.
    ///   RANGE      : 4H and 1H both flat (drift below threshold).
    ///   NO_TRADE   : conflicting timeframes, or insufficient data.
    pub fn detect_regime(&self, mid: f64) -> (DaySwingRegime, String) {
        if self.mid_history.len() < self.cfg.min_samples {
            return (
                DaySwingRegime::NoTrade,
                format!("warming_up:{}/{}_samples", self.mid_history.len(), self.cfg.min_samples),
            );
        }

        let td = match self.drift(self.cfg.trend_window_secs, mid) {
            Some(d) => d,
            None    => return (DaySwingRegime::NoTrade, "no_trend_history".into()),
        };
        let sd = match self.drift(self.cfg.setup_window_secs, mid) {
            Some(d) => d,
            None    => return (DaySwingRegime::NoTrade, "no_setup_history".into()),
        };

        let min = self.cfg.min_trend_drift;

        if td >= min && sd >= 0.0 {
            (
                DaySwingRegime::TrendUp,
                format!("4H_drift={:+.4}% 1H_drift={:+.4}%", td * 100.0, sd * 100.0),
            )
        } else if td <= -min && sd <= 0.0 {
            (
                DaySwingRegime::TrendDown,
                format!("4H_drift={:+.4}% 1H_drift={:+.4}%", td * 100.0, sd * 100.0),
            )
        } else if td.abs() < min && sd.abs() < min * self.cfg.range_setup_factor {
            (
                DaySwingRegime::Range,
                format!("4H_drift={:+.4}% 1H_drift={:+.4}% (flat)", td * 100.0, sd * 100.0),
            )
        } else {
            (
                DaySwingRegime::NoTrade,
                format!("conflicting_timeframes:4H={:+.4}%_1H={:+.4}%", td * 100.0, sd * 100.0),
            )
        }
    }

    // ── Setup detection ───────────────────────────────────────────────────────

    /// PULLBACK_LONG: uptrend on 1H, 15M execution window has dipped and is
    /// showing early signs of recovery (5s momentum still positive but
    /// shorter-window momentum is transitioning).
    fn detect_pullback_long(&self, metrics: &SignalMetrics) -> Option<Phase1Setup> {
        let mid = metrics.mid;

        // 1H drift must confirm upward setup.
        let sd = self.drift(self.cfg.setup_window_secs, mid)?;
        if sd < self.cfg.min_trend_drift { return None; }

        // Trigger: 15M window must show a dip (negative drift but not a catastrophic flush).
        let td = self.drift(self.cfg.trigger_window_secs, mid)?;
        if td >= 0.0 { return None; }             // not pulling back
        // Ignore if flush is deeper than configured max (may be a trend break, not a pullback).
        if td < -self.cfg.pullback_max_flush_pct { return None; }

        // Medium-term momentum (5s) must still be positive — trend intact.
        if metrics.momentum_5s <= 0.0 { return None; }

        let entry  = metrics.ask;
        let stop   = entry * (1.0 - self.cfg.pullback_stop_pct);
        let target = entry * (1.0 + self.cfg.pullback_tp_pct);

        let trend_score    = (sd / self.cfg.min_trend_drift).min(1.0).max(0.0);
        let pullback_depth = ((-td) / 0.006).min(1.0).max(0.0);
        let spread_score   = (1.0 - metrics.spread_bps / self.cfg.max_spread_bps).max(0.0);
        let quality = (trend_score * 0.45 + pullback_depth * 0.30 + spread_score * 0.25)
            .clamp(0.0, 1.0);

        // Shallow pullback (< pullback_swing_depth_pct) = DAY trade; deeper = SWING.
        let mode = if td.abs() < self.cfg.pullback_swing_depth_pct { TradeMode::Day } else { TradeMode::Swing };

        Some(Phase1Setup {
            setup_type: SetupType::PullbackLong,
            side:       "BUY",
            mode,
            entry,
            stop,
            target,
            setup_quality: quality,
            block_reason: None,
        })
    }

    /// BREAKOUT_LONG: 15M execution window shows price accelerating above the
    /// recent range with confirmed buy volume. Both 1H and 15M must be positive.
    fn detect_breakout_long(&self, metrics: &SignalMetrics) -> Option<Phase1Setup> {
        let mid = metrics.mid;

        // 1H drift must confirm upward trend.
        let sd = self.drift(self.cfg.setup_window_secs, mid)?;
        if sd < self.cfg.min_trend_drift { return None; }

        // 15M window must show positive and accelerating drift.
        let td = self.drift(self.cfg.trigger_window_secs, mid)?;
        if td < self.cfg.min_trend_drift * 0.5 { return None; }

        // Strong buy imbalance required — volume must confirm the move.
        if metrics.imbalance_1s < self.cfg.breakout_min_imbalance { return None; }

        // 1s momentum must meet the configured minimum to confirm acceleration.
        // Without acceleration, the "breakout" could be a false start.
        if metrics.momentum_1s < self.cfg.breakout_min_momentum_1s { return None; }

        let entry  = metrics.ask;
        let stop   = entry * (1.0 - self.cfg.breakout_stop_pct);
        let target = entry * (1.0 + self.cfg.breakout_tp_pct);

        let momentum_score  = (metrics.momentum_1s / 0.001).min(1.0).max(0.0);
        let imbalance_score = (metrics.imbalance_1s / 0.5).min(1.0).max(0.0);
        let trend_score     = (td / self.cfg.min_trend_drift).min(1.0).max(0.0);
        let spread_score    = (1.0 - metrics.spread_bps / self.cfg.max_spread_bps).max(0.0);
        let quality = (momentum_score  * 0.30
            + imbalance_score * 0.30
            + trend_score     * 0.20
            + spread_score    * 0.20)
            .clamp(0.0, 1.0);

        // Breakouts are always DAY trades in Phase 1.
        Some(Phase1Setup {
            setup_type: SetupType::BreakoutLong,
            side:       "BUY",
            mode:       TradeMode::Day,
            entry,
            stop,
            target,
            setup_quality: quality,
            block_reason: None,
        })
    }

    // ── Position management ───────────────────────────────────────────────────

    /// Manage an open position on each evaluation tick.
    ///
    /// Returns `Some(exit_reason)` when an exit should be triggered.
    /// Exit reasons: STOP_LOSS, TAKE_PROFIT, BREAK_EVEN_STOP, TRAILING_STOP, TIME_EXIT.
    pub fn manage_position(&mut self, metrics: &SignalMetrics) -> Option<String> {
        let pos = self.open_position.as_mut()?;
        let mid = metrics.mid;

        // Update high-water mark.
        if mid > pos.high_water_mark {
            pos.high_water_mark = mid;
        }

        // Break-even: when price rises above trigger, move stop to entry cost.
        let be_trigger = pos.entry_price * (1.0 + self.cfg.break_even_trigger_pct);
        if !pos.break_even_triggered && mid >= be_trigger {
            pos.stop_price = pos.entry_price;
            pos.break_even_triggered = true;
            debug!(
                entry = pos.entry_price, be_trigger,
                "[PHASE1] Break-even triggered — stop moved to entry"
            );
        }

        // Trailing stop: trail below high-water mark.
        let trail_level = pos.high_water_mark * (1.0 - self.cfg.trailing_stop_pct);
        if trail_level > pos.stop_price {
            pos.stop_price = trail_level;
        }

        // Check stop hit.
        if mid <= pos.stop_price {
            let reason = if pos.break_even_triggered {
                "BREAK_EVEN_STOP"
            } else if pos.stop_price >= pos.entry_price {
                "TRAILING_STOP"
            } else {
                "STOP_LOSS"
            };
            return Some(reason.into());
        }

        // Check take-profit hit.
        if mid >= pos.target_price {
            return Some("TAKE_PROFIT".into());
        }

        // Time-based exit.
        let max_hold = match pos.mode {
            TradeMode::Day   => self.cfg.day_max_hold,
            TradeMode::Swing => self.cfg.swing_max_hold,
        };
        if pos.opened_at.elapsed() >= max_hold {
            return Some("TIME_EXIT".into());
        }

        None
    }

    // ── Risk sizing ───────────────────────────────────────────────────────────

    /// Compute position size from stop distance and fixed dollar risk.
    ///
    /// size = dollar_risk / (entry - stop)
    /// This is the only sizing method for Phase 1 — no vibe sizing.
    pub fn size_from_risk(&self, entry: f64, stop: f64) -> f64 {
        let stop_distance = (entry - stop).abs();
        if stop_distance <= 0.0 {
            return 0.0;
        }
        self.cfg.dollar_risk_per_trade / stop_distance
    }

    // ── Main evaluation ───────────────────────────────────────────────────────

    /// Evaluate the current market snapshot.
    ///
    /// Execution path: setup → regime check → guards → setup detection → result.
    /// Block reason is always explicit — never returns a generic HOLD.
    pub fn evaluate(
        &mut self,
        metrics: &SignalMetrics,
        truth:   &TruthState,
    ) -> Phase1Result {
        // Ingest current mid price into our rolling buffer.
        self.update(metrics.mid);

        // ── Common guards ─────────────────────────────────────────────────────
        if truth.state_dirty {
            let r = "state_dirty".to_string();
            self.last_block_reason = r.clone();
            self.last_regime = DaySwingRegime::NoTrade;
            return Phase1Result::blocked(r);
        }
        if truth.recon_in_progress {
            let r = "recon_in_progress".to_string();
            self.last_block_reason = r.clone();
            self.last_regime = DaySwingRegime::NoTrade;
            return Phase1Result::blocked(r);
        }
        if metrics.feed_age_ms > 5_000.0 {
            let r = format!("feed_stale:{:.0}ms", metrics.feed_age_ms);
            self.last_block_reason = r.clone();
            self.last_regime = DaySwingRegime::NoTrade;
            return Phase1Result::blocked(r);
        }
        if metrics.spread_bps > self.cfg.max_spread_bps {
            let r = format!(
                "spread_too_wide:{:.2}bps_limit:{:.2}bps",
                metrics.spread_bps, self.cfg.max_spread_bps
            );
            self.last_block_reason = r.clone();
            return Phase1Result::blocked(r);
        }

        // ── Compute regime ────────────────────────────────────────────────────
        let (regime, regime_note) = self.detect_regime(metrics.mid);
        self.last_regime = regime;

        // ── Manage open position if any ───────────────────────────────────────
        if !truth.position.is_flat() {
            if let Some(exit_reason) = self.manage_position(metrics) {
                self.last_block_reason = String::new();
                return Phase1Result::Exit { reason: exit_reason };
            }
            let note = format!("managing_open_position regime={}", regime.as_str());
            self.last_block_reason = note.clone();
            return Phase1Result::HoldPosition {
                regime,
                position_note: note,
            };
        }

        // Position is flat — clear any stale position state.
        if self.open_position.is_some() {
            self.open_position = None;
        }

        // ── Phase 1: entries only on TREND_UP ─────────────────────────────────
        if regime != DaySwingRegime::TrendUp {
            let r = format!("regime={}:{}", regime.as_str(), regime_note);
            self.last_block_reason = r.clone();
            return Phase1Result::NoTrade { regime, block_reason: r };
        }

        // Block if open BUY order already exists.
        let has_open_buy = truth.orders.values()
            .any(|r| r.side.eq_ignore_ascii_case("BUY") && r.status.is_active());
        if has_open_buy {
            let r = "open_buy_order_exists".to_string();
            self.last_block_reason = r.clone();
            return Phase1Result::NoTrade { regime, block_reason: r };
        }

        // ── Detect setups ─────────────────────────────────────────────────────
        let pullback = self.detect_pullback_long(metrics);
        let breakout = self.detect_breakout_long(metrics);

        // Select highest-quality setup.
        let best = match (pullback, breakout) {
            (Some(p), Some(b)) => {
                if b.setup_quality >= p.setup_quality { Some(b) } else { Some(p) }
            }
            (Some(p), None) => Some(p),
            (None, Some(b)) => Some(b),
            (None, None)    => None,
        };

        match best {
            Some(setup) => {
                debug!(
                    setup_type = setup.setup_type.as_str(),
                    mode       = setup.mode.as_str(),
                    entry      = setup.entry,
                    stop       = setup.stop,
                    target     = setup.target,
                    quality    = setup.setup_quality,
                    "[PHASE1] Setup detected"
                );
                self.last_setup = Some(setup.clone());
                self.last_block_reason = String::new();
                Phase1Result::Setup(setup)
            }
            None => {
                let r = format!(
                    "no_setup_in_TREND_UP:pullback=none breakout=none 15M_drift={:.4}%",
                    self.drift(self.cfg.trigger_window_secs, metrics.mid)
                        .unwrap_or(0.0) * 100.0
                );
                self.last_block_reason = r.clone();
                Phase1Result::NoTrade { regime, block_reason: r }
            }
        }
    }

    // ── Entry / exit hooks ────────────────────────────────────────────────────

    /// Called when a Phase 1 entry has been confirmed (order filled).
    pub fn on_entry_confirmed(&mut self, setup: &Phase1Setup) {
        self.open_position = Some(Phase1PositionState {
            entry_price:          setup.entry,
            stop_price:           setup.stop,
            target_price:         setup.target,
            mode:                 setup.mode,
            setup_type:           setup.setup_type,
            opened_at:            Instant::now(),
            high_water_mark:      setup.entry,
            break_even_triggered: false,
        });
    }

    /// Called when the Phase 1 position has been closed.
    pub fn on_exit_confirmed(&mut self) {
        self.open_position = None;
    }
}

impl Default for Phase1Engine {
    fn default() -> Self {
        Self::new(Phase1Config::default())
    }
}

// ── Status snapshot ───────────────────────────────────────────────────────────

/// Read-only snapshot of Phase1Engine state for UI display.
#[derive(Debug, Clone)]
pub struct Phase1Status {
    pub enabled:          bool,
    pub regime:           DaySwingRegime,
    pub block_reason:     String,
    /// Last setup detected (even if not taken).
    pub last_setup_type:  Option<SetupType>,
    pub last_setup_mode:  Option<TradeMode>,
    pub last_entry:       Option<f64>,
    pub last_stop:        Option<f64>,
    pub last_target:      Option<f64>,
    pub last_quality:     Option<f64>,
    /// True when a position is currently being managed by Phase 1.
    pub in_position:      bool,
    pub position_stop:    Option<f64>,
    pub position_target:  Option<f64>,
    pub break_even:       bool,
    pub high_water:       Option<f64>,
    /// Active behavior mode label (e.g. "MICRO_ACTIVE", "STANDARD").
    pub behavior_mode:    String,
    /// Effective trigger window in seconds (profile-adjusted).
    pub effective_trigger_window_secs: f64,
    /// Effective minimum trend drift (profile-adjusted).
    pub effective_min_trend_drift: f64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconciler::TruthState;
    use crate::signal::SignalMetrics;
    use std::time::Duration;

    fn base_metrics() -> SignalMetrics {
        SignalMetrics {
            bid:           50000.0,
            ask:           50001.0,
            mid:           50000.5,
            spread_bps:    2.0,
            momentum_1s:   0.0005,
            momentum_3s:   0.0004,
            momentum_5s:   0.0003,
            imbalance_1s:  0.40,
            imbalance_3s:  0.30,
            imbalance_5s:  0.20,
            feed_age_ms:   50.0,
            mid_samples:   10,
            trade_samples: 8,
        }
    }

    fn clean_truth() -> TruthState {
        let mut t = TruthState::new("BTCUSDT", 0.0);
        t.state_dirty       = false;
        t.recon_in_progress = false;
        t.last_reconciled_at = Some(std::time::Instant::now());
        t
    }

    #[test]
    fn test_regime_no_trade_on_startup() {
        let engine = Phase1Engine::default();
        let (regime, _) = engine.detect_regime(50000.0);
        // No history yet → should be NoTrade
        assert_eq!(regime, DaySwingRegime::NoTrade);
    }

    #[test]
    fn test_regime_as_str() {
        assert_eq!(DaySwingRegime::TrendUp.as_str(), "TREND_UP");
        assert_eq!(DaySwingRegime::TrendDown.as_str(), "TREND_DOWN");
        assert_eq!(DaySwingRegime::Range.as_str(), "RANGE");
        assert_eq!(DaySwingRegime::NoTrade.as_str(), "NO_TRADE");
    }

    #[test]
    fn test_setup_type_as_str() {
        assert_eq!(SetupType::PullbackLong.as_str(), "PULLBACK_LONG");
        assert_eq!(SetupType::BreakoutLong.as_str(), "BREAKOUT_LONG");
    }

    #[test]
    fn test_trade_mode_as_str() {
        assert_eq!(TradeMode::Day.as_str(), "DAY");
        assert_eq!(TradeMode::Swing.as_str(), "SWING");
    }

    #[test]
    fn test_size_from_risk() {
        let engine = Phase1Engine::default(); // dollar_risk = $25
        // Entry 50000, stop 49850 → distance $150 → size = 25/150 ≈ 0.1667
        let size = engine.size_from_risk(50000.0, 49850.0);
        assert!((size - 25.0 / 150.0).abs() < 1e-10, "size={}", size);
    }

    #[test]
    fn test_size_from_risk_zero_distance() {
        let engine = Phase1Engine::default();
        let size = engine.size_from_risk(50000.0, 50000.0);
        assert_eq!(size, 0.0);
    }

    #[test]
    fn test_evaluate_blocks_on_state_dirty() {
        let mut engine = Phase1Engine::default();
        let m = base_metrics();
        let mut t = clean_truth();
        t.state_dirty = true;
        let result = engine.evaluate(&m, &t);
        assert!(matches!(result, Phase1Result::NoTrade { regime: DaySwingRegime::NoTrade, .. }));
    }

    #[test]
    fn test_evaluate_blocks_on_stale_feed() {
        let mut engine = Phase1Engine::default();
        let mut m = base_metrics();
        m.feed_age_ms = 10_000.0;
        let t = clean_truth();
        let result = engine.evaluate(&m, &t);
        assert!(matches!(result, Phase1Result::NoTrade { regime: DaySwingRegime::NoTrade, .. }));
    }

    #[test]
    fn test_evaluate_blocks_on_wide_spread() {
        let mut engine = Phase1Engine::default();
        let mut m = base_metrics();
        m.spread_bps = 20.0; // above 8.0 limit
        let t = clean_truth();
        let result = engine.evaluate(&m, &t);
        assert!(matches!(result, Phase1Result::NoTrade { .. }));
        assert!(engine.last_block_reason.contains("spread_too_wide"));
    }

    #[test]
    fn test_evaluate_no_trade_warming_up() {
        let mut engine = Phase1Engine::default();
        let m = base_metrics();
        let t = clean_truth();
        // Only 1 sample — not enough for min_samples
        let result = engine.evaluate(&m, &t);
        assert!(matches!(result, Phase1Result::NoTrade { regime: DaySwingRegime::NoTrade, .. }));
        assert!(engine.last_block_reason.contains("warming_up") || engine.last_block_reason.contains("no_trend_history") || engine.last_block_reason.contains("regime=NO_TRADE"));
    }

    #[test]
    fn test_engine_starting_clears_after_warmup_samples_present() {
        let mut engine = Phase1Engine::default();
        let t = clean_truth();
        assert_eq!(engine.last_block_reason, "engine_starting");

        for i in 0..=engine.cfg.min_samples {
            let mut m = base_metrics();
            m.mid = 50_000.0 + i as f64;
            m.bid = m.mid - 1.0;
            m.ask = m.mid + 1.0;
            m.feed_age_ms = 10.0;
            let _ = engine.evaluate(&m, &t);
        }

        assert!(!engine.last_block_reason.contains("engine_starting"));
        assert!(!engine.last_block_reason.contains("warming_up"));
    }

    #[test]
    fn test_block_reason_never_empty_on_no_trade() {
        let mut engine = Phase1Engine::default();
        let m = base_metrics();
        let t = clean_truth();
        let result = engine.evaluate(&m, &t);
        if matches!(result, Phase1Result::NoTrade { .. }) {
            assert!(
                !engine.last_block_reason.is_empty(),
                "block_reason must never be empty on NoTrade"
            );
        }
    }

    #[test]
    fn test_position_management_stop_loss() {
        let cfg = Phase1Config {
            pullback_stop_pct: 0.0030,
            pullback_tp_pct:   0.0090,
            ..Default::default()
        };
        let mut engine = Phase1Engine::new(cfg);
        // Simulate confirmed entry at 50000, stop at 49850
        engine.open_position = Some(Phase1PositionState {
            entry_price:          50_000.0,
            stop_price:           49_850.0,
            target_price:         50_450.0,
            mode:                 TradeMode::Day,
            setup_type:           SetupType::PullbackLong,
            opened_at:            Instant::now(),
            high_water_mark:      50_000.0,
            break_even_triggered: false,
        });
        // Price drops to stop.
        let mut m = base_metrics();
        m.mid = 49_840.0;
        let exit = engine.manage_position(&m);
        assert_eq!(exit.as_deref(), Some("STOP_LOSS"));
    }

    #[test]
    fn test_position_management_take_profit() {
        let mut engine = Phase1Engine::default();
        engine.open_position = Some(Phase1PositionState {
            entry_price:          50_000.0,
            stop_price:           49_850.0,
            target_price:         50_450.0,
            mode:                 TradeMode::Day,
            setup_type:           SetupType::PullbackLong,
            opened_at:            Instant::now(),
            high_water_mark:      50_000.0,
            break_even_triggered: false,
        });
        let mut m = base_metrics();
        m.mid = 50_500.0; // above target
        let exit = engine.manage_position(&m);
        assert_eq!(exit.as_deref(), Some("TAKE_PROFIT"));
    }

    #[test]
    fn test_position_management_break_even() {
        let cfg = Phase1Config {
            break_even_trigger_pct: 0.005, // 0.5%
            trailing_stop_pct:      0.004, // 0.4%
            ..Default::default()
        };
        let mut engine = Phase1Engine::new(cfg);
        engine.open_position = Some(Phase1PositionState {
            entry_price:          50_000.0,
            stop_price:           49_850.0,
            target_price:         50_900.0,
            mode:                 TradeMode::Day,
            setup_type:           SetupType::BreakoutLong,
            opened_at:            Instant::now(),
            high_water_mark:      50_000.0,
            break_even_triggered: false,
        });
        // Price rises above BE trigger (50000 * 1.005 = 50250).
        let mut m = base_metrics();
        m.mid = 50_300.0;
        let _ = engine.manage_position(&m);
        let pos = engine.open_position.as_ref().unwrap();
        assert!(pos.break_even_triggered, "Break-even should be triggered");
        // After break-even, stop is at entry (50000).
        // The trailing stop level = 50300 * (1 - 0.004) = 50098.8, which is above
        // entry, so the trailing stop overrides to keep the best protection.
        assert!(
            pos.stop_price >= 50_000.0,
            "Stop should be at or above entry (break-even) after trigger, got {}",
            pos.stop_price
        );
    }

    #[test]
    fn test_on_entry_confirmed_sets_position() {
        let mut engine = Phase1Engine::default();
        let setup = Phase1Setup {
            setup_type:    SetupType::PullbackLong,
            side:          "BUY",
            mode:          TradeMode::Day,
            entry:         50_000.0,
            stop:          49_850.0,
            target:        50_450.0,
            setup_quality: 0.75,
            block_reason:  None,
        };
        engine.on_entry_confirmed(&setup);
        assert!(engine.open_position.is_some());
        let pos = engine.open_position.as_ref().unwrap();
        assert_eq!(pos.entry_price,  50_000.0);
        assert_eq!(pos.stop_price,   49_850.0);
        assert_eq!(pos.target_price, 50_450.0);
    }

    #[test]
    fn test_on_exit_confirmed_clears_position() {
        let mut engine = Phase1Engine::default();
        engine.open_position = Some(Phase1PositionState {
            entry_price:          50_000.0,
            stop_price:           49_850.0,
            target_price:         50_450.0,
            mode:                 TradeMode::Day,
            setup_type:           SetupType::PullbackLong,
            opened_at:            Instant::now(),
            high_water_mark:      50_000.0,
            break_even_triggered: false,
        });
        engine.on_exit_confirmed();
        assert!(engine.open_position.is_none());
    }

    // ── MICRO_ACTIVE Phase1Config preset tests ────────────────────────────────

    #[test]
    fn test_micro_active_config_trigger_window_shorter() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();
        assert!(ma.trigger_window_secs < def.trigger_window_secs,
            "micro_active trigger_window_secs {} should be < default {}",
            ma.trigger_window_secs, def.trigger_window_secs);
    }

    #[test]
    fn test_micro_active_config_min_trend_drift_lower() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();
        assert!(ma.min_trend_drift < def.min_trend_drift,
            "micro_active min_trend_drift {} should be < default {}",
            ma.min_trend_drift, def.min_trend_drift);
    }

    #[test]
    fn test_micro_active_config_min_samples_fewer() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();
        assert!(ma.min_samples < def.min_samples,
            "micro_active min_samples {} should be < default {}",
            ma.min_samples, def.min_samples);
    }

    #[test]
    fn test_micro_active_config_breakout_momentum_lower() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();
        assert!(ma.breakout_min_momentum_1s < def.breakout_min_momentum_1s,
            "micro_active breakout_min_momentum_1s {} should be < default {}",
            ma.breakout_min_momentum_1s, def.breakout_min_momentum_1s);
    }

    #[test]
    fn test_micro_active_config_breakout_imbalance_lower() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();
        assert!(ma.breakout_min_imbalance < def.breakout_min_imbalance,
            "micro_active breakout_min_imbalance {} should be < default {}",
            ma.breakout_min_imbalance, def.breakout_min_imbalance);
    }

    /// Safety rails unchanged: stop/TP percentages, spread limit, sizing.
    #[test]
    fn test_micro_active_preserves_safety_parameters() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();
        assert_eq!(ma.pullback_stop_pct,     def.pullback_stop_pct,   "stop_pct unchanged");
        assert_eq!(ma.breakout_stop_pct,     def.breakout_stop_pct,   "breakout stop_pct unchanged");
        assert_eq!(ma.max_spread_bps,        def.max_spread_bps,      "max_spread unchanged");
        assert_eq!(ma.dollar_risk_per_trade, def.dollar_risk_per_trade, "dollar_risk unchanged");
    }

    /// MICRO_ACTIVE Phase1Config fires a breakout setup where default would not.
    ///
    /// We verify this at the config level:
    ///   - MICRO_ACTIVE has a lower breakout_min_imbalance (0.12 < 0.20 default)
    ///   - MICRO_ACTIVE has a lower breakout_min_momentum_1s (0.00005 < 0.0001 default)
    ///   - A metrics snapshot with imbalance=0.15 and momentum_1s=0.00007 would pass
    ///     MICRO_ACTIVE thresholds but fail default thresholds.
    #[test]
    fn test_micro_active_detects_breakout_where_default_does_not() {
        let ma  = Phase1Config::micro_active();
        let def = Phase1Config::default();

        // Target scenario: imbalance_1s = 0.15, momentum_1s = 0.00007
        let imbalance    = 0.15_f64;
        let momentum_1s  = 0.00007_f64;

        // MICRO_ACTIVE accepts these values
        assert!(
            imbalance >= ma.breakout_min_imbalance,
            "MICRO_ACTIVE should accept imbalance={} (threshold={})",
            imbalance, ma.breakout_min_imbalance
        );
        assert!(
            momentum_1s >= ma.breakout_min_momentum_1s,
            "MICRO_ACTIVE should accept momentum_1s={} (threshold={})",
            momentum_1s, ma.breakout_min_momentum_1s
        );

        // Default engine rejects these values
        assert!(
            imbalance < def.breakout_min_imbalance,
            "Default should reject imbalance={} (threshold={})",
            imbalance, def.breakout_min_imbalance
        );
        assert!(
            momentum_1s < def.breakout_min_momentum_1s,
            "Default should reject momentum_1s={} (threshold={})",
            momentum_1s, def.breakout_min_momentum_1s
        );
    }
}

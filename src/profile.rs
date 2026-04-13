// profile.rs
//
// Runtime trading profiles for the RW-Trader agent.
//
// Profiles adjust aggressiveness thresholds without touching safety rails.
// Spread guard, kill switch, risk engine, and side-aware balance checks are
// always active regardless of profile.

use std::time::Duration;

// ── RuntimeProfile enum ───────────────────────────────────────────────────────

/// Agent runtime profile.  Selects a named bundle of thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeProfile {
    /// High confidence bar, long cooldowns.  Suitable for large accounts or
    /// highly volatile sessions where patience reduces noise trades.
    #[default]
    Conservative,
    /// Balanced defaults.  Normal-confidence, standard cooldowns.
    Active,
    /// Small-balance, fast-iteration mode.  Lower confidence bar, shortened
    /// re-entry and failed-breakout cooldowns.  All safety rails remain on.
    MicroTest,
    /// High-frequency micro-account mode for balances under $100.
    ///
    /// Lowers entry strictness (momentum / imbalance thresholds, trigger window,
    /// minimum confirmation samples), shortens all cooldowns, and uses a faster
    /// cycle interval to capture more micro-moves.  All hard safety rails
    /// (kill switch, exchange filters, valid sizing, balance checks, authority
    /// mode, dispatcher / executor protections) remain fully active.
    MicroActive,
}

impl RuntimeProfile {
    /// Parse from an env-var / UI string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_uppercase().as_str() {
            "CONSERVATIVE" => Self::Conservative,
            "ACTIVE"       => Self::Active,
            "MICRO_TEST" | "MICROTEST"     => Self::MicroTest,
            "MICRO_ACTIVE" | "MICROACTIVE" => Self::MicroActive,
            _              => Self::default(),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "CONSERVATIVE",
            Self::Active       => "ACTIVE",
            Self::MicroTest    => "MICRO_TEST",
            Self::MicroActive  => "MICRO_ACTIVE",
        }
    }

    /// Human-readable label shown in the web UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::Conservative => "Conservative — cautious, long cooldowns",
            Self::Active       => "Active — balanced defaults",
            Self::MicroTest    => "Micro-Test — faster decisions for small balances",
            Self::MicroActive  => "⚡ Micro Active — high-frequency mode for balances under $100",
        }
    }

    /// Returns true when the profile is a micro-account mode.
    ///
    /// Used by the UI to render mode-specific badges and by the strategy
    /// engine to apply the micro-account Phase1 configuration.
    pub fn is_micro(self) -> bool {
        matches!(self, Self::MicroTest | Self::MicroActive)
    }
}

impl std::fmt::Display for RuntimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── ProfileConfig ─────────────────────────────────────────────────────────────

/// Resolved threshold bundle for the chosen profile.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Minimum signal confidence accepted for execution.
    pub signal_min_confidence: f64,
    /// Cooldown after an exit (or losing trade) before re-entry is allowed.
    pub entry_cooldown_after_exit: Duration,
    /// Cooldown after a failed breakout attempt.
    pub failed_breakout_cooldown: Duration,
    /// Autonomous cycle interval.
    pub cycle_interval: Duration,
    // ── Phase 1 parameter overrides (applied only for micro profiles) ──────────
    /// Phase1 trigger window in seconds (acts as 15M execution trigger).
    /// Shorter = faster trigger acceptance.
    pub phase1_trigger_window_secs: Option<f64>,
    /// Phase1 minimum trend drift to declare TREND_UP/TREND_DOWN.
    /// Lower = allow earlier momentum confirmation.
    pub phase1_min_trend_drift: Option<f64>,
    /// Phase1 minimum samples before computing regime or setups.
    /// Fewer = shorter required confirmation window.
    pub phase1_min_samples: Option<usize>,
    /// Phase1 minimum 1s momentum required to confirm a breakout.
    /// Lower = fewer "wait for stronger trigger" holds.
    pub phase1_breakout_min_momentum_1s: Option<f64>,
    /// Phase1 minimum buy imbalance required for a breakout setup.
    /// Lower = more permissive breakout entry.
    pub phase1_breakout_min_imbalance: Option<f64>,
    // ── Signal engine threshold overrides ─────────────────────────────────────
    /// Momentum threshold for the signal engine.
    /// Lower = more permissive signal threshold.
    pub signal_momentum_threshold: Option<f64>,
    /// Imbalance threshold for the signal engine.
    /// Lower = more permissive signal threshold.
    pub signal_imbalance_threshold: Option<f64>,
    /// Maximum hold duration for the signal engine.
    /// Shorter = faster re-entry cycle.
    pub signal_max_hold_secs: Option<u64>,
    // ── StrategyEngine threshold overrides ────────────────────────────────────
    /// Base confidence threshold in the StrategyEngine dynamic gate.
    /// Lower = more frequent execution opportunities.
    pub strategy_base_confidence_threshold: Option<f64>,
    /// How long without a trade before the dynamic threshold is lowered.
    /// Shorter = fewer "wait for stronger trigger" holds.
    pub strategy_no_trade_lowering_after_secs: Option<u64>,
    /// Minimum absolute 1s imbalance required for participation.
    /// Lower = more permissive participation check.
    pub strategy_min_abs_imbalance_1s: Option<f64>,
}

impl ProfileConfig {
    pub fn for_profile(profile: RuntimeProfile) -> Self {
        match profile {
            RuntimeProfile::Conservative => Self {
                signal_min_confidence:       0.82,
                entry_cooldown_after_exit:   Duration::from_secs(300),
                failed_breakout_cooldown:    Duration::from_secs(30),
                cycle_interval:              Duration::from_millis(2_000),
                // No Phase1 / signal / strategy overrides — use defaults.
                phase1_trigger_window_secs:      None,
                phase1_min_trend_drift:          None,
                phase1_min_samples:              None,
                phase1_breakout_min_momentum_1s: None,
                phase1_breakout_min_imbalance:   None,
                signal_momentum_threshold:       None,
                signal_imbalance_threshold:      None,
                signal_max_hold_secs:            None,
                strategy_base_confidence_threshold:      None,
                strategy_no_trade_lowering_after_secs:   None,
                strategy_min_abs_imbalance_1s:           None,
            },
            RuntimeProfile::Active => Self {
                signal_min_confidence:       0.70,
                entry_cooldown_after_exit:   Duration::from_secs(300),
                failed_breakout_cooldown:    Duration::from_secs(20),
                cycle_interval:              Duration::from_millis(1_000),
                phase1_trigger_window_secs:      None,
                phase1_min_trend_drift:          None,
                phase1_min_samples:              None,
                phase1_breakout_min_momentum_1s: None,
                phase1_breakout_min_imbalance:   None,
                signal_momentum_threshold:       None,
                signal_imbalance_threshold:      None,
                signal_max_hold_secs:            None,
                strategy_base_confidence_threshold:      None,
                strategy_no_trade_lowering_after_secs:   None,
                strategy_min_abs_imbalance_1s:           None,
            },
            RuntimeProfile::MicroTest => Self {
                signal_min_confidence:       0.64,
                entry_cooldown_after_exit:   Duration::from_secs(8),
                failed_breakout_cooldown:    Duration::from_secs(12),
                cycle_interval:              Duration::from_millis(500),
                phase1_trigger_window_secs:      None,
                phase1_min_trend_drift:          None,
                phase1_min_samples:              None,
                phase1_breakout_min_momentum_1s: None,
                phase1_breakout_min_imbalance:   None,
                signal_momentum_threshold:       None,
                signal_imbalance_threshold:      None,
                signal_max_hold_secs:            None,
                strategy_base_confidence_threshold:      None,
                strategy_no_trade_lowering_after_secs:   None,
                strategy_min_abs_imbalance_1s:           None,
            },
            // ── MICRO_ACTIVE: high-frequency micro-account mode ────────────────
            //
            // Before → After (per dimension):
            //
            // Profile-level:
            //   signal_min_confidence         0.70 (Active) → 0.52   lower entry bar
            //   entry_cooldown_after_exit    300s (Active) → 5s     faster re-entry
            //   failed_breakout_cooldown      20s (Active) → 6s     faster re-entry
            //   cycle_interval              1000ms (Active) → 250ms  faster cycle
            //
            // Phase1 engine:
            //   trigger_window_secs          15.0 → 8.0    faster trigger acceptance
            //   min_trend_drift             0.0008 → 0.0004 earlier momentum confirmation
            //   min_samples                   10 → 5       shorter confirmation window
            //   breakout_min_momentum_1s   0.0001 → 0.00005 fewer "wait" holds
            //   breakout_min_imbalance       0.20 → 0.12   more permissive breakout
            //
            // Signal engine:
            //   momentum_threshold         0.00005 → 0.00002 more permissive signal
            //   imbalance_threshold          0.10 → 0.05   more permissive signal
            //   max_hold_secs               120 → 60       faster position cycling
            //
            // StrategyEngine:
            //   base_confidence_threshold   0.72 → 0.52   more execution opportunities
            //   no_trade_lowering_after    120s → 30s     fewer "wait for stronger" holds
            //   min_abs_imbalance_1s        0.06 → 0.03   more permissive participation
            RuntimeProfile::MicroActive => Self {
                signal_min_confidence:       0.52,
                entry_cooldown_after_exit:   Duration::from_secs(5),
                failed_breakout_cooldown:    Duration::from_secs(6),
                cycle_interval:              Duration::from_millis(250),
                phase1_trigger_window_secs:      Some(8.0),
                phase1_min_trend_drift:          Some(0.0004),
                phase1_min_samples:              Some(5),
                phase1_breakout_min_momentum_1s: Some(0.00005),
                phase1_breakout_min_imbalance:   Some(0.12),
                signal_momentum_threshold:       Some(0.00002),
                signal_imbalance_threshold:      Some(0.05),
                signal_max_hold_secs:            Some(60),
                strategy_base_confidence_threshold:      Some(0.52),
                strategy_no_trade_lowering_after_secs:   Some(30),
                strategy_min_abs_imbalance_1s:           Some(0.03),
            },
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str_round_trip() {
        for (input, expected) in [
            ("CONSERVATIVE",  RuntimeProfile::Conservative),
            ("conservative",  RuntimeProfile::Conservative),
            ("ACTIVE",        RuntimeProfile::Active),
            ("MICRO_TEST",    RuntimeProfile::MicroTest),
            ("MICROTEST",     RuntimeProfile::MicroTest),
            ("MICRO_ACTIVE",  RuntimeProfile::MicroActive),
            ("MICROACTIVE",   RuntimeProfile::MicroActive),
            ("unknown",       RuntimeProfile::Conservative), // falls back to default
        ] {
            assert_eq!(RuntimeProfile::from_str(input), expected, "input={input}");
        }
    }

    #[test]
    fn test_micro_test_thresholds() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroTest);
        assert!((cfg.signal_min_confidence - 0.64).abs() < 1e-9);
        assert_eq!(cfg.entry_cooldown_after_exit.as_secs(), 8);
        assert_eq!(cfg.failed_breakout_cooldown.as_secs(), 12);
        assert!(cfg.cycle_interval.as_millis() <= 500);
    }

    #[test]
    fn test_conservative_thresholds() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::Conservative);
        assert!((cfg.signal_min_confidence - 0.82).abs() < 1e-9);
        assert!(cfg.entry_cooldown_after_exit.as_secs() >= 60);
    }

    #[test]
    fn test_active_thresholds() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::Active);
        assert!((cfg.signal_min_confidence - 0.70).abs() < 1e-9);
    }

    #[test]
    fn test_micro_test_more_aggressive_than_active() {
        let micro = ProfileConfig::for_profile(RuntimeProfile::MicroTest);
        let active = ProfileConfig::for_profile(RuntimeProfile::Active);
        // Lower confidence bar → more trades executed
        assert!(micro.signal_min_confidence < active.signal_min_confidence);
        // Shorter cooldowns → faster re-entry
        assert!(micro.entry_cooldown_after_exit < active.entry_cooldown_after_exit);
        assert!(micro.cycle_interval <= active.cycle_interval);
    }

    #[test]
    fn test_display() {
        assert_eq!(RuntimeProfile::MicroTest.to_string(), "MICRO_TEST");
        assert_eq!(RuntimeProfile::Conservative.to_string(), "CONSERVATIVE");
        assert_eq!(RuntimeProfile::MicroActive.to_string(), "MICRO_ACTIVE");
    }

    #[test]
    fn test_label_contains_micro_test_text() {
        assert!(RuntimeProfile::MicroTest.label().contains("faster decisions"));
        assert!(RuntimeProfile::MicroTest.label().contains("small balances"));
    }

    // ── MICRO_ACTIVE specific tests ───────────────────────────────────────────

    #[test]
    fn test_micro_active_thresholds() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        assert!((cfg.signal_min_confidence - 0.52).abs() < 1e-9,
            "expected 0.52 got {}", cfg.signal_min_confidence);
        assert_eq!(cfg.entry_cooldown_after_exit.as_secs(), 5,
            "entry cooldown should be 5s");
        assert_eq!(cfg.failed_breakout_cooldown.as_secs(), 6,
            "failed breakout cooldown should be 6s");
        assert!(cfg.cycle_interval.as_millis() <= 250,
            "cycle interval should be ≤250ms");
    }

    #[test]
    fn test_micro_active_more_aggressive_than_micro_test() {
        let ma  = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        let mt  = ProfileConfig::for_profile(RuntimeProfile::MicroTest);
        // Lower confidence bar → even more trades executed
        assert!(ma.signal_min_confidence < mt.signal_min_confidence,
            "MICRO_ACTIVE should have lower confidence bar than MICRO_TEST");
        // Shorter cooldowns → faster re-entry
        assert!(ma.entry_cooldown_after_exit < mt.entry_cooldown_after_exit,
            "MICRO_ACTIVE should have shorter entry cooldown than MICRO_TEST");
        assert!(ma.failed_breakout_cooldown < mt.failed_breakout_cooldown,
            "MICRO_ACTIVE should have shorter failed-breakout cooldown than MICRO_TEST");
        assert!(ma.cycle_interval < mt.cycle_interval,
            "MICRO_ACTIVE should have shorter cycle interval than MICRO_TEST");
    }

    #[test]
    fn test_micro_active_phase1_overrides_present() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        // All Phase1 overrides must be Some(...)
        assert!(cfg.phase1_trigger_window_secs.is_some(), "trigger_window_secs override must be set");
        assert!(cfg.phase1_min_trend_drift.is_some(), "min_trend_drift override must be set");
        assert!(cfg.phase1_min_samples.is_some(), "min_samples override must be set");
        assert!(cfg.phase1_breakout_min_momentum_1s.is_some(), "breakout_min_momentum_1s override must be set");
        assert!(cfg.phase1_breakout_min_imbalance.is_some(), "breakout_min_imbalance override must be set");
    }

    #[test]
    fn test_micro_active_phase1_trigger_window_shorter() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        let default_trigger = 15.0_f64;  // Phase1Config::default().trigger_window_secs
        assert!(cfg.phase1_trigger_window_secs.unwrap() < default_trigger,
            "MICRO_ACTIVE trigger window should be shorter than default 15s");
    }

    #[test]
    fn test_micro_active_phase1_min_drift_lower() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        let default_drift = 0.0008_f64;  // Phase1Config::default().min_trend_drift
        assert!(cfg.phase1_min_trend_drift.unwrap() < default_drift,
            "MICRO_ACTIVE min_trend_drift should be lower than default 0.0008");
    }

    #[test]
    fn test_micro_active_signal_overrides_more_permissive() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        // Signal momentum threshold lower than default 0.00005
        assert!(cfg.signal_momentum_threshold.unwrap() < 0.00005,
            "MICRO_ACTIVE momentum threshold should be below default");
        // Signal imbalance threshold lower than default 0.10
        assert!(cfg.signal_imbalance_threshold.unwrap() < 0.10,
            "MICRO_ACTIVE imbalance threshold should be below default");
    }

    #[test]
    fn test_micro_active_strategy_overrides_present() {
        let cfg = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        assert!(cfg.strategy_base_confidence_threshold.is_some());
        assert!(cfg.strategy_no_trade_lowering_after_secs.is_some());
        assert!(cfg.strategy_min_abs_imbalance_1s.is_some());
        // More permissive than default 0.72
        assert!(cfg.strategy_base_confidence_threshold.unwrap() < 0.72);
        // Shorter no-trade window than default 120s
        assert!(cfg.strategy_no_trade_lowering_after_secs.unwrap() < 120);
        // Lower imbalance floor than default 0.06
        assert!(cfg.strategy_min_abs_imbalance_1s.unwrap() < 0.06);
    }

    #[test]
    fn test_micro_active_label() {
        assert!(RuntimeProfile::MicroActive.label().contains("Micro Active"),
            "label should contain 'Micro Active'");
        assert!(RuntimeProfile::MicroActive.label().contains("$100"),
            "label should mention $100 threshold");
    }

    #[test]
    fn test_is_micro_flag() {
        assert!(!RuntimeProfile::Conservative.is_micro());
        assert!(!RuntimeProfile::Active.is_micro());
        assert!(RuntimeProfile::MicroTest.is_micro());
        assert!(RuntimeProfile::MicroActive.is_micro());
    }

    /// Verify that MICRO_ACTIVE produces more execution opportunities than
    /// Active by having lower signal bars and shorter cooldowns.
    #[test]
    fn test_micro_active_expected_higher_trade_frequency_vs_active() {
        let ma = ProfileConfig::for_profile(RuntimeProfile::MicroActive);
        let ac = ProfileConfig::for_profile(RuntimeProfile::Active);

        // Lower confidence bar
        assert!(ma.signal_min_confidence < ac.signal_min_confidence,
            "MICRO_ACTIVE confidence bar must be lower than Active");
        // Shorter re-entry cooldown
        assert!(ma.entry_cooldown_after_exit < ac.entry_cooldown_after_exit,
            "MICRO_ACTIVE re-entry cooldown must be shorter than Active");
        // Shorter cycle
        assert!(ma.cycle_interval < ac.cycle_interval,
            "MICRO_ACTIVE cycle interval must be shorter than Active");
        // Phase1: faster trigger acceptance (shorter window = fire sooner)
        assert!(ma.phase1_trigger_window_secs.is_some());
        // Signal: more permissive thresholds
        assert!(ma.signal_momentum_threshold.unwrap() < 0.00005,
            "Signal momentum threshold must be more permissive than Active default");
        assert!(ma.signal_imbalance_threshold.unwrap() < 0.10,
            "Signal imbalance threshold must be more permissive than Active default");
        // StrategyEngine: lower confidence gate
        assert!(ma.strategy_base_confidence_threshold.unwrap() < 0.72,
            "StrategyEngine confidence gate must be lower than default");
    }
}


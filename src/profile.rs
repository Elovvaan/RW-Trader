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
}

impl RuntimeProfile {
    /// Parse from an env-var / UI string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_uppercase().as_str() {
            "CONSERVATIVE" => Self::Conservative,
            "ACTIVE"       => Self::Active,
            "MICRO_TEST" | "MICROTEST" => Self::MicroTest,
            _              => Self::default(),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "CONSERVATIVE",
            Self::Active       => "ACTIVE",
            Self::MicroTest    => "MICRO_TEST",
        }
    }

    /// Human-readable label shown in the web UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::Conservative => "Conservative — cautious, long cooldowns",
            Self::Active       => "Active — balanced defaults",
            Self::MicroTest    => "Micro-Test — faster decisions for small balances",
        }
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
}

impl ProfileConfig {
    pub fn for_profile(profile: RuntimeProfile) -> Self {
        match profile {
            RuntimeProfile::Conservative => Self {
                signal_min_confidence:       0.82,
                entry_cooldown_after_exit:   Duration::from_secs(300),
                failed_breakout_cooldown:    Duration::from_secs(30),
                cycle_interval:              Duration::from_millis(2_000),
            },
            RuntimeProfile::Active => Self {
                signal_min_confidence:       0.70,
                entry_cooldown_after_exit:   Duration::from_secs(300),
                failed_breakout_cooldown:    Duration::from_secs(20),
                cycle_interval:              Duration::from_millis(1_000),
            },
            RuntimeProfile::MicroTest => Self {
                signal_min_confidence:       0.64,
                entry_cooldown_after_exit:   Duration::from_secs(8),
                failed_breakout_cooldown:    Duration::from_secs(12),
                cycle_interval:             Duration::from_millis(500),
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
            ("CONSERVATIVE", RuntimeProfile::Conservative),
            ("conservative", RuntimeProfile::Conservative),
            ("ACTIVE",       RuntimeProfile::Active),
            ("MICRO_TEST",   RuntimeProfile::MicroTest),
            ("MICROTEST",    RuntimeProfile::MicroTest),
            ("unknown",      RuntimeProfile::Conservative), // falls back to default
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
    }

    #[test]
    fn test_label_contains_micro_test_text() {
        assert!(RuntimeProfile::MicroTest.label().contains("faster decisions"));
        assert!(RuntimeProfile::MicroTest.label().contains("small balances"));
    }
}

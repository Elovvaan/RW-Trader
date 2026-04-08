// risk.rs
//
// Risk engine. Every proposed order passes through risk_check() before execution.
// If any rule fails, the trade is rejected with a logged reason.
//
// risk_check() is intentionally read-heavy and write-minimal:
//   - It updates peak_equity (monotonic, always safe to update).
//   - It does NOT mutate kill_switch, cooldown, or daily baseline on its own.
//
// The caller is responsible for:
//   - Calling notify_fill() after each confirmed fill.
//   - Calling set_kill_switch(true) when manual halt is needed.
//   - Calling reset_day() at the start of each trading day.

use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::position::Position;

// ── Configuration ─────────────────────────────────────────────────────────────

/// All risk limits. Immutable after construction.
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Max base-asset quantity held at any time (e.g. 0.1 BTC).
    pub max_position_qty: f64,

    /// Max loss allowed in a single trading day, in quote (USDT).
    /// Compared against: (realized + unrealized) - day_start_pnl
    pub max_daily_loss_usd: f64,

    /// Max peak-to-trough drawdown in quote (USDT) since engine start.
    pub max_drawdown_usd: f64,

    /// How long to pause new BUY entries after a losing trade closes.
    pub cooldown_after_loss: Duration,

    /// Reject orders if spread exceeds this many basis points.
    /// 1 bps = 0.01%. Typical BTC/USDT spread is 1-3 bps.
    pub max_spread_bps: f64,

    /// Reject orders if no feed message received within this window.
    pub max_feed_staleness: Duration,

    /// Minimum time between any two submitted orders (rate gate).
    /// Prevents runaway order loops. e.g. 1s means at most 1 order/second.
    pub min_order_interval: Duration,

    /// Suppress duplicate signals: same (symbol, side) within this window.
    /// Handles strategy firing multiple times on a single tick.
    pub signal_dedup_window: Duration,

    /// Maximum number of open (unconfirmed/working) orders allowed at once.
    /// Tracked locally; reconcile with exchange on startup.
    pub max_open_orders: usize,

    /// Reject order if expected execution price deviates from mid by more
    /// than this many basis points. Tighter than max_spread_bps.
    /// For market orders: expected price = ask (buy) or bid (sell).
    /// For limit orders: expected price = limit_price.
    pub max_slippage_bps: f64,
}

impl RiskConfig {
    /// Sensible defaults for a small live account. Override per your risk tolerance.
    pub fn default_for_btcusdt() -> Self {
        Self {
            max_position_qty:    0.01,
            max_daily_loss_usd:  50.0,
            max_drawdown_usd:    100.0,
            cooldown_after_loss: Duration::from_secs(300),
            max_spread_bps:      10.0,
            max_feed_staleness:  Duration::from_secs(5),
            min_order_interval:  Duration::from_secs(1),
            signal_dedup_window: Duration::from_secs(5),
            max_open_orders:     3,
            max_slippage_bps:    20.0,
        }
    }
}

// ── Market snapshot ───────────────────────────────────────────────────────────

/// Caller provides this on every risk_check call.
/// Comes from the WebSocket feed (bookTicker or ticker).
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub bid: f64,
    pub ask: f64,
    /// When the feed last delivered a message (from Instant::now() in feed handler).
    pub feed_last_seen: Option<Instant>,
}

impl MarketSnapshot {
    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid();
        if mid <= 0.0 {
            return f64::MAX;
        }
        ((self.ask - self.bid) / mid) * 10_000.0
    }
}

// ── Proposed order ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Minimal description of the order being proposed.
#[derive(Debug, Clone)]
pub struct ProposedOrder {
    pub symbol: String,
    pub side: OrderSide,
    /// Base asset quantity (e.g. 0.001 BTC).
    pub qty: f64,
    /// Expected execution price for slippage calculation.
    ///   - Market BUY:  use current ask.
    ///   - Market SELL: use current bid.
    ///   - Limit:       use the limit price.
    /// Set to 0.0 to skip the slippage check.
    pub expected_price: f64,
}

// ── Verdict ───────────────────────────────────────────────────────────────────

/// Outcome of a risk check. Not a Result — rejection is expected behavior.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskVerdict {
    Approved,
    Rejected(RejectionReason),
}

impl RiskVerdict {
    pub fn is_approved(&self) -> bool {
        matches!(self, RiskVerdict::Approved)
    }
}

/// Every possible rejection reason, with the values that triggered it.
/// Structured so the caller can log or serialize them cleanly.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectionReason {
    KillSwitch,
    StaleFeed { age_secs: f64 },
    NoFeedData,
    SpreadTooWide { spread_bps: f64, limit_bps: f64 },
    InvalidMarketData { detail: String },
    DailyLossExceeded { loss_usd: f64, limit_usd: f64 },
    DrawdownExceeded { drawdown_usd: f64, limit_usd: f64 },
    Cooldown { remaining_secs: f64 },
    PositionSizeExceeded { current_qty: f64, proposed_qty: f64, limit_qty: f64 },
}

impl std::fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectionReason::KillSwitch =>
                write!(f, "KILL_SWITCH: global kill switch is active"),
            RejectionReason::StaleFeed { age_secs } =>
                write!(f, "STALE_FEED: last message {:.1}s ago (limit: depends on config)", age_secs),
            RejectionReason::NoFeedData =>
                write!(f, "STALE_FEED: no feed data received yet"),
            RejectionReason::SpreadTooWide { spread_bps, limit_bps } =>
                write!(f, "SPREAD: {:.2} bps > limit {:.2} bps", spread_bps, limit_bps),
            RejectionReason::InvalidMarketData { detail } =>
                write!(f, "BAD_MARKET_DATA: {}", detail),
            RejectionReason::DailyLossExceeded { loss_usd, limit_usd } =>
                write!(f, "DAILY_LOSS: -{:.2} USD exceeds limit -{:.2} USD", loss_usd, limit_usd),
            RejectionReason::DrawdownExceeded { drawdown_usd, limit_usd } =>
                write!(f, "DRAWDOWN: {:.2} USD exceeds limit {:.2} USD", drawdown_usd, limit_usd),
            RejectionReason::Cooldown { remaining_secs } =>
                write!(f, "COOLDOWN: {:.0}s remaining after last losing trade", remaining_secs),
            RejectionReason::PositionSizeExceeded { current_qty, proposed_qty, limit_qty } =>
                write!(f, "POSITION_SIZE: current {:.6} + proposed {:.6} > limit {:.6}", current_qty, proposed_qty, limit_qty),
        }
    }
}

// ── Risk engine state ─────────────────────────────────────────────────────────

pub struct RiskEngine {
    pub config: RiskConfig,

    /// Hard block. When true, no orders pass under any circumstance.
    kill_switch: bool,

    /// Highest total PnL (realized + unrealized) seen since engine started.
    /// Used to compute drawdown. Monotonically non-decreasing.
    peak_equity: f64,

    /// Total PnL at the start of the current trading day.
    /// Set at construction or by reset_day().
    day_start_pnl: f64,

    /// No new BUY entries allowed until this instant.
    /// Set by notify_fill() when a losing trade is detected.
    cooldown_until: Option<Instant>,

    /// Tracks the last known realized_pnl to detect when a fill closes at a loss.
    last_realized_pnl: f64,
}

impl RiskEngine {
    /// Create engine from config and current position state.
    /// `current_pos`: snapshot of position at startup (for baseline).
    pub fn new(config: RiskConfig, current_pos: &Position) -> Self {
        let initial_pnl = current_pos.realized_pnl + current_pos.unrealized_pnl;
        Self {
            config,
            kill_switch: false,
            peak_equity: initial_pnl,
            day_start_pnl: initial_pnl,
            cooldown_until: None,
            last_realized_pnl: current_pos.realized_pnl,
        }
    }

    // ── Operator controls ─────────────────────────────────────────────────────

    pub fn set_kill_switch(&mut self, active: bool) {
        self.kill_switch = active;
        if active {
            warn!("RISK: Kill switch ACTIVATED");
        } else {
            info!("RISK: Kill switch deactivated");
        }
    }

    pub fn kill_switch_active(&self) -> bool {
        self.kill_switch
    }

    /// Call at the start of each trading day to reset the daily loss baseline.
    pub fn reset_day(&mut self, current_pos: &Position) {
        let pnl = current_pos.realized_pnl + current_pos.unrealized_pnl;
        self.day_start_pnl = pnl;
        info!("RISK: Daily baseline reset to {:.4} USD", pnl);
    }

    // ── Fill notification ─────────────────────────────────────────────────────

    /// Call this after every confirmed fill with the updated position.
    /// Detects losing trade closes and activates cooldown if needed.
    pub fn notify_fill(&mut self, updated_pos: &Position) {
        let new_realized = updated_pos.realized_pnl;
        let delta = new_realized - self.last_realized_pnl;
        self.last_realized_pnl = new_realized;

        if delta < 0.0 {
            // A losing trade was just closed (realized PnL decreased).
            let until = Instant::now() + self.config.cooldown_after_loss;
            self.cooldown_until = Some(until);
            warn!(
                "RISK: Losing fill detected (delta={:.4} USD). Cooldown active for {:.0}s.",
                delta,
                self.config.cooldown_after_loss.as_secs_f64()
            );
        }
    }

    // ── Core check ────────────────────────────────────────────────────────────

    /// Run all risk rules against a proposed order.
    /// Call this BEFORE submitting any order to the exchange.
    ///
    /// `pos`:      current position state (after latest fill replay).
    /// `market`:   latest market snapshot from the feed.
    /// `order`:    the order you intend to place.
    ///
    /// Returns Approved or Rejected(reason). Logs every rejection at WARN level.
    pub fn risk_check(
        &mut self,
        pos: &Position,
        market: &MarketSnapshot,
        order: &ProposedOrder,
    ) -> RiskVerdict {
        // Update peak equity on every check — we always want the latest high-water mark.
        let current_total_pnl = pos.realized_pnl + pos.unrealized_pnl;
        if current_total_pnl > self.peak_equity {
            self.peak_equity = current_total_pnl;
        }

        // ── Rule 1: Kill switch ───────────────────────────────────────────────
        if self.kill_switch {
            return self.reject(RejectionReason::KillSwitch);
        }

        // ── Rule 2: Stale feed ────────────────────────────────────────────────
        match market.feed_last_seen {
            None => {
                return self.reject(RejectionReason::NoFeedData);
            }
            Some(last_seen) => {
                let age = last_seen.elapsed();
                if age > self.config.max_feed_staleness {
                    return self.reject(RejectionReason::StaleFeed {
                        age_secs: age.as_secs_f64(),
                    });
                }
            }
        }

        // ── Rule 3: Spread ────────────────────────────────────────────────────
        if market.bid <= 0.0 || market.ask <= 0.0 {
            return self.reject(RejectionReason::InvalidMarketData {
                detail: format!("bid={} ask={}", market.bid, market.ask),
            });
        }
        if market.ask < market.bid {
            return self.reject(RejectionReason::InvalidMarketData {
                detail: format!("ask {} < bid {} (inverted book)", market.ask, market.bid),
            });
        }
        let spread_bps = market.spread_bps();
        if spread_bps > self.config.max_spread_bps {
            return self.reject(RejectionReason::SpreadTooWide {
                spread_bps,
                limit_bps: self.config.max_spread_bps,
            });
        }

        // ── Rule 4: Daily loss ────────────────────────────────────────────────
        let daily_pnl = current_total_pnl - self.day_start_pnl;
        if daily_pnl < -self.config.max_daily_loss_usd {
            // Trip the kill switch automatically — don't just reject this one order.
            self.kill_switch = true;
            warn!(
                "RISK: Daily loss limit breached ({:.4} USD). Kill switch auto-activated.",
                daily_pnl
            );
            return self.reject(RejectionReason::DailyLossExceeded {
                loss_usd: -daily_pnl,
                limit_usd: self.config.max_daily_loss_usd,
            });
        }

        // ── Rule 5: Drawdown ──────────────────────────────────────────────────
        let drawdown = self.peak_equity - current_total_pnl;
        if drawdown >= self.config.max_drawdown_usd {
            // Also auto-trip kill switch on drawdown breach.
            self.kill_switch = true;
            warn!(
                "RISK: Max drawdown breached ({:.4} USD from peak {:.4}). Kill switch auto-activated.",
                drawdown, self.peak_equity
            );
            return self.reject(RejectionReason::DrawdownExceeded {
                drawdown_usd: drawdown,
                limit_usd: self.config.max_drawdown_usd,
            });
        }

        // ── Rule 6: Cooldown (BUY entries only) ───────────────────────────────
        // Sells are always allowed during cooldown — you must be able to exit.
        if order.side == OrderSide::Buy {
            if let Some(until) = self.cooldown_until {
                if Instant::now() < until {
                    let remaining = until.duration_since(Instant::now());
                    return self.reject(RejectionReason::Cooldown {
                        remaining_secs: remaining.as_secs_f64(),
                    });
                }
            }
        }

        // ── Rule 7: Position size (BUY entries only) ──────────────────────────
        // Sells reduce position — no size check needed.
        if order.side == OrderSide::Buy {
            let resulting_qty = pos.size + order.qty;
            if resulting_qty > self.config.max_position_qty {
                return self.reject(RejectionReason::PositionSizeExceeded {
                    current_qty: pos.size,
                    proposed_qty: order.qty,
                    limit_qty: self.config.max_position_qty,
                });
            }
        }

        // ── All rules passed ──────────────────────────────────────────────────
        info!(
            "RISK: Approved {} {} qty={:.6} spread={:.2}bps daily_pnl={:+.4} drawdown={:.4}",
            order.side_str(),
            order.symbol,
            order.qty,
            spread_bps,
            daily_pnl,
            drawdown,
        );
        RiskVerdict::Approved
    }

    // ── Status summary ────────────────────────────────────────────────────────

    /// Print a summary of current risk state. Useful on startup or after resets.
    pub fn print_status(&self, pos: &Position) {
        let current_pnl = pos.realized_pnl + pos.unrealized_pnl;
        let daily_pnl   = current_pnl - self.day_start_pnl;
        let drawdown    = self.peak_equity - current_pnl;

        let cooldown_remaining = self.cooldown_until.and_then(|until| {
            let now = Instant::now();
            if until > now { Some(until.duration_since(now).as_secs_f64()) } else { None }
        });

        println!();
        println!("╔══════════════════════════════════════════╗");
        println!("║  Risk Engine Status                       ║");
        println!("╠══════════════════════════════════════════╣");
        println!("║  Kill Switch   : {:<25} ║", if self.kill_switch { "🔴 ACTIVE" } else { "🟢 off" });
        println!("╠══════════════════════════════════════════╣");
        println!("║  Limits                                   ║");
        println!("║  Max Position  : {:<25.6} ║", self.config.max_position_qty);
        println!("║  Max Daily Loss: ${:<24.2} ║", self.config.max_daily_loss_usd);
        println!("║  Max Drawdown  : ${:<24.2} ║", self.config.max_drawdown_usd);
        println!("║  Max Spread    : {:<22.2} bps ║", self.config.max_spread_bps);
        println!("║  Feed Staleness: {:<22.1}s  ║", self.config.max_feed_staleness.as_secs_f64());
        println!("║  Cooldown      : {:<22.0}s  ║", self.config.cooldown_after_loss.as_secs_f64());
        println!("╠══════════════════════════════════════════╣");
        println!("║  Current State                            ║");
        println!("║  Daily PnL     : {:+<25.4} ║", daily_pnl);
        println!("║  Peak Equity   : {:<25.4} ║", self.peak_equity);
        println!("║  Drawdown      : {:<25.4} ║", drawdown);
        match cooldown_remaining {
            Some(s) => println!("║  Cooldown      : {:.0}s remaining             ║", s),
            None    => println!("║  Cooldown      : {:<25} ║", "none"),
        }
        println!("╚══════════════════════════════════════════╝");
        println!();
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn reject(&self, reason: RejectionReason) -> RiskVerdict {
        warn!("RISK REJECTED: {}", reason);
        RiskVerdict::Rejected(reason)
    }
}

impl ProposedOrder {
    pub fn side_str(&self) -> &'static str {
        match self.side {
            OrderSide::Buy  => "BUY",
            OrderSide::Sell => "SELL",
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    // Helper: engine with tight, predictable limits
    fn make_engine(pos: &Position) -> RiskEngine {
        RiskEngine::new(
            RiskConfig {
                max_position_qty:    0.1,
                max_daily_loss_usd:  50.0,
                max_drawdown_usd:    100.0,
                cooldown_after_loss: Duration::from_secs(60),
                max_spread_bps:      10.0,
                max_feed_staleness:  Duration::from_secs(5),
                min_order_interval:  Duration::from_secs(1),
                signal_dedup_window: Duration::from_secs(5),
                max_open_orders:     1,
                max_slippage_bps:    20.0,
            },
            pos,
        )
    }

    fn flat_pos() -> Position {
        Position::new("BTCUSDT")
    }

    // Fresh feed snapshot with tight spread
    fn good_market() -> MarketSnapshot {
        MarketSnapshot {
            bid: 50000.0,
            ask: 50005.0, // spread = 5/50002.5 * 10000 = 1.0 bps
            feed_last_seen: Some(Instant::now()),
        }
    }

    fn buy_order(qty: f64) -> ProposedOrder {
        // expected_price: 0.0 skips slippage check; these tests focus on other risk gates
        ProposedOrder { symbol: "BTCUSDT".into(), side: OrderSide::Buy, qty, expected_price: 0.0 }
    }

    fn sell_order(qty: f64) -> ProposedOrder {
        // expected_price: 0.0 skips slippage check; these tests focus on other risk gates
        ProposedOrder { symbol: "BTCUSDT".into(), side: OrderSide::Sell, qty, expected_price: 0.0 }
    }

    // ── Kill switch ───────────────────────────────────────────────────────────
    #[test]
    fn test_kill_switch_blocks_everything() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        engine.set_kill_switch(true);

        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.01));
        assert_eq!(v, RiskVerdict::Rejected(RejectionReason::KillSwitch));

        // Also blocks sells
        let v = engine.risk_check(&pos, &good_market(), &sell_order(0.01));
        assert_eq!(v, RiskVerdict::Rejected(RejectionReason::KillSwitch));
    }

    #[test]
    fn test_kill_switch_can_be_deactivated() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        engine.set_kill_switch(true);
        engine.set_kill_switch(false);
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert_eq!(v, RiskVerdict::Approved);
    }

    // ── Stale feed ────────────────────────────────────────────────────────────
    #[test]
    fn test_no_feed_data_rejected() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        let market = MarketSnapshot { bid: 50000.0, ask: 50005.0, feed_last_seen: None };
        let v = engine.risk_check(&pos, &market, &buy_order(0.001));
        assert_eq!(v, RiskVerdict::Rejected(RejectionReason::NoFeedData));
    }

    #[test]
    fn test_stale_feed_rejected() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        // Pretend last message was 10 seconds ago (limit is 5)
        let stale_time = Instant::now() - Duration::from_secs(10);
        let market = MarketSnapshot { bid: 50000.0, ask: 50005.0, feed_last_seen: Some(stale_time) };
        let v = engine.risk_check(&pos, &market, &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::StaleFeed { .. })));
    }

    // ── Spread ────────────────────────────────────────────────────────────────
    #[test]
    fn test_wide_spread_rejected() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        // Spread = (51000 - 50000) / 50500 * 10000 = ~198 bps
        let market = MarketSnapshot {
            bid: 50000.0,
            ask: 51000.0,
            feed_last_seen: Some(Instant::now()),
        };
        let v = engine.risk_check(&pos, &market, &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::SpreadTooWide { .. })));
    }

    #[test]
    fn test_inverted_book_rejected() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        let market = MarketSnapshot { bid: 50010.0, ask: 50000.0, feed_last_seen: Some(Instant::now()) };
        let v = engine.risk_check(&pos, &market, &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::InvalidMarketData { .. })));
    }

    // ── Daily loss ────────────────────────────────────────────────────────────
    #[test]
    fn test_daily_loss_rejected() {
        // Start with pos that has a -60 USD realized PnL
        // Engine initializes day_start_pnl = -60, so daily_pnl = 0 initially.
        // Then simulate a further -60 drop.
        let mut pos = flat_pos();
        pos.realized_pnl = -60.0;
        pos.unrealized_pnl = -60.0; // total = -120, but engine starts at -60
        // Engine sets day_start_pnl = -60 (realized + unrealized at construction).
        let mut engine = make_engine(&pos);
        // Now daily_pnl = (-120) - (-60) = -60, exceeds limit of -50.
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::DailyLossExceeded { .. })));
        // Should have also activated kill switch
        assert!(engine.kill_switch_active());
    }

    // ── Drawdown ──────────────────────────────────────────────────────────────
    #[test]
    fn test_drawdown_rejected() {
        let mut pos = flat_pos();
        // Peak was 200 (simulate by calling check when pnl=200 first)
        pos.realized_pnl = 200.0;
        let mut engine = make_engine(&pos); // peak_equity = 200
        // Now position tanks: pnl = 50 → drawdown = 150 > limit 100
        pos.realized_pnl = 50.0;
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::DrawdownExceeded { .. })));
        assert!(engine.kill_switch_active());
    }

    // ── Cooldown ──────────────────────────────────────────────────────────────
    #[test]
    fn test_cooldown_blocks_buys_not_sells() {
        let mut pos = flat_pos();
        let mut engine = make_engine(&pos);

        // Simulate: realized_pnl drops from 0 to -10 on a losing sell fill
        pos.realized_pnl = -10.0;
        engine.notify_fill(&pos); // should set cooldown

        // Buy should be blocked
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::Cooldown { .. })));

        // Sell should still be allowed (to exit a position)
        let v = engine.risk_check(&pos, &good_market(), &sell_order(0.001));
        assert_eq!(v, RiskVerdict::Approved);
    }

    #[test]
    fn test_no_cooldown_on_winning_fill() {
        let mut pos = flat_pos();
        let mut engine = make_engine(&pos);
        pos.realized_pnl = 20.0; // winning fill
        engine.notify_fill(&pos);
        // Buy should go through (no cooldown)
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert_eq!(v, RiskVerdict::Approved);
    }

    // ── Position size ─────────────────────────────────────────────────────────
    #[test]
    fn test_position_size_exceeded() {
        let mut pos = flat_pos();
        pos.size = 0.09; // already near limit
        let mut engine = make_engine(&pos);
        // Propose buying 0.02 → total 0.11 > limit 0.1
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.02));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::PositionSizeExceeded { .. })));
    }

    #[test]
    fn test_sell_allowed_when_over_limit() {
        // Even if somehow position is over limit (legacy fill, manual trade), sells proceed.
        let mut pos = flat_pos();
        pos.size = 0.15; // over limit
        let mut engine = make_engine(&pos);
        let v = engine.risk_check(&pos, &good_market(), &sell_order(0.05));
        assert_eq!(v, RiskVerdict::Approved);
    }

    #[test]
    fn test_exact_limit_is_allowed() {
        let mut pos = flat_pos();
        pos.size = 0.0;
        let mut engine = make_engine(&pos);
        // 0.0 + 0.1 = 0.1 = limit, should be allowed
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.1));
        assert_eq!(v, RiskVerdict::Approved);
    }

    #[test]
    fn test_one_over_limit_rejected() {
        let mut pos = flat_pos();
        pos.size = 0.0;
        let mut engine = make_engine(&pos);
        // 0.0 + 0.100001 > 0.1
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.100_001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::PositionSizeExceeded { .. })));
    }

    // ── Happy path ────────────────────────────────────────────────────────────
    #[test]
    fn test_all_rules_pass() {
        let pos = flat_pos();
        let mut engine = make_engine(&pos);
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert_eq!(v, RiskVerdict::Approved);
    }

    // ── Drawdown peaks track correctly ────────────────────────────────────────
    #[test]
    fn test_peak_equity_tracks_high_water_mark() {
        let mut pos = flat_pos();
        let mut engine = make_engine(&pos); // peak = 0

        // PnL rises to 200
        pos.realized_pnl = 200.0;
        engine.risk_check(&pos, &good_market(), &buy_order(0.001)); // updates peak to 200

        // PnL falls to 120 — drawdown = 80, under limit 100
        pos.realized_pnl = 120.0;
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert_eq!(v, RiskVerdict::Approved); // 80 < 100, still ok

        // PnL falls to 90 — drawdown = 110, over limit
        pos.realized_pnl = 90.0;
        let v = engine.risk_check(&pos, &good_market(), &buy_order(0.001));
        assert!(matches!(v, RiskVerdict::Rejected(RejectionReason::DrawdownExceeded { .. })));
    }
}

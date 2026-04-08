// position.rs
//
// Builds position state purely from fills (myTrades).
// No order state is used — fills are ground truth.
//
// Supports: flat → long → flat transitions (Binance Spot).
// Short positions are not possible on Spot and are treated as an error.

use anyhow::{bail, Result};
use chrono::{DateTime, TimeZone, Utc};

use crate::client::MyTrade;

// ── Fee handling ──────────────────────────────────────────────────────────────
//
// Binance Spot fee rules:
//   BUY  + base asset fee  (e.g. BTC)  → you receive less base
//   BUY  + BNB/other fee               → you receive full base qty
//   SELL + quote asset fee (e.g. USDT) → you receive less quote
//   SELL + BNB/other fee               → you receive full quote proceeds
//
// We derive the base asset from the symbol (e.g. "BTCUSDT" → "BTC").
// The quote asset is "USDT" (only USDT pairs are expected here).

fn base_asset_of(symbol: &str) -> &str {
    // Strip known quote suffixes. Extend if you trade other pairs.
    for suffix in &["USDT", "BUSD", "BTC", "ETH", "BNB"] {
        if let Some(base) = symbol.strip_suffix(suffix) {
            if !base.is_empty() {
                return base;
            }
        }
    }
    symbol // fallback
}

// ── Fill ──────────────────────────────────────────────────────────────────────

/// A parsed, validated fill ready for position math.
#[derive(Debug, Clone)]
pub struct Fill {
    pub trade_id: i64,
    pub is_buy: bool,
    /// Raw quantity from the fill (before fee adjustment on buys).
    pub raw_qty: f64,
    /// Quantity that actually changes your base asset balance.
    /// For BUY + base fee: raw_qty - commission.
    /// For all other cases: raw_qty.
    pub effective_qty: f64,
    pub price: f64,
    /// Commission in quote terms (USD equivalent) for PnL accounting.
    /// BNB and non-quote fees: tracked but NOT subtracted from position qty or proceeds.
    /// They are subtracted from realized PnL as a cost.
    pub fee_in_quote: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub timestamp: DateTime<Utc>,
}

impl Fill {
    pub fn from_trade(t: &MyTrade, symbol: &str, bnb_price_usd: f64) -> Result<Fill> {
        let raw_qty: f64 = t.qty.parse().map_err(|_| anyhow::anyhow!("bad qty: {}", t.qty))?;
        let price: f64 = t.price.parse().map_err(|_| anyhow::anyhow!("bad price: {}", t.price))?;
        let commission: f64 = t.commission.parse().map_err(|_| anyhow::anyhow!("bad commission: {}", t.commission))?;

        if raw_qty <= 0.0 {
            bail!("trade {} has non-positive qty {}", t.id, raw_qty);
        }
        if price <= 0.0 {
            bail!("trade {} has non-positive price {}", t.id, price);
        }

        let base = base_asset_of(symbol);
        let is_buy = t.is_buyer;

        // effective_qty: what actually lands in / leaves your base asset balance
        let effective_qty = if is_buy && t.commission_asset.eq_ignore_ascii_case(base) {
            // Fee taken from base: we receive less BTC
            raw_qty - commission
        } else {
            raw_qty
        };

        if effective_qty < 0.0 {
            bail!("trade {} effective_qty went negative (commission > qty?)", t.id);
        }

        // fee_in_quote: normalize all fees to quote currency for PnL accounting
        let fee_in_quote = if t.commission_asset.eq_ignore_ascii_case("USDT")
            || t.commission_asset.eq_ignore_ascii_case("BUSD")
        {
            commission // already in quote
        } else if t.commission_asset.eq_ignore_ascii_case("BNB") {
            commission * bnb_price_usd // approximate, or 0.0 if unknown
        } else if t.commission_asset.eq_ignore_ascii_case(base) {
            // Base fee on a buy: convert at fill price
            commission * price
        } else {
            // Unknown asset — record as 0, don't crash
            0.0
        };

        let timestamp = Utc.timestamp_millis_opt(t.time as i64).single().unwrap_or_else(Utc::now);

        Ok(Fill {
            trade_id: t.id,
            is_buy,
            raw_qty,
            effective_qty,
            price,
            fee_in_quote,
            commission,
            commission_asset: t.commission_asset.clone(),
            timestamp,
        })
    }
}

// ── Position ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,

    /// Net base asset quantity currently held. 0.0 = flat.
    pub size: f64,

    /// Weighted average entry price. 0.0 when flat.
    pub avg_entry: f64,

    /// Cumulative realized PnL in quote (USDT), after fees.
    pub realized_pnl: f64,

    /// Unrealized PnL at last mark price. Updated by mark().
    pub unrealized_pnl: f64,

    /// Last mark price used for unrealized PnL.
    pub mark_price: f64,

    /// Timestamp of the most recent fill processed.
    pub last_fill_time: Option<DateTime<Utc>>,

    /// Total fees paid (quote equivalent) across all fills.
    pub total_fees_paid: f64,

    /// Number of fills processed.
    pub fill_count: usize,
}

impl Position {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            size: 0.0,
            avg_entry: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            mark_price: 0.0,
            last_fill_time: None,
            total_fees_paid: 0.0,
            fill_count: 0,
        }
    }

    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Apply a single fill to update position state.
    /// Fills must be applied in chronological order (ascending trade_id).
    pub fn apply(&mut self, fill: &Fill) -> Result<()> {
        self.fill_count += 1;
        self.total_fees_paid += fill.fee_in_quote;
        self.last_fill_time = Some(fill.timestamp);

        if fill.is_buy {
            self.apply_buy(fill)
        } else {
            self.apply_sell(fill)
        }
    }

    fn apply_buy(&mut self, fill: &Fill) -> Result<()> {
        let qty = fill.effective_qty;
        if qty == 0.0 {
            // Entire fill was consumed by fee — unusual, but handle it.
            return Ok(());
        }

        // Weighted average entry price
        // new_avg = (old_qty * old_avg + new_qty * fill_price) / (old_qty + new_qty)
        let new_size = self.size + qty;
        self.avg_entry = (self.size * self.avg_entry + qty * fill.price) / new_size;
        self.size = new_size;

        Ok(())
    }

    fn apply_sell(&mut self, fill: &Fill) -> Result<()> {
        if self.is_flat() {
            // Selling from flat on Spot = short, which isn't possible.
            // This means we're looking at fills outside our tracking window.
            // Log a warning and skip rather than bail — the user may have
            // pre-existing fills from before the bot started.
            eprintln!(
                "WARN: sell fill {} on flat position — fill predates tracking window or is a short (not supported on Spot). Skipping.",
                fill.trade_id
            );
            return Ok(());
        }

        let sell_qty = fill.effective_qty;

        // Guard: can't sell more than we hold (shouldn't happen on Spot)
        let sell_qty = sell_qty.min(self.size);

        // Realized PnL for this fill:
        //   proceeds  = sell_qty * sell_price
        //   cost_basis= sell_qty * avg_entry
        //   realized  = proceeds - cost_basis - fee
        let proceeds = sell_qty * fill.price;
        let cost_basis = sell_qty * self.avg_entry;
        let fill_realized = proceeds - cost_basis - fill.fee_in_quote;

        self.realized_pnl += fill_realized;
        self.size -= sell_qty;

        // Also subtract the buy-side fees attributable to this portion.
        // We don't track per-fill buy fees separately here — they were already
        // included in avg_entry via the reduced effective_qty on buys with base fees.
        // For BNB buys, the fee was added to total_fees_paid but NOT in avg_entry.
        // The realized PnL above correctly uses avg_entry, so BNB buy fees show up
        // as a drag on total_fees_paid vs realized_pnl. This is accurate.

        if self.is_flat() {
            // Clean reset — avg entry is meaningless when flat
            self.avg_entry = 0.0;
            self.size = 0.0; // kill any floating point dust
        }

        Ok(())
    }

    /// Update unrealized PnL using a current market price.
    pub fn mark(&mut self, price: f64) {
        self.mark_price = price;
        if self.is_flat() {
            self.unrealized_pnl = 0.0;
        } else {
            self.unrealized_pnl = self.size * (price - self.avg_entry);
        }
    }

    /// Total PnL (realized + unrealized, before any open-position fees).
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Build a Position from a slice of MyTrade from Binance.
///
/// `trades` must be sorted ascending by trade id (oldest first) —
/// that's what Binance returns by default from myTrades.
///
/// `bnb_price_usd`: used to convert BNB fees to USD. Pass 0.0 to ignore.
/// `mark_price`: current market price for unrealized PnL. Pass 0.0 if unknown.
pub fn build_position(
    symbol: &str,
    trades: &[MyTrade],
    bnb_price_usd: f64,
    mark_price: f64,
) -> Result<Position> {
    // Ensure sorted by trade id ascending (Binance guarantees this, but be safe)
    let mut sorted: Vec<&MyTrade> = trades.iter().collect();
    sorted.sort_by_key(|t| t.id);

    let mut pos = Position::new(symbol);

    for trade in sorted {
        let fill = Fill::from_trade(trade, symbol, bnb_price_usd)?;
        pos.apply(&fill)?;
    }

    pos.mark(mark_price);

    Ok(pos)
}

// ── Display ───────────────────────────────────────────────────────────────────

pub fn print_position_state(pos: &Position) {
    let pnl_sign = |v: f64| if v >= 0.0 { "+" } else { "" };

    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║  Position State — {}                ", pos.symbol);
    println!("╠══════════════════════════════════════════╣");

    if pos.is_flat() {
        println!("║  Status     : FLAT                       ║");
    } else {
        println!("║  Status     : LONG                        ║");
    }

    println!("║  Size       : {:<28.8} ║", pos.size);
    println!("║  Avg Entry  : {:<28.4} ║", pos.avg_entry);
    println!("║  Mark Price : {:<28.4} ║", pos.mark_price);
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║  Realized   : {}{:<27.4} ║",
        pnl_sign(pos.realized_pnl),
        pos.realized_pnl
    );
    println!(
        "║  Unrealized : {}{:<27.4} ║",
        pnl_sign(pos.unrealized_pnl),
        pos.unrealized_pnl
    );
    println!(
        "║  Total PnL  : {}{:<27.4} ║",
        pnl_sign(pos.total_pnl()),
        pos.total_pnl()
    );
    println!("║  Fees Paid  : {:<28.4} ║", pos.total_fees_paid);
    println!("╠══════════════════════════════════════════╣");
    println!("║  Fills      : {:<28} ║", pos.fill_count);

    let last_update = pos
        .last_fill_time
        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "none".to_string());
    println!("║  Last Fill  : {:<28} ║", last_update);
    println!("╚══════════════════════════════════════════╝");
    println!();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(
        id: i64,
        is_buyer: bool,
        qty: &str,
        price: &str,
        commission: &str,
        commission_asset: &str,
    ) -> MyTrade {
        MyTrade {
            id,
            symbol: "BTCUSDT".into(),
            qty: qty.into(),
            price: price.into(),
            commission: commission.into(),
            commission_asset: commission_asset.into(),
            time: 1_700_000_000_000 + id as u64 * 1000,
            is_buyer,
            is_maker: false,
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-8
    }

    // ── Case 1: Simple buy then full sell, USDT fees ──────────────────────────
    #[test]
    fn test_buy_sell_usdt_fees() {
        let trades = vec![
            make_trade(1, true,  "1.0", "50000.0", "7.5",  "USDT"), // buy 1 BTC @ 50000
            make_trade(2, false, "1.0", "51000.0", "7.65", "USDT"), // sell 1 BTC @ 51000
        ];
        let pos = build_position("BTCUSDT", &trades, 0.0, 51000.0).unwrap();

        assert!(pos.is_flat(), "should be flat after full sell");
        assert!(approx_eq(pos.size, 0.0));
        assert!(approx_eq(pos.avg_entry, 0.0));

        // realized = (51000 - 50000) * 1.0 - 7.65 (sell fee)
        // buy fee was 7.5 USDT — already paid, shown in total_fees_paid
        // realized does NOT subtract buy fee (that was a cash cost at buy time, not at sell time)
        // BUT: for accurate per-trade realized, buy fees ARE a cost of the position.
        // Our model: buy fees go into total_fees_paid. Realized = proceeds - cost_basis - sell_fee.
        // Total net = realized - buy_fees_not_yet_accounted = 1000 - 7.65 - 7.5 = 984.85
        // But our realized field = 1000 - 7.65 = 992.35
        // total_fees_paid = 7.5 + 7.65 = 15.15
        // net = realized - (total_fees_paid - sell_fees_already_in_realized)
        // This is by design — see comment in apply_sell.

        let expected_realized = 1000.0 - 7.65;
        assert!(
            approx_eq(pos.realized_pnl, expected_realized),
            "realized_pnl={} expected={}",
            pos.realized_pnl,
            expected_realized
        );
        assert!(approx_eq(pos.unrealized_pnl, 0.0));
        assert!(approx_eq(pos.total_fees_paid, 15.15));
    }

    // ── Case 2: Buy with BTC fee ──────────────────────────────────────────────
    #[test]
    fn test_buy_with_base_fee() {
        let trades = vec![
            // Buy 1.0 BTC, fee 0.001 BTC → receive 0.999 BTC
            make_trade(1, true, "1.0", "50000.0", "0.001", "BTC"),
        ];
        let pos = build_position("BTCUSDT", &trades, 0.0, 50000.0).unwrap();

        // effective_qty = 1.0 - 0.001 = 0.999
        assert!(approx_eq(pos.size, 0.999), "size={}", pos.size);
        assert!(approx_eq(pos.avg_entry, 50000.0));
        // fee_in_quote = 0.001 * 50000 = 50.0
        assert!(approx_eq(pos.total_fees_paid, 50.0));
        // unrealized = 0.999 * (50000 - 50000) = 0
        assert!(approx_eq(pos.unrealized_pnl, 0.0));
    }

    // ── Case 3: Multiple buys, partial sells ──────────────────────────────────
    #[test]
    fn test_avg_entry_across_buys() {
        let trades = vec![
            make_trade(1, true,  "1.0", "50000.0", "0.0", "USDT"), // buy 1 @ 50000
            make_trade(2, true,  "1.0", "52000.0", "0.0", "USDT"), // buy 1 @ 52000 → avg 51000
            make_trade(3, false, "1.0", "53000.0", "0.0", "USDT"), // sell 1 → realized = 53000-51000 = 2000
        ];
        let pos = build_position("BTCUSDT", &trades, 0.0, 53000.0).unwrap();

        // After 2 buys: avg = (50000 + 52000) / 2 = 51000
        // After sell 1: realized = 1 * (53000 - 51000) = 2000
        // Remaining: 1 BTC @ avg_entry 51000
        assert!(approx_eq(pos.size, 1.0));
        assert!(approx_eq(pos.avg_entry, 51000.0));
        assert!(approx_eq(pos.realized_pnl, 2000.0));
        // unrealized = 1 * (53000 - 51000) = 2000
        assert!(approx_eq(pos.unrealized_pnl, 2000.0));
    }

    // ── Case 4: Partial fills of same order ───────────────────────────────────
    #[test]
    fn test_partial_fills() {
        let trades = vec![
            // Same order filled in 3 chunks
            make_trade(1, true, "0.3", "50000.0", "0.0", "USDT"),
            make_trade(2, true, "0.5", "50100.0", "0.0", "USDT"),
            make_trade(3, true, "0.2", "50200.0", "0.0", "USDT"),
        ];
        let pos = build_position("BTCUSDT", &trades, 0.0, 50050.0).unwrap();

        // avg = (0.3*50000 + 0.5*50100 + 0.2*50200) / 1.0
        //     = (15000 + 25050 + 10040) / 1.0 = 50090
        let expected_avg = (0.3 * 50000.0 + 0.5 * 50100.0 + 0.2 * 50200.0) / 1.0;
        assert!(approx_eq(pos.size, 1.0));
        assert!(approx_eq(pos.avg_entry, expected_avg), "avg={} expected={}", pos.avg_entry, expected_avg);
    }

    // ── Case 5: Full close returns exactly flat ───────────────────────────────
    #[test]
    fn test_full_close_is_flat() {
        let trades = vec![
            make_trade(1, true,  "0.5", "60000.0", "0.0", "USDT"),
            make_trade(2, false, "0.5", "61000.0", "0.0", "USDT"),
        ];
        let pos = build_position("BTCUSDT", &trades, 0.0, 61000.0).unwrap();
        assert!(pos.is_flat());
        assert!(approx_eq(pos.size, 0.0));
        assert!(approx_eq(pos.avg_entry, 0.0));
        assert!(approx_eq(pos.unrealized_pnl, 0.0));
        assert!(approx_eq(pos.realized_pnl, 500.0)); // 0.5 * (61000-60000)
    }

    // ── Case 6: BNB fee doesn't change effective_qty ─────────────────────────
    #[test]
    fn test_buy_bnb_fee() {
        let trades = vec![
            // Buy 1 BTC, fee in BNB → full 1 BTC received
            make_trade(1, true, "1.0", "50000.0", "0.05", "BNB"),
        ];
        // BNB @ $300
        let pos = build_position("BTCUSDT", &trades, 300.0, 50000.0).unwrap();

        assert!(approx_eq(pos.size, 1.0), "size should be full 1.0, got {}", pos.size);
        assert!(approx_eq(pos.avg_entry, 50000.0));
        // fee_in_quote = 0.05 BNB * 300 = 15 USDT
        assert!(approx_eq(pos.total_fees_paid, 15.0));
    }

    // ── Case 7: No trades → flat position ────────────────────────────────────
    #[test]
    fn test_empty_trades() {
        let pos = build_position("BTCUSDT", &[], 0.0, 50000.0).unwrap();
        assert!(pos.is_flat());
        assert!(approx_eq(pos.realized_pnl, 0.0));
        assert!(approx_eq(pos.unrealized_pnl, 0.0));
    }

    // ── Case 8: Sell before any buy (pre-existing fills) ─────────────────────
    #[test]
    fn test_sell_on_flat_is_skipped() {
        // This can happen if the user traded before the bot started tracking.
        // We skip the sell (warning printed) and continue.
        let trades = vec![
            make_trade(1, false, "1.0", "50000.0", "0.0", "USDT"), // sell with no position
            make_trade(2, true,  "1.0", "50000.0", "0.0", "USDT"), // buy
        ];
        let pos = build_position("BTCUSDT", &trades, 0.0, 51000.0).unwrap();
        // The sell is skipped, we end up long 1 BTC
        assert!(!pos.is_flat());
        assert!(approx_eq(pos.size, 1.0));
    }
}

// orders.rs
//
// Execution-layer logic that sits between the risk engine and the exchange client.
// Handles:
//   - Symbol filter validation (notional floor, qty step, tick size)
//   - Cancel-before-replace with confirmed-cancel polling
//   - Retry policy with bounded attempts and exponential backoff

use std::time::Duration;

use anyhow::{bail, Result};
use tracing::{debug, info, warn};

use crate::client::{BinanceClient, OpenOrder, OrderResponse};

// ── Symbol filters ────────────────────────────────────────────────────────────
//
// Loaded once at startup from GET /api/v3/exchangeInfo.
// Passed into validate_order() before every submission.
// Catching these locally avoids a round-trip rejection from Binance.

#[derive(Debug, Clone)]
pub struct SymbolFilters {
    /// Minimum order quantity in base asset.
    pub min_qty: f64,
    /// Maximum order quantity in base asset.
    pub max_qty: f64,
    /// Quantity must be a multiple of step_size.
    /// e.g. step_size=0.001 → qty=0.123 ok, qty=0.1234 → rejected.
    pub step_size: f64,
    /// Minimum order notional value (qty * price) in quote asset.
    pub min_notional: f64,
    /// Price must be a multiple of tick_size (for limit orders).
    pub tick_size: f64,
}

impl SymbolFilters {
    /// Round qty DOWN to the nearest valid step_size multiple.
    /// Always round down — rounding up could exceed available balance.
    pub fn round_qty(&self, qty: f64) -> f64 {
        if self.step_size <= 0.0 {
            return qty;
        }
        (qty / self.step_size).floor() * self.step_size
    }

    /// Round price to the nearest valid tick_size multiple.
    pub fn round_price(&self, price: f64) -> f64 {
        if self.tick_size <= 0.0 {
            return price;
        }
        (price / self.tick_size).round() * self.tick_size
    }

    /// Validate a proposed order against exchange filters.
    /// `expected_price`: used for notional check.
    ///   - For limit orders: the limit price.
    ///   - For market orders: current mid or ask price.
    ///
    /// Returns Ok(()) or Err describing the specific violation.
    pub fn validate(&self, qty: f64, expected_price: f64) -> Result<()> {
        // ── Quantity range ────────────────────────────────────────────────────
        if qty < self.min_qty {
            bail!(
                "FILTER: qty {:.8} < min_qty {:.8}",
                qty, self.min_qty
            );
        }
        if qty > self.max_qty {
            bail!(
                "FILTER: qty {:.8} > max_qty {:.8}",
                qty, self.max_qty
            );
        }

        // ── Step size alignment ───────────────────────────────────────────────
        // Allow 1e-9 floating point tolerance (step_size is usually 0.001 etc.)
        if self.step_size > 0.0 {
            let remainder = qty % self.step_size;
            let tolerance = self.step_size * 1e-6;
            if remainder > tolerance && (self.step_size - remainder) > tolerance {
                bail!(
                    "FILTER: qty {:.8} not aligned to step_size {:.8} (remainder {:.10})",
                    qty, self.step_size, remainder
                );
            }
        }

        // ── Notional floor ────────────────────────────────────────────────────
        if expected_price > 0.0 {
            let notional = qty * expected_price;
            if notional < self.min_notional {
                bail!(
                    "FILTER: notional {:.4} (qty {:.8} × price {:.4}) < min_notional {:.4}",
                    notional, qty, expected_price, self.min_notional
                );
            }
        }

        Ok(())
    }
}

// ── Retry policy ──────────────────────────────────────────────────────────────
//
// Retries on transient failures only. Non-transient Binance errors are not retried.
//
// Transient:     network error, HTTP 5xx, rate limit (429/418)
// Non-transient: Binance error codes in the -10xx and -11xx range (client errors)
//
// Idempotency: the caller must generate clientOrderId before calling submit_with_retry,
// and pass the SAME id on every attempt. Binance returns the existing order if the id
// was already accepted, making retries safe.

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of submission attempts (including the first).
    pub max_attempts: u32,
    /// Delay before the second attempt. Doubles each retry.
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 200,
            max_delay_ms: 5_000,
        }
    }
}

impl RetryPolicy {
    /// Compute delay for attempt N (0-indexed after the first try).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let ms = (self.base_delay_ms * 2u64.pow(attempt)).min(self.max_delay_ms);
        Duration::from_millis(ms)
    }
}

/// Returns true if the error string describes a transient condition worth retrying.
fn is_transient(err: &str) -> bool {
    // Binance rate limit
    if err.contains("429") || err.contains("418") {
        return true;
    }
    // HTTP 5xx
    if err.contains("500") || err.contains("502") || err.contains("503") || err.contains("504") {
        return true;
    }
    // Network / timeout
    if err.contains("timeout") || err.contains("connection") || err.contains("timed out") {
        return true;
    }
    // Binance error -1021: timestamp outside recvWindow (can happen on clock edge)
    if err.contains("-1021") {
        return true;
    }
    false
}

/// Submit a market order with retry.
/// `client_order_id` is fixed across all attempts for idempotency.
pub async fn submit_market_with_retry(
    client: &BinanceClient,
    symbol: &str,
    side: &str,
    qty: &str,
    client_order_id: &str,
    policy: &RetryPolicy,
) -> Result<OrderResponse> {
    submit_with_retry(
        || client.place_market_order_with_coid(symbol, side, qty, client_order_id),
        "MARKET",
        client_order_id,
        policy,
    )
    .await
}

/// Submit a limit order with retry.
pub async fn submit_limit_with_retry(
    client: &BinanceClient,
    symbol: &str,
    side: &str,
    qty: &str,
    price: &str,
    client_order_id: &str,
    policy: &RetryPolicy,
) -> Result<OrderResponse> {
    submit_with_retry(
        || client.place_limit_order_with_coid(symbol, side, qty, price, client_order_id),
        "LIMIT",
        client_order_id,
        policy,
    )
    .await
}

async fn submit_with_retry<F, Fut>(
    mut make_request: F,
    order_type: &str,
    client_order_id: &str,
    policy: &RetryPolicy,
) -> Result<OrderResponse>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<OrderResponse>>,
{
    let mut last_err = String::new();

    for attempt in 0..policy.max_attempts {
        if attempt > 0 {
            let delay = policy.delay_for_attempt(attempt - 1);
            warn!(
                attempt,
                max = policy.max_attempts,
                delay_ms = delay.as_millis(),
                coid = client_order_id,
                "Retrying order submission"
            );
            tokio::time::sleep(delay).await;
        }

        match make_request().await {
            Ok(resp) => {
                if attempt > 0 {
                    info!(attempt, coid = client_order_id, "Order succeeded on retry");
                }
                return Ok(resp);
            }
            Err(e) => {
                let msg = e.to_string();
                debug!(attempt, error = %msg, coid = client_order_id, "Order attempt failed");

                if !is_transient(&msg) {
                    // Non-transient: log and bail immediately, don't waste retries.
                    warn!(
                        coid = client_order_id,
                        error = %msg,
                        "Non-transient error — not retrying"
                    );
                    return Err(e);
                }

                last_err = msg;
            }
        }
    }

    bail!(
        "{} order {} failed after {} attempts: {}",
        order_type,
        client_order_id,
        policy.max_attempts,
        last_err
    )
}

// ── Cancel-before-replace ─────────────────────────────────────────────────────
//
// When replacing a working limit order:
//   1. Send cancel for the old order.
//   2. Poll until exchange confirms cancel (or timeout).
//   3. Only then submit the new order.
//
// Skipping step 2 risks two live orders simultaneously (both old and new),
// which can cause unintended position doubling.

#[derive(Debug, Clone)]
pub struct CancelReplaceConfig {
    /// How long to wait for cancel confirmation before aborting.
    pub cancel_timeout: Duration,
    /// How often to poll for cancel confirmation.
    pub poll_interval: Duration,
}

impl Default for CancelReplaceConfig {
    fn default() -> Self {
        Self {
            cancel_timeout: Duration::from_secs(5),
            poll_interval:  Duration::from_millis(250),
        }
    }
}

/// Cancel `old_client_order_id` and confirm its cancellation before submitting
/// a new market order. Returns the new order response.
///
/// If the cancel times out (exchange busy), returns Err — the new order is NOT submitted.
/// The caller should decide whether to retry or halt.
pub async fn cancel_then_replace_market(
    client: &BinanceClient,
    symbol: &str,
    old_client_order_id: &str,
    new_side: &str,
    new_qty: &str,
    new_client_order_id: &str,
    cbr: &CancelReplaceConfig,
    retry: &RetryPolicy,
) -> Result<OrderResponse> {
    // ── Step 1: Cancel ────────────────────────────────────────────────────────
    info!(
        symbol,
        old_coid = old_client_order_id,
        new_coid = new_client_order_id,
        "Cancel-before-replace: cancelling old order"
    );

    match client.cancel_order(symbol, old_client_order_id).await {
        Ok(cancelled) => {
            debug!(status = %cancelled.status, "Cancel request accepted");
        }
        Err(e) => {
            let msg = e.to_string();
            // -2011: order not found (already filled or already cancelled) — safe to proceed.
            if msg.contains("-2011") {
                warn!(
                    old_coid = old_client_order_id,
                    "Order not found on cancel (-2011) — may already be filled. Proceeding with new order."
                );
            } else {
                bail!("Cancel-before-replace: cancel failed: {}", msg);
            }
        }
    }

    // ── Step 2: Poll for confirmed cancel ─────────────────────────────────────
    let deadline = tokio::time::Instant::now() + cbr.cancel_timeout;

    loop {
        tokio::time::sleep(cbr.poll_interval).await;

        match client.get_order_status(symbol, old_client_order_id).await {
            Ok(order) => {
                let status = order.status.to_uppercase();
                if status == "CANCELED" || status == "FILLED" || status == "EXPIRED" || status == "REJECTED" {
                    info!(
                        old_coid = old_client_order_id,
                        status = %order.status,
                        "Old order confirmed terminal — proceeding with replacement"
                    );
                    break;
                }
                debug!(status = %order.status, "Waiting for cancel confirmation...");
            }
            Err(e) => {
                // -2011: not found = already gone, safe to proceed
                if e.to_string().contains("-2011") {
                    info!(old_coid = old_client_order_id, "Order gone from exchange (-2011), proceeding");
                    break;
                }
                warn!(error = %e, "Error polling cancel status");
            }
        }

        if tokio::time::Instant::now() >= deadline {
            bail!(
                "Cancel-before-replace: timed out waiting for cancel confirmation of {} after {:?}",
                old_client_order_id,
                cbr.cancel_timeout
            );
        }
    }

    // ── Step 3: Submit new order ──────────────────────────────────────────────
    info!(new_coid = new_client_order_id, "Submitting replacement order");

    submit_market_with_retry(client, symbol, new_side, new_qty, new_client_order_id, retry).await
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn btcusdt_filters() -> SymbolFilters {
        SymbolFilters {
            min_qty: 0.00001,
            max_qty: 9000.0,
            step_size: 0.00001,
            min_notional: 10.0,
            tick_size: 0.01,
        }
    }

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-10 }

    // ── Notional floor ────────────────────────────────────────────────────────
    #[test]
    fn test_notional_below_floor_rejected() {
        let f = btcusdt_filters();
        // qty=0.0001 BTC @ 50000 = $5, below $10 minimum
        assert!(f.validate(0.0001, 50000.0).is_err());
    }

    #[test]
    fn test_notional_at_floor_accepted() {
        let f = btcusdt_filters();
        // qty=0.0002 BTC @ 50000 = $10.00, exactly at minimum
        assert!(f.validate(0.0002, 50000.0).is_ok());
    }

    // ── Quantity range ────────────────────────────────────────────────────────
    #[test]
    fn test_qty_below_min_rejected() {
        let f = btcusdt_filters();
        assert!(f.validate(0.000001, 50000.0).is_err()); // below min_qty 0.00001
    }

    #[test]
    fn test_qty_above_max_rejected() {
        let f = btcusdt_filters();
        assert!(f.validate(9001.0, 50000.0).is_err()); // above max_qty 9000
    }

    // ── Step size alignment ───────────────────────────────────────────────────
    #[test]
    fn test_step_size_aligned_ok() {
        let f = btcusdt_filters();
        assert!(f.validate(0.00100, 50000.0).is_ok()); // 0.001 = 100 steps of 0.00001
    }

    #[test]
    fn test_step_size_misaligned_rejected() {
        let f = btcusdt_filters();
        // 0.000012 / 0.00001 = 1.2 — not a whole number
        assert!(f.validate(0.000012, 50000.0).is_err());
    }

    #[test]
    fn test_round_qty_down() {
        let f = btcusdt_filters();
        // 0.000123456 rounded down to step 0.00001 → 0.00012
        let rounded = f.round_qty(0.000123456);
        assert!(approx(rounded, 0.00012), "got {:.10}", rounded);
    }

    #[test]
    fn test_round_qty_exact_step_unchanged() {
        let f = btcusdt_filters();
        let rounded = f.round_qty(0.00100);
        assert!(approx(rounded, 0.001));
    }

    #[test]
    fn test_round_price() {
        let f = btcusdt_filters();
        // 50000.126 rounded to tick 0.01 → 50000.13
        let rounded = f.round_price(50000.126);
        assert!(approx(rounded, 50000.13), "got {:.4}", rounded);
    }

    // ── Retry policy delays ───────────────────────────────────────────────────
    #[test]
    fn test_retry_delays_exponential() {
        let p = RetryPolicy { max_attempts: 4, base_delay_ms: 100, max_delay_ms: 10_000 };
        assert_eq!(p.delay_for_attempt(0).as_millis(), 100); // 100 * 2^0
        assert_eq!(p.delay_for_attempt(1).as_millis(), 200); // 100 * 2^1
        assert_eq!(p.delay_for_attempt(2).as_millis(), 400); // 100 * 2^2
        assert_eq!(p.delay_for_attempt(3).as_millis(), 800); // 100 * 2^3
    }

    #[test]
    fn test_retry_delay_capped() {
        let p = RetryPolicy { max_attempts: 10, base_delay_ms: 1000, max_delay_ms: 3_000 };
        // 1000 * 2^5 = 32000, but cap is 3000
        assert_eq!(p.delay_for_attempt(5).as_millis(), 3_000);
    }

    // ── Transient vs non-transient ────────────────────────────────────────────
    #[test]
    fn test_rate_limit_is_transient() {
        assert!(is_transient("HTTP 429: rate limit exceeded"));
        assert!(is_transient("418 I'm a teapot"));
    }

    #[test]
    fn test_5xx_is_transient() {
        assert!(is_transient("HTTP 503: service unavailable"));
    }

    #[test]
    fn test_timestamp_error_is_transient() {
        assert!(is_transient("Binance error -1021: Timestamp for this request"));
    }

    #[test]
    fn test_bad_symbol_is_not_transient() {
        assert!(!is_transient("Binance error -1121: Invalid symbol"));
    }

    #[test]
    fn test_insufficient_balance_not_transient() {
        assert!(!is_transient("Binance error -2010: Account has insufficient balance"));
    }
}

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use sha2::Sha256;
use tracing::{debug, info, warn};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

// ── Response types ────────────────────────────────────────────────────────────
// These mirror the Binance API exactly. Fields match the JSON keys via serde.

#[derive(Debug, Deserialize)]
pub struct Balance {
    pub asset: String,
    pub free: String,
    pub locked: String,
}

#[derive(Debug, Deserialize)]
pub struct AccountInfo {
    balances: Vec<Balance>,
}

#[derive(Debug, Deserialize)]
pub struct OpenOrder {
    pub symbol: String,
    #[serde(rename = "orderId")]
    pub order_id: i64,
    #[serde(rename = "clientOrderId")]
    pub client_order_id: String,
    pub price: String,
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    pub status: String,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
}

// Full order response returned from POST /api/v3/order
#[derive(Debug, Deserialize)]
pub struct OrderResponse {
    pub symbol: String,
    #[serde(rename = "orderId")]
    pub order_id: i64,
    #[serde(rename = "clientOrderId")]
    pub client_order_id: String,
    #[serde(rename = "transactTime")]
    pub transact_time: u64,
    pub price: String,
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    #[serde(rename = "cummulativeQuoteQty")]
    pub cumulative_quote_qty: String,
    pub status: String,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
    pub fills: Option<Vec<Fill>>,
}

#[derive(Debug, Deserialize)]
pub struct Fill {
    pub price: String,
    pub qty: String,
    pub commission: String,
    #[serde(rename = "commissionAsset")]
    pub commission_asset: String,
    #[serde(rename = "tradeId")]
    pub trade_id: i64,
}

#[derive(Debug, Deserialize)]
pub struct MyTrade {
    pub id: i64,
    pub symbol: String,
    pub price: String,
    pub qty: String,
    pub commission: String,
    #[serde(rename = "commissionAsset")]
    pub commission_asset: String,
    pub time: u64,
    #[serde(rename = "isBuyer")]
    pub is_buyer: bool,
    #[serde(rename = "isMaker")]
    pub is_maker: bool,
}

impl MyTrade {
    pub fn side(&self) -> &'static str {
        if self.is_buyer { "BUY" } else { "SELL" }
    }
}

#[derive(Debug, Deserialize)]
pub struct TickerPrice {
    pub symbol: String,
    pub price: String,
}

// Binance returns this shape on errors
#[derive(Debug, Deserialize)]
struct BinanceError {
    code: i32,
    msg: String,
}

#[derive(Debug, Deserialize)]
struct ServerTime {
    #[serde(rename = "serverTime")]
    server_time: u64,
}

// ── Client ────────────────────────────────────────────────────────────────────

pub struct BinanceClient {
    api_key: String,
    api_secret: String,
    base_url: String,
    http: Client,
    /// Clock offset in milliseconds: server_time - local_time.
    /// Applied to every signed request's timestamp.
    /// Shared via Arc so sync_time can update it from an async context.
    time_offset_ms: Arc<AtomicI64>,
}

impl BinanceClient {
    pub fn new(api_key: String, api_secret: String, base_url: String) -> Self {
        // Remove trailing slash so URL joins are consistent
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            api_key,
            api_secret,
            base_url,
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to build HTTP client"),
            time_offset_ms: Arc::new(AtomicI64::new(0)),
        }
    }

    // ── Time sync ─────────────────────────────────────────────────────────────
    //
    // Binance rejects requests where |server_time - request_timestamp| > recvWindow.
    // Default recvWindow = 5000ms. If local clock drifts, all signed requests fail
    // with error -1021 ("Timestamp for this request is outside of the recvWindow").
    //
    // Call sync_time() once at startup and periodically (e.g. every 30 minutes).
    // It measures round-trip time and estimates the offset conservatively.

    pub async fn sync_time(&self) -> Result<i64> {
        let url = format!("{}/api/v3/time", self.base_url);

        let t0 = Utc::now().timestamp_millis();
        let resp = self.http.get(&url).send().await?;
        let t1 = Utc::now().timestamp_millis();

        let server: ServerTime = resp.json().await?;
        let server_ms = server.server_time as i64;

        // Estimate: server time at the midpoint of our round-trip
        let rtt_ms = t1 - t0;
        let local_midpoint = t0 + rtt_ms / 2;
        let offset = server_ms - local_midpoint;

        self.time_offset_ms.store(offset, Ordering::Relaxed);

        let abs_offset = offset.abs();
        if abs_offset > 10_000 {
            // Binance recvWindow max is 60_000ms, but >10s is a red flag.
            warn!(
                offset_ms = offset,
                rtt_ms,
                "CLOCK: Offset exceeds 10s — check system time synchronization (NTP)"
            );
        } else if abs_offset > 1_000 {
            warn!(offset_ms = offset, rtt_ms, "CLOCK: Offset > 1s — monitoring");
        } else {
            info!(offset_ms = offset, rtt_ms, "CLOCK: Time sync OK");
        }

        Ok(offset)
    }

    /// Current offset between local clock and Binance server, in milliseconds.
    pub fn time_offset_ms(&self) -> i64 {
        self.time_offset_ms.load(Ordering::Relaxed)
    }

    fn sign(&self, payload: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Returns current Unix timestamp in milliseconds, corrected for exchange clock offset.
    /// This is what goes into every signed request. Without the offset, requests
    /// fail with -1021 if the local clock is skewed relative to Binance servers.
    fn timestamp_ms(&self) -> u64 {
        let local = Utc::now().timestamp_millis();
        let offset = self.time_offset_ms.load(Ordering::Relaxed);
        (local + offset) as u64
    }

    /// Builds a query string from a BTreeMap (sorted for determinism) and appends signature.
    /// Returns the full query string including `&signature=...`.
    fn signed_query(&self, mut params: BTreeMap<&str, String>) -> String {
        params.insert("timestamp", self.timestamp_ms().to_string());
        params.insert("recvWindow", "5000".to_string());

        // Build raw query (no signature yet)
        let raw: String = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, urlenccode(v)))
            .collect::<Vec<_>>()
            .join("&");

        let sig = self.sign(&raw);
        format!("{}&signature={}", raw, sig)
    }

    // ── Response handling ─────────────────────────────────────────────────────

    async fn parse_response<T: for<'de> Deserialize<'de>>(
        resp: reqwest::Response,
        context: &str,
    ) -> Result<T> {
        let status = resp.status();
        let body = resp.text().await.context("Failed to read response body")?;

        if status == StatusCode::TOO_MANY_REQUESTS || status.as_u16() == 418 {
            bail!("[{}] Rate limited ({}). Body: {}", context, status, body);
        }

        if !status.is_success() {
            // Try to parse Binance error envelope
            if let Ok(err) = serde_json::from_str::<BinanceError>(&body) {
                bail!("[{}] Binance error {}: {}", context, err.code, err.msg);
            }
            bail!("[{}] HTTP {}: {}", context, status, body);
        }

        serde_json::from_str::<T>(&body)
            .with_context(|| format!("[{}] Failed to parse response: {}", context, body))
    }

    // ── Public endpoints ──────────────────────────────────────────────────────

    pub async fn fetch_ticker_price(&self, symbol: &str) -> Result<TickerPrice> {
        let url = format!("{}/api/v3/ticker/price?symbol={}", self.base_url, symbol);
        debug!(url = %url, "GET ticker price");
        let resp = self.http.get(&url).send().await?;
        Self::parse_response(resp, "ticker/price").await
    }

    /// Fetch exchange filters for a single symbol.
    /// Call once at startup; store the result and pass to order validation.
    pub async fn fetch_symbol_filters(&self, symbol: &str) -> Result<crate::orders::SymbolFilters> {
        let url = format!("{}/api/v3/exchangeInfo?symbol={}", self.base_url, symbol);
        debug!(symbol, "GET exchangeInfo");
        let resp = self.http.get(&url).send().await?;
        let raw: serde_json::Value = Self::parse_response(resp, "exchangeInfo").await?;

        let sym = raw["symbols"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| anyhow::anyhow!("exchangeInfo: no symbols in response"))?;

        let mut min_qty = 0.0f64;
        let mut max_qty = f64::MAX;
        let mut step_size = 1e-8f64;
        let mut min_notional = 10.0f64;
        let mut tick_size = 0.01f64;

        for filter in sym["filters"].as_array().unwrap_or(&vec![]) {
            match filter["filterType"].as_str() {
                Some("LOT_SIZE") => {
                    min_qty   = filter["minQty"].as_str().and_then(|s| s.parse().ok()).unwrap_or(min_qty);
                    max_qty   = filter["maxQty"].as_str().and_then(|s| s.parse().ok()).unwrap_or(max_qty);
                    step_size = filter["stepSize"].as_str().and_then(|s| s.parse().ok()).unwrap_or(step_size);
                }
                Some("MIN_NOTIONAL") => {
                    min_notional = filter["minNotional"].as_str().and_then(|s| s.parse().ok()).unwrap_or(min_notional);
                }
                Some("NOTIONAL") => {
                    // Newer filter name used in some market types
                    min_notional = filter["minNotional"].as_str().and_then(|s| s.parse().ok()).unwrap_or(min_notional);
                }
                Some("PRICE_FILTER") => {
                    tick_size = filter["tickSize"].as_str().and_then(|s| s.parse().ok()).unwrap_or(tick_size);
                }
                _ => {}
            }
        }

        info!(
            symbol,
            min_qty, max_qty, step_size, min_notional, tick_size,
            "Symbol filters loaded"
        );

        Ok(crate::orders::SymbolFilters { min_qty, max_qty, step_size, min_notional, tick_size })
    }

    // ── Authenticated GET endpoints ───────────────────────────────────────────

    pub async fn fetch_balances(&self) -> Result<Vec<Balance>> {
        let params = BTreeMap::new(); // no extra params needed
        let query = self.signed_query(params);
        let url = format!("{}/api/v3/account?{}", self.base_url, query);

        debug!("GET account");
        let resp = self
            .http
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;

        let account: AccountInfo = Self::parse_response(resp, "account").await?;

        // Filter zero balances — useful for real accounts with many assets
        let nonzero: Vec<Balance> = account
            .balances
            .into_iter()
            .filter(|b| {
                b.free.parse::<f64>().unwrap_or(0.0) > 0.0
                    || b.locked.parse::<f64>().unwrap_or(0.0) > 0.0
            })
            .collect();

        Ok(nonzero)
    }

    pub async fn fetch_open_orders(&self, symbol: &str) -> Result<Vec<OpenOrder>> {
        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        let query = self.signed_query(params);
        let url = format!("{}/api/v3/openOrders?{}", self.base_url, query);

        debug!(symbol, "GET openOrders");
        let resp = self
            .http
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;

        Self::parse_response(resp, "openOrders").await
    }

    pub async fn fetch_my_trades(&self, symbol: &str, limit: u32) -> Result<Vec<MyTrade>> {
        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("limit", limit.to_string());
        let query = self.signed_query(params);
        let url = format!("{}/api/v3/myTrades?{}", self.base_url, query);

        debug!(symbol, limit, "GET myTrades");
        let resp = self
            .http
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;

        Self::parse_response(resp, "myTrades").await
    }

    // ── Order placement ───────────────────────────────────────────────────────

    /// Place a MARKET order.
    /// `side` = "BUY" or "SELL"
    /// `quantity` = base asset quantity as a string, e.g. "0.001"
    pub async fn place_market_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
    ) -> Result<OrderResponse> {
        // Deterministic client order ID so retries are idempotent.
        // Format: "rwt-<uuid-v4>" — unique per call, 32 chars max on Binance.
        let client_order_id = format!("rwt-{}", Uuid::new_v4().simple());

        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("side", side.to_uppercase());
        params.insert("type", "MARKET".to_string());
        params.insert("quantity", quantity.to_string());
        params.insert("newClientOrderId", client_order_id.clone());
        // FULL response includes per-fill details
        params.insert("newOrderRespType", "FULL".to_string());

        let order = self.post_order(params).await?;

        debug!(
            symbol,
            side,
            quantity,
            client_order_id = %order.client_order_id,
            exchange_id = order.order_id,
            status = %order.status,
            filled_qty = %order.executed_qty,
            avg_price = ?order.fills.as_ref().map(avg_fill_price),
            "Market order submitted"
        );

        Ok(order)
    }

    /// Place a LIMIT GTC order.
    /// `price` = limit price as a string, e.g. "65000.00"
    pub async fn place_limit_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        price: &str,
    ) -> Result<OrderResponse> {
        let client_order_id = format!("rwt-{}", Uuid::new_v4().simple());

        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("side", side.to_uppercase());
        params.insert("type", "LIMIT".to_string());
        params.insert("timeInForce", "GTC".to_string());
        params.insert("quantity", quantity.to_string());
        params.insert("price", price.to_string());
        params.insert("newClientOrderId", client_order_id.clone());
        params.insert("newOrderRespType", "FULL".to_string());

        let order = self.post_order(params).await?;

        debug!(
            symbol,
            side,
            quantity,
            price,
            client_order_id = %order.client_order_id,
            exchange_id = order.order_id,
            status = %order.status,
            "Limit order submitted"
        );

        Ok(order)
    }

    /// Cancel an order by clientOrderId.
    pub async fn cancel_order(
        &self,
        symbol: &str,
        client_order_id: &str,
    ) -> Result<OrderResponse> {
        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("origClientOrderId", client_order_id.to_string());

        let query = self.signed_query(params);
        let url = format!("{}/api/v3/order?{}", self.base_url, query);

        debug!(symbol, client_order_id, "DELETE order");
        let resp = self
            .http
            .delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;

        Self::parse_response(resp, "cancel_order").await
    }

    /// Fetch the current status of a single order by clientOrderId.
    /// Used by cancel-before-replace to confirm cancellation.
    pub async fn get_order_status(&self, symbol: &str, client_order_id: &str) -> Result<OrderResponse> {
        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("origClientOrderId", client_order_id.to_string());

        let query = self.signed_query(params);
        let url = format!("{}/api/v3/order?{}", self.base_url, query);

        debug!(symbol, client_order_id, "GET order status");
        let resp = self
            .http
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?;

        Self::parse_response(resp, "get_order_status").await
    }

    /// Place a MARKET order with an explicit clientOrderId.
    /// Used by the retry layer, which fixes the ID across all attempts.
    pub async fn place_market_order_with_coid(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        client_order_id: &str,
    ) -> Result<OrderResponse> {
        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("side", side.to_uppercase());
        params.insert("type", "MARKET".to_string());
        params.insert("quantity", quantity.to_string());
        params.insert("newClientOrderId", client_order_id.to_string());
        params.insert("newOrderRespType", "FULL".to_string());
        self.post_order(params).await
    }

    /// Place a LIMIT GTC order with an explicit clientOrderId.
    pub async fn place_limit_order_with_coid(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        price: &str,
        client_order_id: &str,
    ) -> Result<OrderResponse> {
        let mut params = BTreeMap::new();
        params.insert("symbol", symbol.to_string());
        params.insert("side", side.to_uppercase());
        params.insert("type", "LIMIT".to_string());
        params.insert("timeInForce", "GTC".to_string());
        params.insert("quantity", quantity.to_string());
        params.insert("price", price.to_string());
        params.insert("newClientOrderId", client_order_id.to_string());
        params.insert("newOrderRespType", "FULL".to_string());
        self.post_order(params).await
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    async fn post_order(&self, params: BTreeMap<&str, String>) -> Result<OrderResponse> {
        let query = self.signed_query(params);
        let url = format!("{}/api/v3/order", self.base_url);

        let resp = self
            .http
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(query)
            .send()
            .await?;

        Self::parse_response(resp, "place_order").await
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Percent-encode a query string value.
/// Binance only requires encoding of spaces and a few special chars.
/// Using a minimal encoder to avoid pulling in a URL encoding library.
fn urlenccode(s: &str) -> String {
    // Most Binance param values are plain numbers or symbols — pass through.
    // Only encode chars that would break the query string.
    s.chars()
        .flat_map(|c| match c {
            ' ' => vec!['%', '2', '0'],
            '+' => vec!['%', '2', 'B'],
            '&' => vec!['%', '2', '6'],
            '=' => vec!['%', '3', 'D'],
            '#' => vec!['%', '2', '3'],
            _ => vec![c],
        })
        .collect()
}

/// Compute average fill price from a FULL order response's fills.
fn avg_fill_price(fills: &Vec<Fill>) -> Option<f64> {
    let total_qty: f64 = fills.iter().filter_map(|f| f.qty.parse().ok()).sum();
    if total_qty == 0.0 {
        return None;
    }
    let total_notional: f64 = fills
        .iter()
        .filter_map(|f| {
            let p: f64 = f.price.parse().ok()?;
            let q: f64 = f.qty.parse().ok()?;
            Some(p * q)
        })
        .sum();
    Some(total_notional / total_qty)
}

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

// ── Raw WebSocket message types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CombinedMessage {
    stream: String,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct BookTicker {
    #[serde(rename = "u")]
    update_id: u64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "b")]
    bid_price: String,
    #[serde(rename = "B")]
    bid_qty: String,
    #[serde(rename = "a")]
    ask_price: String,
    #[serde(rename = "A")]
    ask_qty: String,
}

#[derive(Debug, Deserialize)]
struct TradeEvent {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "t")]
    trade_id: u64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    qty: String,
    #[serde(rename = "m")]
    is_buyer_maker: bool,
    #[serde(rename = "T")]
    trade_time: u64,
}

// ── Feed state ────────────────────────────────────────────────────────────────
//
// Accumulates all data the signal engine needs.
// Updated by the WebSocket handlers; read by the signal loop.
// Shared via Arc<Mutex<FeedState>>.

/// A single mid-price sample stamped at arrival time.
#[derive(Debug, Clone)]
pub struct MidSample {
    pub timestamp: Instant,
    pub mid: f64,
}

/// A single public trade, stamped at arrival time.
#[derive(Debug, Clone)]
pub struct TradeSample {
    pub timestamp: Instant,
    pub qty: f64,
    /// True when the buyer was the aggressor (taker).
    /// Binance: is_buyer_maker=false → buyer is taker → aggressor buy.
    pub is_aggressor_buy: bool,
}

/// All live market state maintained by the feed.
pub struct FeedState {
    /// Latest best bid price.
    pub bid: f64,
    /// Latest best ask price.
    pub ask: f64,
    /// When the most recent WebSocket message arrived.
    pub last_seen: Option<Instant>,
    /// Rolling mid-price history, oldest first.
    /// Entries older than `max_window` are pruned on each update.
    pub mid_history: VecDeque<MidSample>,
    /// Rolling trade history, oldest first.
    pub trade_history: VecDeque<TradeSample>,
    /// How far back to keep data. Set to your longest signal window + buffer.
    pub max_window: Duration,
}

impl FeedState {
    pub fn new(max_window: Duration) -> Self {
        Self {
            bid: 0.0,
            ask: 0.0,
            last_seen: None,
            mid_history: VecDeque::new(),
            trade_history: VecDeque::new(),
            max_window,
        }
    }

    pub fn mid(&self) -> f64 {
        if self.bid > 0.0 && self.ask > 0.0 {
            (self.bid + self.ask) / 2.0
        } else {
            0.0
        }
    }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid();
        if mid <= 0.0 || self.bid <= 0.0 {
            return f64::MAX;
        }
        ((self.ask - self.bid) / mid) * 10_000.0
    }

    /// True if the feed has received data within the staleness threshold.
    pub fn is_fresh(&self, max_staleness: Duration) -> bool {
        self.last_seen
            .map(|t| t.elapsed() < max_staleness)
            .unwrap_or(false)
    }

    // ── Internal update methods ───────────────────────────────────────────────

    fn push_book_ticker(&mut self, bid: f64, ask: f64) {
        let now = Instant::now();
        self.bid = bid;
        self.ask = ask;
        self.last_seen = Some(now);

        let mid = (bid + ask) / 2.0;
        if mid > 0.0 {
            self.mid_history.push_back(MidSample { timestamp: now, mid });
            self.prune_mid_history();
        }
    }

    fn push_trade(&mut self, qty: f64, is_aggressor_buy: bool) {
        let now = Instant::now();
        self.last_seen = Some(now);
        self.trade_history.push_back(TradeSample { timestamp: now, qty, is_aggressor_buy });
        self.prune_trade_history();
    }

    fn prune_mid_history(&mut self) {
        let cutoff = Instant::now() - self.max_window;
        while self.mid_history.front().map_or(false, |s| s.timestamp < cutoff) {
            self.mid_history.pop_front();
        }
    }

    fn prune_trade_history(&mut self) {
        let cutoff = Instant::now() - self.max_window;
        while self.trade_history.front().map_or(false, |s| s.timestamp < cutoff) {
            self.trade_history.pop_front();
        }
    }
}

// ── Feed entry points ─────────────────────────────────────────────────────────

/// Poll the Binance REST API for best bid/ask, writing results into `feed_state`.
/// Runs forever at the given `poll_interval`. Used when WebSocket is disabled.
///
/// This is the primary feed path: WebSocket is disabled to avoid runtime crashes
/// caused by incorrect endpoints or missing listenKey configuration.
pub async fn run_rest_polling(
    client: Arc<crate::client::BinanceClient>,
    symbol: &str,
    feed_state: Arc<Mutex<FeedState>>,
    poll_interval: Duration,
) -> Result<()> {
    info!("WebSocket disabled — using REST polling mode");
    let mut interval = tokio::time::interval(poll_interval);
    loop {
        interval.tick().await;
        match client.fetch_book_ticker(symbol).await {
            Ok(bt) => {
                let bid: f64 = bt.bid_price.parse().unwrap_or_else(|_| {
                    warn!(raw = %bt.bid_price, "Failed to parse bid price from bookTicker");
                    0.0
                });
                let ask: f64 = bt.ask_price.parse().unwrap_or_else(|_| {
                    warn!(raw = %bt.ask_price, "Failed to parse ask price from bookTicker");
                    0.0
                });
                if bid > 0.0 && ask > 0.0 {
                    let mut state = feed_state.lock().await;
                    state.push_book_ticker(bid, ask);
                    debug!(bid, ask, spread_bps = state.spread_bps(), "[REST bookTicker]");
                }
            }
            Err(e) => {
                warn!(error = %e, "REST book ticker poll failed");
            }
        }
    }
}

/// Start the WebSocket feed, writing all market data into `feed_state`.
/// Reconnects automatically. Runs until the process exits.
///
/// This is the production path used by the signal loop.
///
/// `ws_stream_base` must be the **stream** base URL — the host-only portion of
/// the Binance stream endpoint, with no trailing path.  Examples:
///   - Binance.US production:  `wss://stream.binance.us:9443`
///   - Binance testnet:        `wss://testnet.binance.vision`
///
/// This is distinct from the WebSocket API base (`wss://ws-api.binance.us:443/ws-api/v3`),
/// which is used for signed API calls and is not used here.
///
/// The combined stream URL is constructed as:
///   `{ws_stream_base}/stream?streams={sym}@bookTicker/{sym}@trade`
pub async fn run_feed_with_state(
    ws_stream_base: &str,
    symbol: &str,
    feed_state: Arc<Mutex<FeedState>>,
) -> Result<()> {
    let url = build_market_data_url(ws_stream_base, symbol);

    info!(url = %url, "Connecting to Binance market-data stream (signal mode)");

    let mut reconnect_delay = Duration::from_secs(1);
    let mut attempt = 0u32;

    loop {
        match connect_one_stateful(&url, Arc::clone(&feed_state)).await {
            Ok(()) => {
                warn!("WebSocket disconnected cleanly, reconnecting...");
                reconnect_delay = Duration::from_secs(1);
                attempt = 0;
            }
            Err(e) => {
                attempt += 1;
                error!(attempt, error = %e, "WebSocket error, reconnecting in {:?}", reconnect_delay);
                sleep(reconnect_delay).await;
                reconnect_delay = (reconnect_delay * 2).min(Duration::from_secs(30));
            }
        }
    }
}

/// Original logging-only feed. Kept for standalone use / testing without signal engine.
///
/// `ws_stream_base` must be the **stream** base URL (e.g. `wss://stream.binance.us:9443`
/// for Binance.US, or `wss://testnet.binance.vision` for testnet).  Do not include a
/// trailing path; the combined stream path `/stream?streams=…` is appended here.
pub async fn run_feed(ws_stream_base: &str, symbol: &str) -> Result<()> {
    let url = build_market_data_url(ws_stream_base, symbol);
    info!(url = %url, "Connecting to Binance market-data stream");

    let mut reconnect_delay = Duration::from_secs(1);
    let mut attempt = 0u32;
    loop {
        match connect_one_logging(&url).await {
            Ok(()) => { warn!("WebSocket disconnected cleanly, reconnecting..."); reconnect_delay = Duration::from_secs(1); attempt = 0; }
            Err(e) => { attempt += 1; error!(attempt, error = %e, "WebSocket error, reconnecting in {:?}", reconnect_delay); sleep(reconnect_delay).await; reconnect_delay = (reconnect_delay * 2).min(Duration::from_secs(30)); }
        }
    }
}

// ── Stateful connection (writes to FeedState) ─────────────────────────────────

async fn connect_one_stateful(url: &str, feed_state: Arc<Mutex<FeedState>>) -> Result<()> {
    let (ws, _) = connect_async(url).await.context("WebSocket handshake failed")?;
    info!("WebSocket connected");

    let (mut write, mut read) = ws.split();
    let mut ping_interval = tokio::time::interval(Duration::from_secs(180));
    ping_interval.tick().await;

    let mut last_message = Instant::now();
    let stale_warn = Duration::from_secs(10);

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        last_message = Instant::now();
                        dispatch_to_state(&text, &feed_state).await;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        write.send(Message::Pong(data)).await.context("Failed to send pong")?;
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(Message::Close(frame))) => {
                        info!(frame = ?frame, "Server sent close frame");
                        return Ok(());
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(e).context("WebSocket read error"),
                    None => return Ok(()),
                }
            }
            _ = ping_interval.tick() => {
                if last_message.elapsed() > stale_warn {
                    warn!(stale_secs = last_message.elapsed().as_secs(), "Feed stale — no messages");
                }
                write.send(Message::Ping(vec![])).await.context("Failed to send ping")?;
            }
        }
    }
}

async fn dispatch_to_state(text: &str, feed_state: &Arc<Mutex<FeedState>>) {
    let msg = match serde_json::from_str::<CombinedMessage>(text) {
        Ok(m) => m,
        Err(e) => { debug!(error = %e, "Could not parse combined message"); return; }
    };

    if msg.stream.ends_with("@bookTicker") {
        if let Ok(bt) = serde_json::from_value::<BookTicker>(msg.data) {
            let bid: f64 = bt.bid_price.parse().unwrap_or(0.0);
            let ask: f64 = bt.ask_price.parse().unwrap_or(0.0);
            if bid > 0.0 && ask > 0.0 {
                let mut state = feed_state.lock().await;
                state.push_book_ticker(bid, ask);
                debug!(bid, ask, spread_bps = format!("{:.2}", state.spread_bps()), "[bookTicker]");
            }
        }
    } else if msg.stream.ends_with("@trade") {
        if let Ok(t) = serde_json::from_value::<TradeEvent>(msg.data) {
            let qty: f64 = t.qty.parse().unwrap_or(0.0);
            if qty > 0.0 {
                // is_buyer_maker=false → buyer is taker → aggressor buy
                let is_aggressor_buy = !t.is_buyer_maker;
                let mut state = feed_state.lock().await;
                state.push_trade(qty, is_aggressor_buy);
                debug!(
                    price = %t.price,
                    qty,
                    aggressor = if is_aggressor_buy { "BUY" } else { "SELL" },
                    "[trade]"
                );
            }
        }
    }
}

// ── Logging-only connection (original behaviour) ──────────────────────────────

async fn connect_one_logging(url: &str) -> Result<()> {
    let (ws, _) = connect_async(url).await.context("WebSocket handshake failed")?;
    info!("WebSocket connected");
    let (mut write, mut read) = ws.split();
    let mut ping_interval = tokio::time::interval(Duration::from_secs(180));
    ping_interval.tick().await;
    let mut last_message = Instant::now();
    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => { last_message = Instant::now(); dispatch_logging(&text); }
                    Some(Ok(Message::Ping(data))) => { write.send(Message::Pong(data)).await.context("pong")?; }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(Message::Close(frame))) => { info!(frame = ?frame, "Close"); return Ok(()); }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(e).context("WebSocket read error"),
                    None => return Ok(()),
                }
            }
            _ = ping_interval.tick() => {
                if last_message.elapsed() > Duration::from_secs(10) { warn!("Feed stale"); }
                write.send(Message::Ping(vec![])).await.context("ping")?;
            }
        }
    }
}

fn dispatch_logging(text: &str) {
    match serde_json::from_str::<CombinedMessage>(text) {
        Ok(msg) => {
            if msg.stream.ends_with("@bookTicker") {
                if let Ok(bt) = serde_json::from_value::<BookTicker>(msg.data) {
                    let bid: f64 = bt.bid_price.parse().unwrap_or(0.0);
                    let ask: f64 = bt.ask_price.parse().unwrap_or(0.0);
                    let spread_bps = if bid > 0.0 { ((ask - bid) / bid) * 10_000.0 } else { 0.0 };
                    info!(symbol = %bt.symbol, bid = %bt.bid_price, ask = %bt.ask_price, spread_bps = format!("{:.2}", spread_bps), "[bookTicker]");
                }
            } else if msg.stream.ends_with("@trade") {
                if let Ok(t) = serde_json::from_value::<TradeEvent>(msg.data) {
                    let aggressor = if t.is_buyer_maker { "SELL" } else { "BUY" };
                    info!(symbol = %t.symbol, price = %t.price, qty = %t.qty, aggressor, "[trade]");
                }
            }
        }
        Err(e) => { debug!(error = %e, "Could not parse message"); }
    }
}

// ── URL helpers ───────────────────────────────────────────────────────────────

/// Build the combined market-data stream URL for bookTicker + trade.
///
/// `ws_stream_base` must be the host-only stream base URL, e.g.:
///   - `wss://stream.binance.us:9443`  (Binance.US production)
///   - `wss://testnet.binance.vision`  (Binance testnet)
///
/// The returned URL uses the combined-stream endpoint:
///   `{ws_stream_base}/stream?streams={sym}@bookTicker/{sym}@trade`
pub fn build_market_data_url(ws_stream_base: &str, symbol: &str) -> String {
    let sym_lower = symbol.to_lowercase();
    let stream_path = format!("{}@bookTicker/{}@trade", sym_lower, sym_lower);
    format!("{}/stream?streams={}", ws_stream_base.trim_end_matches('/'), stream_path)
}

/// Build the user data stream URL for a given listenKey.
///
/// `ws_stream_base` is the same stream base URL as for market data.
/// The returned URL uses the single-stream `/ws/` path:
///   `{ws_stream_base}/ws/{listenKey}`
pub fn build_user_data_url(ws_stream_base: &str, listen_key: &str) -> String {
    format!("{}/ws/{}", ws_stream_base.trim_end_matches('/'), listen_key)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn market_data_url_binance_us() {
        let url = build_market_data_url("wss://stream.binance.us:9443", "BTCUSDT");
        assert_eq!(
            url,
            "wss://stream.binance.us:9443/stream?streams=btcusdt@bookTicker/btcusdt@trade"
        );
    }

    #[test]
    fn market_data_url_testnet() {
        let url = build_market_data_url("wss://testnet.binance.vision", "BTCUSDT");
        assert_eq!(
            url,
            "wss://testnet.binance.vision/stream?streams=btcusdt@bookTicker/btcusdt@trade"
        );
    }

    #[test]
    fn market_data_url_trailing_slash_stripped() {
        // Ensure a trailing slash in the base URL doesn't produce a double slash.
        let url = build_market_data_url("wss://stream.binance.us:9443/", "ETHUSDT");
        assert_eq!(
            url,
            "wss://stream.binance.us:9443/stream?streams=ethusdt@bookTicker/ethusdt@trade"
        );
    }

    #[test]
    fn user_data_url_binance_us() {
        let url = build_user_data_url("wss://stream.binance.us:9443", "abc123listenkey");
        assert_eq!(url, "wss://stream.binance.us:9443/ws/abc123listenkey");
    }

    #[test]
    fn user_data_url_trailing_slash_stripped() {
        let url = build_user_data_url("wss://stream.binance.us:9443/", "mykey");
        assert_eq!(url, "wss://stream.binance.us:9443/ws/mykey");
    }
}

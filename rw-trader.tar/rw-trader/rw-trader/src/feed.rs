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

/// Start the WebSocket feed, writing all market data into `feed_state`.
/// Reconnects automatically. Runs until the process exits.
///
/// This is the production path used by the signal loop.
pub async fn run_feed_with_state(
    ws_base: &str,
    symbol: &str,
    feed_state: Arc<Mutex<FeedState>>,
) -> Result<()> {
    let sym_lower = symbol.to_lowercase();
    let stream_path = format!("{}@bookTicker/{}@trade", sym_lower, sym_lower);
    let url = format!("{}/stream?streams={}", ws_base.trim_end_matches('/'), stream_path);

    info!(url = %url, "Connecting to Binance WebSocket (signal mode)");

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
pub async fn run_feed(ws_base: &str, symbol: &str) -> Result<()> {
    let sym_lower = symbol.to_lowercase();
    let stream_path = format!("{}@bookTicker/{}@trade", sym_lower, sym_lower);
    let url = format!("{}/stream?streams={}", ws_base.trim_end_matches('/'), stream_path);
    info!(url = %url, "Connecting to Binance WebSocket");

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

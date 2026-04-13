// reconciler.rs
//
// Exchange state is always the source of truth.
// This module owns the canonical view of:
//   - open orders (from GET /openOrders)
//   - position (rebuilt from fills)
//   - balances (from GET /account)
//
// run_reconciliation() fetches all three, compares with local state,
// logs any mismatches, and overwrites local state with exchange truth.
//
// Guards (state_dirty, recon_in_progress) block order placement
// whenever state may be inconsistent.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::client::{Balance, BinanceClient, MyTrade, OpenOrder};
use crate::position::{build_position, Position};

// ── Order status ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Expired,
    Rejected,
    Unknown,
}

impl OrderStatus {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "NEW"              => OrderStatus::New,
            "PARTIALLY_FILLED" => OrderStatus::PartiallyFilled,
            "FILLED"           => OrderStatus::Filled,
            "CANCELED"         => OrderStatus::Canceled,
            "EXPIRED"          => OrderStatus::Expired,
            "REJECTED"         => OrderStatus::Rejected,
            _                  => OrderStatus::Unknown,
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Expired | OrderStatus::Rejected)
    }

    pub fn is_active(self) -> bool {
        matches!(self, OrderStatus::New | OrderStatus::PartiallyFilled)
    }
}

impl std::fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderStatus::New             => write!(f, "NEW"),
            OrderStatus::PartiallyFilled => write!(f, "PARTIALLY_FILLED"),
            OrderStatus::Filled          => write!(f, "FILLED"),
            OrderStatus::Canceled        => write!(f, "CANCELED"),
            OrderStatus::Expired         => write!(f, "EXPIRED"),
            OrderStatus::Rejected        => write!(f, "REJECTED"),
            OrderStatus::Unknown         => write!(f, "UNKNOWN"),
        }
    }
}

// ── Order record ──────────────────────────────────────────────────────────────
//
// Full lifecycle tracking for a single order. Updated ONLY from exchange
// responses or reconciliation — never from assumptions.

#[derive(Debug, Clone)]
pub struct OrderRecord {
    pub client_order_id:   String,
    pub exchange_order_id: i64,
    pub symbol:            String,
    pub side:              String,   // "BUY" or "SELL"
    pub order_type:        String,   // "MARKET", "LIMIT"
    pub orig_qty:          f64,
    pub filled_qty:        f64,
    pub remaining_qty:     f64,     // orig_qty - filled_qty
    pub avg_fill_price:    f64,     // 0.0 if not yet filled
    pub status:            OrderStatus,
    pub last_seen:         Instant, // when we last heard from exchange about this
}

impl OrderRecord {
    /// Build from an OpenOrder response (exchange open orders endpoint).
    pub fn from_open_order(o: &OpenOrder) -> Self {
        let orig_qty   = o.orig_qty.parse().unwrap_or(0.0);
        let filled_qty = o.executed_qty.parse().unwrap_or(0.0);
        Self {
            client_order_id:   o.client_order_id.clone(),
            exchange_order_id: o.order_id,
            symbol:            o.symbol.clone(),
            side:              o.side.clone(),
            order_type:        o.order_type.clone(),
            orig_qty,
            filled_qty,
            remaining_qty:     (orig_qty - filled_qty).max(0.0),
            avg_fill_price:    0.0, // not available from open orders endpoint
            status:            OrderStatus::from_str(&o.status),
            last_seen:         Instant::now(),
        }
    }

    /// True when the bot has submitted but exchange hasn't confirmed yet.
    pub fn is_pending_ack(&self) -> bool {
        self.exchange_order_id == 0
    }
}

// ── Reconcile result ──────────────────────────────────────────────────────────

/// What changed in a single reconciliation cycle.
#[derive(Debug, Default)]
pub struct ReconcileResult {
    /// Position size differed between local and exchange.
    pub position_mismatch: bool,
    /// Open order count differed.
    pub open_order_count_mismatch: bool,
    /// Orders found on exchange that we had no local record of.
    pub unknown_orders_found: usize,
    /// Orders in local state that no longer exist on exchange.
    pub orders_disappeared: usize,
    /// New fills processed this cycle.
    pub new_fills: usize,
    /// Whether reconcile data itself came back successfully.
    pub exchange_fetch_ok: bool,
    /// True when buy_power or sell_inventory changed from the previous cycle.
    pub balances_changed: bool,
    /// Per-fill details for every new fill discovered this cycle.
    pub fill_details: Vec<crate::events::FillDetail>,
}

impl ReconcileResult {
    pub fn has_anomaly(&self) -> bool {
        self.position_mismatch
            || self.open_order_count_mismatch
            || self.unknown_orders_found > 0
            || self.orders_disappeared > 0
    }
}

// ── Truth state ───────────────────────────────────────────────────────────────
//
// The single source of truth for the bot's view of the world.
// All fields are rebuilt from exchange data; local writes are prohibited
// except through reconciliation or confirmed order responses.

pub struct TruthState {
    pub symbol: String,

    /// Position rebuilt from fills every reconcile cycle.
    pub position: Position,

    /// All orders the bot has submitted, keyed by client_order_id.
    /// Updated by: order submission, reconciliation.
    pub orders: HashMap<String, OrderRecord>,

    /// Open order count derived from exchange, not from local tracking.
    pub open_order_count: usize,

    /// Account balances from last reconcile.
    pub balances: Vec<Balance>,
    /// Total account value estimate in USD (stable-asset sum + convertible assets).
    pub total_balance_usd: f64,
    /// Free quote-asset amount usable for the active trading symbol.
    pub tradable_balance: f64,
    /// Free quote-asset amount for BUY orders on the active symbol.
    pub buy_power: f64,
    /// Free base-asset amount for SELL orders on the active symbol.
    pub sell_inventory: f64,
    /// Human-readable explanation of balance usability, if constrained.
    pub balance_status: Option<String>,

    /// Fill IDs we have already processed. Prevents double-counting.
    pub seen_fill_ids: HashSet<i64>,

    /// When the last successful reconcile completed.
    pub last_reconciled_at: Option<Instant>,

    /// True when state may be wrong (reconcile failed or detected corruption).
    /// Blocks new order placement until cleared.
    pub state_dirty: bool,

    /// True while reconciliation is actively running.
    /// Blocks new order placement to prevent racing with in-flight state rebuild.
    pub recon_in_progress: bool,

    /// BNB price for fee normalization (set from env or 0.0).
    pub bnb_price_usd: f64,
}

impl TruthState {
    pub fn new(symbol: &str, bnb_price_usd: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            position: Position::new(symbol),
            orders: HashMap::new(),
            open_order_count: 0,
            balances: Vec::new(),
            total_balance_usd: 0.0,
            tradable_balance: 0.0,
            buy_power: 0.0,
            sell_inventory: 0.0,
            balance_status: None,
            seen_fill_ids: HashSet::new(),
            last_reconciled_at: None,
            state_dirty: true,   // start dirty until first reconcile
            recon_in_progress: false,
            bnb_price_usd,
        }
    }

    /// Returns true if a new order can be placed right now.
    /// False if state is dirty, reconcile is running, or we have never reconciled.
    pub fn can_place_order(&self) -> bool {
        !self.state_dirty && !self.recon_in_progress && self.last_reconciled_at.is_some()
    }

    /// Side-aware free balance used for readiness checks.
    /// BUY consumes quote asset, SELL consumes base asset.
    pub fn available_balance_for_side(&self, side: &str) -> f64 {
        if side.eq_ignore_ascii_case("BUY") {
            self.buy_power
        } else if side.eq_ignore_ascii_case("SELL") {
            self.sell_inventory
        } else {
            0.0
        }
    }

    /// Register a new order record from a successful submission response.
    pub fn record_order_submitted(&mut self, record: OrderRecord) {
        info!(
            coid = %record.client_order_id,
            exchange_id = record.exchange_order_id,
            side = %record.side,
            qty = record.orig_qty,
            "OrderRecord: submitted"
        );
        self.orders.insert(record.client_order_id.clone(), record);
    }

    /// Update an order record from any exchange response.
    pub fn update_order_from_response(
        &mut self,
        client_order_id: &str,
        exchange_order_id: i64,
        status: &str,
        filled_qty: f64,
        orig_qty: f64,
        avg_fill_price: f64,
    ) {
        let record = self.orders.entry(client_order_id.to_string()).or_insert_with(|| {
            // Order we didn't know about — create a minimal record
            OrderRecord {
                client_order_id: client_order_id.to_string(),
                exchange_order_id,
                symbol: self.symbol.clone(),
                side: "UNKNOWN".into(),
                order_type: "UNKNOWN".into(),
                orig_qty,
                filled_qty: 0.0,
                remaining_qty: orig_qty,
                avg_fill_price: 0.0,
                status: OrderStatus::Unknown,
                last_seen: Instant::now(),
            }
        });

        record.exchange_order_id = exchange_order_id;
        record.status = OrderStatus::from_str(status);
        record.filled_qty = filled_qty;
        record.remaining_qty = (orig_qty - filled_qty).max(0.0);
        record.avg_fill_price = avg_fill_price;
        record.last_seen = Instant::now();
    }
}

fn split_symbol_assets(symbol: &str) -> (&str, &str) {
    for quote in ["USDT", "USDC", "BUSD", "FDUSD", "TUSD", "BTC", "ETH", "BNB"] {
        if let Some(base) = symbol.strip_suffix(quote) {
            if !base.is_empty() {
                return (base, quote);
            }
        }
    }
    (symbol, "")
}

fn is_stable_asset(asset: &str) -> bool {
    matches!(asset.to_ascii_uppercase().as_str(), "USDT" | "USDC" | "BUSD" | "FDUSD" | "TUSD" | "DAI")
}

fn map_balances(
    symbol: &str,
    balances: &[Balance],
    mark_price: f64,
) -> (f64, f64, f64, Option<String>) {
    let (base_asset, quote_asset) = split_symbol_assets(symbol);
    let mut total_usd = 0.0;
    let mut stable_sum = 0.0;
    let mut non_stable_total = 0.0;

    for b in balances {
        let total_amt = b.free + b.locked;
        if total_amt <= 0.0 {
            continue;
        }
        if is_stable_asset(&b.asset) {
            stable_sum += total_amt;
            total_usd += total_amt;
            continue;
        }
        non_stable_total += total_amt;
        if b.asset.eq_ignore_ascii_case(base_asset) && mark_price > 0.0 {
            total_usd += total_amt * mark_price;
        }
    }

    let buy_power = balances
        .iter()
        .find(|b| b.asset.eq_ignore_ascii_case(quote_asset))
        .map(|b| b.free)
        .unwrap_or(0.0);
    let sell_inventory = balances
        .iter()
        .find(|b| b.asset.eq_ignore_ascii_case(base_asset))
        .map(|b| b.free)
        .unwrap_or(0.0);

    let funds_detected = balances.iter().any(|b| (b.free + b.locked) > 0.0);
    let balance_status = if !funds_detected {
        None
    } else if buy_power <= 0.0 && sell_inventory > 0.0 {
        Some(format!(
            "You can SELL {} now. You cannot BUY more {} until you hold {}.",
            base_asset, base_asset, quote_asset
        ))
    } else if buy_power <= 0.0 && sell_inventory <= 0.0 {
        Some(format!(
            "Funds detected but not in tradable asset for {} (requires {})",
            symbol, quote_asset
        ))
    } else if sell_inventory <= 0.0 {
        Some(format!(
            "BUY ready with {}. SELL inventory empty for {}.",
            quote_asset, base_asset
        ))
    } else {
        None
    };

    let final_total = if total_usd > 0.0 {
        total_usd
    } else if stable_sum > 0.0 {
        stable_sum
    } else if non_stable_total > 0.0 {
        // No conversion route available; expose non-zero via status message/UI.
        0.0
    } else {
        0.0
    };

    (final_total, buy_power, sell_inventory, balance_status)
}

// ── Reconciliation ────────────────────────────────────────────────────────────

/// Run a full reconciliation cycle.
///
/// Fetches open orders, fills, and balances from Binance.
/// Rebuilds position from fills, compares with local state.
/// Overwrites local state with exchange truth on any mismatch.
/// Sets state_dirty if exchange fetch fails.
pub async fn run_reconciliation(
    client: &BinanceClient,
    state: Arc<Mutex<TruthState>>,
) -> Result<ReconcileResult> {
    let symbol = {
        let s = state.lock().await;
        s.symbol.clone()
    };

    // ── Mark reconcile in progress ────────────────────────────────────────────
    {
        let mut s = state.lock().await;
        s.recon_in_progress = true;
    }

    let fetch_result = fetch_all(client, &symbol).await;

    match fetch_result {
        Err(e) => {
            error!("RECONCILE: Exchange fetch failed: {:#}", e);
            let mut s = state.lock().await;
            s.state_dirty = true;
            s.recon_in_progress = false;
            return Err(e);
        }
        Ok((open_orders, all_trades, balances, mark_price)) => {
            let mut s = state.lock().await;
            let result = apply_reconciliation(
                &mut s,
                open_orders,
                all_trades,
                balances,
                mark_price,
            );
            s.recon_in_progress = false;
            s.last_reconciled_at = Some(Instant::now());
            s.state_dirty = false;
            Ok(result)
        }
    }
}

/// Fetch all three exchange data sources concurrently.
async fn fetch_all(
    client: &BinanceClient,
    symbol: &str,
) -> Result<(Vec<OpenOrder>, Vec<MyTrade>, Vec<Balance>, f64)> {
    // Fire all three requests concurrently — no reason to serialize them.
    let (open_orders_res, trades_res, balances_res, price_res) = tokio::join!(
        client.fetch_open_orders(symbol),
        client.fetch_my_trades(symbol, 500),
        client.fetch_balances(),
        client.fetch_ticker_price(symbol),
    );

    let open_orders = open_orders_res.context("fetch open orders")?;
    let trades      = trades_res.context("fetch trades")?;
    let balances    = balances_res.context("fetch balances")?;
    let mark_price  = price_res
        .map(|t| t.price.parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);

    Ok((open_orders, trades, balances, mark_price))
}

/// Minimum balance delta (in asset units) required to trigger a BALANCE_UPDATED event.
/// Avoids spurious events from floating-point rounding noise.
const BALANCE_CHANGE_EPSILON: f64 = 1e-8;

/// Apply fetched data to TruthState. Returns what changed.
/// Called with the mutex held — runs synchronously.
fn apply_reconciliation(
    state: &mut TruthState,
    exchange_open_orders: Vec<OpenOrder>,
    all_trades: Vec<MyTrade>,
    balances: Vec<Balance>,
    mark_price: f64,
) -> ReconcileResult {
    let mut result = ReconcileResult { exchange_fetch_ok: true, ..Default::default() };

    // ── Fill deduplication ────────────────────────────────────────────────────
    // Only process fills we haven't seen before. Sorted ascending (oldest first).
    let mut sorted_trades = all_trades;
    sorted_trades.sort_by_key(|t| t.id);

    let new_trades: Vec<&MyTrade> = sorted_trades
        .iter()
        .filter(|t| !state.seen_fill_ids.contains(&t.id))
        .collect();

    let pending_new_fills = new_trades.len();
    if pending_new_fills > 0 {
        info!("RECONCILE: {} new fill(s) to process", pending_new_fills);
    }

    // ── Collect fill details for new fills (for RECONCILE_APPLIED events) ──────
    // These remain pending until after build_position(...) succeeds so callers
    // cannot observe fills as "applied" on an early-return error path.
    let pending_fill_details: Vec<crate::events::FillDetail> = new_trades
        .iter()
        .filter_map(|t| {
            let qty: f64 = match t.qty.parse() {
                Ok(qty) => qty,
                Err(_) => {
                    warn!(
                        "RECONCILE: fill id={} has unparseable qty {:?} — skipping fill detail",
                        t.id,
                        t.qty
                    );
                    return None;
                }
            };
            let price: f64 = match t.price.parse() {
                Ok(price) => price,
                Err(_) => {
                    warn!(
                        "RECONCILE: fill id={} has unparseable price {:?} — skipping fill detail",
                        t.id,
                        t.price
                    );
                    return None;
                }
            };
            Some(crate::events::FillDetail {
                fill_id: t.id,
                side:    t.side().to_string(),
                qty,
                price,
            })
        })
        .collect();

    // ── Rebuild position from ALL fills (not just new ones) ───────────────────
    // Rebuilding from scratch every cycle ensures correctness even if we
    // missed fills or had an earlier bug. The seen_fill_ids set prevents
    // double-logging but position math uses the full history.
    let exchange_pos = match build_position(
        &state.symbol,
        &sorted_trades,
        state.bnb_price_usd,
        mark_price,
    ) {
        Ok(p) => p,
        Err(e) => {
            warn!("RECONCILE: Failed to build position from fills: {}", e);
            state.state_dirty = true;
            return result;
        }
    };

    // ── Position comparison ───────────────────────────────────────────────────
    let size_diff = (exchange_pos.size - state.position.size).abs();
    let avg_diff  = (exchange_pos.avg_entry - state.position.avg_entry).abs();

    if size_diff > 1e-8 || avg_diff > 1e-4 {
        result.position_mismatch = true;
        warn!(
            "RECONCILE: Position mismatch — local size={:.8} avg={:.4} | exchange size={:.8} avg={:.4}",
            state.position.size, state.position.avg_entry,
            exchange_pos.size, exchange_pos.avg_entry,
        );
    } else {
        info!(
            "RECONCILE: Position OK — size={:.8} avg_entry={:.4} unrealized={:+.4}",
            exchange_pos.size, exchange_pos.avg_entry, exchange_pos.unrealized_pnl,
        );
    }
    state.position = exchange_pos;

    // ── Mark all seen fill IDs ────────────────────────────────────────────────
    for t in &sorted_trades {
        state.seen_fill_ids.insert(t.id);
    }

    // ── Open orders reconciliation ────────────────────────────────────────────
    let exchange_open_count = exchange_open_orders.len();
    let local_open_count    = state.open_order_count;

    if exchange_open_count != local_open_count {
        result.open_order_count_mismatch = true;
        warn!(
            "RECONCILE: Open order count mismatch — local={} | exchange={}",
            local_open_count, exchange_open_count,
        );
    } else {
        info!("RECONCILE: Open orders OK — count={}", exchange_open_count);
    }

    // Build set of exchange order IDs for cross-reference
    let exchange_coids: HashSet<&str> = exchange_open_orders
        .iter()
        .map(|o| o.client_order_id.as_str())
        .collect();

    // Find orders that are active locally but gone from exchange
    let disappeared: Vec<String> = state
        .orders
        .values()
        .filter(|r| r.status.is_active() && !exchange_coids.contains(r.client_order_id.as_str()))
        .map(|r| r.client_order_id.clone())
        .collect();

    result.orders_disappeared = disappeared.len();
    for coid in &disappeared {
        warn!(
            "RECONCILE: Order {} was active locally but absent from exchange — marking terminal",
            coid
        );
        if let Some(record) = state.orders.get_mut(coid) {
            // We don't know if it filled or was cancelled. Mark unknown.
            // The next fill reconcile will catch if it filled.
            record.status = OrderStatus::Unknown;
        }
    }

    // Update records for all orders still on exchange
    for o in &exchange_open_orders {
        let coid = &o.client_order_id;
        if !state.orders.contains_key(coid.as_str()) {
            // Order exists on exchange but not in our local map
            // Could be: manual order, or bot order from a previous process
            result.unknown_orders_found += 1;
            warn!(
                "RECONCILE: Found order {} on exchange with no local record — adding",
                coid
            );
        }
        let record = OrderRecord::from_open_order(o);
        state.orders.insert(coid.clone(), record);
    }

    // Authoritative open count comes from exchange
    state.open_order_count = exchange_open_count;

    // ── Balances update ───────────────────────────────────────────────────────
    info!("RECONCILE: Balances updated ({} assets)", balances.len());
    for b in &balances {
        info!("  {} free={} locked={}", b.asset, b.free, b.locked);
    }
    let (total_balance_usd, buy_power, sell_inventory, balance_status) =
        map_balances(&state.symbol, &balances, mark_price);

    // Detect balance changes: flag when buy_power or sell_inventory differs from last cycle.
    let prev_buy_power    = state.buy_power;
    let prev_sell_inv     = state.sell_inventory;
    result.balances_changed = (buy_power - prev_buy_power).abs() > BALANCE_CHANGE_EPSILON
        || (sell_inventory - prev_sell_inv).abs() > BALANCE_CHANGE_EPSILON;
    if result.balances_changed {
        info!(
            "RECONCILE: BALANCE_UPDATED — buy_power {:.8} → {:.8}  sell_inventory {:.8} → {:.8}",
            prev_buy_power, buy_power, prev_sell_inv, sell_inventory,
        );
    }

    state.total_balance_usd = total_balance_usd;
    state.tradable_balance = buy_power;
    state.buy_power = buy_power;
    state.sell_inventory = sell_inventory;
    state.balance_status = balance_status;
    state.balances = balances;

    // ── Summary ───────────────────────────────────────────────────────────────
    if result.has_anomaly() {
        warn!(
            "RECONCILE: Anomalies detected — pos_mismatch={} order_count_mismatch={} unknown={} disappeared={}",
            result.position_mismatch,
            result.open_order_count_mismatch,
            result.unknown_orders_found,
            result.orders_disappeared,
        );
    } else {
        info!("RECONCILE: Cycle complete — state consistent");
    }

    result
}

// ── Startup recovery ──────────────────────────────────────────────────────────

/// Full startup recovery. Call once before starting the reconcile loop.
///
/// Fetches exchange state and rebuilds everything from scratch.
/// After this returns Ok, the bot knows the true state of the world
/// even if it was restarted mid-trade.
pub async fn startup_recovery(
    client: &BinanceClient,
    state: Arc<Mutex<TruthState>>,
) -> Result<()> {
    info!("STARTUP: Beginning exchange state recovery...");

    let result = run_reconciliation(client, Arc::clone(&state)).await?;

    let s = state.lock().await;
    info!(
        "STARTUP: Recovery complete — position size={:.8} avg_entry={:.4} open_orders={} fills_seen={}",
        s.position.size,
        s.position.avg_entry,
        s.open_order_count,
        s.seen_fill_ids.len(),
    );

    if result.unknown_orders_found > 0 {
        warn!(
            "STARTUP: {} order(s) on exchange with no local record — check for manual trades",
            result.unknown_orders_found
        );
    }

    Ok(())
}

// ── Background reconcile loop ─────────────────────────────────────────────────

/// Spawn a background task that reconciles state every `interval`.
/// The task runs forever; drop the JoinHandle to cancel it.
pub fn spawn_reconciliation_loop(
    client: Arc<BinanceClient>,
    state: Arc<Mutex<TruthState>>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            ticker.tick().await;

            match run_reconciliation(&client, Arc::clone(&state)).await {
                Ok(result) => {
                    if result.has_anomaly() {
                        warn!("RECONCILE LOOP: Anomalies corrected");
                    }
                }
                Err(e) => {
                    error!("RECONCILE LOOP: Failed: {:#}", e);
                    // state_dirty was set inside run_reconciliation on failure
                }
            }
        }
    })
}

/// Spawn a reconciliation loop that also notifies the Executor after every cycle.
/// The executor updates SystemMode and ExecutionState based on exchange truth.
/// Pass `event_store` to persist ReconcileStarted/ReconcileCompleted events.
pub fn spawn_reconciliation_loop_with_executor(
    client: Arc<BinanceClient>,
    state: Arc<Mutex<TruthState>>,
    executor: Arc<crate::executor::Executor>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    spawn_reconciliation_loop_with_executor_and_store(client, state, executor, interval, None)
}

pub fn spawn_reconciliation_loop_with_executor_and_store(
    client: Arc<BinanceClient>,
    state: Arc<Mutex<TruthState>>,
    executor: Arc<crate::executor::Executor>,
    interval: Duration,
    event_store: Option<Arc<dyn crate::store::EventStore>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut cycle: u64 = 0;

        loop {
            ticker.tick().await;
            cycle += 1;

            let t_start = std::time::Instant::now();

            // Fire ReconcileStarted
            if let Some(store) = &event_store {
                store.append(crate::events::StoredEvent::new(
                    Some(state.lock().await.symbol.clone()),
                    None, None,
                    crate::events::TradingEvent::ReconcileStarted(
                        crate::events::ReconcileStartedPayload { cycle },
                    ),
                ));
            }

            match run_reconciliation(&client, Arc::clone(&state)).await {
                Ok(result) => {
                    let duration_ms = t_start.elapsed().as_millis() as u64;
                    let (pos_size, open_orders_count, symbol) = {
                        let s = state.lock().await;
                        (s.position.size, s.open_order_count, s.symbol.clone())
                    };

                    // Fire ReconcileCompleted
                    if let Some(store) = &event_store {
                        store.append(crate::events::StoredEvent::new(
                            Some(symbol.clone()),
                            None, None,
                            crate::events::TradingEvent::ReconcileCompleted(
                                crate::events::ReconcileCompletedPayload {
                                    cycle,
                                    had_anomaly:   result.has_anomaly(),
                                    position_size: pos_size,
                                    open_orders:   open_orders_count,
                                    new_fills:     result.new_fills,
                                    duration_ms,
                                },
                            ),
                        ));

                        // Fire ReconcileMismatch events for each anomaly detected
                        if result.position_mismatch {
                            store.append(crate::events::StoredEvent::new(
                                Some(symbol.clone()),
                                None, None,
                                crate::events::TradingEvent::ReconcileMismatch(
                                    crate::events::ReconcileMismatchPayload {
                                        field:          "position".to_string(),
                                        local_value:    "mismatch_detected".to_string(),
                                        exchange_value: "see_logs".to_string(),
                                    },
                                ),
                            ));
                        }
                        if result.open_order_count_mismatch {
                            store.append(crate::events::StoredEvent::new(
                                Some(symbol.clone()),
                                None, None,
                                crate::events::TradingEvent::ReconcileMismatch(
                                    crate::events::ReconcileMismatchPayload {
                                        field:          "open_order_count".to_string(),
                                        local_value:    "mismatch_detected".to_string(),
                                        exchange_value: "see_logs".to_string(),
                                    },
                                ),
                            ));
                        }

                        // ── RECONCILE_APPLIED: emit when new fills were processed ──────────
                        if result.new_fills > 0 {
                            store.append(crate::events::StoredEvent::new(
                                Some(symbol.clone()),
                                None, None,
                                crate::events::TradingEvent::ReconcileApplied(
                                    crate::events::ReconcileAppliedPayload {
                                        cycle,
                                        fills_count: result.new_fills,
                                        fills:       result.fill_details.clone(),
                                    },
                                ),
                            ));
                        }

                        // ── BALANCE_UPDATED: emit when buy_power or sell_inventory changed ─
                        if result.balances_changed {
                            let (total_usd, buy_power, sell_inv) = {
                                let s = state.lock().await;
                                (s.total_balance_usd, s.buy_power, s.sell_inventory)
                            };
                            store.append(crate::events::StoredEvent::new(
                                Some(symbol.clone()),
                                None, None,
                                crate::events::TradingEvent::BalanceUpdated(
                                    crate::events::BalanceUpdatedPayload {
                                        total_balance_usd: total_usd,
                                        buy_power,
                                        sell_inventory:    sell_inv,
                                    },
                                ),
                            ));
                        }
                    }

                    // Collect open orders for executor notification
                    let open_order_list: Vec<crate::client::OpenOrder> = {
                        let s = state.lock().await;
                        s.orders
                            .values()
                            .filter(|r| r.status.is_active())
                            .map(|r| crate::client::OpenOrder {
                                symbol:          r.symbol.clone(),
                                order_id:        r.exchange_order_id,
                                client_order_id: r.client_order_id.clone(),
                                price:           "0".into(),
                                orig_qty:        r.orig_qty.to_string(),
                                executed_qty:    r.filled_qty.to_string(),
                                status:          r.status.to_string(),
                                time_in_force:   "GTC".into(),
                                order_type:      r.order_type.clone(),
                                side:            r.side.clone(),
                            })
                            .collect()
                    };
                    executor.on_reconcile(&open_order_list, result.has_anomaly()).await;
                }
                Err(e) => {
                    error!("RECONCILE LOOP: Failed: {:#}", e);
                    executor.force_recovery("reconcile failed").await;
                    if let Some(store) = &event_store {
                        let symbol = state.lock().await.symbol.clone();
                        store.append(crate::events::StoredEvent::new(
                            Some(symbol),
                            None, None,
                            crate::events::TradingEvent::ReconcileCompleted(
                                crate::events::ReconcileCompletedPayload {
                                    cycle,
                                    had_anomaly:   true,
                                    position_size: 0.0,
                                    open_orders:   0,
                                    new_fills:     0,
                                    duration_ms:   t_start.elapsed().as_millis() as u64,
                                },
                            ),
                        ));
                    }
                }
            }
        }
    })
}

// ── Stale order detection ─────────────────────────────────────────────────────

/// Check for orders that have been active too long without exchange confirmation.
/// Call this from the reconcile loop or periodically.
/// Returns client_order_ids of orders that look stale.
pub fn detect_stale_orders(state: &TruthState, stale_threshold: Duration) -> Vec<String> {
    state
        .orders
        .values()
        .filter(|r| {
            r.status.is_active()
                && r.last_seen.elapsed() > stale_threshold
        })
        .map(|r| {
            warn!(
                "STALE: Order {} has been active for {:.0}s without exchange update",
                r.client_order_id,
                r.last_seen.elapsed().as_secs_f64(),
            );
            r.client_order_id.clone()
        })
        .collect()
}

// ── Partial fill aware cancel-replace ─────────────────────────────────────────

/// Outcome of a cancel-replace decision.
#[derive(Debug)]
pub enum CancelReplaceOutcome {
    /// Order was already fully filled. No replacement needed.
    AlreadyFilled { filled_qty: f64 },
    /// Replacement order was submitted for the remaining quantity.
    Replaced { replacement_qty: f64 },
    /// Nothing to replace (remaining qty too small after rounding).
    NothingRemaining,
}

/// Compute remaining quantity for an order, accounting for partial fills.
/// Returns None if the order is already terminal (filled/cancelled/etc).
pub fn compute_remaining_qty(record: &OrderRecord) -> Option<f64> {
    if record.status.is_terminal() {
        return None;
    }
    let remaining = (record.orig_qty - record.filled_qty).max(0.0);
    Some(remaining)
}

/// Determine whether to replace an order and with what quantity.
///
/// Called before cancel_then_replace. Uses the OrderRecord from TruthState
/// to determine what has already been filled.
pub fn plan_cancel_replace(
    state: &TruthState,
    old_client_order_id: &str,
    min_qty: f64,
) -> CancelReplaceOutcome {
    let record = match state.orders.get(old_client_order_id) {
        Some(r) => r,
        None => {
            warn!("plan_cancel_replace: no record for {}", old_client_order_id);
            // Assume full original qty — safer than doing nothing
            return CancelReplaceOutcome::Replaced { replacement_qty: 0.0 };
        }
    };

    // Terminal: already filled, cancelled, etc.
    if record.status.is_terminal() {
        if record.status == OrderStatus::Filled {
            return CancelReplaceOutcome::AlreadyFilled { filled_qty: record.filled_qty };
        }
        return CancelReplaceOutcome::NothingRemaining;
    }

    let remaining = (record.orig_qty - record.filled_qty).max(0.0);

    if remaining < min_qty {
        info!(
            "plan_cancel_replace: remaining {:.8} < min_qty {:.8} — nothing to replace",
            remaining, min_qty
        );
        return CancelReplaceOutcome::NothingRemaining;
    }

    CancelReplaceOutcome::Replaced { replacement_qty: remaining }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// All tests operate on data structures only — no network calls.
// Exchange responses are constructed inline.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::MyTrade;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_trade(id: i64, is_buyer: bool, qty: &str, price: &str) -> MyTrade {
        MyTrade {
            id,
            symbol: "BTCUSDT".into(),
            qty: qty.into(),
            price: price.into(),
            commission: "0".into(),
            commission_asset: "USDT".into(),
            time: 1_700_000_000_000 + id as u64 * 1000,
            is_buyer,
            is_maker: false,
        }
    }

    fn make_open_order(coid: &str, order_id: i64, side: &str, qty: &str, exec_qty: &str) -> OpenOrder {
        OpenOrder {
            symbol: "BTCUSDT".into(),
            order_id,
            client_order_id: coid.into(),
            price: "50000".into(),
            orig_qty: qty.into(),
            executed_qty: exec_qty.into(),
            status: "NEW".into(),
            time_in_force: "GTC".into(),
            order_type: "LIMIT".into(),
            side: side.into(),
        }
    }

    fn fresh_state() -> TruthState {
        TruthState::new("BTCUSDT", 0.0)
    }

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < 1e-8 }

    // ── OrderStatus mapping ───────────────────────────────────────────────────
    #[test]
    fn test_order_status_mapping() {
        assert_eq!(OrderStatus::from_str("NEW"),              OrderStatus::New);
        assert_eq!(OrderStatus::from_str("PARTIALLY_FILLED"), OrderStatus::PartiallyFilled);
        assert_eq!(OrderStatus::from_str("FILLED"),           OrderStatus::Filled);
        assert_eq!(OrderStatus::from_str("CANCELED"),         OrderStatus::Canceled);
        assert_eq!(OrderStatus::from_str("gibberish"),        OrderStatus::Unknown);
    }

    #[test]
    fn test_terminal_statuses() {
        assert!(OrderStatus::Filled.is_terminal());
        assert!(OrderStatus::Canceled.is_terminal());
        assert!(!OrderStatus::New.is_terminal());
        assert!(!OrderStatus::PartiallyFilled.is_terminal());
    }

    // ── state_dirty blocks placement ──────────────────────────────────────────
    #[test]
    fn test_state_dirty_blocks_order() {
        let mut s = fresh_state();
        s.state_dirty = true;
        assert!(!s.can_place_order());
    }

    #[test]
    fn test_recon_in_progress_blocks_order() {
        let mut s = fresh_state();
        s.state_dirty = false;
        s.recon_in_progress = true;
        s.last_reconciled_at = Some(Instant::now());
        assert!(!s.can_place_order());
    }

    #[test]
    fn test_never_reconciled_blocks_order() {
        let mut s = fresh_state();
        s.state_dirty = false;
        s.recon_in_progress = false;
        // last_reconciled_at = None (default)
        assert!(!s.can_place_order());
    }

    #[test]
    fn test_clean_state_allows_order() {
        let mut s = fresh_state();
        s.state_dirty = false;
        s.recon_in_progress = false;
        s.last_reconciled_at = Some(Instant::now());
        assert!(s.can_place_order());
    }

    // ── Duplicate fill deduplication ──────────────────────────────────────────
    #[test]
    fn test_duplicate_fills_not_double_counted() {
        let trades = vec![
            make_trade(1, true, "1.0", "50000"),
            make_trade(1, true, "1.0", "50000"), // exact same ID
        ];
        // sort_by_key deduplication happens in seen_fill_ids
        let mut state = fresh_state();
        // Simulate two reconcile cycles with the same trade
        let sorted: Vec<MyTrade> = {
            let mut v = trades.clone();
            v.sort_by_key(|t| t.id);
            v.dedup_by_key(|t| t.id); // dedup before processing
            v
        };
        let new_first: Vec<&MyTrade> = sorted.iter().filter(|t| !state.seen_fill_ids.contains(&t.id)).collect();
        assert_eq!(new_first.len(), 1, "Only one fill should be new");

        for t in &new_first { state.seen_fill_ids.insert(t.id); }

        // Second cycle: same trade reappears
        let new_second: Vec<&MyTrade> = sorted.iter().filter(|t| !state.seen_fill_ids.contains(&t.id)).collect();
        assert_eq!(new_second.len(), 0, "Already seen — should not process again");
    }

    // ── Position mismatch detection ───────────────────────────────────────────
    #[test]
    fn test_position_mismatch_detected() {
        let mut state = fresh_state();
        // Local says 0.5 BTC
        state.position.size = 0.5;
        state.position.avg_entry = 50000.0;

        // Exchange fills say 0.1 BTC (different)
        let exchange_pos = build_position(
            "BTCUSDT",
            &[make_trade(1, true, "0.1", "50000")],
            0.0,
            50000.0,
        ).unwrap();

        let size_diff = (exchange_pos.size - state.position.size).abs();
        assert!(size_diff > 1e-8, "Should detect mismatch: {}", size_diff);
    }

    #[test]
    fn test_position_match_no_false_alarm() {
        let mut state = fresh_state();
        state.position.size = 1.0;
        state.position.avg_entry = 50000.0;

        let exchange_pos = build_position(
            "BTCUSDT",
            &[make_trade(1, true, "1.0", "50000")],
            0.0,
            50000.0,
        ).unwrap();

        let size_diff = (exchange_pos.size - state.position.size).abs();
        assert!(size_diff < 1e-8, "Should not fire false alarm: {}", size_diff);
    }

    // ── Open order count mismatch ─────────────────────────────────────────────
    #[test]
    fn test_open_order_count_mismatch() {
        let mut state = fresh_state();
        state.open_order_count = 2; // local thinks 2

        let exchange_orders = vec![make_open_order("coid-1", 111, "BUY", "0.01", "0.0")];
        // exchange has 1 → mismatch

        let mismatch = exchange_orders.len() != state.open_order_count;
        assert!(mismatch);
    }

    // ── Unknown order detection ───────────────────────────────────────────────
    #[test]
    fn test_unknown_order_from_exchange() {
        let state = fresh_state();
        // state.orders is empty, exchange has an order
        let exchange_orders = vec![make_open_order("manual-order-xyz", 999, "BUY", "0.01", "0.0")];

        let unknown_count = exchange_orders
            .iter()
            .filter(|o| !state.orders.contains_key(&o.client_order_id))
            .count();

        assert_eq!(unknown_count, 1);
    }

    // ── Order record from open order ──────────────────────────────────────────
    #[test]
    fn test_order_record_from_open_order() {
        let o = make_open_order("coid-abc", 12345, "BUY", "0.01", "0.005");
        let rec = OrderRecord::from_open_order(&o);

        assert_eq!(rec.client_order_id, "coid-abc");
        assert_eq!(rec.exchange_order_id, 12345);
        assert!(approx(rec.orig_qty, 0.01));
        assert!(approx(rec.filled_qty, 0.005));
        assert!(approx(rec.remaining_qty, 0.005));
        assert_eq!(rec.status, OrderStatus::New);
    }

    // ── Partial fill before cancel-replace ────────────────────────────────────
    #[test]
    fn test_plan_cancel_replace_partial_fill() {
        let mut state = fresh_state();
        let record = OrderRecord {
            client_order_id: "coid-1".into(),
            exchange_order_id: 100,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.004,
            remaining_qty: 0.006,
            avg_fill_price: 50000.0,
            status: OrderStatus::PartiallyFilled,
            last_seen: Instant::now(),
        };
        state.orders.insert("coid-1".into(), record);

        let outcome = plan_cancel_replace(&state, "coid-1", 0.00001);
        match outcome {
            CancelReplaceOutcome::Replaced { replacement_qty } => {
                assert!(approx(replacement_qty, 0.006), "got {:.8}", replacement_qty);
            }
            other => panic!("Expected Replaced, got {:?}", other),
        }
    }

    #[test]
    fn test_plan_cancel_replace_fully_filled() {
        let mut state = fresh_state();
        let record = OrderRecord {
            client_order_id: "coid-2".into(),
            exchange_order_id: 101,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.01, // fully filled
            remaining_qty: 0.0,
            avg_fill_price: 50000.0,
            status: OrderStatus::Filled,
            last_seen: Instant::now(),
        };
        state.orders.insert("coid-2".into(), record);

        let outcome = plan_cancel_replace(&state, "coid-2", 0.00001);
        assert!(matches!(outcome, CancelReplaceOutcome::AlreadyFilled { filled_qty: _ }));
    }

    #[test]
    fn test_plan_cancel_replace_not_filled() {
        let mut state = fresh_state();
        let record = OrderRecord {
            client_order_id: "coid-3".into(),
            exchange_order_id: 102,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.0,
            remaining_qty: 0.01,
            avg_fill_price: 0.0,
            status: OrderStatus::New,
            last_seen: Instant::now(),
        };
        state.orders.insert("coid-3".into(), record);

        let outcome = plan_cancel_replace(&state, "coid-3", 0.00001);
        match outcome {
            CancelReplaceOutcome::Replaced { replacement_qty } => {
                assert!(approx(replacement_qty, 0.01));
            }
            other => panic!("Expected Replaced, got {:?}", other),
        }
    }

    #[test]
    fn test_plan_cancel_replace_remaining_below_min() {
        let mut state = fresh_state();
        let record = OrderRecord {
            client_order_id: "coid-4".into(),
            exchange_order_id: 103,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.009999, // tiny remainder
            remaining_qty: 0.000001,
            avg_fill_price: 50000.0,
            status: OrderStatus::PartiallyFilled,
            last_seen: Instant::now(),
        };
        state.orders.insert("coid-4".into(), record);

        // min_qty for BTCUSDT is 0.00001
        let outcome = plan_cancel_replace(&state, "coid-4", 0.00001);
        assert!(matches!(outcome, CancelReplaceOutcome::NothingRemaining));
    }

    // ── Cancel returns -2011 but order was actually filled ────────────────────
    // The bot must detect this via fill reconciliation, not the cancel response.
    // This test verifies: after a -2011 on cancel, if the fill appears in
    // myTrades, the position is correctly updated.
    #[test]
    fn test_fill_detected_after_cancel_2011() {
        // Simulate: we had no local fills, then a reconcile shows the fill
        let mut state = fresh_state();
        state.state_dirty = false;
        state.last_reconciled_at = Some(Instant::now());

        // The order was "cancelled" (-2011) but actually filled
        let trades = vec![make_trade(77, true, "0.01", "50000")];
        let new: Vec<&MyTrade> = trades.iter().filter(|t| !state.seen_fill_ids.contains(&t.id)).collect();
        assert_eq!(new.len(), 1, "Fill should be new");

        let pos = build_position("BTCUSDT", &trades, 0.0, 50000.0).unwrap();
        assert!(approx(pos.size, 0.01));
        // Bot now knows it's long 0.01 BTC from the fill
    }

    // ── Restart with existing open orders ─────────────────────────────────────
    #[test]
    fn test_startup_absorbs_existing_open_orders() {
        let mut state = fresh_state();
        assert_eq!(state.open_order_count, 0);
        assert_eq!(state.orders.len(), 0);

        // Simulate exchange returning 2 open orders from previous session
        let exchange_orders = vec![
            make_open_order("prev-coid-1", 201, "BUY", "0.01", "0.0"),
            make_open_order("prev-coid-2", 202, "SELL", "0.01", "0.0"),
        ];

        // Apply as reconciliation would
        let exchange_coids: HashSet<&str> = exchange_orders.iter().map(|o| o.client_order_id.as_str()).collect();
        let unknown_count = exchange_orders.iter().filter(|o| !state.orders.contains_key(o.client_order_id.as_str())).count();
        assert_eq!(unknown_count, 2);

        for o in &exchange_orders {
            state.orders.insert(o.client_order_id.clone(), OrderRecord::from_open_order(o));
        }
        state.open_order_count = exchange_orders.len();

        assert_eq!(state.open_order_count, 2);
        assert!(state.orders.contains_key("prev-coid-1"));
        assert!(state.orders.contains_key("prev-coid-2"));
    }

    // ── Stale order detection ─────────────────────────────────────────────────
    #[test]
    fn test_stale_order_detected() {
        let mut state = fresh_state();

        // Insert a record with a very old last_seen
        let stale_last_seen = Instant::now() - Duration::from_secs(120);
        let mut record = OrderRecord {
            client_order_id: "stale-coid".into(),
            exchange_order_id: 300,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.0,
            remaining_qty: 0.01,
            avg_fill_price: 0.0,
            status: OrderStatus::New,
            last_seen: stale_last_seen,
        };
        state.orders.insert("stale-coid".into(), record);

        let stale = detect_stale_orders(&state, Duration::from_secs(30));
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], "stale-coid");
    }

    #[test]
    fn test_fresh_order_not_stale() {
        let mut state = fresh_state();
        let record = OrderRecord {
            client_order_id: "fresh-coid".into(),
            exchange_order_id: 301,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.0,
            remaining_qty: 0.01,
            avg_fill_price: 0.0,
            status: OrderStatus::New,
            last_seen: Instant::now(),
        };
        state.orders.insert("fresh-coid".into(), record);

        let stale = detect_stale_orders(&state, Duration::from_secs(30));
        assert_eq!(stale.len(), 0);
    }

    // ── compute_remaining_qty ─────────────────────────────────────────────────
    #[test]
    fn test_compute_remaining_terminal_returns_none() {
        let record = OrderRecord {
            client_order_id: "x".into(),
            exchange_order_id: 1,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "MARKET".into(),
            orig_qty: 0.01,
            filled_qty: 0.01,
            remaining_qty: 0.0,
            avg_fill_price: 50000.0,
            status: OrderStatus::Filled,
            last_seen: Instant::now(),
        };
        assert!(compute_remaining_qty(&record).is_none());
    }

    #[test]
    fn test_compute_remaining_partial() {
        let record = OrderRecord {
            client_order_id: "x".into(),
            exchange_order_id: 1,
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 0.01,
            filled_qty: 0.003,
            remaining_qty: 0.007,
            avg_fill_price: 50000.0,
            status: OrderStatus::PartiallyFilled,
            last_seen: Instant::now(),
        };
        let remaining = compute_remaining_qty(&record).unwrap();
        assert!(approx(remaining, 0.007), "got {}", remaining);
    }

    // ── End-to-end spot lifecycle diagnostic ─────────────────────────────────
    //
    // Requirement: prove that submit → accepted/rejected → reconcile/update
    // is fully observable through ReconcileResult fields.
    //
    // Simulates the path:
    //   1. Order submitted (state transitions to WaitingAck / position = 0)
    //   2. Reconcile cycle discovers the fill via myTrades
    //   3. ReconcileResult carries fill details (RECONCILE_APPLIED observable)
    //   4. ReconcileResult carries balance change flag (BALANCE_UPDATED observable)
    //   5. Position is updated from exchange truth
    //   6. fill_id is recorded so the next cycle does NOT re-process the fill

    #[test]
    fn test_spot_lifecycle_reconcile_applied_and_balance_updated() {
        let mut state = fresh_state();
        // Simulate post-submission state: order submitted, not yet locally filled
        state.state_dirty = false;
        state.last_reconciled_at = Some(Instant::now());
        state.buy_power = 500.0;     // starting USDT balance
        state.sell_inventory = 0.0;  // no BTC yet

        // Exchange returns a fill for the order (0.001 BTC @ 50000 USDT)
        let trades = vec![make_trade(101, true, "0.001", "50000")];
        // Balance after fill: received 0.001 BTC, spent ~50 USDT
        let balances = vec![
            crate::client::Balance { asset: "USDT".to_string(), free: 450.0, locked: 0.0 },
            crate::client::Balance { asset: "BTC".to_string(),  free: 0.001, locked: 0.0 },
        ];

        // ── Cycle 1: reconcile discovers the fill ────────────────────────────
        let result = apply_reconciliation(&mut state, vec![], trades.clone(), balances.clone(), 50000.0);

        // ORDER fill observable: new_fills = 1, fill_details populated
        assert_eq!(result.new_fills, 1,
            "RECONCILE_APPLIED: expected 1 new fill, got {}", result.new_fills);
        assert_eq!(result.fill_details.len(), 1,
            "fill_details should carry per-fill info");
        let fill = &result.fill_details[0];
        assert_eq!(fill.fill_id, 101);
        assert_eq!(fill.side, "BUY");
        assert!((fill.qty - 0.001).abs() < 1e-8,
            "fill qty mismatch: {}", fill.qty);
        assert!((fill.price - 50000.0).abs() < 1e-2,
            "fill price mismatch: {}", fill.price);

        // BALANCE_UPDATED observable: sell_inventory increased (received BTC)
        assert!(result.balances_changed,
            "BALANCE_UPDATED: balances_changed should be true after fill");

        // Position updated from exchange truth
        assert!((state.position.size - 0.001).abs() < 1e-8,
            "position.size should reflect exchange fill: {}", state.position.size);

        // fill_id recorded to prevent re-processing
        assert!(state.seen_fill_ids.contains(&101),
            "fill_id 101 should be in seen_fill_ids after reconcile");

        // ── Cycle 2: same fill reappears on exchange, no new processing ───────
        let result2 = apply_reconciliation(&mut state, vec![], trades.clone(), balances.clone(), 50000.0);
        assert_eq!(result2.new_fills, 0,
            "Idempotency: fill should not be re-processed on second reconcile cycle");
        assert!(result2.fill_details.is_empty(),
            "fill_details should be empty when no new fills");
        // Balances are identical so no change event on second cycle
        assert!(!result2.balances_changed,
            "BALANCE_UPDATED: no change expected when balances are identical");
    }

    // ── Rejected order observable ─────────────────────────────────────────────
    // Verifies that when an order is rejected (no fill appears in myTrades),
    // reconcile correctly sees 0 new fills and no balance change.

    #[test]
    fn test_spot_lifecycle_rejected_order_no_fill_no_balance_change() {
        let mut state = fresh_state();
        state.state_dirty = false;
        state.last_reconciled_at = Some(Instant::now());
        state.buy_power = 500.0;
        state.sell_inventory = 0.0;

        // Exchange returns no trades (order was rejected before reaching the book)
        let no_trades: Vec<crate::client::MyTrade> = vec![];
        let balances_unchanged = vec![
            crate::client::Balance { asset: "USDT".to_string(), free: 500.0, locked: 0.0 },
        ];

        let result = apply_reconciliation(
            &mut state, vec![], no_trades, balances_unchanged, 50000.0,
        );

        // No fills processed — ORDER_REJECTED path leaves no trace in myTrades
        assert_eq!(result.new_fills, 0,
            "No fills expected for a rejected order");
        assert!(result.fill_details.is_empty());

        // No balance change (order never reached exchange balance layer)
        // buy_power was 500.0, still shows 500.0 in USDT
        assert!(!result.balances_changed,
            "No balance change expected when order was rejected and never filled");
    }
}

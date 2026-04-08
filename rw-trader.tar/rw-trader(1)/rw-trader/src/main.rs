mod assistant;
mod authority;
mod client;
mod events;
mod executor;
mod feed;
mod orders;
mod position;
mod reader;
mod reconciler;
mod replay;
mod risk;
mod signal;
mod store;
mod strategy;
mod suggestions;
mod webui;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // ── CLI subcommand: --timeline <correlation_id> ───────────────────────────
    // Must be checked before any trading startup or logging setup so it can be
    // used as a standalone read-only tool without exchange credentials.
    {
        let args: Vec<String> = std::env::args().collect();
        if let Some(pos) = args.iter().position(|a| a == "--timeline") {
            let corr_id = args.get(pos + 1)
                .ok_or_else(|| anyhow::anyhow!("--timeline requires a correlation_id argument"))?;

            // Minimal logging: warnings only, no noise.
            tracing_subscriber::fmt()
                .with_env_filter("warn")
                .init();

            let db_path = std::env::var("EVENT_DB_PATH")
                .unwrap_or_else(|_| "rw-trader-events.db".into());
            let store = store::SqliteEventStore::open(std::path::Path::new(&db_path))
                .map_err(|e| anyhow::anyhow!(
                    "Cannot open event store at '{}': {}\n\
                     Set EVENT_DB_PATH to point to the correct database file.",
                    db_path, e
                ))?;

            let timeline = reader::get_trade_timeline(&*store, corr_id)?;
            reader::print_trade_timeline(&timeline);
            return Ok(());
        }

        // --serve: read-only web UI against an existing database (no trading)
        if let Some(pos) = args.iter().position(|a| a == "--serve") {
            let addr = args.get(pos + 1)
                .map(|s| s.as_str())
                .unwrap_or("127.0.0.1:8080");

            tracing_subscriber::fmt().with_env_filter("info").init();

            let db_path = std::env::var("EVENT_DB_PATH")
                .unwrap_or_else(|_| "rw-trader-events.db".into());
            let store = store::SqliteEventStore::open(std::path::Path::new(&db_path))
                .map_err(|e| anyhow::anyhow!("Cannot open event store at '{}': {}", db_path, e))?;
            let store: Arc<dyn store::EventStore> = store;

            // In serve-only mode we have no live trading state.
            // Construct minimal stubs so AppState is satisfied.
            let stub_pos = position::Position::new("BTCUSDT");
            let stub_risk = risk::RiskEngine::new(risk::RiskConfig {
                max_position_qty:    0.0,
                max_daily_loss_usd:  0.0,
                max_drawdown_usd:    0.0,
                cooldown_after_loss: Duration::from_secs(0),
                max_spread_bps:      0.0,
                max_feed_staleness:  Duration::from_secs(0),
                min_order_interval:  Duration::from_secs(0),
                signal_dedup_window: Duration::from_secs(0),
                max_open_orders:     0,
                max_slippage_bps:    0.0,
            }, &stub_pos);
            let stub_exec = executor::Executor::new(
                "BTCUSDT".into(),
                executor::CircuitBreakerConfig::default(),
                executor::WatchdogConfig::default(),
            );
            let state = webui::AppState {
                store:     Arc::clone(&store),
                exec:      Arc::new(stub_exec),
                truth:     Arc::new(Mutex::new(reconciler::TruthState::new("BTCUSDT", 0.0))),
                risk:      Arc::new(Mutex::new(stub_risk)),
                authority: Arc::new(authority::AuthorityLayer::new()),
                strategy:  Arc::new(Mutex::new(strategy::StrategyEngine::new())),
            };
            webui::run(addr, state).await?;
            return Ok(());
        }
        if let Some(pos) = args.iter().position(|a| a == "--recent") {
            let n: usize = args.get(pos + 1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);

            tracing_subscriber::fmt().with_env_filter("warn").init();

            let db_path = std::env::var("EVENT_DB_PATH")
                .unwrap_or_else(|_| "rw-trader-events.db".into());
            let store = store::SqliteEventStore::open(std::path::Path::new(&db_path))
                .map_err(|e| anyhow::anyhow!("Cannot open event store at '{}': {}", db_path, e))?;

            let events = store.fetch_recent(n)?;
            // Give the background writer a moment if we just opened the store
            std::thread::sleep(std::time::Duration::from_millis(50));

            println!();
            println!("{}", "═".repeat(100));
            println!("  RECENT EVENTS  (newest first, n={})", n);
            println!("{}", "═".repeat(100));
            println!("  {:26}  {:26}  {:<24}  {}",
                "occurred_at", "correlation_id", "event_type", "summary");
            println!("{}", "─".repeat(100));
            for e in &events {
                let corr = e.correlation_id.as_deref().unwrap_or("—");
                // Truncate correlation_id display to 24 chars
                let corr_short = if corr.len() > 24 { &corr[..24] } else { corr };
                let ts = e.occurred_at.format("%Y-%m-%d %H:%M:%S%.3f");
                let summary = reader::summarise_event(&e);
                // Truncate summary to fit terminal
                let summary_short = if summary.len() > 60 { format!("{}…", &summary[..59]) } else { summary };
                println!("  {}  {:<26}  {:<24}  {}",
                    ts, corr_short, e.event_type, summary_short);
            }
            println!("{}", "═".repeat(100));
            println!("  {} event(s)", events.len());
            println!();
            return Ok(());
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rw_trader=info".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    // ── Config ───────────────────────────────────────────────────────────────
    let api_key    = require_env("BINANCE_API_KEY")?;
    let api_secret = require_env("BINANCE_API_SECRET")?;
    let rest_url   = require_env("BINANCE_REST_URL")?;
    let ws_url     = require_env("BINANCE_WS_URL")?;
    let symbol     = std::env::var("SYMBOL").unwrap_or_else(|_| "BTCUSDT".into());
    let live_trade = std::env::var("LIVE_TRADE")
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let bnb_price: f64 = env_f64("BNB_PRICE_USD", 0.0);

    let client = Arc::new(client::BinanceClient::new(api_key, api_secret, rest_url));

    // ── 1. Clock sync ────────────────────────────────────────────────────────
    info!("=== Clock sync ===");
    client.sync_time().await?;

    // ── 1b. Event store ───────────────────────────────────────────────────────
    let db_path = std::env::var("EVENT_DB_PATH").unwrap_or_else(|_| "rw-trader-events.db".into());
    let event_store: Arc<dyn store::EventStore> = store::SqliteEventStore::open(
        std::path::Path::new(&db_path)
    ).map(|s| s as Arc<dyn store::EventStore>)
    .unwrap_or_else(|e| {
        warn!("Failed to open event store at '{}': {} — using no-op store", db_path, e);
        Arc::new(store::NoopEventStore)
    });

    event_store.append(events::mode_change_event("None", "Booting", "startup"));

    // ── 1c. Web UI address (optional) ────────────────────────────────────────
    // Spawn is deferred to step 6b, after exec/truth/risk are constructed.
    let web_ui_addr = std::env::var("WEB_UI_ADDR").ok();

    // ── 2. Executor (starts in Booting, wired to event store) ───────────────
    let cb_config = executor::CircuitBreakerConfig {
        max_attempts_per_min: env_u64("CB_MAX_ATTEMPTS",  10) as u32,
        max_rejects_per_min:  env_u64("CB_MAX_REJECTS",    3) as u32,
        max_errors_per_min:   env_u64("CB_MAX_ERRORS",     3) as u32,
        max_slippage_per_min: env_u64("CB_MAX_SLIPPAGE",   2) as u32,
    };
    let wd_config = executor::WatchdogConfig {
        waiting_ack_timeout: Duration::from_secs(env_u64("WD_ACK_TIMEOUT_SECS",     10)),
        canceling_timeout:   Duration::from_secs(env_u64("WD_CANCEL_TIMEOUT_SECS",   5)),
        replacing_timeout:   Duration::from_secs(env_u64("WD_REPLACE_TIMEOUT_SECS", 15)),
        check_interval:      Duration::from_millis(500),
    };
    let exec = Arc::new(executor::Executor::with_store(
        symbol.clone(),
        cb_config,
        wd_config,
        Some(Arc::clone(&event_store)),
    ));

    // ── 3. Startup recovery ──────────────────────────────────────────────────
    info!("=== Startup recovery ===");
    exec.set_mode_reconciling().await;
    event_store.append(events::mode_change_event("Booting", "Reconciling", "startup_recovery"));

    let truth = Arc::new(Mutex::new(reconciler::TruthState::new(&symbol, bnb_price)));
    reconciler::startup_recovery(&client, Arc::clone(&truth)).await?;

    {
        let s = truth.lock().await;
        position::print_position_state(&s.position);
        info!(
            open_orders = s.open_order_count,
            fills_seen  = s.seen_fill_ids.len(),
            "Recovery complete"
        );
        // Notify executor of startup reconcile result
        // Collect active orders for executor
        let open_orders: Vec<client::OpenOrder> = s.orders.values()
            .filter(|r| r.status.is_active())
            .map(|r| client::OpenOrder {
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
            .collect();
        drop(s);
        exec.on_reconcile(&open_orders, false).await;
    }

    // ── 4. Risk engine ───────────────────────────────────────────────────────
    let risk_config = risk::RiskConfig {
        max_position_qty:    env_f64("RISK_MAX_QTY",          0.01),
        max_daily_loss_usd:  env_f64("RISK_MAX_DAILY_LOSS",   50.0),
        max_drawdown_usd:    env_f64("RISK_MAX_DRAWDOWN",      100.0),
        max_spread_bps:      env_f64("RISK_MAX_SPREAD_BPS",    10.0),
        cooldown_after_loss: Duration::from_secs(env_u64("RISK_COOLDOWN_SECS", 300)),
        max_feed_staleness:  Duration::from_secs(env_u64("RISK_FEED_STALE_SECS", 5)),
        min_order_interval:  Duration::from_secs(1),
        signal_dedup_window: Duration::from_secs(5),
        max_open_orders:     1,
        max_slippage_bps:    20.0,
    };
    let risk_engine = Arc::new(Mutex::new({
        let s = truth.lock().await;
        risk::RiskEngine::new(risk_config, &s.position)
    }));

    // ── 5. Signal engine + Strategy engine ───────────────────────────────────
    let signal_config = signal::SignalConfig {
        order_qty:             env_f64("SIGNAL_QTY",              0.001),
        momentum_threshold:    env_f64("SIGNAL_MOMENTUM_THRESH",  0.00005),
        imbalance_threshold:   env_f64("SIGNAL_IMBALANCE_THRESH", 0.10),
        max_entry_spread_bps:  env_f64("SIGNAL_MAX_SPREAD_BPS",   5.0),
        max_feed_staleness:    Duration::from_secs(env_u64("RISK_FEED_STALE_SECS", 3)),
        stop_loss_pct:         env_f64("SIGNAL_STOP_LOSS_PCT",    0.0020),
        take_profit_pct:       env_f64("SIGNAL_TAKE_PROFIT_PCT",  0.0040),
        max_hold_duration:     Duration::from_secs(env_u64("SIGNAL_MAX_HOLD_SECS", 120)),
        min_mid_samples:       3,
        min_trade_samples:     3,
    };
    let order_qty      = signal_config.order_qty;
    // signal_engine retained for compute_metrics (SignalEngine still computes
    // the shared SignalMetrics that all strategies receive).
    let signal_engine  = Arc::new(Mutex::new(signal::SignalEngine::new(signal_config)));
    // StrategyEngine replaces direct use of signal_engine for decision-making.
    let strategy_engine = Arc::new(Mutex::new(strategy::StrategyEngine::new()));

    // ── 6. Feed state ────────────────────────────────────────────────────────
    let feed_state = Arc::new(Mutex::new(feed::FeedState::new(Duration::from_secs(10))));

    // ── 6b. Spawn web UI now that all live components exist ───────────────────
    // AppState holds Arc refs to event_store, exec, truth, risk_engine, authority.
    // All are Arc-cloned — no ownership transferred, no lock held at spawn time.
    let authority = Arc::new(authority::AuthorityLayer::new());
    if let Some(ref addr) = web_ui_addr {
        let ui_state = webui::AppState {
            store:    Arc::clone(&event_store),
            exec:     Arc::clone(&exec),
            truth:    Arc::clone(&truth),
            risk:     Arc::clone(&risk_engine),
            authority: Arc::clone(&authority),
            strategy: Arc::clone(&strategy_engine),
        };
        let addr = addr.clone();
        tokio::spawn(async move {
            if let Err(e) = webui::run(&addr, ui_state).await {
                tracing::error!("[WEBUI] Server error: {}", e);
            }
        });
        info!("Web UI available at http://{}", web_ui_addr.as_deref().unwrap_or(""));
    }

    // ── 7. Spawn reconciliation loop (with executor + event store) ──────────
    let recon_secs = env_u64("RECONCILE_INTERVAL_SECS", 2);
    info!("=== Starting reconciliation loop (every {}s) ===", recon_secs);
    let _recon_handle = reconciler::spawn_reconciliation_loop_with_executor_and_store(
        Arc::clone(&client),
        Arc::clone(&truth),
        Arc::clone(&exec),
        Duration::from_secs(recon_secs),
        Some(Arc::clone(&event_store)),
    );

    // ── 8. Spawn WebSocket feed ───────────────────────────────────────────────
    info!("=== Starting WebSocket feed ===");
    {
        let fs = Arc::clone(&feed_state);
        let ws = ws_url.clone();
        let sym = symbol.clone();
        tokio::spawn(async move {
            if let Err(e) = feed::run_feed_with_state(&ws, &sym, fs).await {
                tracing::error!("Feed task exited: {:#}", e);
            }
        });
    }

    // ── 9. Spawn timeout watchdog ─────────────────────────────────────────────
    info!("=== Starting execution watchdog ===");
    let _wd_handle = exec.spawn_watchdog(Arc::clone(&client), Arc::clone(&truth));

    if !live_trade {
        info!("LIVE_TRADE=false — monitor mode. Set LIVE_TRADE=true to enable signal loop.");
        feed::run_feed(&ws_url, &symbol).await?;
        return Ok(());
    }

    // ── 10. Signal + execution loop ───────────────────────────────────────────
    info!("=== LIVE_TRADE=true — signal loop active ===");
    info!("Symbol={} qty={} SL={:.2}% TP={:.2}%",
        symbol, order_qty,
        env_f64("SIGNAL_STOP_LOSS_PCT",   0.0020) * 100.0,
        env_f64("SIGNAL_TAKE_PROFIT_PCT", 0.0040) * 100.0,
    );

    // Retry policy for order submissions
    let retry_policy = orders::RetryPolicy::default();

    // Monotonic sequence counter for deterministic coid generation
    let mut order_seq: u64 = 0;

    // Give feed time to collect initial data
    tokio::time::sleep(Duration::from_secs(2)).await;
    let mut eval_interval = tokio::time::interval(Duration::from_millis(100));
    eval_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        eval_interval.tick().await;

        // ── Gate 1: Executor system mode ──────────────────────────────────────
        // Checked first — cheapest check, no other locks needed.
        if !exec.can_trade().await {
            let mode = exec.system_mode().await;
            let exec_state = exec.execution_state().await;
            tracing::debug!(mode = %mode, exec = %exec_state, "Not tradeable");
            continue;
        }

        // ── Snapshot state (brief locks, released before any exchange call) ──
        let (market_snapshot, position_snapshot, feed_fresh) = {
            let feed = feed_state.lock().await;
            let truth = truth.lock().await;
            (
                risk::MarketSnapshot {
                    bid:            feed.bid,
                    ask:            feed.ask,
                    feed_last_seen: feed.last_seen,
                },
                truth.position.clone(),
                feed.is_fresh(Duration::from_secs(env_u64("RISK_FEED_STALE_SECS", 5))),
            )
        };

        // ── Gate 2: Feed freshness ────────────────────────────────────────────
        if !feed_fresh {
            tracing::debug!("Feed not fresh, skipping");
            continue;
        }

        // ── Evaluate strategies ───────────────────────────────────────────────
        // Compute SignalMetrics from feed (shared by all strategies).
        // Then ask the StrategyEngine which strategy fires and at what confidence.
        let (strategy_decision, strategy_all, signal_result) = {
            let feed  = feed_state.lock().await;
            let truth = truth.lock().await;
            let sig   = signal_engine.lock().await;
            let strat = strategy_engine.lock().await;
            // compute_metrics is a method on SignalEngine
            let metrics = sig.compute_metrics_pub(&feed);
            let (chosen, all) = strat.evaluate(&metrics, &truth);
            let sr = chosen.to_signal_result(&metrics);
            (chosen, all, sr)
        };

        match &signal_result.decision {
            signal::SignalDecision::Hold => {
                tracing::debug!(
                    reason   = %signal_result.reason,
                    strategy = %strategy_decision.strategy_id,
                    "[STRATEGY] Hold"
                );
                continue;
            }
            signal::SignalDecision::Buy => {
                info!(
                    reason   = %signal_result.reason,
                    conf     = signal_result.confidence,
                    strategy = %strategy_decision.strategy_id,
                    "[STRATEGY] BUY"
                );
            }
            signal::SignalDecision::Exit { reason } => {
                info!(
                    reason   = %signal_result.reason,
                    exit     = %reason,
                    strategy = %strategy_decision.strategy_id,
                    "[STRATEGY] EXIT"
                );
            }
        }

        // Generate correlation_id that ties this signal → risk → order → fill together.
        // Every event in this decision cycle carries the same correlation_id.
        let correlation_id = Uuid::new_v4().to_string();

        // Persist market snapshot and signal decision.
        event_store.append(events::market_snapshot_event(
            &signal_result.metrics, &symbol, &correlation_id,
        ));
        event_store.append(events::signal_event(
            &signal_result, &symbol, &correlation_id,
        ));

        // ── Gate 3: TruthState dirty/recon guards ────────────────────────────
        {
            let t = truth.lock().await;
            if !t.can_place_order() {
                signal::log_decision(&signal_result, None, "BLOCKED_DIRTY_STATE");
                continue;
            }
        }

        // ── Route through risk engine ─────────────────────────────────────────
        let (order_side, proposed_price) = match &signal_result.decision {
            signal::SignalDecision::Buy       => (risk::OrderSide::Buy,  market_snapshot.ask),
            signal::SignalDecision::Exit { .. }=> (risk::OrderSide::Sell, market_snapshot.bid),
            signal::SignalDecision::Hold       => unreachable!(),
        };

        let proposed = risk::ProposedOrder {
            symbol:         symbol.clone(),
            side:           order_side,
            qty:            order_qty,
            expected_price: proposed_price,
        };

        let risk_verdict = {
            let mut risk = risk_engine.lock().await;
            risk.risk_check(&position_snapshot, &market_snapshot, &proposed)
        };

        match &risk_verdict {
            risk::RiskVerdict::Approved => {
                signal::log_decision(&signal_result, Some("APPROVED"), "SUBMITTING");
            }
            risk::RiskVerdict::Rejected(r) => {
                signal::log_decision(&signal_result, Some("REJECTED"), "NO_ORDER");
                info!(reason = %r, "[RISK] Rejected");
                // If risk tripped due to halting condition, sync executor mode
                if matches!(r, risk::RejectionReason::DailyLossExceeded { .. }
                              | risk::RejectionReason::DrawdownExceeded { .. }
                              | risk::RejectionReason::KillSwitch) {
                    exec.set_mode_halted(&r.to_string()).await;
                }
                event_store.append(events::risk_event(
                    &risk_verdict, &proposed, &position_snapshot, &symbol, &correlation_id,
                ));
                continue;
            }
        }

        // Persist risk approval event
        event_store.append(events::risk_event(
            &risk_verdict, &proposed, &position_snapshot, &symbol, &correlation_id,
        ));

        // Compute side_str here so it's available to the authority gate below.
        let side_str = match order_side { risk::OrderSide::Buy => "BUY", risk::OrderSide::Sell => "SELL" };

        // ── Gate 4: Authority layer ───────────────────────────────────────────
        // OFF   → block (no execution from suggestion path)
        // ASSIST → create proposal, halt here until operator approves via web UI
        // AUTO  → proceed only when all system/risk/exec gates are clear
        // This gate fires AFTER risk approval so risk always runs first.
        {
            let sys_mode     = exec.system_mode().await;
            let exec_is_idle = matches!(exec.execution_state().await,
                                        executor::ExecutionState::Idle);
            let kill_active  = { let r = risk_engine.lock().await; r.kill_switch_active() };
            let can_place    = { let t = truth.lock().await; t.can_place_order() };

            let auth_result = authority.check(
                &symbol,
                side_str,
                order_qty,
                &signal_result.reason,
                signal_result.confidence,
                sys_mode,
                exec_is_idle,
                kill_active,
                can_place,
                &*event_store,
            ).await;

            match auth_result {
                authority::AuthorityResult::Proceed => {
                    // AUTO gate cleared — fall through to execution below
                    info!(mode = "AUTO", "[AUTHORITY] Execution permitted");
                }
                authority::AuthorityResult::ProposalCreated(p) => {
                    // ASSIST — proposal stored, waiting for operator approval via /authority
                    info!(
                        id      = %p.id,
                        symbol  = %p.symbol,
                        side    = %p.side,
                        ttl_s   = p.ttl_remaining_secs(),
                        "[AUTHORITY] ASSIST proposal created — awaiting operator approval"
                    );
                    // Decrement order_seq so the next real approval gets a fresh coid
                    order_seq = order_seq.saturating_sub(1);
                    continue;
                }
                authority::AuthorityResult::Blocked(reason) => {
                    info!(reason = %reason, "[AUTHORITY] Execution blocked");
                    order_seq = order_seq.saturating_sub(1);
                    continue;
                }
            }
        }

        // ── Generate deterministic coid for this logical order ────────────────
        // Incremented once per signal, not per retry. Retries reuse the same coid.
        order_seq += 1;
        let client_order_id = executor::make_client_order_id(order_seq);
        let qty_str  = format!("{:.5}", order_qty);

        // Persist order submission intent
        event_store.append(events::order_submitted_event(
            &client_order_id, side_str, &qty_str, proposed_price, &symbol, &correlation_id,
        ));

        // ── Execute via Executor (enforces all in-flight protections) ─────────
        match exec.submit_market_order(
            &symbol,
            side_str,
            &qty_str,
            &client_order_id,
            &client,
            &truth,
            &risk_engine,
            &retry_policy,
        ).await {
            Ok(resp) => {
                let avg_price: f64 = {
                    let filled: f64 = resp.executed_qty.parse().unwrap_or(0.0);
                    if filled > 0.0 {
                        resp.cumulative_quote_qty.parse::<f64>().unwrap_or(0.0) / filled
                    } else { 0.0 }
                };

                // OrderAcked — exchange accepted the order
                event_store.append(events::StoredEvent::new(
                    Some(symbol.clone()),
                    Some(correlation_id.clone()),
                    Some(resp.client_order_id.clone()),
                    events::TradingEvent::OrderAcked(events::OrderAckedPayload {
                        client_order_id:   resp.client_order_id.clone(),
                        exchange_order_id: resp.order_id,
                        status:            resp.status.clone(),
                    }),
                ));

                // OrderFilled — market orders always fill synchronously
                if resp.status.to_uppercase() == "FILLED" {
                    event_store.append(events::order_filled_event(&resp, &symbol, &correlation_id));
                }

                // Check slippage vs expected price
                if proposed_price > 0.0 && avg_price > 0.0 {
                    let slippage_bps = ((avg_price - proposed_price).abs() / proposed_price) * 10_000.0;
                    if slippage_bps > env_f64("RISK_MAX_SPREAD_BPS", 10.0) * 2.0 {
                        warn!(
                            slippage_bps = format!("{:.2}", slippage_bps),
                            expected = proposed_price,
                            actual   = avg_price,
                            "[EXEC] Slippage breach"
                        );
                        exec.record_slippage_breach().await;
                    }
                }

                // Update strategy engine entry/exit tracking
                {
                    let mut strat = strategy_engine.lock().await;
                    let mut sig   = signal_engine.lock().await;
                    match order_side {
                        risk::OrderSide::Buy  => {
                            strat.on_entry_submitted(&strategy_decision.strategy_id, avg_price.max(proposed_price));
                            sig.on_entry_submitted(avg_price.max(proposed_price));
                        }
                        risk::OrderSide::Sell => {
                            strat.on_exit_submitted(&strategy_decision.strategy_id);
                            sig.on_exit_submitted();
                        }
                    }
                }
                let _ = &strategy_all; // comparison view available to UI via Arc<Mutex<StrategyEngine>>

                signal::log_decision(&signal_result, Some("APPROVED"), "FILLED");
            }
            Err(e) => {
                warn!(error = %e, "[EXEC] Submission failed");
                // OrderRejected — exchange refused or network failed
                event_store.append(events::StoredEvent::new(
                    Some(symbol.clone()),
                    Some(correlation_id.clone()),
                    Some(client_order_id.clone()),
                    events::TradingEvent::OrderRejected(events::OrderRejectedPayload {
                        client_order_id: client_order_id.clone(),
                        reason:          e.to_string(),
                    }),
                ));
                signal::log_decision(&signal_result, Some("APPROVED"), "ORDER_FAILED");
            }
        }
    }
}

fn require_env(key: &str) -> Result<String> {
    std::env::var(key).map_err(|_| anyhow::anyhow!("Missing required env var: {}", key))
}
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

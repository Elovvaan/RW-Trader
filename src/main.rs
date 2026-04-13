mod assistant;
mod agent;
mod authority;
mod client;
mod events;
mod npc;
mod executor;
mod feed;
mod orders;
mod phase1;
mod portfolio;
mod position;
mod profile;
mod reader;
mod reconciler;
mod replay;
mod risk;
mod signal;
mod store;
mod strategy;
mod suggestions;
mod webui;
mod withdrawal;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use crate::store::EventStore;

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
            let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
            let default_addr = format!("0.0.0.0:{}", port);
            let addr = args.get(pos + 1)
                .map(|s| s.clone())
                .unwrap_or(default_addr);

            tracing_subscriber::fmt().with_env_filter("info").init();
            info!("Server running on port {}", port);

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
                max_consecutive_losses: 0,
                cooldown_after_loss: Duration::from_secs(0),
                max_spread_bps:      0.0,
                max_feed_staleness:  Duration::from_secs(0),
                min_order_interval:  Duration::from_secs(0),
                signal_dedup_window: Duration::from_secs(0),
                max_open_orders:     0,
                max_slippage_bps:    0.0,
            }, &stub_pos);
            let stub_exec = Arc::new(executor::Executor::new(
                "BTCUSDT".into(),
                executor::CircuitBreakerConfig::default(),
                executor::WatchdogConfig::default(),
            ));
            let stub_truth = Arc::new(Mutex::new(reconciler::TruthState::new("BTCUSDT", 0.0)));
            let stub_feed = Arc::new(Mutex::new(feed::FeedState::new(Duration::from_secs(10))));
            let stub_signal = Arc::new(Mutex::new(signal::SignalEngine::new(signal::SignalConfig {
                order_qty: 0.0,
                momentum_threshold: 0.0,
                imbalance_threshold: 0.0,
                max_entry_spread_bps: 0.0,
                max_feed_staleness: Duration::from_secs(1),
                stop_loss_pct: 0.0,
                take_profit_pct: 0.0,
                max_hold_duration: Duration::from_secs(1),
                min_mid_samples: 1,
                min_trade_samples: 1,
            })));
            let stub_withdrawals = Arc::new(withdrawal::WithdrawalManager::new(withdrawal::WithdrawalConfig::default()));
            let authority = Arc::new(authority::AuthorityLayer::new());
            let npc_controller = Arc::new(npc::NpcAutonomousController::new(
                npc::NpcConfig::from_trade_cfg(&agent::TradeAgentConfig {
                    enabled: false,
                    trade_size: 0.0,
                    momentum_threshold: 0.0,
                    poll_interval: Duration::from_millis(1000),
                    max_spread_bps: 0.0,
                }),
                agent::AgentState {
                    store: Arc::clone(&store),
                    exec: Arc::clone(&stub_exec),
                    feed: Arc::clone(&stub_feed),
                    signal: Arc::clone(&stub_signal),
                    truth: Arc::clone(&stub_truth),
                    authority: Arc::clone(&authority),
                    withdrawals: Arc::clone(&stub_withdrawals),
                    client: Arc::new(client::BinanceClient::new(String::new(), String::new(), String::new())),
                    symbol: "BTCUSDT".into(),
                    web_base_url: None,
                },
            ));
            let state = webui::AppState {
                store:     Arc::clone(&store),
                exec:      Arc::clone(&stub_exec),
                truth:     Arc::clone(&stub_truth),
                risk:      Arc::new(Mutex::new(stub_risk)),
                authority: Arc::clone(&authority),
                strategy:  Arc::new(Mutex::new(strategy::StrategyEngine::new())),
                client:    None,
                withdrawals: Arc::clone(&stub_withdrawals),
                npc: npc_controller,
                profile:   Arc::new(Mutex::new(profile::RuntimeProfile::default())),
            };
            webui::run(&addr, state).await?;
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
    let symbol     = std::env::var("SYMBOL").unwrap_or_else(|_| "BTCUSDT".into());
    let symbols: Vec<String> = std::env::var("SYMBOLS")
        .unwrap_or_else(|_| symbol.clone())
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect();
    let live_trade = std::env::var("LIVE_TRADE")
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let bnb_price: f64 = env_f64("BNB_PRICE_USD", 0.0);

    // ── Runtime profile ──────────────────────────────────────────────────────
    // RUNTIME_PROFILE env-var: CONSERVATIVE | ACTIVE | MICRO_TEST (default: ACTIVE)
    let runtime_profile = profile::RuntimeProfile::from_str(
        &std::env::var("RUNTIME_PROFILE").unwrap_or_else(|_| "ACTIVE".into()),
    );
    let profile_cfg = profile::ProfileConfig::for_profile(runtime_profile);
    info!(
        profile = runtime_profile.as_str(),
        label   = runtime_profile.label(),
        min_confidence            = profile_cfg.signal_min_confidence,
        entry_cooldown_secs       = profile_cfg.entry_cooldown_after_exit.as_secs(),
        failed_breakout_cooldown_secs = profile_cfg.failed_breakout_cooldown.as_secs(),
        cycle_interval_ms         = profile_cfg.cycle_interval.as_millis() as u64,
        "=== Runtime profile ==="
    );
    // Wrap in Arc<Mutex> so the web UI can update it at runtime.
    let active_profile = Arc::new(Mutex::new(runtime_profile));

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
    // If WEB_UI_ADDR is not set, fall back to 0.0.0.0:{PORT} when PORT is available.
    let web_ui_addr = std::env::var("WEB_UI_ADDR")
        .ok()
        .or_else(|| std::env::var("PORT").ok().map(|p| format!("0.0.0.0:{}", p)));

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
    // RISK_COOLDOWN_SECS is overridden by the profile's entry_cooldown_after_exit
    // unless the operator has explicitly set RISK_COOLDOWN_SECS in the environment.
    let risk_cooldown_secs = if std::env::var("RISK_COOLDOWN_SECS").is_ok() {
        env_u64("RISK_COOLDOWN_SECS", 300)
    } else {
        profile_cfg.entry_cooldown_after_exit.as_secs()
    };
    let risk_config = risk::RiskConfig {
        max_position_qty:    env_f64("RISK_MAX_QTY",          0.01),
        max_daily_loss_usd:  env_f64("RISK_MAX_DAILY_LOSS",   50.0),
        max_drawdown_usd:    env_f64("RISK_MAX_DRAWDOWN",      100.0),
        max_consecutive_losses: env_u64("RISK_MAX_CONSECUTIVE_LOSSES", 3) as u32,
        max_spread_bps:      env_f64("RISK_MAX_SPREAD_BPS",    10.0),
        cooldown_after_loss: Duration::from_secs(risk_cooldown_secs),
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
    // min_confidence_normal is overridden by the profile unless explicitly set.
    let min_conf_normal = if std::env::var("PORTFOLIO_MIN_CONF_NORMAL").is_ok() {
        env_f64("PORTFOLIO_MIN_CONF_NORMAL", 0.70)
    } else {
        profile_cfg.signal_min_confidence
    };
    let portfolio_config = portfolio::PortfolioConfig {
        initial_equity_usd: env_f64("PORTFOLIO_INITIAL_EQUITY_USD", 10_000.0),
        reserve_cash_buffer: env_f64("PORTFOLIO_RESERVE_BUFFER", 0.20),
        max_total_exposure: env_f64("PORTFOLIO_MAX_TOTAL_EXPOSURE", 0.95),
        max_correlated_exposure: env_f64("PORTFOLIO_MAX_CORRELATED_EXPOSURE", 0.45),
        max_symbol_exposure: env_f64("PORTFOLIO_MAX_SYMBOL_EXPOSURE", 0.35),
        max_strategy_exposure: env_f64("PORTFOLIO_MAX_STRATEGY_EXPOSURE", 0.45),
        max_intraday_drawdown: env_f64("PORTFOLIO_MAX_INTRADAY_DRAWDOWN", 0.06),
        max_strategy_loss: env_f64("PORTFOLIO_MAX_STRATEGY_LOSS", 0.03),
        max_symbol_loss: env_f64("PORTFOLIO_MAX_SYMBOL_LOSS", 0.03),
        max_turnover_per_hour: env_f64("PORTFOLIO_MAX_TURNOVER_PER_HOUR", 4.0),
        defensive_drawdown: env_f64("PORTFOLIO_DEFENSIVE_DRAWDOWN", 0.04),
        recovery_drawdown: env_f64("PORTFOLIO_RECOVERY_DRAWDOWN", 0.02),
        defensive_hit_rate: env_f64("PORTFOLIO_DEFENSIVE_HIT_RATE", 0.45),
        min_confidence_normal: min_conf_normal,
        min_confidence_defensive: env_f64("PORTFOLIO_MIN_CONF_DEFENSIVE", 0.82),
        max_concurrent_positions_normal: env_u64("PORTFOLIO_MAX_CONCURRENT_NORMAL", 6) as usize,
        max_concurrent_positions_defensive: env_u64("PORTFOLIO_MAX_CONCURRENT_DEFENSIVE", 3) as usize,
        hard_cap_active_positions: env_u64("PORTFOLIO_HARD_CAP_ACTIVE_POSITIONS", 8) as usize,
        min_expected_value: env_f64("PORTFOLIO_MIN_EXPECTED_VALUE", 0.03),
        max_signal_age_secs: env_f64("PORTFOLIO_MAX_SIGNAL_AGE_SECS", 2.0),
        max_spread_bps_gate: env_f64("PORTFOLIO_MAX_SPREAD_BPS_GATE", 8.0),
        min_execution_quality: env_f64("PORTFOLIO_MIN_EXEC_QUALITY", 0.45),
        defensive_size_multiplier: env_f64("PORTFOLIO_DEFENSIVE_SIZE_MULT", 0.55),
        recovery_size_multiplier: env_f64("PORTFOLIO_RECOVERY_SIZE_MULT", 0.80),
    };
    let portfolio_allocator = Arc::new(portfolio::PortfolioAllocator::new(portfolio_config.clone()));
    let portfolio_risk = Arc::new(portfolio::PortfolioRiskEngine::new(portfolio_config.clone()));
    let portfolio_state = Arc::new(Mutex::new(portfolio::PortfolioState::new(&portfolio_config)));
    let strategy_scoreboard = Arc::new(Mutex::new(portfolio::StrategyScoreboard::default()));
    let execution_quality = Arc::new(Mutex::new(portfolio::ExecutionQualityTracker::default()));

    // ── 6. Feed state ────────────────────────────────────────────────────────
    let feed_state = Arc::new(Mutex::new(feed::FeedState::new(Duration::from_secs(10))));

    // ── 6b. Spawn web UI now that all live components exist ───────────────────
    // AppState holds Arc refs to event_store, exec, truth, risk_engine, authority.
    // All are Arc-cloned — no ownership transferred, no lock held at spawn time.
    let authority = Arc::new(authority::AuthorityLayer::new());
    let allowed_withdrawal_destinations: std::collections::HashSet<String> = std::env::var("WITHDRAW_ALLOWED_DESTINATIONS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let withdrawals = Arc::new(withdrawal::WithdrawalManager::new(withdrawal::WithdrawalConfig {
        auto_execute_enabled: env_bool("WITHDRAW_AUTO_ENABLED", false),
        max_withdrawal_amount: env_f64("WITHDRAW_MAX_AMOUNT", 1000.0),
        allowed_destinations: allowed_withdrawal_destinations,
        cooldown: Duration::from_secs(env_u64("WITHDRAW_COOLDOWN_SECS", 300)),
        duplicate_window: Duration::from_secs(env_u64("WITHDRAW_DUP_WINDOW_SECS", 600)),
        default_fee: env_f64("WITHDRAW_DEFAULT_FEE", 0.0),
    }));
    // TRADE_INTERVAL_SECS is overridden by the profile's cycle_interval unless
    // the operator has explicitly set TRADE_INTERVAL_SECS.
    let trade_interval_secs = if std::env::var("TRADE_INTERVAL_SECS").is_ok() {
        env_u64("TRADE_INTERVAL_SECS", 1)
    } else {
        profile_cfg.cycle_interval.as_secs().max(1)
    };
    let trade_cfg = agent::TradeAgentConfig {
        enabled: env_bool("TRADE_ENABLED", false),
        trade_size: env_f64("TRADE_SIZE", order_qty),
        momentum_threshold: env_f64("MOMENTUM_THRESHOLD", 0.00005),
        poll_interval: Duration::from_secs(trade_interval_secs),
        max_spread_bps: env_f64("SIGNAL_MAX_SPREAD_BPS", 5.0),
    };
    let web_base_url = resolve_web_base_url();
    if let Some(ref base) = web_base_url {
        info!("[DISPATCH] Resolved dispatch base URL: {}", base);
        info!("[DISPATCH] Trade requests will be sent to: {}/trade/request", base);
    } else {
        warn!("[DISPATCH] No dispatch base URL resolved — trade dispatch will be blocked (WEB_UI_ADDR_MISSING)");
    }
    let npc_controller = Arc::new(npc::NpcAutonomousController::new(
        npc::NpcConfig::from_trade_cfg(&trade_cfg),
        agent::AgentState {
            store: Arc::clone(&event_store),
            exec: Arc::clone(&exec),
            feed: Arc::clone(&feed_state),
            signal: Arc::clone(&signal_engine),
            truth: Arc::clone(&truth),
            authority: Arc::clone(&authority),
            withdrawals: Arc::clone(&withdrawals),
            client: Arc::clone(&client),
            symbol: symbol.clone(),
            web_base_url: web_base_url.clone(),
        },
    ));
    if let Some(ref addr) = web_ui_addr {
        let ui_state = webui::AppState {
            store:    Arc::clone(&event_store),
            exec:     Arc::clone(&exec),
            truth:    Arc::clone(&truth),
            risk:     Arc::clone(&risk_engine),
            authority: Arc::clone(&authority),
            strategy: Arc::clone(&strategy_engine),
            client: Some(Arc::clone(&client)),
            withdrawals: Arc::clone(&withdrawals),
            npc: Arc::clone(&npc_controller),
            profile: Arc::clone(&active_profile),
        };
        // Capture the port from the env var directly; fall back to parsing the address.
        let port_str = std::env::var("PORT")
            .unwrap_or_else(|_| std::env::var("WEB_UI_ADDR")
                .ok()
                .and_then(|a| a.rsplit(':').next().map(str::to_string))
                .unwrap_or_else(|| "8080".to_string()));
        let addr = addr.clone();
        tokio::spawn(async move {
            if let Err(e) = webui::run(&addr, ui_state).await {
                tracing::error!("[WEBUI] Server error: {}", e);
            }
        });
        info!("Web UI available at http://{}", web_ui_addr.as_deref().unwrap_or(""));
        info!("Server running on port {}", port_str);
    }

    // ── 6c. Profit Sweep Agent loop ──────────────────────────────────────────
    let sweep_cfg = agent::AgentConfig {
        sweep_threshold: env_f64("SWEEP_THRESHOLD", 0.0),
        sweep_asset: std::env::var("SWEEP_ASSET").unwrap_or_else(|_| "USDT".to_string()),
        sweep_interval: Duration::from_secs(env_u64("SWEEP_INTERVAL", 30)),
        sweep_network: std::env::var("WITHDRAW_DEFAULT_NETWORK").unwrap_or_else(|_| "ETH".to_string()),
    };
    let web_base_url = resolve_web_base_url();
    agent::spawn_profit_sweep_agent(
        sweep_cfg,
        agent::AgentState {
            store: Arc::clone(&event_store),
            exec: Arc::clone(&exec),
            feed: Arc::clone(&feed_state),
            signal: Arc::clone(&signal_engine),
            truth: Arc::clone(&truth),
            authority: Arc::clone(&authority),
            withdrawals: Arc::clone(&withdrawals),
            client: Arc::clone(&client),
            symbol: symbol.clone(),
            web_base_url,
        },
    );

    npc::spawn_npc_trading_layer(&npc_controller).await;

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

    // ── 8. REST polling feed (WebSocket disabled) ────────────────────────────
    // WebSocket is disabled to avoid runtime crashes from incorrect endpoints
    // or missing listenKey configuration. REST polling keeps feed_state fresh.
    info!("WebSocket disabled — using REST polling mode");
    {
        let fs  = Arc::clone(&feed_state);
        let cl  = Arc::clone(&client);
        let sym = symbol.clone();
        tokio::spawn(async move {
            if let Err(e) = feed::run_rest_polling(cl, &sym, fs, Duration::from_secs(1)).await {
                tracing::error!("REST polling task exited: {:#}", e);
            }
        });
    }

    // ── 8b. Snapshot recorder (continuous market snapshots) ──────────────────
    // Persist market snapshots every second even when no entry/exit signal fires.
    // This keeps the snapshot store warm for watchlists, suggestions, and replay.
    {
        let fs = Arc::clone(&feed_state);
        let sig = Arc::clone(&signal_engine);
        let store = Arc::clone(&event_store);
        let sym = symbol.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;

                let metrics = {
                    let feed = fs.lock().await;
                    if feed.last_seen.is_none() {
                        continue;
                    }
                    let signal = sig.lock().await;
                    signal.compute_metrics_pub(&feed)
                };

                let correlation_id = Uuid::new_v4().to_string();
                store.append(events::market_snapshot_event(&metrics, &sym, &correlation_id));
                info!(
                    symbol = %sym,
                    bid = metrics.bid,
                    ask = metrics.ask,
                    spread_bps = metrics.spread_bps,
                    feed_age_ms = metrics.feed_age_ms,
                    "Market snapshot updated"
                );
            }
        });
    }

    // ── 9. Spawn timeout watchdog ─────────────────────────────────────────────
    info!("=== Starting execution watchdog ===");
    let _wd_handle = exec.spawn_watchdog(Arc::clone(&client), Arc::clone(&truth));

    if !live_trade {
        info!("LIVE_TRADE=false — monitor mode. Set LIVE_TRADE=true to enable signal loop.");
        // REST polling is already running in a spawned task (step 8).
        // Park main() indefinitely so the tokio runtime — and all spawned tasks
        // (REST poller, web UI, reconciler, watchdog) — stay alive.
        std::future::pending::<()>().await;
        return Ok(());
    }

    // ── 10. Signal + execution loop ───────────────────────────────────────────
    info!("=== LIVE_TRADE=true — signal loop active ===");
    info!("Symbols={:?} primary={} qty={} SL={:.2}% TP={:.2}%",
        symbols, symbol, order_qty,
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
        let (strategy_decision, strategy_all, metrics, regime_weight) = {
            let feed  = feed_state.lock().await;
            let truth = truth.lock().await;
            let sig   = signal_engine.lock().await;
            let mut strat = strategy_engine.lock().await;
            // compute_metrics is a method on SignalEngine
            let metrics = sig.compute_metrics_pub(&feed);
            let regime_weight = strat.regime_weight(&metrics);
            let (chosen, all) = strat.evaluate(&metrics, &truth);
            (chosen, all, metrics, regime_weight)
        };
        let mut signal_result = strategy_decision.to_signal_result(&metrics);

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
            &metrics, &symbol, &correlation_id,
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

        // ── Portfolio ranking + allocation (signal -> ranking -> allocation) ─
        let candidates: Vec<portfolio::OpportunityCandidate> = strategy_all.iter()
            .filter(|d| d.action.is_actionable())
            .map(|d| {
                let local_signal = d.to_signal_result(&metrics);
                let side = match &local_signal.decision {
                    signal::SignalDecision::Buy => risk::OrderSide::Buy,
                    signal::SignalDecision::Exit { .. } => risk::OrderSide::Sell,
                    signal::SignalDecision::Hold => risk::OrderSide::Buy,
                };
                let expected_reward_risk = ((metrics.momentum_3s.abs() * 15_000.0)
                    / (metrics.spread_bps.max(0.1))).clamp(0.0, 2.0);
                let expected_value = d.confidence * regime_weight * expected_reward_risk
                    - (metrics.spread_bps / 10_000.0);
                portfolio::OpportunityCandidate {
                    symbol: symbol.clone(),
                    strategy: d.strategy_id.clone(),
                    side,
                    confidence: d.confidence,
                    regime_fit: regime_weight,
                    expected_reward_risk,
                    volatility: metrics.momentum_1s.abs() + metrics.momentum_3s.abs() + metrics.momentum_5s.abs(),
                    current_price: metrics.mid,
                    reason: local_signal.reason,
                    signal_age_secs: (metrics.feed_age_ms / 1_000.0).max(0.0),
                    spread_bps: metrics.spread_bps,
                    expected_value,
                    execution_path: "binance:market".into(),
                }
            })
            .collect();
        if candidates.is_empty() {
            continue;
        }

        let ranked = {
            let mut pstate = portfolio_state.lock().await;
            pstate.update_mode(&portfolio_config);
            let scores = strategy_scoreboard.lock().await;
            let eq = execution_quality.lock().await;
            portfolio_allocator.rank_opportunities(&pstate, &scores, &eq, candidates)
        };
        let Some(top_opp) = ranked.first().cloned() else { continue; };
        let selected_strategy_id = top_opp.candidate.strategy.clone();
        signal_result = strategy_all.iter()
            .find(|d| d.strategy_id == selected_strategy_id)
            .map(|d| d.to_signal_result(&metrics))
            .unwrap_or_else(|| strategy_decision.to_signal_result(&metrics));
        let portfolio_gate = {
            let pstate = portfolio_state.lock().await;
            portfolio_risk.check(&pstate, &top_opp)
        };
        if let Err(reason) = portfolio_gate {
            info!(reason = %reason, symbol = %top_opp.candidate.symbol, "[PORTFOLIO] Rejected opportunity");
            event_store.append(events::StoredEvent::new(
                Some(symbol.clone()),
                Some(correlation_id.clone()),
                None,
                events::TradingEvent::OperatorAction(events::OperatorActionPayload {
                    action: "portfolio_rejection".into(),
                    reason: format!("{} rank={:.3} mode={}", reason, top_opp.rank_score, {
                        let p = portfolio_state.lock().await;
                        p.mode.as_str()
                    }),
                }),
            ));
            continue;
        }

        event_store.append(events::StoredEvent::new(
            Some(symbol.clone()),
            Some(correlation_id.clone()),
            None,
            events::TradingEvent::OperatorAction(events::OperatorActionPayload {
                action: "portfolio_ranked_opportunity".into(),
                reason: format!(
                    "rank={:.3} alloc_usd={:.2} div={:.2} strat_score={:.2} corr_penalty={:.2} exec_penalty={:.2} mode={}",
                    top_opp.rank_score,
                    top_opp.allocated_notional_usd,
                    top_opp.diversification_benefit,
                    top_opp.strategy_score,
                    top_opp.correlation_penalty,
                    top_opp.execution_penalty,
                    { let p = portfolio_state.lock().await; p.mode.as_str().to_string() }
                ),
            }),
        ));

        // ── Route through risk engine ─────────────────────────────────────────
        let (order_side, proposed_price) = match &signal_result.decision {
            signal::SignalDecision::Buy       => (risk::OrderSide::Buy,  market_snapshot.ask),
            signal::SignalDecision::Exit { .. }=> (risk::OrderSide::Sell, market_snapshot.bid),
            signal::SignalDecision::Hold       => unreachable!(),
        };

        // Hard risk constraints are checked before adaptive sizing.
        let hard_verdict = {
            let mut risk = risk_engine.lock().await;
            risk.check_hard_constraints(&position_snapshot)
        };
        if let risk::RiskVerdict::Rejected(r) = &hard_verdict {
            info!(reason = %r, "[RISK] Hard constraint rejected before adaptive sizing");
            if matches!(r, risk::RejectionReason::DailyLossExceeded { .. }
                          | risk::RejectionReason::DrawdownExceeded { .. }
                          | risk::RejectionReason::KillSwitch) {
                exec.set_mode_halted(&r.to_string()).await;
            }
            // Log event with zero quantity because adaptive sizing was never applied.
            let proposed = risk::ProposedOrder {
                symbol:         symbol.clone(),
                side:           order_side,
                qty:            0.0,
                expected_price: proposed_price,
            };
            event_store.append(events::risk_event(
                &hard_verdict, &proposed, &position_snapshot, &symbol, &correlation_id,
            ));
            continue;
        }

        let sell_side_free_balance = if matches!(order_side, risk::OrderSide::Sell) {
            let t = truth.lock().await;
            t.available_balance_for_side("SELL").max(0.0)
        } else {
            0.0
        };

        let dynamic_qty = {
            let strat = strategy_engine.lock().await;
            let notional_budget_qty = if proposed_price > 0.0 {
                top_opp.allocated_notional_usd / proposed_price
            } else { 0.0 };
            match order_side {
                risk::OrderSide::Buy => strat.compute_position_size(
                    order_qty.min(notional_budget_qty.max(0.0)),
                    signal_result.confidence,
                    regime_weight,
                    &signal_result.metrics,
                ),
                risk::OrderSide::Sell => {
                    let pos_qty = position_snapshot.size.max(0.0);
                    if pos_qty > 0.0 {
                        pos_qty
                    } else {
                        // Position-aware immediate SELL path:
                        // if we are flat but hold base inventory from balances,
                        // allow execution from free base balance.
                        sell_side_free_balance
                    }
                }
            }
        };
        info!(
            pre_risk_qty = order_qty,
            post_risk_qty = dynamic_qty,
            side = %match order_side { risk::OrderSide::Buy => "BUY", risk::OrderSide::Sell => "SELL" },
            "[RISK] Pre-risk vs post-adaptive quantity"
        );

        if dynamic_qty <= 0.0 {
            tracing::debug!("Dynamic quantity <= 0, skipping");
            continue;
        }

        let side_str = match order_side { risk::OrderSide::Buy => "BUY", risk::OrderSide::Sell => "SELL" };
        let side_available_balance = {
            let t = truth.lock().await;
            t.available_balance_for_side(side_str)
        };
        if side_available_balance <= 0.0 {
            info!(
                side = side_str,
                available = side_available_balance,
                "Skipping execution: side-aware free balance is zero"
            );
            order_seq = order_seq.saturating_sub(1);
            continue;
        }

        let proposed = risk::ProposedOrder {
            symbol:         symbol.clone(),
            side:           order_side,
            qty:            dynamic_qty,
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
                info!(
                    pre_risk_qty = order_qty,
                    post_risk_qty = 0.0_f64,
                    side = %match order_side { risk::OrderSide::Buy => "BUY", risk::OrderSide::Sell => "SELL" },
                    "[RISK] Final quantity after rejection"
                );
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
                dynamic_qty,
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
        let qty_str  = format!("{:.5}", dynamic_qty);

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
                    {
                        let mut eq = execution_quality.lock().await;
                        eq.record("binance:market", slippage_bps, 0.0, false);
                    }
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
                let holding_secs = {
                    let strat = strategy_engine.lock().await;
                    strat.entry_age_secs(&selected_strategy_id).unwrap_or(0.0)
                };
                {
                    let mut strat = strategy_engine.lock().await;
                    let mut sig   = signal_engine.lock().await;
                    let mut pstate = portfolio_state.lock().await;
                    match order_side {
                        risk::OrderSide::Buy  => {
                            strat.on_entry_submitted(&selected_strategy_id, avg_price.max(proposed_price));
                            sig.on_entry_submitted(avg_price.max(proposed_price));
                            pstate.record_fill(&symbol, &selected_strategy_id, dynamic_qty * avg_price.max(proposed_price), true);
                        }
                        risk::OrderSide::Sell => {
                            strat.on_exit_submitted(&selected_strategy_id, avg_price.max(proposed_price));
                            sig.on_exit_submitted();
                            pstate.record_fill(&symbol, &selected_strategy_id, dynamic_qty * avg_price.max(proposed_price), false);
                        }
                    }
                }
                let _ = &strategy_all; // comparison view available to UI via Arc<Mutex<StrategyEngine>>
                {
                    let mut pstate = portfolio_state.lock().await;
                    let mut scores = strategy_scoreboard.lock().await;
                    let realized = if proposed_price > 0.0 {
                        (avg_price - proposed_price) / proposed_price
                    } else { 0.0 };
                    pstate.record_trade_result(dynamic_qty * (avg_price - proposed_price));
                    scores.update(&selected_strategy_id, realized, pstate.drawdown());
                    event_store.append(events::StoredEvent::new(
                        Some(symbol.clone()),
                        Some(correlation_id.clone()),
                        Some(resp.client_order_id.clone()),
                        events::TradingEvent::OperatorAction(events::OperatorActionPayload {
                            action: "trade_lifecycle_analytics".into(),
                            reason: format!(
                                "strategy={} symbol={} regime_weight={:.3} confidence={:.3} entry_reason={} exit_reason={} edge_decay={:.4} slippage_bps={:.2} holding_secs={:.2} realized_edge={:.5} pnl_usd={:.4}",
                                selected_strategy_id,
                                symbol,
                                regime_weight,
                                signal_result.confidence,
                                signal_result.reason.replace(' ', "_"),
                                match &signal_result.decision { signal::SignalDecision::Exit { reason } => reason.to_string(), _ => "N/A".into() },
                                signal_result.metrics.momentum_1s - signal_result.metrics.momentum_5s,
                                if proposed_price > 0.0 { ((avg_price - proposed_price).abs() / proposed_price) * 10_000.0 } else { 0.0 },
                                holding_secs,
                                realized,
                                dynamic_qty * (avg_price - proposed_price),
                            ),
                        }),
                    ));
                }

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
fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

/// Resolve the base URL used for all internal dispatch requests (e.g. POST /trade/request).
///
/// Priority (first non-empty value wins):
///   1. `WEB_BASE_URL`          — explicit absolute URL with scheme (e.g. https://host)
///   2. `RAILWAY_PUBLIC_DOMAIN` — Railway-injected public hostname → https://{domain}
///   3. `WEB_UI_ADDR`           — bind address, with scheme auto-added when absent
///   4. `PORT`                  — fallback loopback URL http://127.0.0.1:{port}
///
/// Returns `None` when no address source is configured at all.
/// When `Some` is returned, the value is guaranteed to be non-empty, start with
/// "http://" or "https://", and have no trailing slash, so that appending
/// "/trade/request" produces a valid absolute URL without double slashes.
fn resolve_web_base_url() -> Option<String> {
    // 1. Explicit absolute base URL (highest priority).
    if let Ok(v) = std::env::var("WEB_BASE_URL") {
        let v = v.trim().trim_end_matches('/').to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }

    // 2. Railway-injected public domain (no scheme).
    if let Ok(domain) = std::env::var("RAILWAY_PUBLIC_DOMAIN") {
        let domain = domain.trim().trim_end_matches('/').to_string();
        if !domain.is_empty() {
            return Some(format!("https://{}", domain));
        }
    }

    // 3. WEB_UI_ADDR — may be a bare host:port or a full URL.
    if let Ok(addr) = std::env::var("WEB_UI_ADDR") {
        let addr = addr.trim().trim_end_matches('/').to_string();
        if !addr.is_empty() {
            if addr.starts_with("http://") || addr.starts_with("https://") {
                return Some(addr);
            }

            let normalized_addr = if let Some(rest) = addr.strip_prefix("0.0.0.0:") {
                format!("127.0.0.1:{}", rest)
            } else if addr == "0.0.0.0" {
                "127.0.0.1".to_string()
            } else {
                addr
            };

            return Some(format!("http://{}", normalized_addr));
        }
    }

    // 4. PORT only — dispatch to self via loopback.
    if let Ok(port) = std::env::var("PORT") {
        let port = port.trim().to_string();
        if !port.is_empty() {
            return Some(format!("http://127.0.0.1:{}", port));
        }
    }

    None
}

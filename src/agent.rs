use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::authority::{AuthorityLayer, AuthorityMode};
use crate::client::BinanceClient;
use crate::events::{OperatorActionPayload, StoredEvent, TradingEvent};
use crate::executor::{ExecutionState, Executor};
use crate::feed::FeedState;
use crate::reconciler::TruthState;
use crate::signal::SignalEngine;
use crate::store::EventStore;
use crate::withdrawal::{WithdrawalManager, WithdrawalStatus};

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub sweep_threshold: f64,
    pub sweep_asset: String,
    pub sweep_interval: Duration,
    pub sweep_network: String,
}

impl AgentConfig {
    pub fn enabled(&self) -> bool {
        self.sweep_threshold > 0.0 && !self.sweep_asset.trim().is_empty()
    }
}

#[derive(Clone)]
pub struct AgentState {
    pub store: Arc<dyn EventStore>,
    pub exec: Arc<Executor>,
    pub feed: Arc<Mutex<FeedState>>,
    pub signal: Arc<Mutex<SignalEngine>>,
    pub truth: Arc<Mutex<TruthState>>,
    pub authority: Arc<AuthorityLayer>,
    pub withdrawals: Arc<WithdrawalManager>,
    pub client: Arc<BinanceClient>,
    pub symbol: String,
    pub web_base_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TradeAgentConfig {
    pub enabled: bool,
    pub trade_size: f64,
    pub momentum_threshold: f64,
    pub poll_interval: Duration,
    pub max_spread_bps: f64,
}

impl TradeAgentConfig {
    pub fn active(&self) -> bool {
        self.enabled && self.trade_size > 0.0
    }
}

pub fn spawn_profit_sweep_agent(cfg: AgentConfig, state: AgentState) {
    if !cfg.enabled() {
        info!("[AGENT] Profit sweep agent disabled (threshold<=0 or asset missing)");
        return;
    }

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(cfg.sweep_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        log_agent_action(
            &*state.store,
            "started",
            &format!(
                "profit_sweep threshold={:.8} asset={} interval_s={}",
                cfg.sweep_threshold,
                cfg.sweep_asset.to_uppercase(),
                cfg.sweep_interval.as_secs()
            ),
        );

        loop {
            interval.tick().await;

            let mode = state.authority.mode().await;
            let (position_size, pnl_total, pending_withdrawals) = {
                let t = state.truth.lock().await;
                let position_size = t.position.size;
                let pnl_total = t.position.realized_pnl + t.position.unrealized_pnl;
                drop(t);

                let pending = state
                    .withdrawals
                    .proposals()
                    .await
                    .into_iter()
                    .filter(|w| {
                        w.asset.eq_ignore_ascii_case(&cfg.sweep_asset)
                            && matches!(
                                w.status,
                                WithdrawalStatus::Requested
                                    | WithdrawalStatus::Approved
                                    | WithdrawalStatus::Executing
                            )
                    })
                    .count();
                (position_size, pnl_total, pending)
            };

            if mode == AuthorityMode::Off {
                log_agent_action(
                    &*state.store,
                    "skipped_mode_off",
                    &format!(
                        "asset={} pos_size={:.8} pnl_total={:+.4} pending_withdrawals={}",
                        cfg.sweep_asset.to_uppercase(),
                        position_size,
                        pnl_total,
                        pending_withdrawals
                    ),
                );
                continue;
            }

            if pending_withdrawals > 0 {
                log_agent_action(
                    &*state.store,
                    "skipped_pending_exists",
                    &format!(
                        "asset={} pending_withdrawals={} pos_size={:.8} pnl_total={:+.4}",
                        cfg.sweep_asset.to_uppercase(),
                        pending_withdrawals,
                        position_size,
                        pnl_total
                    ),
                );
                continue;
            }

            let free_balance = match fetch_free_balance(&state.client, &cfg.sweep_asset).await {
                Ok(v) => v,
                Err(e) => {
                    log_agent_action(
                        &*state.store,
                        "balance_read_failed",
                        &format!("asset={} error={}", cfg.sweep_asset.to_uppercase(), e),
                    );
                    continue;
                }
            };

            if free_balance <= cfg.sweep_threshold {
                log_agent_action(
                    &*state.store,
                    "skipped_below_threshold",
                    &format!(
                        "asset={} free={:.8} threshold={:.8} pos_size={:.8} pnl_total={:+.4}",
                        cfg.sweep_asset.to_uppercase(),
                        free_balance,
                        cfg.sweep_threshold,
                        position_size,
                        pnl_total
                    ),
                );
                continue;
            }

            let Some(url) = state.web_base_url.as_ref().map(|b| format!("{}/withdraw/request", b)) else {
                log_agent_action(
                    &*state.store,
                    "skipped_no_web_ui",
                    "WEB_UI_ADDR missing; cannot POST /withdraw/request",
                );
                continue;
            };

            let sweep_amount = free_balance - cfg.sweep_threshold;
            let destination = state
                .withdrawals
                .allowed_destinations()
                .into_iter()
                .next()
                .unwrap_or_default();

            if destination.is_empty() {
                log_agent_action(
                    &*state.store,
                    "skipped_no_destination",
                    "no allowed withdrawal destination configured",
                );
                continue;
            }

            let mut form = HashMap::new();
            form.insert("amount".to_string(), format!("{:.8}", sweep_amount.max(0.0)));
            form.insert("asset".to_string(), cfg.sweep_asset.to_uppercase());
            form.insert("destination".to_string(), destination.clone());
            form.insert("network".to_string(), cfg.sweep_network.to_uppercase());
            form.insert(
                "reason".to_string(),
                format!(
                    "agent_profit_sweep free={:.8} threshold={:.8} mode={}",
                    free_balance,
                    cfg.sweep_threshold,
                    mode
                ),
            );
            form.insert("confirm_text".to_string(), "CONFIRM".to_string());
            form.insert("confirm_checkbox".to_string(), "on".to_string());
            if sweep_amount >= 500.0 {
                form.insert("confirm_large_text".to_string(), "WITHDRAW LARGE".to_string());
            }

            let post_result = reqwest::Client::new().post(&url).form(&form).send().await;
            match post_result {
                Ok(resp) => {
                    let status = resp.status();
                    let stage = if status.is_success() {
                        match mode {
                            AuthorityMode::Assist => "assist_proposal_posted",
                            AuthorityMode::Auto => "auto_request_posted",
                            AuthorityMode::Off => "skipped_mode_off",
                        }
                    } else {
                        "request_failed"
                    };
                    log_agent_action(
                        &*state.store,
                        stage,
                        &format!(
                            "url={} status={} mode={} asset={} free={:.8} threshold={:.8} amount={:.8}",
                            url,
                            status,
                            mode,
                            cfg.sweep_asset.to_uppercase(),
                            free_balance,
                            cfg.sweep_threshold,
                            sweep_amount
                        ),
                    );
                }
                Err(e) => {
                    warn!(error = %e, "[AGENT] Profit sweep POST failed");
                    log_agent_action(
                        &*state.store,
                        "request_failed",
                        &format!(
                            "mode={} asset={} free={:.8} threshold={:.8} amount={:.8} error={}",
                            mode,
                            cfg.sweep_asset.to_uppercase(),
                            free_balance,
                            cfg.sweep_threshold,
                            sweep_amount,
                            e
                        ),
                    );
                }
            }
        }
    });
}

pub fn spawn_trade_agent(cfg: TradeAgentConfig, state: AgentState) {
    if !cfg.active() {
        info!("[AGENT] Trade agent disabled (TRADE_ENABLED=false or TRADE_SIZE<=0)");
        return;
    }

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(cfg.poll_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut mid_history: VecDeque<f64> = VecDeque::with_capacity(32);

        // Tunable trigger parameters (kept local to decision logic only).
        const TAKE_PROFIT_MULTIPLIER: f64 = 1.01;
        const STOP_LOSS_MULTIPLIER: f64 = 0.99;
        const DIP_LOOKBACK_CYCLES: usize = 5;
        const DIP_TRIGGER_PCT: f64 = 0.003; // 0.30%

        log_agent_action(
            &*state.store,
            "started",
            &format!(
                "trade enabled={} symbol={} size={:.8} momentum_threshold={:.6} spread_max_bps={:.2} interval_s={}",
                cfg.enabled,
                state.symbol,
                cfg.trade_size,
                cfg.momentum_threshold,
                cfg.max_spread_bps,
                cfg.poll_interval.as_secs()
            ),
        );

        loop {
            interval.tick().await;

            let mode = state.authority.mode().await;
            let exec_state = state.exec.execution_state().await;
            let pending = state.authority.pending_proposals().await;

            let metrics = {
                let feed = state.feed.lock().await;
                let signal = state.signal.lock().await;
                signal.compute_metrics_pub(&feed)
            };

            let (position_size, buy_power, sell_inventory, avg_entry) = {
                let t = state.truth.lock().await;
                (
                    t.position.size.max(0.0),
                    t.buy_power.max(0.0),
                    t.sell_inventory.max(0.0),
                    t.position.avg_entry.max(0.0),
                )
            };
            let has_base = sell_inventory > 0.0;
            let has_quote = buy_power > 0.0;
            let market_buy_ok = metrics.momentum_5s > cfg.momentum_threshold
                && metrics.spread_bps <= cfg.max_spread_bps;
            let market_bearish = metrics.momentum_5s < -cfg.momentum_threshold;

            // Track recent mid prices for dip-based entry logic.
            if metrics.mid.is_finite() && metrics.mid > 0.0 {
                mid_history.push_back(metrics.mid);
                while mid_history.len() > 32 {
                    mid_history.pop_front();
                }
            }
            let dip_reference = mid_history
                .len()
                .checked_sub(DIP_LOOKBACK_CYCLES)
                .and_then(|idx| mid_history.get(idx).copied());
            let dip_pct = dip_reference
                .filter(|r| *r > 0.0)
                .map(|r| (metrics.mid / r) - 1.0);

            let (decision, side, reason_for_trade, trigger_type) = if has_base {
                let entry_price = if avg_entry > 0.0 { avg_entry } else { metrics.mid };
                let no_entry_price = avg_entry <= 0.0;
                let take_profit_hit = metrics.mid >= entry_price * TAKE_PROFIT_MULTIPLIER;
                let stop_loss_hit = metrics.mid <= entry_price * STOP_LOSS_MULTIPLIER;
                let momentum_exit_hit = market_bearish;

                if take_profit_hit {
                    (
                        "INVENTORY_SELL",
                        Some("SELL"),
                        format!(
                            "take_profit price={:.2} entry={:.2} target={:.2} pos={:.8} sell_inventory={:.8}",
                            metrics.mid,
                            entry_price,
                            entry_price * TAKE_PROFIT_MULTIPLIER,
                            position_size,
                            sell_inventory
                        ),
                        Some("profit"),
                    )
                } else if stop_loss_hit {
                    (
                        "INVENTORY_SELL",
                        Some("SELL"),
                        format!(
                            "stop_loss price={:.2} entry={:.2} floor={:.2} pos={:.8} sell_inventory={:.8}",
                            metrics.mid,
                            entry_price,
                            entry_price * STOP_LOSS_MULTIPLIER,
                            position_size,
                            sell_inventory
                        ),
                        Some("stop_loss"),
                    )
                } else if momentum_exit_hit || no_entry_price {
                    (
                        "INVENTORY_SELL",
                        Some("SELL"),
                        format!(
                            "momentum_exit price={:.2} entry_ref={:.2} momentum={:+.6} threshold={:.6} no_entry_price={} pos={:.8} sell_inventory={:.8}",
                            metrics.mid,
                            entry_price,
                            metrics.momentum_5s,
                            cfg.momentum_threshold,
                            no_entry_price,
                            position_size,
                            sell_inventory
                        ),
                        Some("momentum"),
                    )
                } else {
                    (
                        "HOLD",
                        None,
                        format!(
                            "waiting_sell_trigger price={:.2} entry={:.2} tp={:.2} sl={:.2} momentum={:+.6} threshold={:.6}",
                            metrics.mid,
                            entry_price,
                            entry_price * TAKE_PROFIT_MULTIPLIER,
                            entry_price * STOP_LOSS_MULTIPLIER,
                            metrics.momentum_5s,
                            cfg.momentum_threshold
                        ),
                        None,
                    )
                }
            } else if has_quote {
                let dip_trigger_hit = dip_pct.map(|v| v <= -DIP_TRIGGER_PCT).unwrap_or(false);
                if dip_trigger_hit {
                    (
                        "INVENTORY_BUY",
                        Some("BUY"),
                        format!(
                            "buy_dip price={:.2} dip_pct={:+.4}% lookback_cycles={} trigger={:.4}% buy_power={:.8}",
                            metrics.mid,
                            dip_pct.unwrap_or(0.0) * 100.0,
                            DIP_LOOKBACK_CYCLES,
                            -DIP_TRIGGER_PCT * 100.0,
                            buy_power
                        ),
                        Some("momentum"),
                    )
                } else if market_buy_ok {
                    (
                        "INVENTORY_BUY",
                        Some("BUY"),
                        format!(
                            "momentum_entry price={:.2} momentum={:+.6}>{:.6} spread_bps={:.2}<={:.2} buy_power={:.8}",
                            metrics.mid,
                            metrics.momentum_5s,
                            cfg.momentum_threshold,
                            metrics.spread_bps,
                            cfg.max_spread_bps,
                            buy_power
                        ),
                        Some("momentum"),
                    )
                } else {
                    (
                        "HOLD",
                        None,
                        format!(
                            "waiting_buy_trigger price={:.2} momentum={:+.6} threshold={:.6} spread_bps={:.2}/{:.2} dip_pct={}",
                            metrics.mid,
                            metrics.momentum_5s,
                            cfg.momentum_threshold,
                            metrics.spread_bps,
                            cfg.max_spread_bps,
                            dip_pct
                                .map(|v| format!("{:+.4}%", v * 100.0))
                                .unwrap_or_else(|| "n/a".to_string())
                        ),
                        None,
                    )
                }
            } else if market_buy_ok {
                (
                    "LONG",
                    Some("BUY"),
                    format!(
                        "fallback momentum_entry momentum={:+.6}>{:.6} spread_bps={:.2}<={:.2} imbalance_1s={:+.3} price={:.2}",
                        metrics.momentum_5s,
                        cfg.momentum_threshold,
                        metrics.spread_bps,
                        cfg.max_spread_bps,
                        metrics.imbalance_1s,
                        metrics.mid
                    ),
                    Some("momentum"),
                )
            } else if market_bearish {
                (
                    "SHORT",
                    Some("SELL"),
                    format!(
                        "fallback momentum_exit momentum={:+.6}<-{:.6} spread_bps={:.2} imbalance_1s={:+.3} price={:.2}",
                        metrics.momentum_5s,
                        cfg.momentum_threshold,
                        metrics.spread_bps,
                        metrics.imbalance_1s,
                        metrics.mid
                    ),
                    Some("momentum"),
                )
            } else {
                (
                    "HOLD",
                    None,
                    format!(
                        "momentum={:+.6} threshold={:.6} spread_bps={:.2} imbalance_1s={:+.3} price={:.2} pos={:.8} buy_power={:.8} sell_inventory={:.8}",
                        metrics.momentum_5s, cfg.momentum_threshold, metrics.spread_bps, metrics.imbalance_1s, metrics.mid, position_size, buy_power, sell_inventory
                    ),
                    None,
                )
            };

            log_agent_action(
                &*state.store,
                "trade_decision",
                &format!(
                    "mode={} decision={} symbol={} trigger_type={} reason_for_trade={}",
                    mode,
                    decision,
                    state.symbol,
                    trigger_type.unwrap_or("none"),
                    reason_for_trade
                ),
            );

            let Some(side) = side else { continue; };

            if mode == AuthorityMode::Off {
                continue;
            }

            if !matches!(exec_state, ExecutionState::Idle) || !pending.is_empty() {
                log_agent_action(
                    &*state.store,
                    "trade_skipped_idempotent_guard",
                    &format!(
                        "mode={} exec_state={} pending_proposals={} decision={} side={}",
                        mode,
                        exec_state,
                        pending.len(),
                        decision,
                        side
                    ),
                );
                continue;
            }

            let Some(url) = state.web_base_url.as_ref().map(|b| format!("{}/trade/request", b)) else {
                log_agent_action(
                    &*state.store,
                    "trade_skipped_no_web_ui",
                    "WEB_UI_ADDR missing; cannot POST /trade/request",
                );
                continue;
            };

            let mut form = HashMap::new();
            form.insert("symbol".to_string(), state.symbol.clone());
            form.insert("side".to_string(), side.to_string());
            form.insert("size".to_string(), format!("{:.8}", cfg.trade_size));
            form.insert(
                "reason".to_string(),
                format!(
                    "agent_trade decision={} trigger_type={} reason_for_trade={}",
                    decision,
                    trigger_type.unwrap_or("none"),
                    reason_for_trade
                ),
            );

            match reqwest::Client::new().post(&url).form(&form).send().await {
                Ok(resp) => {
                    log_agent_action(
                        &*state.store,
                        "trade_request_posted",
                        &format!(
                            "status={} mode={} decision={} side={} symbol={} size={:.8}",
                            resp.status(),
                            mode,
                            decision,
                            side,
                            state.symbol,
                            cfg.trade_size
                        ),
                    );
                }
                Err(e) => {
                    warn!(error = %e, "[AGENT] Trade request POST failed");
                    log_agent_action(
                        &*state.store,
                        "trade_request_failed",
                        &format!(
                            "mode={} decision={} side={} symbol={} size={:.8} error={}",
                            mode, decision, side, state.symbol, cfg.trade_size, e
                        ),
                    );
                }
            }
        }
    });
}

async fn fetch_free_balance(client: &BinanceClient, asset: &str) -> Result<f64, String> {
    let balances = client
        .fetch_balances()
        .await
        .map_err(|e| format!("fetch_balances failed: {e:#}"))?;
    Ok(balances
        .into_iter()
        .find(|b| b.asset.eq_ignore_ascii_case(asset))
        .map(|b| b.free)
        .unwrap_or(0.0))
}

fn log_agent_action(store: &dyn EventStore, action: &str, reason: &str) {
    store.append(StoredEvent::new(
        None,
        None,
        None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("agent_action:{action}"),
            reason: reason.to_string(),
        }),
    ));
}

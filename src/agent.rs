use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::authority::{AuthorityLayer, AuthorityMode};
use crate::client::BinanceClient;
use crate::events::{OperatorActionPayload, StoredEvent, TradingEvent};
use crate::reconciler::TruthState;
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
    pub truth: Arc<Mutex<TruthState>>,
    pub authority: Arc<AuthorityLayer>,
    pub withdrawals: Arc<WithdrawalManager>,
    pub client: Arc<BinanceClient>,
    pub web_base_url: Option<String>,
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

async fn fetch_free_balance(client: &BinanceClient, asset: &str) -> Result<f64, String> {
    let balances = client
        .fetch_balances()
        .await
        .map_err(|e| format!("fetch_balances failed: {e:#}"))?;
    Ok(balances
        .into_iter()
        .find(|b| b.asset.eq_ignore_ascii_case(asset))
        .and_then(|b| b.free.parse::<f64>().ok())
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

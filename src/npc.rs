use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::info;

use crate::agent::{AgentState, TradeAgentConfig};
use crate::authority::AuthorityMode;
use crate::events::{OperatorActionPayload, StoredEvent, TradingEvent};
use crate::executor::ExecutionState;
use crate::store::EventStore;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NpcRole {
    Scout,
    DipBuyer,
    MomentumExecutor,
    RiskManager,
    InventoryManager,
    Learner,
}

impl NpcRole {
    fn as_str(&self) -> &'static str {
        match self {
            NpcRole::Scout => "scout",
            NpcRole::DipBuyer => "dip_buyer",
            NpcRole::MomentumExecutor => "momentum_executor",
            NpcRole::RiskManager => "risk_manager",
            NpcRole::InventoryManager => "inventory_manager",
            NpcRole::Learner => "learner",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpcLifecycleState {
    Proposed,
    Authorized,
    Queued,
    Executing,
    Executed,
    Observed,
    Learned,
    Rejected,
    Blocked,
    Conflict,
    Expired,
    Superseded,
}

impl NpcLifecycleState {
    fn as_str(&self) -> &'static str {
        match self {
            NpcLifecycleState::Proposed => "PROPOSED",
            NpcLifecycleState::Authorized => "AUTHORIZED",
            NpcLifecycleState::Queued => "QUEUED",
            NpcLifecycleState::Executing => "EXECUTING",
            NpcLifecycleState::Executed => "EXECUTED",
            NpcLifecycleState::Observed => "OBSERVED",
            NpcLifecycleState::Learned => "LEARNED",
            NpcLifecycleState::Rejected => "REJECTED",
            NpcLifecycleState::Blocked => "BLOCKED",
            NpcLifecycleState::Conflict => "CONFLICT",
            NpcLifecycleState::Expired => "EXPIRED",
            NpcLifecycleState::Superseded => "SUPERSEDED",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpcTradingMode {
    Simulation,
    Paper,
    Live,
}

impl NpcTradingMode {
    fn from_env() -> Self {
        let mode = std::env::var("NPC_TRADING_MODE").unwrap_or_else(|_| "paper".to_string());
        match mode.to_ascii_lowercase().as_str() {
            "simulation" | "sim" => Self::Simulation,
            "live" => Self::Live,
            _ => Self::Paper,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Simulation => "simulation",
            Self::Paper => "paper",
            Self::Live => "live",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NpcGuardConfig {
    pub max_spread_bps: f64,
    pub min_liquidity_score: f64,
    pub max_slippage_bps: f64,
    pub cooldown_secs: u64,
    pub max_position_qty: f64,
    pub kill_switch: bool,
}

#[derive(Clone, Debug)]
pub struct NpcConfig {
    pub enabled: bool,
    pub cycle_interval: Duration,
    pub trade_size: f64,
    pub momentum_threshold: f64,
    pub dip_lookback_cycles: usize,
    pub dip_trigger_pct: f64,
    pub mode: NpcTradingMode,
    pub guards: NpcGuardConfig,
}

impl NpcConfig {
    pub fn from_trade_cfg(cfg: &TradeAgentConfig) -> Self {
        Self {
            enabled: cfg.active(),
            cycle_interval: cfg.poll_interval,
            trade_size: cfg.trade_size,
            momentum_threshold: cfg.momentum_threshold,
            dip_lookback_cycles: env_usize("NPC_DIP_LOOKBACK_CYCLES", 5),
            dip_trigger_pct: env_f64("NPC_DIP_TRIGGER_PCT", 0.003),
            mode: NpcTradingMode::from_env(),
            guards: NpcGuardConfig {
                max_spread_bps: env_f64("NPC_MAX_SPREAD_BPS", cfg.max_spread_bps),
                min_liquidity_score: env_f64("NPC_MIN_LIQUIDITY_SCORE", 3.0),
                max_slippage_bps: env_f64("NPC_MAX_SLIPPAGE_BPS", 15.0),
                cooldown_secs: env_u64("NPC_COOLDOWN_SECS", 2),
                max_position_qty: env_f64("NPC_MAX_POSITION_QTY", cfg.trade_size.max(0.0) * 4.0),
                kill_switch: env_bool("NPC_KILL_SWITCH", false),
            },
        }
    }
}

#[derive(Clone, Debug)]
struct WorkerProposal {
    action_id: String,
    role: NpcRole,
    side: String,
    score: f64,
    reason: String,
    expected_slippage_bps: f64,
}

#[derive(Clone, Debug, Default)]
struct ActionState {
    status: Option<NpcLifecycleState>,
    actor: Option<NpcRole>,
    created_at: Option<Instant>,
}

#[derive(Clone, Debug, Default)]
struct AgentPerformance {
    proposed: u64,
    executed: u64,
    blocked: u64,
    wins: u64,
    gross_pnl: f64,
    total_slippage_bps: f64,
    total_spread_bps: f64,
    total_hold_ms: f64,
}

#[derive(Default)]
struct NpcRuntimeState {
    // ephemeral runtime/action state layer
    action_state: HashMap<String, ActionState>,
    last_action_at: HashMap<NpcRole, Instant>,
    mid_history: VecDeque<f64>,
    open_actions: HashMap<String, (NpcRole, String, f64, Instant, f64)>, // role, side, entry_mid, ts, spread
    paper_executions: u64,
    // derived metrics layer
    perf: HashMap<NpcRole, AgentPerformance>,
}

pub fn spawn_npc_trading_layer(cfg: NpcConfig, state: AgentState) {
    if !cfg.enabled {
        info!("[NPC] Trading layer disabled");
        return;
    }

    tokio::spawn(async move {
        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        let mut interval = tokio::time::interval(cfg.cycle_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        log_npc_event(&*state.store, "layer_started", &format!(
            "mode={} interval_s={} trade_size={:.8}",
            cfg.mode.as_str(),
            cfg.cycle_interval.as_secs(),
            cfg.trade_size
        ));

        loop {
            interval.tick().await;
            run_cycle(&cfg, &state, Arc::clone(&runtime)).await;
        }
    });
}

async fn run_cycle(cfg: &NpcConfig, state: &AgentState, runtime: Arc<Mutex<NpcRuntimeState>>) {
    let mode = state.authority.mode().await;
    if mode == AuthorityMode::Off {
        return;
    }

    let exec_state = state.exec.execution_state().await;
    let pending = state.authority.pending_proposals().await;
    let metrics = {
        let feed = state.feed.lock().await;
        let signal = state.signal.lock().await;
        signal.compute_metrics_pub(&feed)
    };

    let (position_size, buy_power, sell_inventory) = {
        let t = state.truth.lock().await;
        (t.position.size.max(0.0), t.buy_power.max(0.0), t.sell_inventory.max(0.0))
    };

    let mut rt = runtime.lock().await;
    if metrics.mid.is_finite() && metrics.mid > 0.0 {
        rt.mid_history.push_back(metrics.mid);
        while rt.mid_history.len() > 64 {
            rt.mid_history.pop_front();
        }
    }

    let mut candidates = build_worker_candidates(cfg, &rt.mid_history, &metrics, position_size, buy_power, sell_inventory);
    for c in &candidates {
        rt.perf.entry(c.role).or_default().proposed += 1;
        lifecycle(&*state.store, c.role, &c.action_id, NpcLifecycleState::Proposed, &c.reason);
        rt.action_state.insert(c.action_id.clone(), ActionState { status: Some(NpcLifecycleState::Proposed), actor: Some(c.role), created_at: Some(Instant::now()) });
    }

    if candidates.is_empty() {
        observe_and_learn(&mut rt, &*state.store, metrics.mid);
        return;
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal).then_with(|| a.role.cmp(&b.role)));

    let chosen = candidates[0].clone();
    for stale in candidates.iter().skip(1) {
        lifecycle(&*state.store, stale.role, &stale.action_id, NpcLifecycleState::Superseded, "lower ranked candidate superseded by orchestrator");
    }

    if !matches!(exec_state, ExecutionState::Idle) || !pending.is_empty() {
        lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Conflict, "executor non-idle or pending proposal conflict");
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        return;
    }

    let guard_reasons = evaluate_guards(cfg, &rt, chosen.role, &chosen.side, position_size, &metrics, chosen.expected_slippage_bps);
    if !guard_reasons.is_empty() {
        lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Blocked, &guard_reasons.join("|"));
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        return;
    }

    lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Authorized, "all orchestrator and guard checks passed");

    if cfg.mode == NpcTradingMode::Live && rt.paper_executions == 0 {
        lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Rejected, "live mode requires at least one successful paper execution first");
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        return;
    }

    lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Queued, "ready for dispatch");

    if cfg.mode == NpcTradingMode::Simulation {
        lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Executing, "simulation mode dispatch");
        lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Executed, "simulation fill accepted");
        rt.last_action_at.insert(chosen.role, Instant::now());
        rt.perf.entry(chosen.role).or_default().executed += 1;
    } else {
        let Some(url) = state.web_base_url.as_ref().map(|b| format!("{}/trade/request", b)) else {
            lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Expired, "WEB_UI_ADDR missing; cannot dispatch trade request");
            return;
        };

        let mut form = HashMap::new();
        form.insert("symbol".to_string(), state.symbol.clone());
        form.insert("side".to_string(), chosen.side.clone());
        form.insert("size".to_string(), format!("{:.8}", cfg.trade_size));
        form.insert("reason".to_string(), format!("npc role={} action_id={} reason={}", chosen.role.as_str(), chosen.action_id, chosen.reason));

        lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Executing, "dispatching trade request to web layer");
        match reqwest::Client::new().post(&url).form(&form).send().await {
            Ok(resp) if resp.status().is_success() => {
                lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Executed, &format!("http_status={}", resp.status()));
                rt.last_action_at.insert(chosen.role, Instant::now());
                rt.perf.entry(chosen.role).or_default().executed += 1;
                rt.paper_executions += 1;
            }
            Ok(resp) => {
                lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Rejected, &format!("http_status={}", resp.status()));
                return;
            }
            Err(e) => {
                lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Rejected, &format!("request_error={}", e));
                return;
            }
        }
    }

    rt.open_actions.insert(chosen.action_id.clone(), (chosen.role, chosen.side.clone(), metrics.mid, Instant::now(), metrics.spread_bps));
    observe_and_learn(&mut rt, &*state.store, metrics.mid);
}

fn build_worker_candidates(
    cfg: &NpcConfig,
    mid_history: &VecDeque<f64>,
    metrics: &crate::signal::SignalMetrics,
    position_size: f64,
    buy_power: f64,
    sell_inventory: f64,
) -> Vec<WorkerProposal> {
    let mut proposals = Vec::new();
    let momentum = metrics.momentum_5s;
    let liquidity_score = (metrics.trade_samples as f64).max(0.0);

    proposals.push(WorkerProposal {
        action_id: format!("scout-{}", uuid::Uuid::new_v4()),
        role: NpcRole::Scout,
        side: if momentum >= 0.0 { "BUY".into() } else { "SELL".into() },
        score: momentum.abs() * 1000.0 + liquidity_score * 0.01,
        reason: format!("scout_opportunity momentum_5s={:+.6} liquidity_score={:.2}", momentum, liquidity_score),
        expected_slippage_bps: metrics.spread_bps * 0.5,
    });

    if buy_power > 0.0 {
        if let Some(dip) = dip_pct_from_history(mid_history, cfg.dip_lookback_cycles) {
            if dip <= -cfg.dip_trigger_pct {
                proposals.push(WorkerProposal {
                    action_id: format!("dip-{}", uuid::Uuid::new_v4()),
                    role: NpcRole::DipBuyer,
                    side: "BUY".into(),
                    score: (-dip * 1500.0).max(0.0),
                    reason: format!("dip_trigger dip_pct={:+.4}% lookback={} trigger=-{:.4}%", dip * 100.0, cfg.dip_lookback_cycles, cfg.dip_trigger_pct * 100.0),
                    expected_slippage_bps: metrics.spread_bps * 0.6,
                });
            }
        }
    }

    if momentum > cfg.momentum_threshold {
        proposals.push(WorkerProposal {
            action_id: format!("mom-{}", uuid::Uuid::new_v4()),
            role: NpcRole::MomentumExecutor,
            side: "BUY".into(),
            score: momentum * 2000.0,
            reason: format!("momentum_breakout momentum_5s={:+.6} threshold={:.6}", momentum, cfg.momentum_threshold),
            expected_slippage_bps: metrics.spread_bps * 0.7,
        });
    }

    if sell_inventory > 0.0 && momentum < -cfg.momentum_threshold {
        proposals.push(WorkerProposal {
            action_id: format!("inv-{}", uuid::Uuid::new_v4()),
            role: NpcRole::InventoryManager,
            side: "SELL".into(),
            score: momentum.abs() * 1800.0 + position_size,
            reason: format!("inventory_risk_reduction momentum_5s={:+.6}", momentum),
            expected_slippage_bps: metrics.spread_bps * 0.5,
        });
    }

    proposals.push(WorkerProposal {
        action_id: format!("risk-{}", uuid::Uuid::new_v4()),
        role: NpcRole::RiskManager,
        side: "SELL".into(),
        score: if metrics.spread_bps > cfg.guards.max_spread_bps { 9999.0 } else { 0.0 },
        reason: format!("risk_watch spread_bps={:.2} cap={:.2}", metrics.spread_bps, cfg.guards.max_spread_bps),
        expected_slippage_bps: metrics.spread_bps,
    });

    proposals.push(WorkerProposal {
        action_id: format!("learn-{}", uuid::Uuid::new_v4()),
        role: NpcRole::Learner,
        side: "BUY".into(),
        score: 0.01,
        reason: "learner_observe_only".into(),
        expected_slippage_bps: 0.0,
    });

    proposals
}

fn evaluate_guards(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    role: NpcRole,
    side: &str,
    position_size: f64,
    metrics: &crate::signal::SignalMetrics,
    expected_slippage_bps: f64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if cfg.guards.kill_switch {
        reasons.push("KILL_SWITCH_ACTIVE".to_string());
    }
    if metrics.spread_bps > cfg.guards.max_spread_bps {
        reasons.push(format!("MAX_SPREAD_BPS_EXCEEDED:{:.2}>{:.2}", metrics.spread_bps, cfg.guards.max_spread_bps));
    }
    let liquidity_score = metrics.trade_samples as f64;
    if liquidity_score < cfg.guards.min_liquidity_score {
        reasons.push(format!("LIQUIDITY_TOO_LOW:{:.2}<{:.2}", liquidity_score, cfg.guards.min_liquidity_score));
    }
    if expected_slippage_bps > cfg.guards.max_slippage_bps {
        reasons.push(format!("SLIPPAGE_CEILING_EXCEEDED:{:.2}>{:.2}", expected_slippage_bps, cfg.guards.max_slippage_bps));
    }
    if let Some(last_at) = rt.last_action_at.get(&role) {
        if last_at.elapsed() < Duration::from_secs(cfg.guards.cooldown_secs) {
            reasons.push("ROLE_COOLDOWN_ACTIVE".to_string());
        }
    }
    if side.eq_ignore_ascii_case("BUY") && position_size + cfg.trade_size > cfg.guards.max_position_qty {
        reasons.push(format!("POSITION_LIMIT_EXCEEDED:{:.8}>{:.8}", position_size + cfg.trade_size, cfg.guards.max_position_qty));
    }
    reasons
}

fn observe_and_learn(rt: &mut NpcRuntimeState, store: &dyn EventStore, current_mid: f64) {
    let mut close_ids = Vec::new();
    for (id, (role, side, entry_mid, ts, spread)) in rt.open_actions.iter() {
        if ts.elapsed() >= Duration::from_secs(1) {
            lifecycle(store, *role, id, NpcLifecycleState::Observed, "one-cycle post-execution observation complete");
            let pnl = if side == "BUY" { current_mid - *entry_mid } else { *entry_mid - current_mid };
            let hold_ms = ts.elapsed().as_secs_f64() * 1000.0;
            let perf = rt.perf.entry(*role).or_default();
            perf.gross_pnl += pnl;
            if pnl > 0.0 { perf.wins += 1; }
            perf.total_hold_ms += hold_ms;
            perf.total_spread_bps += *spread;
            perf.total_slippage_bps += spread * 0.5;
            lifecycle(
                store,
                NpcRole::Learner,
                id,
                NpcLifecycleState::Learned,
                &format!(
                    "derived_metrics role={} pnl={:+.6} hold_ms={:.0} win_rate={:.2}%",
                    role.as_str(),
                    pnl,
                    hold_ms,
                    if perf.executed > 0 { (perf.wins as f64 / perf.executed as f64) * 100.0 } else { 0.0 }
                ),
            );
            log_npc_event(
                store,
                "agent_metrics",
                &format!(
                    "role={} proposed={} executed={} blocked={} pnl={:+.6} win_rate={:.2}% avg_spread_entry={:.2} avg_slippage={:.2} avg_hold_ms={:.0}",
                    role.as_str(),
                    perf.proposed,
                    perf.executed,
                    perf.blocked,
                    perf.gross_pnl,
                    if perf.executed > 0 { (perf.wins as f64 / perf.executed as f64) * 100.0 } else { 0.0 },
                    if perf.executed > 0 { perf.total_spread_bps / perf.executed as f64 } else { 0.0 },
                    if perf.executed > 0 { perf.total_slippage_bps / perf.executed as f64 } else { 0.0 },
                    if perf.executed > 0 { perf.total_hold_ms / perf.executed as f64 } else { 0.0 },
                ),
            );
            close_ids.push(id.clone());
        }
    }
    for id in close_ids {
        rt.open_actions.remove(&id);
    }
}

fn lifecycle(store: &dyn EventStore, role: NpcRole, action_id: &str, state: NpcLifecycleState, reason: &str) {
    log_npc_event(store, "lifecycle", &format!(
        "action_id={} role={} state={} reason_code={}",
        action_id,
        role.as_str(),
        state.as_str(),
        reason,
    ));
}

fn log_npc_event(store: &dyn EventStore, action: &str, reason: &str) {
    store.append(StoredEvent::new(
        None,
        None,
        None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("npc:{}", action),
            reason: reason.to_string(),
        }),
    ));
}

fn dip_pct_from_history(mid_history: &VecDeque<f64>, lookback_cycles: usize) -> Option<f64> {
    let current_mid = mid_history.back().copied().filter(|mid| *mid > 0.0)?;
    let reference_mid = mid_history
        .len()
        .checked_sub(1 + lookback_cycles)
        .and_then(|idx| mid_history.get(idx).copied())
        .filter(|mid| *mid > 0.0)?;
    Some((current_mid / reference_mid) - 1.0)
}

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_bool(k: &str, d: bool) -> bool {
    std::env::var(k).map(|v| v.eq_ignore_ascii_case("true") || v == "1").unwrap_or(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dip_lookback_requires_true_horizon() {
        let lookback = 5;
        let mut h: VecDeque<f64> = vec![99.9, 100.0, 100.0, 100.0, 100.0, 99.7].into();
        let first = dip_pct_from_history(&h, lookback).unwrap();
        assert!(first > -0.003);
        h.push_back(99.7);
        let second = dip_pct_from_history(&h, lookback).unwrap();
        assert!(second <= -0.003);
    }
}

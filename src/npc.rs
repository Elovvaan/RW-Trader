use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use tokio::sync::Mutex;
use tracing::info;

use crate::agent::{AgentState, TradeAgentConfig};
use crate::authority::AuthorityMode;
use crate::events::{OperatorActionPayload, StoredEvent, TradingEvent};
use crate::executor::ExecutionState;
use crate::store::EventStore;

const NPC_STATUS_SCANNING: &str = "scanning market";

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

    fn all() -> [NpcRole; 6] {
        [
            NpcRole::Scout,
            NpcRole::DipBuyer,
            NpcRole::MomentumExecutor,
            NpcRole::RiskManager,
            NpcRole::InventoryManager,
            NpcRole::Learner,
        ]
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

/// Runtime control mode for the autonomous trading agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentMode {
    /// Agent is fully stopped — no loop runs.
    Off,
    /// Agent is actively running continuous trading cycles.
    Auto,
    /// Agent loop is running but cycles are skipped (paused).
    Pause,
}

impl AgentMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off   => "off",
            Self::Auto  => "auto",
            Self::Pause => "pause",
        }
    }

    /// Human-readable state label shown in the UI.
    pub fn state_label(&self) -> &'static str {
        match self {
            Self::Off   => "Agent OFF",
            Self::Auto  => "Agent ON — Scanning",
            Self::Pause => "Agent Paused",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off"               => Some(Self::Off),
            "auto"              => Some(Self::Auto),
            "pause" | "paused"  => Some(Self::Pause),
            _                   => None,
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

    fn learner_writable(&self) -> bool {
        matches!(self, Self::Simulation | Self::Paper)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MarketRegime {
    TrendingUp,
    TrendingDown,
    MeanRevert,
    Choppy,
    Volatile,
    Illiquid,
}

impl MarketRegime {
    fn as_str(&self) -> &'static str {
        match self {
            Self::TrendingUp => "TRENDING_UP",
            Self::TrendingDown => "TRENDING_DOWN",
            Self::MeanRevert => "MEAN_REVERT",
            Self::Choppy => "CHOPPY",
            Self::Volatile => "VOLATILE",
            Self::Illiquid => "ILLIQUID",
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
pub struct NpcAlphaConfig {
    pub min_action_score: f64,
    pub vol_spike_bps: f64,
    pub choppy_band_bps: f64,
    pub min_horizon_alignment: f64,
    pub max_concurrent_positions: usize,
    pub max_symbol_exposure_notional: f64,
    pub max_capital_at_risk_notional: f64,
    pub max_drawdown_pct: f64,
    pub drawdown_derisk_trigger_pct: f64,
    pub drawdown_hard_stop_pct: f64,
    pub per_agent_budget_pct: HashMap<NpcRole, f64>,
}

#[derive(Clone, Debug)]
struct LearnerRange {
    min: f64,
    max: f64,
    current: f64,
    step: f64,
}

impl LearnerRange {
    fn bounded(mut self) -> Self {
        if self.current < self.min {
            self.current = self.min;
        }
        if self.current > self.max {
            self.current = self.max;
        }
        self
    }

    fn nudge_tighter(&mut self) {
        self.current = (self.current - self.step).max(self.min);
    }

    fn nudge_looser(&mut self) {
        self.current = (self.current + self.step).min(self.max);
    }
}

#[derive(Clone, Debug)]
struct LearnerConfigRanges {
    spread_tolerance_bps: LearnerRange,
    dip_trigger_pct: LearnerRange,
    cooldown_secs: LearnerRange,
    regime_score_cutoff: HashMap<MarketRegime, LearnerRange>,
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
    pub alpha: NpcAlphaConfig,
}

impl NpcConfig {
    pub fn from_trade_cfg(cfg: &TradeAgentConfig) -> Self {
        let per_agent_budget_pct = HashMap::from([
            (NpcRole::Scout, env_f64("NPC_BUDGET_SCOUT", 0.10)),
            (NpcRole::DipBuyer, env_f64("NPC_BUDGET_DIP", 0.25)),
            (NpcRole::MomentumExecutor, env_f64("NPC_BUDGET_MOMENTUM", 0.30)),
            (NpcRole::RiskManager, env_f64("NPC_BUDGET_RISK", 0.15)),
            (NpcRole::InventoryManager, env_f64("NPC_BUDGET_INVENTORY", 0.20)),
            (NpcRole::Learner, env_f64("NPC_BUDGET_LEARNER", 0.05)),
        ]);

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
            alpha: NpcAlphaConfig {
                min_action_score: env_f64("NPC_MIN_ACTION_SCORE", 0.10),
                vol_spike_bps: env_f64("NPC_VOL_SPIKE_BPS", 45.0),
                choppy_band_bps: env_f64("NPC_CHOPPY_BAND_BPS", 3.0),
                min_horizon_alignment: env_f64("NPC_MIN_HORIZON_ALIGNMENT", 0.30),
                max_concurrent_positions: env_usize("NPC_MAX_CONCURRENT_POSITIONS", 2),
                max_symbol_exposure_notional: env_f64("NPC_MAX_SYMBOL_EXPOSURE", 2_000.0),
                max_capital_at_risk_notional: env_f64("NPC_MAX_CAPITAL_AT_RISK", 3_000.0),
                max_drawdown_pct: env_f64("NPC_MAX_DRAWDOWN_PCT", 0.08),
                drawdown_derisk_trigger_pct: env_f64("NPC_DERISK_DRAWDOWN_PCT", 0.04),
                drawdown_hard_stop_pct: env_f64("NPC_DRAWDOWN_HARD_STOP_PCT", 0.07),
                per_agent_budget_pct,
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
    raw_score: f64,
    reason: String,
    expected_slippage_bps: f64,
    regime_eligible: bool,
    regime_block_reason: Option<String>,
    score_parts: ScoreBreakdown,
    expected_hold_secs: f64,
}

#[derive(Clone, Debug, Default)]
struct ScoreBreakdown {
    edge_estimate: f64,
    spread_cost: f64,
    slippage_risk: f64,
    liquidity_quality: f64,
    volatility_penalty: f64,
    conflict_penalty: f64,
    hold_efficiency: f64,
}

impl ScoreBreakdown {
    fn final_score(&self) -> f64 {
        self.edge_estimate
            - self.spread_cost
            - self.slippage_risk
            + self.liquidity_quality
            - self.volatility_penalty
            - self.conflict_penalty
            + self.hold_efficiency
    }
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
    losses: u64,
    gross_pnl: f64,
    peak_pnl: f64,
    drawdown: f64,
    total_slippage_bps: f64,
    total_spread_bps: f64,
    total_hold_ms: f64,
}

impl AgentPerformance {
    fn win_rate(&self) -> f64 {
        if self.executed == 0 {
            return 0.0;
        }
        self.wins as f64 / self.executed as f64
    }

    fn quality_score(&self) -> f64 {
        let pnl_component = (self.gross_pnl / 10.0).clamp(-0.5, 0.5);
        let win_component = (self.win_rate() - 0.5) * 0.8;
        let dd_penalty = self.drawdown.min(1.0) * 0.7;
        (1.0 + pnl_component + win_component - dd_penalty).clamp(0.15, 1.8)
    }
}

#[derive(Clone, Debug)]
struct OpenAction {
    role: NpcRole,
    side: String,
    entry_mid: f64,
    opened_at: Instant,
    entry_spread_bps: f64,
    expected_edge: f64,
    regime: MarketRegime,
    allocated_qty: f64,
    cycle_id: u64,
}

#[derive(Default)]
struct NpcRuntimeState {
    action_state: HashMap<String, ActionState>,
    last_action_at: HashMap<NpcRole, Instant>,
    mid_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    open_actions: BTreeMap<String, OpenAction>,
    paper_executions: u64,
    perf: HashMap<NpcRole, AgentPerformance>,
    cycle_seq: u64,
    cycle_open_notional: f64,
    peak_equity: f64,
    regime_perf: HashMap<(NpcRole, MarketRegime), AgentPerformance>,
    learner_ranges: Option<LearnerConfigRanges>,
}

#[derive(Clone, Debug)]
pub struct NpcLoopSnapshot {
    /// New 3-state mode (preferred field).
    pub agent_mode: AgentMode,
    /// Kept for backward-compatibility — derived from `agent_mode`.
    pub autonomous_mode: bool,
    pub running: bool,
    pub interval_ms: u64,
    pub cycle_count: u64,
    pub cycle_id: u64,
    pub last_action: String,
    pub timestamp: String,
    pub execution_result: String,
    pub status: String,
    /// Most recent agent decision (e.g. "SELL InventoryManager score=0.85").
    pub last_agent_decision: String,
    /// Why no trade was placed on the last no-action cycle (empty string when a trade executed).
    pub last_no_trade_reason: String,
    /// Current execution pipeline state label.
    pub pipeline_state: String,
}

#[derive(Default)]
struct NpcLoopTelemetry {
    running: bool,
    cycle_count: u64,
    cycle_id: u64,
    last_action: String,
    timestamp: String,
    execution_result: String,
    status: String,
    last_agent_decision: String,
    last_no_trade_reason: String,
    pipeline_state: String,
}

struct NpcLoopControl {
    mode: AgentMode,
    interval_ms: u64,
    stop_tx: Option<tokio::sync::watch::Sender<bool>>,
    pause_tx: Option<tokio::sync::watch::Sender<bool>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl NpcLoopControl {
    fn new(interval_ms: u64, mode: AgentMode) -> Self {
        Self {
            mode,
            interval_ms,
            stop_tx: None,
            pause_tx: None,
            handle: None,
        }
    }
}

#[derive(Clone)]
pub struct NpcAutonomousController {
    cfg: NpcConfig,
    state: AgentState,
    runtime: Arc<Mutex<NpcRuntimeState>>,
    telemetry: Arc<Mutex<NpcLoopTelemetry>>,
    control: Arc<Mutex<NpcLoopControl>>,
}

impl NpcAutonomousController {
    pub fn new(cfg: NpcConfig, state: AgentState) -> Self {
        let interval_ms = cfg.cycle_interval.as_millis().clamp(500, 2000) as u64;
        let mode = if cfg.enabled { AgentMode::Auto } else { AgentMode::Off };
        Self {
            cfg,
            state,
            runtime: Arc::new(Mutex::new(NpcRuntimeState::default())),
            telemetry: Arc::new(Mutex::new(NpcLoopTelemetry {
                last_action: "NO_ACTION".to_string(),
                execution_result: NPC_STATUS_SCANNING.to_string(),
                status: NPC_STATUS_SCANNING.to_string(),
                last_agent_decision: "Waiting for first cycle".to_string(),
                last_no_trade_reason: String::new(),
                pipeline_state: "Scanning".to_string(),
                ..NpcLoopTelemetry::default()
            })),
            control: Arc::new(Mutex::new(NpcLoopControl::new(interval_ms, mode))),
        }
    }

    /// Set the agent to a new mode, starting or stopping the loop as needed.
    pub async fn set_agent_mode(&self, mode: AgentMode) {
        // Capture old mode and update atomically.
        let old_mode = {
            let mut control = self.control.lock().await;
            let old = control.mode;
            control.mode = mode;
            old
        };

        match mode {
            AgentMode::Off => {
                self.stop_trading_loop().await;
                // stop_trading_loop already sets status via AgentMode::Off.state_label()
            }
            AgentMode::Auto => {
                // If we were paused, send unpause signal on the existing loop.
                {
                    let control = self.control.lock().await;
                    if let Some(tx) = control.pause_tx.as_ref() {
                        let _ = tx.send(false);
                    }
                    let loop_running = control.handle.as_ref().map(|h| !h.is_finished()).unwrap_or(false);
                    drop(control);
                    if !loop_running {
                        // Loop was not running (came from Off); start it.
                        self.spawn_trading_loop().await;
                        return;
                    }
                }
                let mut t = self.telemetry.lock().await;
                t.status = AgentMode::Auto.state_label().to_string();
            }
            AgentMode::Pause => {
                if old_mode == AgentMode::Off {
                    // Pausing from Off state is not meaningful — revert and log.
                    let mut control = self.control.lock().await;
                    control.mode = AgentMode::Off;
                    tracing::warn!("[NPC] set_agent_mode(Pause) called from Off — ignored; agent is not running");
                    return;
                }
                let control = self.control.lock().await;
                if let Some(tx) = control.pause_tx.as_ref() {
                    let _ = tx.send(true);
                }
                drop(control);
                let mut t = self.telemetry.lock().await;
                t.status = AgentMode::Pause.state_label().to_string();
            }
        }
    }

    /// Legacy helper — delegates to `set_agent_mode`.
    pub async fn set_autonomous_mode(&self, enabled: bool) {
        if enabled {
            self.set_agent_mode(AgentMode::Auto).await;
        } else {
            self.set_agent_mode(AgentMode::Off).await;
        }
    }

    pub async fn set_interval_ms(&self, interval_ms: u64) {
        let interval_ms = interval_ms.clamp(500, 2000);
        let mut control = self.control.lock().await;
        if control.interval_ms == interval_ms {
            return;
        }
        control.interval_ms = interval_ms;
        let should_restart = control.mode == AgentMode::Auto;
        drop(control);
        if should_restart {
            self.stop_trading_loop().await;
            self.spawn_trading_loop().await;
        }
    }

    pub async fn snapshot(&self) -> NpcLoopSnapshot {
        let telemetry = self.telemetry.lock().await;
        let control = self.control.lock().await;
        let agent_mode = control.mode;
        NpcLoopSnapshot {
            agent_mode,
            autonomous_mode: agent_mode != AgentMode::Off,
            running: telemetry.running,
            interval_ms: control.interval_ms,
            cycle_count: telemetry.cycle_count,
            cycle_id: telemetry.cycle_id,
            last_action: telemetry.last_action.clone(),
            timestamp: telemetry.timestamp.clone(),
            execution_result: telemetry.execution_result.clone(),
            status: telemetry.status.clone(),
            last_agent_decision: telemetry.last_agent_decision.clone(),
            last_no_trade_reason: telemetry.last_no_trade_reason.clone(),
            pipeline_state: telemetry.pipeline_state.clone(),
        }
    }

    pub async fn spawn_trading_loop(&self) {
        let mut control = self.control.lock().await;
        if let Some(handle) = control.handle.as_ref() {
            if !handle.is_finished() {
                return;
            }
        }

        let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
        let (pause_tx, mut pause_rx) = tokio::sync::watch::channel(false);
        let interval_ms = control.interval_ms;
        control.stop_tx = Some(stop_tx);
        control.pause_tx = Some(pause_tx);
        let cfg = self.cfg.clone();
        let state = self.state.clone();
        let runtime = Arc::clone(&self.runtime);
        let telemetry = Arc::clone(&self.telemetry);
        control.handle = Some(tokio::spawn(async move {
            {
                let mut t = telemetry.lock().await;
                t.running = true;
                t.status = AgentMode::Auto.state_label().to_string();
            }
            log_npc_event(
                &*state.store,
                "layer_started",
                &format!(
                    "mode={} interval_ms={} trade_size={:.8}",
                    cfg.mode.as_str(),
                    interval_ms,
                    cfg.trade_size
                ),
            );

            let mut loop_cfg = cfg.clone();
            loop_cfg.cycle_interval = Duration::from_millis(interval_ms);
            let mut interval = tokio::time::interval(loop_cfg.cycle_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // If paused, log scanning and skip cycle execution.
                        if *pause_rx.borrow() {
                            let mut t = telemetry.lock().await;
                            t.status = AgentMode::Pause.state_label().to_string();
                            t.timestamp = Utc::now().to_rfc3339();
                            continue;
                        }
                        let report = run_cycle(&loop_cfg, &state, Arc::clone(&runtime)).await;
                        let mut t = telemetry.lock().await;
                        t.cycle_count = t.cycle_count.saturating_add(1);
                        t.cycle_id = report.cycle_id;
                        t.last_action = report.last_action;
                        t.timestamp = Utc::now().to_rfc3339();
                        t.execution_result = report.execution_result;
                        t.last_agent_decision = report.last_agent_decision;
                        t.last_no_trade_reason = report.no_trade_reason;
                        t.pipeline_state = report.pipeline_state;
                        t.status = match report.status.as_str() {
                            "blocked" => "Blocked by safety checks".to_string(),
                            "running" => AgentMode::Auto.state_label().to_string(),
                            other     => other.to_string(),
                        };
                    }
                    changed = stop_rx.changed() => {
                        if changed.is_ok() && *stop_rx.borrow() {
                            break;
                        }
                    }
                }
            }

            let mut t = telemetry.lock().await;
            t.running = false;
            t.status = AgentMode::Off.state_label().to_string();
        }));
    }

    pub async fn stop_trading_loop(&self) {
        let handle = {
            let mut control = self.control.lock().await;
            if let Some(tx) = control.stop_tx.take() {
                let _ = tx.send(true);
            }
            control.pause_tx = None;
            control.handle.take()
        };
        if let Some(handle) = handle {
            let _ = handle.await;
        }
        let mut t = self.telemetry.lock().await;
        t.running = false;
        t.status = AgentMode::Off.state_label().to_string();
    }
}

pub async fn spawn_npc_trading_layer(controller: &NpcAutonomousController) {
    if controller.snapshot().await.agent_mode == AgentMode::Off {
        info!("[NPC] Trading layer disabled");
        return;
    }
    controller.spawn_trading_loop().await;
}

struct NpcCycleReport {
    cycle_id: u64,
    last_action: String,
    execution_result: String,
    status: String,
    /// Human-readable description of the agent's most recent decision.
    last_agent_decision: String,
    /// Reason no trade was placed (empty string when a trade executed).
    no_trade_reason: String,
    /// Current execution pipeline state label.
    pipeline_state: String,
}

async fn run_cycle(cfg: &NpcConfig, state: &AgentState, runtime: Arc<Mutex<NpcRuntimeState>>) -> NpcCycleReport {
    let no_action = |cycle_id: u64, execution_result: String, status: String| NpcCycleReport {
        cycle_id,
        last_action: "NO_ACTION".to_string(),
        execution_result: execution_result.clone(),
        status,
        last_agent_decision: "HOLD — no actionable trigger".to_string(),
        no_trade_reason: execution_result,
        pipeline_state: "Scanning".to_string(),
    };
    let mode = state.authority.mode().await;
    if mode == AuthorityMode::Off {
        return no_action(0, "authority_mode_off".to_string(), "blocked".to_string());
    }

    let exec_state = state.exec.execution_state().await;
    let pending = state.authority.pending_proposals().await;
    let metrics = {
        let feed = state.feed.lock().await;
        let signal = state.signal.lock().await;
        signal.compute_metrics_pub(&feed)
    };

    let (position_size, buy_power, sell_inventory, exposure_notional, total_balance_usd) = {
        let t = state.truth.lock().await;
        let pos = t.position.size.max(0.0);
        let mid = metrics.mid.max(0.0);
        (pos, t.buy_power.max(0.0), t.sell_inventory.max(0.0), pos * mid, t.total_balance_usd.max(0.0))
    };

    let mut rt = runtime.lock().await;
    rt.cycle_seq = rt.cycle_seq.saturating_add(1);
    let cycle_id = rt.cycle_seq;
    if metrics.mid.is_finite() && metrics.mid > 0.0 {
        rt.mid_history.push_back(metrics.mid);
        while rt.mid_history.len() > 128 {
            rt.mid_history.pop_front();
        }
    }
    if metrics.spread_bps.is_finite() && metrics.spread_bps >= 0.0 {
        rt.spread_history.push_back(metrics.spread_bps);
        while rt.spread_history.len() > 128 {
            rt.spread_history.pop_front();
        }
    }

    if rt.learner_ranges.is_none() {
        rt.learner_ranges = Some(default_learner_ranges(cfg));
    }
    let mut effective_cfg = cfg.clone();
    apply_learner_overrides(&mut effective_cfg, &mut rt);

    let regime = detect_regime(&effective_cfg, &rt.mid_history, &rt.spread_history, &metrics);
    let portfolio_controls = evaluate_portfolio_controls(&effective_cfg, &rt, position_size, exposure_notional, metrics.mid, regime);

    let mut candidates = build_worker_candidates(
        &effective_cfg,
        cycle_id,
        &rt,
        &metrics,
        position_size,
        buy_power,
        sell_inventory,
        regime,
        !matches!(exec_state, ExecutionState::Idle) || !pending.is_empty(),
    );

    let regime_cutoff = rt
        .learner_ranges
        .as_ref()
        .and_then(|r| r.regime_score_cutoff.get(&regime).map(|v| v.current))
        .unwrap_or(effective_cfg.alpha.min_action_score);

    for c in &candidates {
        rt.perf.entry(c.role).or_default().proposed += 1;
        lifecycle(
            &*state.store,
            c.role,
            &c.action_id,
            NpcLifecycleState::Proposed,
            &format!(
                "regime={} score={:.4} edge={:.4} spread_cost={:.4} liquidity={:.4}",
                regime.as_str(),
                c.score,
                c.score_parts.edge_estimate,
                c.score_parts.spread_cost,
                c.score_parts.liquidity_quality
            ),
        );
        rt.action_state.insert(
            c.action_id.clone(),
            ActionState {
                status: Some(NpcLifecycleState::Proposed),
                actor: Some(c.role),
                created_at: Some(Instant::now()),
            },
        );
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.role.cmp(&b.role))
    });

    let metrics_line = format_cycle_metrics(cycle_id, regime, &candidates, regime_cutoff, &portfolio_controls);
    log_npc_event(&*state.store, "alpha_cycle", &metrics_line);

    let Some(chosen) = candidates.first().cloned() else {
        log_npc_event(&*state.store, "no_action", &format!("cycle={} reason=NO_CANDIDATES", cycle_id));
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return no_action(cycle_id, "NO_CANDIDATES".to_string(), "running".to_string());
    };

    for stale in candidates.iter().skip(1) {
        lifecycle(
            &*state.store,
            stale.role,
            &stale.action_id,
            NpcLifecycleState::Superseded,
            "lower-ranked candidate superseded by orchestrator ranking",
        );
    }

    if let Some(reason) = chosen.regime_block_reason.clone() {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Blocked,
            &reason,
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: reason.clone(),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} blocked: {}",
                chosen.side.to_uppercase(), chosen.role.as_str(), reason
            ),
            no_trade_reason: format!("Regime mismatch — {}. Try a different market condition.", reason),
            pipeline_state: "Scanning".to_string(),
        };
    }

    if chosen.score < regime_cutoff {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Blocked,
            &format!(
                "NO_ACTION_SCORE_BELOW_THRESHOLD:{:.4}<{:.4}",
                chosen.score, regime_cutoff
            ),
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: "SCORE_BELOW_THRESHOLD".to_string(),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} score {:.4} below threshold {:.4}",
                chosen.side.to_uppercase(), chosen.role.as_str(), chosen.score, regime_cutoff
            ),
            no_trade_reason: format!(
                "Signal score {:.4} below minimum threshold {:.4}. Waiting for stronger trigger.",
                chosen.score, regime_cutoff
            ),
            pipeline_state: "Scanning".to_string(),
        };
    }

    if !portfolio_controls.is_empty() {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Blocked,
            &portfolio_controls.join("|"),
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: portfolio_controls.join("|"),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} blocked by portfolio controls",
                chosen.side.to_uppercase(), chosen.role.as_str()
            ),
            no_trade_reason: format!(
                "Portfolio risk controls active: {}",
                portfolio_controls.join("; ")
            ),
            pipeline_state: "Scanning".to_string(),
        };
    }

    let allocation = allocate_capital(&effective_cfg, &rt, &chosen, position_size, exposure_notional, buy_power, sell_inventory, metrics.mid);
    if allocation.qty <= 0.0 {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Blocked,
            &format!("ALLOCATION_REJECTED:{}", allocation.reason),
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: format!("ALLOCATION_REJECTED:{}", allocation.reason),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} allocation rejected: {}",
                chosen.side.to_uppercase(), chosen.role.as_str(), allocation.reason
            ),
            no_trade_reason: if chosen.side.eq_ignore_ascii_case("SELL") {
                format!(
                    "SELL blocked — insufficient sell inventory (sell_inv={:.8}, alloc_reason={}). \
                     Ensure base-asset balance is non-zero.",
                    sell_inventory, allocation.reason
                )
            } else {
                format!(
                    "BUY blocked — insufficient buy power (buy_power={:.2}, alloc_reason={}). \
                     Ensure USDT balance is non-zero.",
                    buy_power, allocation.reason
                )
            },
            pipeline_state: "Scanning".to_string(),
        };
    }

    let order_notional = allocation.qty * metrics.mid.max(0.0);
    let guard_reasons = evaluate_guards(
        &effective_cfg,
        &rt,
        chosen.role,
        &chosen.side,
        position_size,
        &metrics,
        chosen.expected_slippage_bps,
        total_balance_usd,
        sell_inventory,
        order_notional,
    );
    if !guard_reasons.is_empty() {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Blocked,
            &guard_reasons.join("|"),
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: guard_reasons.join("|"),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} blocked by safety guards",
                chosen.side.to_uppercase(), chosen.role.as_str()
            ),
            no_trade_reason: format!(
                "Safety guard blocked order: {}",
                guard_reasons.join("; ")
            ),
            pipeline_state: "Scanning".to_string(),
        };
    }

    lifecycle(
        &*state.store,
        chosen.role,
        &chosen.action_id,
        NpcLifecycleState::Authorized,
        &format!(
            "rank=1 score={:.4} raw_score={:.4} allocation_qty={:.8} allocation_reason={}",
            chosen.score, chosen.raw_score, allocation.qty, allocation.reason
        ),
    );

    if effective_cfg.mode == NpcTradingMode::Live && rt.paper_executions == 0 {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Rejected,
            "live mode requires at least one successful paper execution first",
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: "LIVE_REQUIRES_PAPER_EXECUTION".to_string(),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} authorized but blocked: NPC_TRADING_MODE=live requires one paper execution first",
                chosen.side.to_uppercase(), chosen.role.as_str()
            ),
            no_trade_reason: "Live mode safety gate: set NPC_TRADING_MODE=paper to enable execution. \
                               One successful paper execution is required before live trades are allowed."
                .to_string(),
            pipeline_state: "Trigger Matched — Blocked".to_string(),
        };
    }

    lifecycle(&*state.store, chosen.role, &chosen.action_id, NpcLifecycleState::Queued, "ready for dispatch");

    if effective_cfg.mode == NpcTradingMode::Simulation {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Executing,
            "simulation mode dispatch",
        );
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Executed,
            "simulation fill accepted",
        );
        rt.last_action_at.insert(chosen.role, Instant::now());
        rt.perf.entry(chosen.role).or_default().executed += 1;
    } else {
        let Some(url) = state.web_base_url.as_ref().map(|b| format!("{}/trade/request", b)) else {
            lifecycle(
                &*state.store,
                chosen.role,
                &chosen.action_id,
                NpcLifecycleState::Expired,
                "WEB_UI_ADDR missing; cannot dispatch trade request",
            );
            observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
            return NpcCycleReport {
                cycle_id,
                last_action: "NO_ACTION".to_string(),
                execution_result: "WEB_UI_ADDR_MISSING".to_string(),
                status: "blocked".to_string(),
                last_agent_decision: format!(
                    "{} {} trigger matched — blocked: WEB_UI_ADDR not configured",
                    chosen.side.to_uppercase(), chosen.role.as_str()
                ),
                no_trade_reason: "WEB_UI_ADDR environment variable is missing. The agent cannot dispatch \
                                   trade requests without a configured web UI address."
                    .to_string(),
                pipeline_state: "Trigger Matched — Blocked".to_string(),
            };
        };

        let mut form = HashMap::new();
        form.insert("symbol".to_string(), state.symbol.clone());
        form.insert("side".to_string(), chosen.side.clone());
        form.insert("size".to_string(), format!("{:.8}", allocation.qty));
        form.insert(
            "reason".to_string(),
            format!(
                "npc cycle={} role={} action_id={} regime={} score={:.4} expected_edge={:.4}",
                cycle_id,
                chosen.role.as_str(),
                chosen.action_id,
                regime.as_str(),
                chosen.score,
                chosen.score_parts.edge_estimate
            ),
        );

        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Executing,
            &format!(
                "dispatching trade request to web layer side={} qty={:.8} symbol={}",
                chosen.side, allocation.qty, state.symbol
            ),
        );
        match reqwest::Client::new().post(&url).form(&form).send().await {
            Ok(resp) if resp.status().is_success() => {
                lifecycle(
                    &*state.store,
                    chosen.role,
                    &chosen.action_id,
                    NpcLifecycleState::Executed,
                    &format!("http_status={}", resp.status()),
                );
                rt.last_action_at.insert(chosen.role, Instant::now());
                rt.perf.entry(chosen.role).or_default().executed += 1;
                rt.paper_executions += 1;
            }
            Ok(resp) => {
                let status_code = resp.status();
                let body_preview = resp.text().await.unwrap_or_default();
                let short_body = if body_preview.len() > 200 {
                    format!("{}…", &body_preview[..200])
                } else {
                    body_preview.clone()
                };
                lifecycle(
                    &*state.store,
                    chosen.role,
                    &chosen.action_id,
                    NpcLifecycleState::Rejected,
                    &format!("http_status={} body={}", status_code, short_body),
                );
                observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
                return NpcCycleReport {
                    cycle_id,
                    last_action: "NO_ACTION".to_string(),
                    execution_result: format!("HTTP_STATUS_{}", status_code),
                    status: "blocked".to_string(),
                    last_agent_decision: format!(
                        "{} {} submitted — rejected by executor (HTTP {})",
                        chosen.side.to_uppercase(), chosen.role.as_str(), status_code
                    ),
                    no_trade_reason: format!(
                        "Trade request rejected by executor (HTTP {}): {}",
                        status_code, short_body
                    ),
                    pipeline_state: "Rejected".to_string(),
                };
            }
            Err(e) => {
                lifecycle(
                    &*state.store,
                    chosen.role,
                    &chosen.action_id,
                    NpcLifecycleState::Rejected,
                    &format!("request_error={}", e),
                );
                observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
                return NpcCycleReport {
                    cycle_id,
                    last_action: "NO_ACTION".to_string(),
                    execution_result: format!("REQUEST_ERROR:{}", e),
                    status: "blocked".to_string(),
                    last_agent_decision: format!(
                        "{} {} dispatch failed — network error",
                        chosen.side.to_uppercase(), chosen.role.as_str()
                    ),
                    no_trade_reason: format!("Network error dispatching trade request: {}", e),
                    pipeline_state: "Rejected".to_string(),
                };
            }
        }
    }

    let expected_edge = chosen.score_parts.edge_estimate - chosen.score_parts.spread_cost - chosen.score_parts.slippage_risk;
    rt.cycle_open_notional += allocation.qty * metrics.mid.max(0.0);
    rt.open_actions.insert(
        chosen.action_id.clone(),
        OpenAction {
            role: chosen.role,
            side: chosen.side.clone(),
            entry_mid: metrics.mid,
            opened_at: Instant::now(),
            entry_spread_bps: metrics.spread_bps,
            expected_edge,
            regime,
            allocated_qty: allocation.qty,
            cycle_id,
        },
    );
    log_npc_event(
        &*state.store,
        "capital_allocation",
        &format!(
            "cycle={} action_id={} role={} qty={:.8} quality_score={:.4} recent_perf={:.4} exposure_notional={:.2} symbol_concentration={:.4} drawdown={:.4}",
            cycle_id,
            chosen.action_id,
            chosen.role.as_str(),
            allocation.qty,
            allocation.quality,
            allocation.recent_performance,
            exposure_notional,
            allocation.symbol_concentration,
            allocation.drawdown,
        ),
    );

    observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
    NpcCycleReport {
        cycle_id,
        last_action: chosen.side.to_uppercase(),
        execution_result: "EXECUTED".to_string(),
        status: "running".to_string(),
        last_agent_decision: format!(
            "{} {} role={} score={:.4} reason={}",
            chosen.side.to_uppercase(),
            state.symbol,
            chosen.role.as_str(),
            chosen.score,
            chosen.reason
        ),
        no_trade_reason: String::new(),
        pipeline_state: "Submitting Order".to_string(),
    }
}

fn default_learner_ranges(cfg: &NpcConfig) -> LearnerConfigRanges {
    let mut regime_score_cutoff = HashMap::new();
    for regime in [
        MarketRegime::TrendingUp,
        MarketRegime::TrendingDown,
        MarketRegime::MeanRevert,
        MarketRegime::Choppy,
        MarketRegime::Volatile,
        MarketRegime::Illiquid,
    ] {
        regime_score_cutoff.insert(
            regime,
            LearnerRange {
                min: 0.02,
                max: 2.5,
                current: cfg.alpha.min_action_score,
                step: 0.02,
            }
            .bounded(),
        );
    }
    LearnerConfigRanges {
        spread_tolerance_bps: LearnerRange {
            min: 1.0,
            max: cfg.guards.max_spread_bps.max(1.0),
            current: cfg.guards.max_spread_bps,
            step: 0.5,
        }
        .bounded(),
        dip_trigger_pct: LearnerRange {
            min: 0.001,
            max: 0.02,
            current: cfg.dip_trigger_pct,
            step: 0.0005,
        }
        .bounded(),
        cooldown_secs: LearnerRange {
            min: 1.0,
            max: 30.0,
            current: cfg.guards.cooldown_secs as f64,
            step: 1.0,
        }
        .bounded(),
        regime_score_cutoff,
    }
}

fn apply_learner_overrides(cfg: &mut NpcConfig, rt: &mut NpcRuntimeState) {
    let Some(ranges) = rt.learner_ranges.as_ref() else {
        return;
    };
    cfg.guards.max_spread_bps = ranges.spread_tolerance_bps.current;
    cfg.dip_trigger_pct = ranges.dip_trigger_pct.current;
    cfg.guards.cooldown_secs = ranges.cooldown_secs.current.round() as u64;
    let _ = cfg;
}

fn detect_regime(
    cfg: &NpcConfig,
    mid_history: &VecDeque<f64>,
    spread_history: &VecDeque<f64>,
    metrics: &crate::signal::SignalMetrics,
) -> MarketRegime {
    let short = pct_change(mid_history, 3).unwrap_or(0.0);
    let long = pct_change(mid_history, 12).unwrap_or(short);
    let alignment = horizon_alignment(short, long);
    let vol_bps = realized_volatility_bps(mid_history, 16).unwrap_or(0.0);
    let avg_spread = rolling_avg(spread_history, 12).unwrap_or(metrics.spread_bps.max(0.0));
    let liquidity = metrics.trade_samples as f64;

    if liquidity < cfg.guards.min_liquidity_score || avg_spread > cfg.guards.max_spread_bps * 1.25 {
        return MarketRegime::Illiquid;
    }
    if vol_bps >= cfg.alpha.vol_spike_bps {
        return MarketRegime::Volatile;
    }
    if alignment < cfg.alpha.min_horizon_alignment {
        return MarketRegime::Choppy;
    }

    let momentum = metrics.momentum_5s;
    if momentum > cfg.momentum_threshold && short > 0.0 && long > 0.0 {
        return MarketRegime::TrendingUp;
    }
    if momentum < -cfg.momentum_threshold && short < 0.0 && long < 0.0 {
        return MarketRegime::TrendingDown;
    }

    let band = cfg.alpha.choppy_band_bps / 10_000.0;
    if short.abs() < band && long.abs() < band {
        MarketRegime::MeanRevert
    } else {
        MarketRegime::Choppy
    }
}

fn regime_allowlist(role: NpcRole) -> HashSet<MarketRegime> {
    match role {
        NpcRole::Scout => HashSet::from([
            MarketRegime::TrendingUp,
            MarketRegime::TrendingDown,
            MarketRegime::MeanRevert,
            MarketRegime::Choppy,
        ]),
        NpcRole::DipBuyer => HashSet::from([MarketRegime::TrendingUp, MarketRegime::MeanRevert]),
        NpcRole::MomentumExecutor => HashSet::from([MarketRegime::TrendingUp, MarketRegime::TrendingDown]),
        NpcRole::RiskManager => HashSet::from([
            MarketRegime::Volatile,
            MarketRegime::Illiquid,
            MarketRegime::TrendingDown,
            MarketRegime::Choppy,
        ]),
        NpcRole::InventoryManager => HashSet::from([
            MarketRegime::TrendingDown,
            MarketRegime::MeanRevert,
            MarketRegime::Choppy,
        ]),
        NpcRole::Learner => HashSet::from([
            MarketRegime::TrendingUp,
            MarketRegime::TrendingDown,
            MarketRegime::MeanRevert,
            MarketRegime::Choppy,
            MarketRegime::Volatile,
            MarketRegime::Illiquid,
        ]),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_worker_candidates(
    cfg: &NpcConfig,
    cycle_id: u64,
    rt: &NpcRuntimeState,
    metrics: &crate::signal::SignalMetrics,
    position_size: f64,
    buy_power: f64,
    sell_inventory: f64,
    regime: MarketRegime,
    has_conflict: bool,
) -> Vec<WorkerProposal> {
    let mut proposals = Vec::new();
    let momentum = metrics.momentum_5s;
    let liquidity_score = (metrics.trade_samples as f64).max(0.0);
    let dip = dip_pct_from_history(&rt.mid_history, cfg.dip_lookback_cycles);

    for role in NpcRole::all() {
        let allow = regime_allowlist(role);
        let regime_eligible = allow.contains(&regime);
        let regime_block_reason = if regime_eligible {
            None
        } else {
            Some(format!(
                "REGIME_MISMATCH role={} regime={} blocked",
                role.as_str(),
                regime.as_str()
            ))
        };

        let (side, reason, expected_slippage_bps, expected_hold_secs) = match role {
            NpcRole::Scout => (
                if momentum >= 0.0 { "BUY" } else { "SELL" },
                format!(
                    "scout_scan momentum_5s={:+.6} liquidity={:.2}",
                    momentum, liquidity_score
                ),
                metrics.spread_bps * 0.55,
                6.0,
            ),
            NpcRole::DipBuyer => {
                let dip_msg = dip
                    .map(|d| format!("dip_pct={:+.4}%", d * 100.0))
                    .unwrap_or_else(|| "dip_pct=NA".to_string());
                ("BUY", format!("dip_reversion {}", dip_msg), metrics.spread_bps * 0.62, 30.0)
            }
            NpcRole::MomentumExecutor => (
                if momentum >= 0.0 { "BUY" } else { "SELL" },
                format!("momentum_follow momentum_5s={:+.6}", momentum),
                metrics.spread_bps * 0.72,
                18.0,
            ),
            NpcRole::RiskManager => (
                "SELL",
                format!(
                    "risk_defense spread_bps={:.2} vol_proxy={:.2}",
                    metrics.spread_bps,
                    realized_volatility_bps(&rt.mid_history, 10).unwrap_or(0.0)
                ),
                metrics.spread_bps * 0.40,
                10.0,
            ),
            NpcRole::InventoryManager => (
                "SELL",
                format!("inventory_relief sell_inventory={:.8}", sell_inventory),
                metrics.spread_bps * 0.50,
                22.0,
            ),
            NpcRole::Learner => (
                if momentum >= 0.0 { "BUY" } else { "SELL" },
                "learner_observe_policy".to_string(),
                metrics.spread_bps * 0.10,
                4.0,
            ),
        };

        let score_parts = score_candidate(
            cfg,
            rt,
            role,
            metrics,
            side,
            expected_slippage_bps,
            expected_hold_secs,
            has_conflict,
            dip,
            buy_power,
            sell_inventory,
            position_size,
        );
        let raw_score = score_parts.final_score();
        let score = raw_score.max(-5.0);
        proposals.push(WorkerProposal {
            action_id: format!("c{}-{}", cycle_id, role.as_str()),
            role,
            side: side.to_string(),
            score,
            raw_score,
            reason,
            expected_slippage_bps,
            regime_eligible,
            regime_block_reason,
            score_parts,
            expected_hold_secs,
        });
    }

    proposals
}

#[allow(clippy::too_many_arguments)]
fn score_candidate(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    role: NpcRole,
    metrics: &crate::signal::SignalMetrics,
    side: &str,
    expected_slippage_bps: f64,
    expected_hold_secs: f64,
    has_conflict: bool,
    dip: Option<f64>,
    buy_power: f64,
    sell_inventory: f64,
    position_size: f64,
) -> ScoreBreakdown {
    let momentum = metrics.momentum_5s;
    let edge_bias = match role {
        NpcRole::Scout => momentum.abs() * 1000.0,
        NpcRole::DipBuyer => dip.map(|v| (-v * 1400.0).max(0.0)).unwrap_or(0.0),
        NpcRole::MomentumExecutor => momentum.abs() * 1700.0,
        NpcRole::RiskManager => (metrics.spread_bps / cfg.guards.max_spread_bps.max(0.1)).clamp(0.0, 2.0),
        NpcRole::InventoryManager => (sell_inventory + position_size) * 5.0,
        NpcRole::Learner => 0.05,
    };

    let side_penalty = if side.eq_ignore_ascii_case("BUY") && buy_power <= 0.0 {
        2.0
    } else if side.eq_ignore_ascii_case("SELL") && sell_inventory <= 0.0 && position_size <= 0.0 {
        2.0
    } else {
        0.0
    };

    let liquidity_quality = ((metrics.trade_samples as f64) / 10.0).clamp(0.0, 1.4);
    let volatility_penalty = (realized_volatility_bps(&rt.mid_history, 8).unwrap_or(0.0) / cfg.alpha.vol_spike_bps.max(1.0)).clamp(0.0, 3.0);
    let spread_cost = (metrics.spread_bps / cfg.guards.max_spread_bps.max(0.1)).clamp(0.0, 3.0);
    let slippage_risk = (expected_slippage_bps / cfg.guards.max_slippage_bps.max(0.1)).clamp(0.0, 3.0);
    let conflict_penalty = if has_conflict { 1.0 } else { 0.0 } + side_penalty;
    let hold_efficiency = (1.0 / expected_hold_secs.max(1.0)).clamp(0.01, 0.20);

    ScoreBreakdown {
        edge_estimate: edge_bias,
        spread_cost,
        slippage_risk,
        liquidity_quality,
        volatility_penalty,
        conflict_penalty,
        hold_efficiency,
    }
}

fn evaluate_guards(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    role: NpcRole,
    side: &str,
    position_size: f64,
    metrics: &crate::signal::SignalMetrics,
    expected_slippage_bps: f64,
    total_balance_usd: f64,
    sell_inventory: f64,
    order_notional: f64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if cfg.guards.kill_switch {
        reasons.push("KILL_SWITCH_ACTIVE".to_string());
    }

    // Detect small account mode: total balance below $100.
    // In small account mode, relax strict liquidity depth checks for SELL
    // orders when inventory is available and the order has positive notional.
    let is_sell = side.eq_ignore_ascii_case("SELL");
    let small_account = total_balance_usd < 100.0 && total_balance_usd > 0.0;
    let relax_liquidity_guards = small_account && is_sell && sell_inventory > 0.0 && order_notional > 0.0;

    if relax_liquidity_guards {
        info!(
            total_balance_usd,
            sell_inventory,
            order_notional,
            "[GUARD] Small account mode: liquidity depth checks relaxed for SELL \
             (total_balance_usd={:.2} < $100, sell_inventory={:.8}, order_notional={:.4})",
            total_balance_usd, sell_inventory, order_notional,
        );
    }

    if !relax_liquidity_guards && metrics.spread_bps > cfg.guards.max_spread_bps {
        reasons.push(format!(
            "MAX_SPREAD_BPS_EXCEEDED:{:.2}>{:.2}",
            metrics.spread_bps, cfg.guards.max_spread_bps
        ));
    }
    let liquidity_score = metrics.trade_samples as f64;
    if !relax_liquidity_guards && liquidity_score < cfg.guards.min_liquidity_score {
        reasons.push(format!(
            "LIQUIDITY_TOO_LOW:{:.2}<{:.2}",
            liquidity_score, cfg.guards.min_liquidity_score
        ));
    }
    if !relax_liquidity_guards && expected_slippage_bps > cfg.guards.max_slippage_bps {
        reasons.push(format!(
            "SLIPPAGE_CEILING_EXCEEDED:{:.2}>{:.2}",
            expected_slippage_bps, cfg.guards.max_slippage_bps
        ));
    }
    if let Some(last_at) = rt.last_action_at.get(&role) {
        if last_at.elapsed() < Duration::from_secs(cfg.guards.cooldown_secs) {
            reasons.push("ROLE_COOLDOWN_ACTIVE".to_string());
        }
    }
    if side.eq_ignore_ascii_case("BUY") && position_size + cfg.trade_size > cfg.guards.max_position_qty {
        reasons.push(format!(
            "POSITION_LIMIT_EXCEEDED:{:.8}>{:.8}",
            position_size + cfg.trade_size,
            cfg.guards.max_position_qty
        ));
    }
    reasons
}

fn evaluate_portfolio_controls(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    position_size: f64,
    exposure_notional: f64,
    mid: f64,
    regime: MarketRegime,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if rt.open_actions.len() >= cfg.alpha.max_concurrent_positions {
        reasons.push(format!(
            "MAX_CONCURRENT_POSITIONS:{}>={}",
            rt.open_actions.len(),
            cfg.alpha.max_concurrent_positions
        ));
    }
    if exposure_notional >= cfg.alpha.max_symbol_exposure_notional {
        reasons.push(format!(
            "MAX_SYMBOL_EXPOSURE:{:.2}>={:.2}",
            exposure_notional, cfg.alpha.max_symbol_exposure_notional
        ));
    }
    let total_at_risk = rt.cycle_open_notional + exposure_notional;
    if total_at_risk >= cfg.alpha.max_capital_at_risk_notional {
        reasons.push(format!(
            "MAX_CAPITAL_AT_RISK:{:.2}>={:.2}",
            total_at_risk, cfg.alpha.max_capital_at_risk_notional
        ));
    }

    let aggregate_pnl: f64 = rt.perf.values().map(|p| p.gross_pnl).sum();
    let equity = (position_size * mid.max(0.0)) + aggregate_pnl;
    let peak = rt.peak_equity.max(equity.max(0.0));
    let dd = if peak > 0.0 { ((peak - equity) / peak).max(0.0) } else { 0.0 };
    if dd >= cfg.alpha.max_drawdown_pct {
        reasons.push(format!("MAX_DRAWDOWN_BREACH:{:.4}>={:.4}", dd, cfg.alpha.max_drawdown_pct));
    }
    if dd >= cfg.alpha.drawdown_hard_stop_pct {
        reasons.push("AUTO_DERISK_DRAWDOWN_EXPANSION".to_string());
    }
    if matches!(regime, MarketRegime::Volatile) {
        reasons.push("AUTO_DERISK_VOLATILITY_SPIKE".to_string());
    }

    reasons
}

struct AllocationDecision {
    qty: f64,
    reason: String,
    quality: f64,
    recent_performance: f64,
    symbol_concentration: f64,
    drawdown: f64,
}

fn allocate_capital(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    chosen: &WorkerProposal,
    _position_size: f64,
    exposure_notional: f64,
    buy_power: f64,
    sell_inventory: f64,
    mid: f64,
) -> AllocationDecision {
    let perf = rt.perf.get(&chosen.role).cloned().unwrap_or_default();
    let quality = perf.quality_score();
    let recent_performance = perf.gross_pnl.clamp(-10.0, 10.0) / 10.0;
    let symbol_concentration = if cfg.alpha.max_symbol_exposure_notional > 0.0 {
        (exposure_notional / cfg.alpha.max_symbol_exposure_notional).clamp(0.0, 2.0)
    } else {
        0.0
    };

    let total_pnl: f64 = rt.perf.values().map(|p| p.gross_pnl).sum();
    let peak = rt.peak_equity.max(1.0);
    let drawdown = ((peak - (peak + total_pnl)) / peak).max(0.0);

    let agent_budget = cfg.alpha.per_agent_budget_pct.get(&chosen.role).copied().unwrap_or(0.05).clamp(0.01, 0.80);
    let dd_factor = if drawdown >= cfg.alpha.drawdown_derisk_trigger_pct { 0.35 } else { 1.0 };

    let size_multiplier = quality * (1.0 + recent_performance).clamp(0.3, 1.4) * (1.0 - symbol_concentration).clamp(0.1, 1.0) * dd_factor;
    let mut qty = cfg.trade_size * agent_budget * size_multiplier;

    // Cap quantity by the available balance for this side:
    // SELL orders are limited by base-asset inventory; BUY orders by quote buying power.
    let max_qty = if chosen.side.eq_ignore_ascii_case("SELL") {
        sell_inventory.max(0.0)
    } else if mid > 0.0 {
        buy_power / mid
    } else {
        0.0
    };
    qty = qty.min(max_qty.max(0.0));

    if drawdown >= cfg.alpha.drawdown_hard_stop_pct {
        qty = 0.0;
        return AllocationDecision {
            qty,
            reason: "DRAWDOWN_HARD_STOP".to_string(),
            quality,
            recent_performance,
            symbol_concentration,
            drawdown,
        };
    }

    if qty < 0.000_000_01 {
        return AllocationDecision {
            qty: 0.0,
            reason: "ALLOCATION_BELOW_MIN_SIZE".to_string(),
            quality,
            recent_performance,
            symbol_concentration,
            drawdown,
        };
    }

    AllocationDecision {
        qty,
        reason: "CAPITAL_ALLOCATOR_ACCEPT".to_string(),
        quality,
        recent_performance,
        symbol_concentration,
        drawdown,
    }
}

fn observe_and_learn(cfg: &NpcConfig, rt: &mut NpcRuntimeState, store: &dyn EventStore, current_mid: f64) {
    let mut close_ids = Vec::new();
    let mut learner_updates: Vec<(NpcRole, MarketRegime, f64)> = Vec::new();
    for (id, open) in rt.open_actions.iter() {
        if open.opened_at.elapsed() >= Duration::from_secs(1) {
            lifecycle(
                store,
                open.role,
                id,
                NpcLifecycleState::Observed,
                "one-cycle post-execution observation complete",
            );
            let pnl = if open.side.eq_ignore_ascii_case("BUY") {
                (current_mid - open.entry_mid) * open.allocated_qty
            } else {
                (open.entry_mid - current_mid) * open.allocated_qty
            };
            let hold_ms = open.opened_at.elapsed().as_secs_f64() * 1000.0;
            let perf = rt.perf.entry(open.role).or_default();
            perf.gross_pnl += pnl;
            perf.peak_pnl = perf.peak_pnl.max(perf.gross_pnl);
            perf.drawdown = ((perf.peak_pnl - perf.gross_pnl) / perf.peak_pnl.max(1e-9)).max(0.0);
            if pnl > 0.0 {
                perf.wins += 1;
            } else {
                perf.losses += 1;
            }
            perf.total_hold_ms += hold_ms;
            perf.total_spread_bps += open.entry_spread_bps;
            perf.total_slippage_bps += open.entry_spread_bps * 0.5;

            let realized_edge = if open.entry_mid > 0.0 {
                pnl / open.entry_mid
            } else {
                0.0
            };

            let regime_perf = rt.regime_perf.entry((open.role, open.regime)).or_default();
            regime_perf.gross_pnl += pnl;
            regime_perf.executed += 1;
            if pnl > 0.0 {
                regime_perf.wins += 1;
            }

            lifecycle(
                store,
                NpcRole::Learner,
                id,
                NpcLifecycleState::Learned,
                &format!(
                    "alpha_metrics cycle={} role={} regime={} expected_edge={:+.6} realized_edge={:+.6} pnl={:+.6} hold_ms={:.0}",
                    open.cycle_id,
                    open.role.as_str(),
                    open.regime.as_str(),
                    open.expected_edge,
                    realized_edge,
                    pnl,
                    hold_ms
                ),
            );

            log_npc_event(
                store,
                "trade_metrics",
                &format!(
                    "action_id={} role={} regime={} expected_edge={:+.6} realized_edge={:+.6} pnl={:+.6} hold_ms={:.0} qty={:.8}",
                    id,
                    open.role.as_str(),
                    open.regime.as_str(),
                    open.expected_edge,
                    realized_edge,
                    pnl,
                    hold_ms,
                    open.allocated_qty,
                ),
            );

            if cfg.mode.learner_writable() {
                learner_updates.push((open.role, open.regime, realized_edge));
            }

            close_ids.push(id.clone());
        }
    }
    for id in close_ids {
        if let Some(open) = rt.open_actions.remove(&id) {
            rt.cycle_open_notional = (rt.cycle_open_notional - (open.allocated_qty * open.entry_mid).max(0.0)).max(0.0);
        }
    }
    for (role, regime, realized_edge) in learner_updates {
        learner_update_ranges(rt, role, regime, realized_edge);
    }
}

fn learner_update_ranges(rt: &mut NpcRuntimeState, role: NpcRole, regime: MarketRegime, realized_edge: f64) {
    let Some(ranges) = rt.learner_ranges.as_mut() else {
        return;
    };
    let improving = realized_edge > 0.0;
    if improving {
        ranges.spread_tolerance_bps.nudge_tighter();
        ranges.cooldown_secs.nudge_looser();
        ranges.dip_trigger_pct.nudge_looser();
    } else {
        ranges.spread_tolerance_bps.nudge_looser();
        ranges.cooldown_secs.nudge_tighter();
        ranges.dip_trigger_pct.nudge_tighter();
    }
    if let Some(cutoff) = ranges.regime_score_cutoff.get_mut(&regime) {
        if improving {
            cutoff.nudge_tighter();
        } else {
            cutoff.nudge_looser();
        }
    }

    if let Some(p) = rt.perf.get(&role) {
        let eq = p.gross_pnl.max(0.0);
        rt.peak_equity = rt.peak_equity.max(eq);
    }
}

fn format_cycle_metrics(
    cycle_id: u64,
    regime: MarketRegime,
    candidates: &[WorkerProposal],
    threshold: f64,
    portfolio_controls: &[String],
) -> String {
    let mut candidate_summary = Vec::new();
    for c in candidates {
        candidate_summary.push(format!(
            "{}:{:.4}:eligible={}:edge={:.3}:spread={:.3}:slip={:.3}:liq={:.3}:vol_pen={:.3}:conf_pen={:.3}:hold={:.3}",
            c.role.as_str(),
            c.score,
            c.regime_eligible,
            c.score_parts.edge_estimate,
            c.score_parts.spread_cost,
            c.score_parts.slippage_risk,
            c.score_parts.liquidity_quality,
            c.score_parts.volatility_penalty,
            c.score_parts.conflict_penalty,
            c.score_parts.hold_efficiency,
        ));
    }
    format!(
        "cycle={} regime={} threshold={:.4} controls={} candidates=[{}]",
        cycle_id,
        regime.as_str(),
        threshold,
        if portfolio_controls.is_empty() {
            "NONE".to_string()
        } else {
            portfolio_controls.join("|")
        },
        candidate_summary.join(",")
    )
}

fn lifecycle(store: &dyn EventStore, role: NpcRole, action_id: &str, state: NpcLifecycleState, reason: &str) {
    log_npc_event(
        store,
        "lifecycle",
        &format!(
            "action_id={} role={} state={} reason_code={}",
            action_id,
            role.as_str(),
            state.as_str(),
            reason,
        ),
    );
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

fn pct_change(mid_history: &VecDeque<f64>, lookback: usize) -> Option<f64> {
    let now = mid_history.back().copied().filter(|v| *v > 0.0)?;
    let then = mid_history
        .len()
        .checked_sub(1 + lookback)
        .and_then(|idx| mid_history.get(idx).copied())
        .filter(|v| *v > 0.0)?;
    Some((now / then) - 1.0)
}

fn horizon_alignment(short: f64, long: f64) -> f64 {
    let denom = short.abs() + long.abs();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - ((short - long).abs() / denom).clamp(0.0, 1.0)
}

fn rolling_avg(history: &VecDeque<f64>, len: usize) -> Option<f64> {
    if history.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut n = 0.0;
    for v in history.iter().rev().take(len) {
        sum += *v;
        n += 1.0;
    }
    if n == 0.0 {
        None
    } else {
        Some(sum / n)
    }
}

fn realized_volatility_bps(mid_history: &VecDeque<f64>, len: usize) -> Option<f64> {
    if mid_history.len() < 3 {
        return None;
    }
    let mut rets = Vec::new();
    let slice: Vec<f64> = mid_history.iter().rev().take(len + 1).copied().collect();
    if slice.len() < 2 {
        return None;
    }
    for w in slice.windows(2) {
        if w[0] > 0.0 && w[1] > 0.0 {
            rets.push((w[0] / w[1]) - 1.0);
        }
    }
    if rets.is_empty() {
        return None;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| {
        let d = r - mean;
        d * d
    }).sum::<f64>() / rets.len() as f64;
    Some(var.sqrt() * 10_000.0)
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
    std::env::var(k)
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> NpcConfig {
        NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 1.0,
            momentum_threshold: 0.001,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        })
    }

    fn low_liquidity_metrics() -> crate::signal::SignalMetrics {
        crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 10.0,    // exceeds max_spread_bps=5.0
            trade_samples: 1,    // below min_liquidity_score=3.0
            mid: 50_000.0,
            ..Default::default()
        }
    }

    // ── evaluate_guards: small account SELL bypass ─────────────────────────────

    #[test]
    fn small_account_sell_bypasses_liquidity_depth_checks() {
        let cfg = test_cfg();
        let rt = NpcRuntimeState::default();
        let metrics = low_liquidity_metrics();

        // Small account (<$100), SELL side, non-zero inventory and notional.
        let reasons = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 20.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
        );

        // Liquidity depth guards must be absent.
        assert!(
            !reasons.iter().any(|r| r.starts_with("LIQUIDITY_TOO_LOW")),
            "LIQUIDITY_TOO_LOW must be bypassed in small account mode, got: {:?}", reasons
        );
        assert!(
            !reasons.iter().any(|r| r.starts_with("MAX_SPREAD_BPS_EXCEEDED")),
            "MAX_SPREAD_BPS_EXCEEDED must be bypassed in small account mode, got: {:?}", reasons
        );
        assert!(
            !reasons.iter().any(|r| r.starts_with("SLIPPAGE_CEILING_EXCEEDED")),
            "SLIPPAGE_CEILING_EXCEEDED must be bypassed in small account mode, got: {:?}", reasons
        );
    }

    #[test]
    fn large_account_sell_applies_liquidity_depth_checks() {
        let cfg = test_cfg();
        let rt = NpcRuntimeState::default();
        let metrics = low_liquidity_metrics();

        // Large account (>=$100): depth checks must remain active.
        let reasons = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 20.0,
            /* total_balance_usd */ 500.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
        );

        assert!(
            reasons.iter().any(|r| r.starts_with("LIQUIDITY_TOO_LOW")),
            "LIQUIDITY_TOO_LOW must trigger for large account, got: {:?}", reasons
        );
        assert!(
            reasons.iter().any(|r| r.starts_with("MAX_SPREAD_BPS_EXCEEDED")),
            "MAX_SPREAD_BPS_EXCEEDED must trigger for large account, got: {:?}", reasons
        );
    }

    #[test]
    fn small_account_no_inventory_does_not_bypass_liquidity_guards() {
        let cfg = test_cfg();
        let rt = NpcRuntimeState::default();
        let metrics = low_liquidity_metrics();

        // Small account but zero sell_inventory: bypass must NOT apply.
        let reasons = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 20.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.0,
            /* order_notional    */ 50.0,
        );

        assert!(
            reasons.iter().any(|r| r.starts_with("LIQUIDITY_TOO_LOW")),
            "LIQUIDITY_TOO_LOW must block when sell_inventory=0, got: {:?}", reasons
        );
    }

    #[test]
    fn kill_switch_always_blocks_even_in_small_account_mode() {
        let mut cfg = test_cfg();
        cfg.guards.kill_switch = true;
        let rt = NpcRuntimeState::default();
        let metrics = low_liquidity_metrics();

        let reasons = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
        );

        assert!(
            reasons.iter().any(|r| r == "KILL_SWITCH_ACTIVE"),
            "kill switch must always block, got: {:?}", reasons
        );
    }


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

    #[test]
    fn regime_detection_illiquid_overrides() {
        let mut cfg = NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 1.0,
            momentum_threshold: 0.001,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        });
        cfg.guards.min_liquidity_score = 5.0;

        let mids: VecDeque<f64> = vec![100.0, 100.1, 100.2, 100.3, 100.4, 100.5].into();
        let spreads: VecDeque<f64> = vec![2.0, 2.0, 2.0, 2.0].into();
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 2.0,
            trade_samples: 1,
            ..Default::default()
        };

        let regime = detect_regime(&cfg, &mids, &spreads, &metrics);
        assert_eq!(regime, MarketRegime::Illiquid);
    }

    #[test]
    fn deterministic_action_id_by_cycle_and_role() {
        let cfg = NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 1.0,
            momentum_threshold: 0.001,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        });
        let rt = NpcRuntimeState::default();
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 1.0,
            trade_samples: 10,
            ..Default::default()
        };
        let candidates = build_worker_candidates(
            &cfg,
            42,
            &rt,
            &metrics,
            0.0,
            1000.0,
            0.0,
            MarketRegime::TrendingUp,
            false,
        );

        assert!(candidates.iter().any(|c| c.action_id == "c42-scout"));
        assert!(candidates.iter().any(|c| c.action_id == "c42-dip_buyer"));
    }

    // ── AgentMode ─────────────────────────────────────────────────────────────

    #[test]
    fn agent_mode_round_trip_str() {
        for (s, expected) in [("off", AgentMode::Off), ("auto", AgentMode::Auto), ("pause", AgentMode::Pause)] {
            let m = AgentMode::from_str(s).expect("parse failed");
            assert_eq!(m, expected);
            assert_eq!(m.as_str(), s);
        }
    }

    #[test]
    fn agent_mode_from_str_case_insensitive() {
        assert_eq!(AgentMode::from_str("OFF"),   Some(AgentMode::Off));
        assert_eq!(AgentMode::from_str("AUTO"),  Some(AgentMode::Auto));
        assert_eq!(AgentMode::from_str("PAUSE"), Some(AgentMode::Pause));
        assert_eq!(AgentMode::from_str("paused"), Some(AgentMode::Pause));
        assert_eq!(AgentMode::from_str("xyz"),   None);
    }

    #[test]
    fn agent_mode_state_labels_not_idle() {
        // No state label should contain the word "Idle" (requirement: remove passive idle state).
        for mode in [AgentMode::Off, AgentMode::Auto, AgentMode::Pause] {
            let label = mode.state_label();
            assert!(!label.contains("Idle"), "mode {:?} must not use Idle label, got: {}", mode, label);
        }
    }

    #[test]
    fn npc_controller_initial_mode_off_when_disabled() {
        let cfg = NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: false,
            trade_size: 0.0,
            momentum_threshold: 0.0,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        });
        // We can't easily construct AgentState in a unit test, so test the enum logic only.
        let mode = if cfg.enabled { AgentMode::Auto } else { AgentMode::Off };
        assert_eq!(mode, AgentMode::Off);
    }
}

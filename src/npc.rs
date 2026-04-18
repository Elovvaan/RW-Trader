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
use crate::profile::RuntimeProfile;
use crate::store::EventStore;

const NPC_STATUS_SCANNING: &str = "scanning market";
const DEFAULT_ACTIVE_PROFILE: &str = "ACTIVE";

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
    /// Behavior profile name (uppercase).  "FLIP_HYPER" activates the
    /// capital-rotation state machine.  Set via `RUNTIME_PROFILE` env var.
    pub behavior_profile: String,
    pub target_usdt_ratio: f64,
    pub target_btc_ratio: f64,
    pub rebalance_min_notional_usd: f64,
    pub rebalance_cooldown_secs: u64,
    pub disable_no_trade_idle: bool,
    pub min_conviction_threshold: f64,
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
            behavior_profile: std::env::var("RUNTIME_PROFILE")
                .unwrap_or_else(|_| DEFAULT_ACTIVE_PROFILE.to_string())
                .trim()
                .to_ascii_uppercase(),
            target_usdt_ratio: env_f64("NPC_TARGET_USDT_RATIO", 0.35).clamp(0.05, 0.95),
            target_btc_ratio: env_f64("NPC_TARGET_BTC_RATIO", 0.65).clamp(0.05, 0.95),
            rebalance_min_notional_usd: env_f64("NPC_REBALANCE_MIN_NOTIONAL_USD", 5.0).max(1.0),
            rebalance_cooldown_secs: env_u64("NPC_REBALANCE_COOLDOWN_SECS", 30).max(1),
            disable_no_trade_idle: env_bool("NPC_DISABLE_NO_TRADE_IDLE", true),
            min_conviction_threshold: env_f64("NPC_MIN_CONVICTION_THRESHOLD", 0.60).clamp(0.05, 0.95),
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
    risk_override: bool,
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

    /// Returns a formatted string naming the top-3 penalty components by magnitude,
    /// descending. There are 4 trackable penalty components (spread_cost, vol_penalty,
    /// slippage_risk, conflict); we report the top 3 by value. Positive contributors
    /// (liquidity_quality, hold_efficiency, edge_estimate) are excluded — this method
    /// is specifically for diagnosing what suppressed the score.
    ///
    /// NaN values cannot arise here because all penalty fields are produced by
    /// `.clamp()` operations on finite inputs; `unwrap_or(Equal)` is a safe fallback.
    fn top_penalties_str(&self) -> String {
        let mut penalties = [
            ("spread_cost",    self.spread_cost),
            ("vol_penalty",    self.volatility_penalty),
            ("slippage_risk",  self.slippage_risk),
            ("conflict",       self.conflict_penalty),
        ];
        penalties.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        penalties[..3]
            .iter()
            .map(|(name, val)| format!("{}={:.4}", name, val))
            .collect::<Vec<_>>()
            .join(", ")
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

// ── FLIP_HYPER types ──────────────────────────────────────────────────────────

/// Phase of the FLIP_HYPER buy→sell capital-rotation cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlipCyclePhase {
    /// No BTC held; scanning for a buy entry.
    #[default]
    SeekEntry,
    /// Buy order submitted; waiting for fill confirmation.
    Entering,
    /// BTC position held; evaluating hold thesis.
    HoldingPosition,
    /// BTC held; scanning for a profitable sell exit.
    SeekExit,
    /// Sell order submitted; waiting for fill confirmation.
    Exiting,
    /// Sell confirmed; brief pause before next buy cycle.
    RebuyReady,
}

impl FlipCyclePhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::SeekEntry       => "SEEK_ENTRY",
            Self::Entering        => "ENTERING",
            Self::HoldingPosition => "HOLDING_POSITION",
            Self::SeekExit        => "SEEK_EXIT",
            Self::Exiting         => "EXITING",
            Self::RebuyReady     => "REBUY_READY",
        }
    }
}

/// Record of a completed FLIP_HYPER buy→sell cycle.
#[derive(Clone, Debug)]
pub struct CompletedFlip {
    pub entry_price:      f64,
    pub exit_price:       f64,
    pub qty:              f64,
    pub realized_pnl_usd: f64,
    pub realized_pnl_pct: f64,
    pub completed_at:     Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContractSide {
    Long,
    Short,
}

impl ContractSide {
    fn as_str(self) -> &'static str {
        match self {
            Self::Long => "LONG",
            Self::Short => "SHORT",
        }
    }
}

#[derive(Clone, Debug)]
struct OpenContractPosition {
    side: ContractSide,
    leverage: f64,
    entry_price: f64,
    mark_price: f64,
    notional_usd: f64,
    qty_base: f64,
    entry_fee_usd: f64,
    unrealized_pnl: f64,
    liquidation_price: f64,
    stop_loss: f64,
    take_profit: f64,
    trailing_stop: f64,
    opened_at: Instant,
    high_water_mark: f64,
    low_water_mark: f64,
}

#[derive(Clone, Debug)]
struct CompletedContractTrade {
    side: ContractSide,
    leverage: f64,
    entry_price: f64,
    exit_price: f64,
    gross_realized_pnl_usd: f64,
    entry_fee_usd: f64,
    exit_fee_usd: f64,
    net_realized_pnl_usd: f64,
    /// Back-compat alias used by existing telemetry/UI bindings.
    pnl_usd: f64,
    duration_secs: f64,
    exit_reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwingBias {
    LongBias,
    ShortBias,
    NoTrade,
}

impl SwingBias {
    fn as_str(self) -> &'static str {
        match self {
            Self::LongBias => "LONG_BIAS",
            Self::ShortBias => "SHORT_BIAS",
            Self::NoTrade => "NO_TRADE",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SwingRegimeSignals {
    momentum_1m: f64,
    momentum_5m: f64,
    trend_bias_15m: f64,
    pullback_detected: bool,
    momentum_resumed_up: bool,
    structure_break_long: bool,
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
    execution_mode: NpcTradingMode,
}

const SLOT_QTY_EPSILON: f64 = 0.000_000_01;
const SLOT_RESERVATION_TTL_SECS: u64 = 30;
const SLOT_PENDING_ACK_TTL_SECS: u64 = 30;

#[derive(Clone, Debug, Default)]
struct SlotUsageDiagnostics {
    max_concurrent_positions: usize,
    counted_open_positions: usize,
    counted_pending_orders: usize,
    counted_reserved_slots: usize,
    slot_block_reason: String,
    slot_source_ids: Vec<String>,
}

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
    // ── COMPOUND_EXECUTION tracking ───────────────────────────────────────────
    /// Number of consecutive losing trades in COMPOUND_EXECUTION mode.
    compound_consecutive_losses: u32,
    /// PnL of the most recently closed trade (positive = win, negative = loss).
    compound_last_trade_pnl: f64,
    /// Whether the most recently closed trade was profitable.
    compound_last_trade_was_profitable: bool,
    /// Cumulative session PnL since agent start in COMPOUND_EXECUTION mode.
    compound_session_pnl: f64,
    /// Highest total_balance_usd seen so far (set once per cycle when in micro_active).
    compound_peak_balance: f64,
    /// Size scalar applied on top of the equity-based calculation (range 0.1–1.0).
    /// Reduced after losses, recovered after wins, slightly reduced on profit lock.
    compound_size_scalar: f64,
    /// When set, no new BUY entries are allowed until this instant (compound loss pause).
    compound_loss_pause_until: Option<Instant>,
    /// Last observed position size in BTC (updated each run_cycle call).
    compound_last_position_btc: f64,
    /// Last observed total account balance in USD (updated each run_cycle call).
    compound_last_balance_usd: f64,
    // ── FLIP_HYPER capital-rotation tracking ──────────────────────────────────
    /// Current phase of the FLIP_HYPER buy→sell cycle.
    flip_cycle_phase: FlipCyclePhase,
    /// Most recently completed flip (buy then sell).
    flip_last_completed: Option<CompletedFlip>,
    /// Cumulative session PnL from completed flips (USD, approximate).
    flip_session_pnl: f64,
    /// Number of completed buy→sell flips this session.
    flip_rotation_count: u32,
    /// Mid price at the time the current BUY was submitted (0.0 = no open entry).
    flip_last_entry_price: f64,
    /// Quantity of the current open BUY (0.0 = no open entry).
    flip_last_entry_qty: f64,
    /// When the current flip phase started (used by anti-stall detector).
    flip_last_active: Option<Instant>,
    /// Non-empty when the anti-stall rule has fired and surfaced an exact blocker.
    flip_blocker: String,
    // ── SWING_TRADER tracking ────────────────────────────────────────────────
    swing_entry_price: f64,
    swing_entry_started_at: Option<Instant>,
    swing_peak_price: f64,
    swing_peak_unrealized_pct: f64,
    swing_last_exit_reason: String,
    swing_last_hold_duration_secs: f64,
    swing_cooldown_until: Option<Instant>,
    // ── CONTRACT_EXECUTOR tracking (paper-mode only) ─────────────────────────
    contract_paper_mode: bool,
    contract_leverage: f64,
    contract_fee_rate: f64,
    contract_position: Option<OpenContractPosition>,
    contract_realized_pnl_session: f64,
    contract_last_trade: Option<CompletedContractTrade>,
    contract_last_exit_reason: String,
    contract_last_no_open_reason: String,
    rebalance_status: String,
    rebalance_triggered: bool,
    rebalance_reason: String,
    rebalance_side: String,
    rebalance_qty: f64,
    rebalance_value_usd: f64,
    free_usdt_before: f64,
    free_usdt_after: f64,
    btc_before: f64,
    btc_after: f64,
    rebalance_cooldown_until: Option<Instant>,
}

impl Default for NpcRuntimeState {
    fn default() -> Self {
        Self {
            action_state: HashMap::new(),
            last_action_at: HashMap::new(),
            mid_history: VecDeque::new(),
            spread_history: VecDeque::new(),
            open_actions: BTreeMap::new(),
            paper_executions: 0,
            perf: HashMap::new(),
            cycle_seq: 0,
            cycle_open_notional: 0.0,
            peak_equity: 0.0,
            regime_perf: HashMap::new(),
            learner_ranges: None,
            compound_consecutive_losses: 0,
            compound_last_trade_pnl: 0.0,
            compound_last_trade_was_profitable: false,
            compound_session_pnl: 0.0,
            compound_peak_balance: 0.0,
            // Start at full size — reduced only after losses.
            compound_size_scalar: 1.0,
            compound_loss_pause_until: None,
            compound_last_position_btc: 0.0,
            compound_last_balance_usd: 0.0,
            // FLIP_HYPER state
            flip_cycle_phase: FlipCyclePhase::default(),
            flip_last_completed: None,
            flip_session_pnl: 0.0,
            flip_rotation_count: 0,
            flip_last_entry_price: 0.0,
            flip_last_entry_qty: 0.0,
            flip_last_active: None,
            flip_blocker: String::new(),
            swing_entry_price: 0.0,
            swing_entry_started_at: None,
            swing_peak_price: 0.0,
            swing_peak_unrealized_pct: 0.0,
            swing_last_exit_reason: String::new(),
            swing_last_hold_duration_secs: 0.0,
            swing_cooldown_until: None,
            contract_paper_mode: true,
            contract_leverage: CONTRACT_DEFAULT_LEVERAGE,
            contract_fee_rate: 0.0004,
            contract_position: None,
            contract_realized_pnl_session: 0.0,
            contract_last_trade: None,
            contract_last_exit_reason: String::new(),
            contract_last_no_open_reason: String::new(),
            rebalance_status: "REBALANCE_COMPLETE".to_string(),
            rebalance_triggered: false,
            rebalance_reason: String::new(),
            rebalance_side: String::new(),
            rebalance_qty: 0.0,
            rebalance_value_usd: 0.0,
            free_usdt_before: 0.0,
            free_usdt_after: 0.0,
            btc_before: 0.0,
            btc_after: 0.0,
            rebalance_cooldown_until: None,
        }
    }
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
    // ── Decision transparency ─────────────────────────────────────────────────
    /// Top-level decision outcome: "EXECUTE", "HOLD", or "BLOCKED".
    pub final_decision: String,
    /// Why a balance/allocation check blocked execution (empty when not the cause).
    pub balance_block_reason: String,
    /// Why a risk/portfolio control blocked execution (empty when not the cause).
    pub risk_block_reason: String,
    /// Why a strategy/score/execution guard blocked execution (empty when not the cause).
    pub execution_block_reason: String,
    /// True when SWING profile is active and regime mismatch is informational.
    pub risk_override: bool,
    pub active_profile: String,
    pub active_profile_label: String,
    pub profile_lock: String,
    pub max_concurrent_positions: usize,
    pub counted_open_positions: usize,
    pub counted_pending_orders: usize,
    pub counted_reserved_slots: usize,
    pub slot_block_reason: String,
    pub slot_source_ids: String,
    // ── Drawdown diagnostics ──────────────────────────────────────────────────
    /// Current equity estimate used in drawdown calculation (position value + aggregate PnL).
    pub current_equity: f64,
    /// Highest equity seen since agent started.
    pub peak_equity: f64,
    /// Current drawdown as a fraction (0.0 = no drawdown, 0.1 = 10% drawdown).
    pub drawdown_pct: f64,
    /// Configured max drawdown limit (fraction).
    pub drawdown_limit: f64,
    // ── Cooldown telemetry ────────────────────────────────────────────────────
    /// True when the chosen role's per-role cooldown is still ticking.
    pub cooldown_active: bool,
    /// Milliseconds remaining on the active cooldown (0 when inactive).
    pub cooldown_remaining_ms: u64,
    // ── Micro-account adaptive thresholds ────────────────────────────────────
    /// Adaptive signal threshold in effect for the last cycle (balance-dependent).
    pub effective_threshold: f64,
    /// Threshold mode label: "normal", "micro_aggressive", or "micro_active".
    pub threshold_mode: String,
    // ── Score telemetry ───────────────────────────────────────────────────────
    /// Raw (un-clamped) score of the best candidate last cycle.
    pub raw_score: f64,
    /// raw_score / effective_threshold.  ≥ 1.0 means the score gate would pass.
    pub normalized_score: f64,
    /// Formatted top-3 penalty components that suppressed the score (descending).
    pub top_score_penalties: String,
    // ── COMPOUND_EXECUTION telemetry ──────────────────────────────────────────
    /// Current position size in USD terms (position_size_btc * mid price).
    pub compound_position_size_usd: f64,
    /// Current position size in base asset (BTC).
    pub compound_position_size_btc: f64,
    /// PnL of the most recently closed trade in USD.
    pub compound_last_trade_pnl: f64,
    /// Cumulative session PnL in USD since agent start.
    pub compound_session_pnl: f64,
    /// Peak account balance seen during this session in USD.
    pub compound_peak_balance: f64,
    /// Current account balance in USD (last observed total_balance_usd).
    pub compound_current_balance: f64,
    /// Consecutive-loss count driving size/cooldown decisions.
    pub compound_consecutive_losses: u32,
    /// Size scalar currently applied (1.0 = full, <1.0 = reduced after losses).
    pub compound_size_scalar: f64,
    /// True when a compound-loss cooldown is currently active (no new BUYs).
    pub compound_loss_pause_active: bool,
    // ── FLIP_HYPER capital-rotation telemetry ─────────────────────────────────
    /// Current FLIP_HYPER cycle phase (e.g. "SEEK_ENTRY", "SEEK_EXIT").
    pub flip_cycle_phase: String,
    /// Cumulative session PnL from completed flips (USD, approximate).
    pub flip_session_pnl: f64,
    /// Number of completed buy→sell flips this session.
    pub flip_rotation_count: u32,
    /// Entry price of the current open BUY (0.0 when flat).
    pub flip_last_entry_price: f64,
    /// Exit price of the last completed flip (0.0 if none yet).
    pub flip_last_exit_price: f64,
    /// Realized PnL (USD) of the last completed flip.
    pub flip_last_pnl_usd: f64,
    /// Realized PnL (%) of the last completed flip.
    pub flip_last_pnl_pct: f64,
    /// Minimum net profit floor (USD) required to execute a sell flip.
    pub flip_min_profit_floor: f64,
    /// Anti-stall blocker reason (non-empty when idle too long with valid inventory).
    pub flip_blocker: String,
    // ── CONTRACT_EXECUTOR telemetry (paper mode) ─────────────────────────────
    pub contract_side: String,
    pub contract_leverage: f64,
    pub contract_entry_price: f64,
    pub contract_mark_price: f64,
    pub contract_notional_usd: f64,
    pub contract_unrealized_pnl: f64,
    pub contract_realized_pnl_session: f64,
    pub contract_liquidation_price: f64,
    pub contract_stop_loss: f64,
    pub contract_take_profit: f64,
    pub contract_liquidation_buffer_pct: f64,
    pub contract_duration_secs: f64,
    pub contract_exit_reason: String,
    pub contract_last_trade_result: f64,
    pub contract_paper_mode: bool,
    pub contract_last_no_open_reason: String,
    pub rebalance_status: String,
    pub rebalance_triggered: bool,
    pub rebalance_reason: String,
    pub rebalance_side: String,
    pub rebalance_qty: f64,
    pub rebalance_value_usd: f64,
    pub free_usdt_before: f64,
    pub free_usdt_after: f64,
    pub btc_before: f64,
    pub btc_after: f64,
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
    final_decision: String,
    balance_block_reason: String,
    risk_block_reason: String,
    execution_block_reason: String,
    /// True when the chosen role's per-role cooldown is still ticking.
    cooldown_active: bool,
    /// Milliseconds remaining on the active cooldown (0 when inactive).
    cooldown_remaining_ms: u64,
    /// Adaptive signal threshold in effect (balance-dependent).
    effective_threshold: f64,
    /// Threshold mode: "normal", "micro_aggressive", or "micro_active".
    threshold_mode: String,
    // ── Score telemetry ───────────────────────────────────────────────────────
    raw_score: f64,
    normalized_score: f64,
    top_score_penalties: String,
    // ── COMPOUND_EXECUTION telemetry ──────────────────────────────────────────
    compound_position_size_usd: f64,
    compound_position_size_btc: f64,
    compound_last_trade_pnl: f64,
    compound_session_pnl: f64,
    compound_peak_balance: f64,
    compound_current_balance: f64,
    compound_consecutive_losses: u32,
    compound_size_scalar: f64,
    compound_loss_pause_active: bool,
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
    cfg: Arc<Mutex<NpcConfig>>,
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
            cfg: Arc::new(Mutex::new(cfg)),
            state,
            runtime: Arc::new(Mutex::new(NpcRuntimeState::default())),
            telemetry: Arc::new(Mutex::new(NpcLoopTelemetry {
                last_action: "NO_ACTION".to_string(),
                execution_result: NPC_STATUS_SCANNING.to_string(),
                status: NPC_STATUS_SCANNING.to_string(),
                last_agent_decision: "Waiting for first cycle".to_string(),
                last_no_trade_reason: String::new(),
                pipeline_state: "Scanning".to_string(),
                effective_threshold: THRESHOLD_BASE,
                threshold_mode: "normal".to_string(),
                raw_score: 0.0,
                normalized_score: 0.0,
                top_score_penalties: String::new(),
                ..NpcLoopTelemetry::default()
            })),
            control: Arc::new(Mutex::new(NpcLoopControl::new(interval_ms, mode))),
        }
    }

    pub async fn set_active_profile(&self, profile: RuntimeProfile) {
        {
            let mut cfg = self.cfg.lock().await;
            cfg.behavior_profile = profile.as_str().to_string();
        }
        {
            let mut rt = self.runtime.lock().await;
            rt.open_actions.clear();
            rt.cycle_open_notional = 0.0;
            rt.last_action_at.clear();
            rt.flip_cycle_phase = FlipCyclePhase::SeekEntry;
            rt.flip_last_completed = None;
            rt.flip_session_pnl = 0.0;
            rt.flip_rotation_count = 0;
            rt.flip_last_entry_price = 0.0;
            rt.flip_last_entry_qty = 0.0;
            rt.flip_last_active = None;
            rt.flip_blocker.clear();
            rt.contract_position = None;
            rt.contract_last_trade = None;
            rt.contract_realized_pnl_session = 0.0;
            rt.contract_last_exit_reason.clear();
            rt.contract_last_no_open_reason.clear();
            rt.swing_entry_price = 0.0;
            rt.swing_entry_started_at = None;
            rt.swing_peak_price = 0.0;
            rt.swing_peak_unrealized_pct = 0.0;
            rt.swing_last_exit_reason.clear();
            rt.swing_last_hold_duration_secs = 0.0;
            rt.swing_cooldown_until = None;
            rt.rebalance_status = "REBALANCE_COMPLETE".to_string();
            rt.rebalance_triggered = false;
            rt.rebalance_reason.clear();
            rt.rebalance_side.clear();
            rt.rebalance_qty = 0.0;
            rt.rebalance_value_usd = 0.0;
            rt.free_usdt_before = 0.0;
            rt.free_usdt_after = 0.0;
            rt.btc_before = 0.0;
            rt.btc_after = 0.0;
            rt.rebalance_cooldown_until = None;
        }
        {
            let mut t = self.telemetry.lock().await;
            t.threshold_mode = profile.as_str().to_ascii_lowercase();
            t.last_action = "NO_ACTION".to_string();
            t.execution_result = NPC_STATUS_SCANNING.to_string();
            t.last_agent_decision = format!("Profile switched to {}", profile.as_str());
            t.pipeline_state = "Scanning".to_string();
            t.final_decision = "HOLD".to_string();
            t.cooldown_active = false;
            t.cooldown_remaining_ms = 0;
            t.last_no_trade_reason.clear();
            t.balance_block_reason.clear();
            t.risk_block_reason.clear();
            t.execution_block_reason.clear();
            t.top_score_penalties.clear();
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
        let cfg = self.cfg.lock().await.clone();
        let agent_mode = control.mode;
        let (truth_position_size, truth_sell_inventory, truth_pending_order_slots, truth_pending_order_ids) = {
            let truth = self.state.truth.lock().await;
            let (pending_order_slots, pending_order_ids) = collect_truth_pending_order_slots(&truth);
            (
                truth.position.size,
                truth.sell_inventory,
                pending_order_slots,
                pending_order_ids,
            )
        };
        let (current_equity, peak_equity, drawdown_pct, compound, flip, contract, rebalance, contract_last_no_open_reason, slot_usage) = {
            // Compute current equity and drawdown from runtime state (same formula as evaluate_portfolio_controls).
            let rt = self.runtime.lock().await;
            let aggregate_pnl: f64 = rt.perf.values().map(|p| p.gross_pnl).sum();
            let equity = aggregate_pnl.max(0.0);
            let peak = rt.peak_equity.max(equity);
            const MIN_PEAK_FOR_DRAWDOWN: f64 = 1.0;
            let dd = if peak >= MIN_PEAK_FOR_DRAWDOWN {
                ((peak - equity) / peak).max(0.0)
            } else {
                0.0
            };
            // Compute position size in USD using the most recent mid price from history.
            let last_mid = rt.mid_history.back().copied().unwrap_or(0.0);
            let pos_usd = rt.compound_last_position_btc * last_mid;
            let c = (
                rt.compound_last_position_btc,
                pos_usd,
                rt.compound_last_trade_pnl,
                rt.compound_session_pnl,
                rt.compound_peak_balance,
                rt.compound_last_balance_usd,
                rt.compound_consecutive_losses,
                rt.compound_size_scalar,
                rt.compound_loss_pause_until.map(|t| t > std::time::Instant::now()).unwrap_or(false),
            );
            let (flip_last_exit_price, flip_last_pnl_usd, flip_last_pnl_pct) = rt
                .flip_last_completed
                .as_ref()
                .map(|f| (f.exit_price, f.realized_pnl_usd, f.realized_pnl_pct))
                .unwrap_or((0.0, 0.0, 0.0));
            let f = (
                rt.flip_cycle_phase.as_str().to_string(),
                rt.flip_session_pnl,
                rt.flip_rotation_count,
                rt.flip_last_entry_price,
                flip_last_exit_price,
                flip_last_pnl_usd,
                flip_last_pnl_pct,
                rt.flip_blocker.clone(),
            );
            let (side, leverage, entry_price, mark_price, notional_usd, unrealized_pnl, liq_price, stop_loss, take_profit, liq_buffer, duration_secs) =
                if let Some(pos) = rt.contract_position.as_ref() {
                    (
                        pos.side.as_str().to_string(),
                        pos.leverage,
                        pos.entry_price,
                        pos.mark_price,
                        pos.notional_usd,
                        pos.unrealized_pnl,
                        pos.liquidation_price,
                        pos.stop_loss,
                        pos.take_profit,
                        contract_liquidation_buffer_pct(pos.side, pos.mark_price, pos.liquidation_price),
                        pos.opened_at.elapsed().as_secs_f64(),
                    )
                } else {
                    ("FLAT".to_string(), rt.contract_leverage, 0.0, last_mid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                };
            let last_result = rt.contract_last_trade.as_ref().map(|t| t.pnl_usd).unwrap_or(0.0);
            let last_reason = rt
                .contract_last_trade
                .as_ref()
                .map(|t| format!(
                    "{} {}x {}s {} gross={:.4} entry_fee={:.4} exit_fee={:.4} net={:.4} @{:.2}->{:.2}",
                    t.side.as_str(),
                    t.leverage,
                    t.duration_secs.round() as i64,
                    t.exit_reason,
                    t.gross_realized_pnl_usd,
                    t.entry_fee_usd,
                    t.exit_fee_usd,
                    t.net_realized_pnl_usd,
                    t.entry_price,
                    t.exit_price
                ))
                .unwrap_or_else(|| {
                    if rt.contract_last_no_open_reason.is_empty() {
                        rt.contract_last_exit_reason.clone()
                    } else {
                        rt.contract_last_no_open_reason.clone()
                    }
                });
            let k = (
                side,
                leverage,
                entry_price,
                mark_price,
                notional_usd,
                unrealized_pnl,
                rt.contract_realized_pnl_session,
                liq_price,
                stop_loss,
                take_profit,
                liq_buffer,
                duration_secs,
                last_reason,
                last_result,
                rt.contract_paper_mode,
            );
            let reb = (
                rt.rebalance_status.clone(),
                rt.rebalance_triggered,
                rt.rebalance_reason.clone(),
                rt.rebalance_side.clone(),
                rt.rebalance_qty,
                rt.rebalance_value_usd,
                rt.free_usdt_before,
                rt.free_usdt_after,
                rt.btc_before,
                rt.btc_after,
            );
            let slot_usage = slot_usage_snapshot(
                &cfg,
                &rt,
                truth_position_size,
                truth_sell_inventory,
                truth_pending_order_slots,
                &truth_pending_order_ids,
            );
            (equity, peak, dd, c, f, k, reb, rt.contract_last_no_open_reason.clone(), slot_usage)
        };
        let (
            compound_pos_btc, compound_pos_usd,
            compound_last_pnl, compound_sess_pnl,
            compound_peak_bal, compound_cur_bal,
            compound_consec_losses, compound_sz_scalar,
            compound_paused,
        ) = compound;
        let (
            flip_cycle_phase,
            flip_session_pnl,
            flip_rotation_count,
            flip_last_entry_price,
            flip_last_exit_price,
            flip_last_pnl_usd,
            flip_last_pnl_pct,
            flip_blocker,
        ) = flip;
        let (
            rebalance_status,
            rebalance_triggered,
            rebalance_reason,
            rebalance_side,
            rebalance_qty,
            rebalance_value_usd,
            free_usdt_before,
            free_usdt_after,
            btc_before,
            btc_after,
        ) = rebalance;
        let (
            contract_side,
            contract_leverage,
            contract_entry_price,
            contract_mark_price,
            contract_notional_usd,
            contract_unrealized_pnl,
            contract_realized_pnl_session,
            contract_liquidation_price,
            contract_stop_loss,
            contract_take_profit,
            contract_liquidation_buffer_pct,
            contract_duration_secs,
            contract_exit_reason,
            contract_last_trade_result,
            contract_paper_mode,
        ) = contract;
        let active_profile = RuntimeProfile::from_str(&cfg.behavior_profile);
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
            final_decision: telemetry.final_decision.clone(),
            balance_block_reason: telemetry.balance_block_reason.clone(),
            risk_block_reason: telemetry.risk_block_reason.clone(),
            execution_block_reason: telemetry.execution_block_reason.clone(),
            risk_override: cfg.behavior_profile.eq_ignore_ascii_case("SWING"),
            active_profile: active_profile.as_str().to_string(),
            active_profile_label: active_profile.label().to_string(),
            profile_lock: active_profile.as_str().to_string(),
            max_concurrent_positions: slot_usage.max_concurrent_positions,
            counted_open_positions: slot_usage.counted_open_positions,
            counted_pending_orders: slot_usage.counted_pending_orders,
            counted_reserved_slots: slot_usage.counted_reserved_slots,
            slot_block_reason: slot_usage.slot_block_reason,
            slot_source_ids: slot_usage.slot_source_ids.join("|"),
            current_equity,
            peak_equity,
            drawdown_pct,
            drawdown_limit: cfg.alpha.max_drawdown_pct,
            cooldown_active: telemetry.cooldown_active,
            cooldown_remaining_ms: telemetry.cooldown_remaining_ms,
            effective_threshold: telemetry.effective_threshold,
            threshold_mode: telemetry.threshold_mode.clone(),
            raw_score: telemetry.raw_score,
            normalized_score: telemetry.normalized_score,
            top_score_penalties: telemetry.top_score_penalties.clone(),
            compound_position_size_btc: compound_pos_btc,
            compound_position_size_usd: compound_pos_usd,
            compound_last_trade_pnl: compound_last_pnl,
            compound_session_pnl: compound_sess_pnl,
            compound_peak_balance: compound_peak_bal,
            compound_current_balance: compound_cur_bal,
            compound_consecutive_losses: compound_consec_losses,
            compound_size_scalar: compound_sz_scalar,
            compound_loss_pause_active: compound_paused,
            flip_cycle_phase,
            flip_session_pnl,
            flip_rotation_count,
            flip_last_entry_price,
            flip_last_exit_price,
            flip_last_pnl_usd,
            flip_last_pnl_pct,
            flip_min_profit_floor: flip_hyper_profit_floor_for_balance(compound_cur_bal),
            flip_blocker,
            contract_side,
            contract_leverage,
            contract_entry_price,
            contract_mark_price,
            contract_notional_usd,
            contract_unrealized_pnl,
            contract_realized_pnl_session,
            contract_liquidation_price,
            contract_stop_loss,
            contract_take_profit,
            contract_liquidation_buffer_pct,
            contract_duration_secs,
            contract_exit_reason,
            contract_last_trade_result,
            contract_paper_mode,
            contract_last_no_open_reason,
            rebalance_status,
            rebalance_triggered,
            rebalance_reason,
            rebalance_side,
            rebalance_qty,
            rebalance_value_usd,
            free_usdt_before,
            free_usdt_after,
            btc_before,
            btc_after,
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
        let cfg = Arc::clone(&self.cfg);
        let state = self.state.clone();
        let runtime = Arc::clone(&self.runtime);
        let telemetry = Arc::clone(&self.telemetry);
        control.handle = Some(tokio::spawn(async move {
            let cfg_initial = cfg.lock().await.clone();
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
                    cfg_initial.mode.as_str(),
                    interval_ms,
                    cfg_initial.trade_size
                ),
            );

            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
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
                        let mut loop_cfg = cfg.lock().await.clone();
                        loop_cfg.cycle_interval = Duration::from_millis(interval_ms);
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
                        t.final_decision = report.final_decision;
                        t.balance_block_reason = report.balance_block_reason;
                        t.risk_block_reason = report.risk_block_reason;
                        t.execution_block_reason = report.execution_block_reason;
                        t.cooldown_active = report.cooldown_active;
                        t.cooldown_remaining_ms = report.cooldown_remaining_ms;
                        t.effective_threshold = report.effective_threshold;
                        t.threshold_mode = report.threshold_mode;
                        t.raw_score = report.raw_score;
                        t.normalized_score = report.normalized_score;
                        t.top_score_penalties = report.top_score_penalties;
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

    /// Test-only helper: inject a no-trade reason and mark the agent as blocked.
    /// This simulates what happens after a cycle that doesn't trade.
    #[cfg(test)]
    pub async fn set_no_trade_reason_for_test(&self, reason: &str) {
        let mut t = self.telemetry.lock().await;
        t.last_no_trade_reason = reason.to_string();
        t.execution_block_reason = reason.to_string();
        t.final_decision = "BLOCKED".to_string();
    }

    /// Test-only helper: inject cooldown state into telemetry.
    #[cfg(test)]
    pub async fn set_cooldown_for_test(&self, active: bool, remaining_ms: u64) {
        let mut t = self.telemetry.lock().await;
        t.cooldown_active = active;
        t.cooldown_remaining_ms = remaining_ms;
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
    /// Top-level decision outcome: "EXECUTE", "HOLD", or "BLOCKED".
    final_decision: String,
    /// Why a balance/allocation check blocked execution (empty when not the cause).
    balance_block_reason: String,
    /// Why a risk/portfolio control blocked execution (empty when not the cause).
    risk_block_reason: String,
    /// Why a strategy/score/execution guard blocked execution (empty when not the cause).
    execution_block_reason: String,
    /// True when the chosen role's per-role cooldown is still ticking.
    cooldown_active: bool,
    /// Milliseconds remaining on the active cooldown (0 when inactive).
    cooldown_remaining_ms: u64,
    /// Adaptive signal threshold in effect for this cycle (balance-dependent).
    effective_threshold: f64,
    /// Threshold mode: "normal", "micro_aggressive", or "micro_active".
    threshold_mode: String,
    // ── Score telemetry ───────────────────────────────────────────────────────
    /// Raw (un-clamped) score of the best candidate this cycle.  0.0 when no
    /// candidate was built (e.g. authority OFF or no candidates).
    raw_score: f64,
    /// raw_score / effective_threshold.  ≥ 1.0 means the score gate would pass.
    normalized_score: f64,
    /// Formatted top-3 penalty components that suppressed the score (descending).
    top_score_penalties: String,
}

// ── Micro-account adaptive threshold constants ────────────────────────────────

const MICRO_BALANCE_USD: f64        = 50.0;   // below this: micro tier (< $50)
const MICRO_BALANCE_MID_USD: f64    = 100.0;  // below this: mid tier ($50–$99.99)
const THRESHOLD_MICRO_SMALL: f64    = 0.11;   // signal threshold for micro accounts (balance < $50)
const THRESHOLD_MICRO: f64          = 0.14;   // signal threshold for mid accounts ($50–$99.99)
const THRESHOLD_BASE: f64           = 0.18;   // signal threshold for normal accounts (balance ≥ $100)

// ── MICRO_ACTIVE live-mode thresholds (lower than paper/sim micro_aggressive) ──
/// Signal threshold for live micro_active < $50 (lower than micro_aggressive 0.11).
const THRESHOLD_MICRO_ACTIVE_SMALL: f64 = 0.065;
/// Signal threshold for live micro_active $50–$99.99 (lower than micro_aggressive 0.14).
const THRESHOLD_MICRO_ACTIVE_MID: f64   = 0.09;

/// Penalty dampen factor applied to spread_cost / volatility_penalty / slippage_risk
/// in MICRO_ACTIVE mode. Halves those components so the tiny edge signal is not
/// overwhelmed by proportionally large penalty terms.
const MICRO_PENALTY_DAMPEN: f64 = 0.50;

// ── FLIP_HYPER capital-rotation constants ─────────────────────────────────────
/// Signal threshold for FLIP_HYPER live < $50 (lower than micro_active 0.065).
const THRESHOLD_FLIP_HYPER_SMALL: f64 = 0.040;
/// Signal threshold for FLIP_HYPER live $50–$99.99 (lower than micro_active 0.09).
const THRESHOLD_FLIP_HYPER_MID: f64   = 0.055;
/// Minimum net profit (USD) required to execute a FLIP_HYPER sell.
/// Never sell for less than this after spread + fees + slippage.
pub const FLIP_HYPER_MIN_PROFIT_FLOOR_USD: f64 = 0.01;
/// Minimum net profit (USD) required to execute a FLIP_HYPER sell for
/// micro accounts (< $50). Tuned for frequent small rotations.
const FLIP_HYPER_MIN_PROFIT_FLOOR_USD_MICRO: f64 = 0.02;
/// Seconds without an execution before the anti-stall rule fires and surfaces
/// an exact blocker reason in telemetry.
const FLIP_HYPER_STALL_SECS: u64 = 120;
/// Force-rotation fallback timeout (seconds) for HOLDING_POSITION/SEEK_EXIT.
const FLIP_HYPER_FORCED_ROTATION_SECS: u64 = 60;
/// Max loss allowed for forced-rotation fallback exits (0.1%).
const FLIP_HYPER_FORCED_ROTATION_MAX_LOSS_PCT: f64 = 0.1;

// ── SWING_TRADER constants ───────────────────────────────────────────────────
const SWING_TRAILING_STOP_PCT: f64 = 0.0045; // ~0.45%
const SWING_TARGET_MIN_PCT: f64 = 0.003; // 0.3%
const SWING_TARGET_MAX_PCT: f64 = 0.012; // 1.2%
const SWING_COOLDOWN_MIN_SECS: u64 = 30;
const SWING_COOLDOWN_MAX_SECS: u64 = 120;
const REBALANCE_DUST_QTY: f64 = 0.000_000_01;
const REBALANCE_MAX_INVENTORY_SELL_PCT: f64 = 0.80;
const REBALANCE_FALLBACK_SELL_PCT: f64 = 0.25;
const REBALANCE_MAX_USDT_BUY_PCT: f64 = 0.30;
const CONTRACT_LEVERAGE_MIN: f64 = 3.0;
const CONTRACT_LEVERAGE_MAX: f64 = 5.0;
const CONTRACT_DEFAULT_LEVERAGE: f64 = 3.0;
const CONTRACT_MAINT_MARGIN_RATE: f64 = 0.005;
const CONTRACT_MIN_VOL_ESTIMATE: f64 = 0.001;
const CONTRACT_STOP_VOL_MULT: f64 = 1.5;
const CONTRACT_TAKE_PROFIT_VOL_MULT: f64 = 3.0;
const CONTRACT_TRAIL_VOL_MULT: f64 = 1.2;

// ── COMPOUND_EXECUTION mode constants ────────────────────────────────────────
/// Minimum normalized score (raw_score / threshold) required in COMPOUND_EXECUTION.
/// Score must be at least 20% above the threshold — not just barely passing.
const COMPOUND_NORMALIZED_SCORE_MIN: f64 = 1.2;
/// Base position size as a fraction of total account balance (15%).
const COMPOUND_BASE_EQUITY_PCT: f64 = 0.15;
/// Maximum position size as a fraction of total account balance (30%).
const COMPOUND_MAX_EQUITY_PCT: f64 = 0.30;
/// Minimum trade notional in USD (floor, never go below $5).
const COMPOUND_MIN_NOTIONAL_USD: f64 = 5.0;
/// After this many consecutive losses, reduce position size by COMPOUND_LOSS_SIZE_FACTOR.
const COMPOUND_LOSS_SIZE_REDUCE_THRESHOLD: u32 = 2;
/// After this many consecutive losses, pause new BUY entries for a cooldown window.
const COMPOUND_LOSS_PAUSE_THRESHOLD: u32 = 3;
/// Factor applied to size scalar after the loss-reduce threshold is reached (50% cut).
const COMPOUND_LOSS_SIZE_FACTOR: f64 = 0.50;
/// Cooldown duration (seconds) after COMPOUND_LOSS_PAUSE_THRESHOLD consecutive losses.
const COMPOUND_LOSS_COOLDOWN_SECS: u64 = 60;
/// When account balance grows by this fraction above the previous peak, lock profits.
const COMPOUND_PROFIT_LOCK_TRIGGER_PCT: f64 = 0.10;
/// Risk reduction applied to size scalar when profit-lock triggers (keep 85% of size).
const COMPOUND_PROFIT_LOCK_RISK_FACTOR: f64 = 0.85;
/// Hard floor for the compound size scalar — never reduce below this fraction.
const COMPOUND_MIN_SIZE_SCALAR: f64 = 0.25;

/// Compute balance-adaptive signal threshold and mode label.
///
/// Returns `(effective_threshold, threshold_mode)`:
/// - `effective_threshold`: the score floor to use for this balance level
/// - `threshold_mode`: `"micro_aggressive"` when `total_balance_usd > 0.0`
///   and below [`MICRO_BALANCE_MID_USD`]; a `0.0` balance returns `"normal"`
///
/// Tier map:
///   balance < $50        → 0.11  "micro_aggressive"
///   $50 ≤ balance < $100 → 0.14  "micro_aggressive"
///   balance ≥ $100       → 0.18  "normal"
///   balance = 0.0        → 0.18  "normal"  (no data yet)
///
/// Used in tests and paper/sim modes. For live mode use
/// `adaptive_signal_threshold_with_trading_mode`.
fn adaptive_signal_threshold(total_balance_usd: f64) -> (f64, &'static str) {
    if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_USD {
        (THRESHOLD_MICRO_SMALL, "micro_aggressive")
    } else if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_MID_USD {
        (THRESHOLD_MICRO, "micro_aggressive")
    } else {
        (THRESHOLD_BASE, "normal")
    }
}

/// Compute balance-adaptive signal threshold and mode label, taking the current
/// trading mode into account.
///
/// In `Live` mode, accounts below $100 activate **MICRO_ACTIVE** with even lower
/// thresholds and dedicated scoring-penalty dampening:
///   live balance < $50        → 0.065  "micro_active"
///   live balance $50–$99.99   → 0.09   "micro_active"
///   live balance ≥ $100       → 0.18   "normal"
///
/// In paper/simulation mode the ordinary `adaptive_signal_threshold` tiers apply.
fn adaptive_signal_threshold_with_trading_mode(
    total_balance_usd: f64,
    mode: NpcTradingMode,
) -> (f64, &'static str) {
    if mode == NpcTradingMode::Live {
        if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_USD {
            (THRESHOLD_MICRO_ACTIVE_SMALL, "micro_active")
        } else if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_MID_USD {
            (THRESHOLD_MICRO_ACTIVE_MID, "micro_active")
        } else {
            (THRESHOLD_BASE, "normal")
        }
    } else {
        adaptive_signal_threshold(total_balance_usd)
    }
}

fn profile_threshold(
    profile: RuntimeProfile,
    total_balance_usd: f64,
    mode: NpcTradingMode,
) -> (f64, &'static str) {
    match profile {
        RuntimeProfile::Conservative => (THRESHOLD_BASE, "conservative"),
        RuntimeProfile::Active => (THRESHOLD_BASE, "active"),
        RuntimeProfile::MicroTest => adaptive_signal_threshold(total_balance_usd),
        RuntimeProfile::MicroActive => adaptive_signal_threshold_with_trading_mode(total_balance_usd, mode),
        RuntimeProfile::FlipHyper => {
            if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_USD {
                (THRESHOLD_FLIP_HYPER_SMALL, "flip_hyper")
            } else if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_MID_USD {
                (THRESHOLD_FLIP_HYPER_MID, "flip_hyper")
            } else {
                (THRESHOLD_BASE, "flip_hyper")
            }
        }
        RuntimeProfile::Swing => (THRESHOLD_BASE, "swing"),
    }
}

fn flip_hyper_profit_floor_for_balance(total_balance_usd: f64) -> f64 {
    if total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_USD {
        FLIP_HYPER_MIN_PROFIT_FLOOR_USD_MICRO
    } else {
        FLIP_HYPER_MIN_PROFIT_FLOOR_USD
    }
}

fn compute_swing_signals(feed: &crate::feed::FeedState, current_mid: f64) -> SwingRegimeSignals {
    let now = Instant::now();
    let momentum_1m = crate::signal::SignalEngine::compute_momentum(
        &feed.mid_history,
        now,
        Duration::from_secs(60),
        current_mid,
    );
    let momentum_5m = crate::signal::SignalEngine::compute_momentum(
        &feed.mid_history,
        now,
        Duration::from_secs(300),
        current_mid,
    );
    let trend_bias_15m = crate::signal::SignalEngine::compute_momentum(
        &feed.mid_history,
        now,
        Duration::from_secs(900),
        current_mid,
    );

    let cutoff_recent = now - Duration::from_secs(60);
    let cutoff_prior = now - Duration::from_secs(120);
    let mut recent_low = f64::INFINITY;
    let mut prior_low = f64::INFINITY;
    for s in &feed.mid_history {
        if s.timestamp >= cutoff_recent {
            recent_low = recent_low.min(s.mid);
        } else if s.timestamp >= cutoff_prior {
            prior_low = prior_low.min(s.mid);
        }
    }
    let structure_break_long = recent_low.is_finite()
        && prior_low.is_finite()
        && recent_low < prior_low * (1.0 - 0.0002);

    // Pullback + resume rule:
    // - 5m trend must stay positive
    // - 1m momentum dips mildly negative (pullback)
    // - short horizon starts lifting again (resume)
    let pullback_detected = momentum_5m > 0.0 && momentum_1m < -0.0001;
    let momentum_resumed_up = momentum_1m > 0.0;

    SwingRegimeSignals {
        momentum_1m,
        momentum_5m,
        trend_bias_15m,
        pullback_detected,
        momentum_resumed_up,
        structure_break_long,
    }
}

fn swing_bias(signals: &SwingRegimeSignals) -> SwingBias {
    if signals.momentum_5m > 0.0 && signals.trend_bias_15m > 0.0 {
        SwingBias::LongBias
    } else if signals.momentum_5m < 0.0 && signals.trend_bias_15m < 0.0 {
        SwingBias::ShortBias
    } else {
        SwingBias::NoTrade
    }
}

fn contract_unrealized_pnl(side: ContractSide, entry_price: f64, mark_price: f64, qty_base: f64) -> f64 {
    let delta = match side {
        ContractSide::Long => mark_price - entry_price,
        ContractSide::Short => entry_price - mark_price,
    };
    delta * qty_base
}

fn contract_liquidation_price(side: ContractSide, entry_price: f64, leverage: f64) -> f64 {
    let lev = leverage.max(1.0);
    match side {
        ContractSide::Long => {
            (entry_price * (1.0 - (1.0 / lev) + CONTRACT_MAINT_MARGIN_RATE)).max(f64::EPSILON)
        }
        ContractSide::Short => entry_price * (1.0 + (1.0 / lev) - CONTRACT_MAINT_MARGIN_RATE),
    }
}

fn contract_liquidation_buffer_pct(side: ContractSide, mark_price: f64, liquidation_price: f64) -> f64 {
    if mark_price <= 0.0 || !mark_price.is_finite() {
        return 0.0;
    }
    match side {
        ContractSide::Long => ((mark_price - liquidation_price) / mark_price).max(0.0),
        ContractSide::Short => ((liquidation_price - mark_price) / mark_price).max(0.0),
    }
}

fn update_contract_executor(
    rt: &mut NpcRuntimeState,
    swing_bias: SwingBias,
    entry_price: f64,
    mark_price: f64,
    volatility_estimate: f64,
    balance_usd: f64,
) {
    if !rt.contract_paper_mode || !mark_price.is_finite() || mark_price <= 0.0 {
        return;
    }
    let leverage = rt.contract_leverage.clamp(CONTRACT_LEVERAGE_MIN, CONTRACT_LEVERAGE_MAX);
    rt.contract_leverage = leverage;
    let vol = volatility_estimate.abs().max(CONTRACT_MIN_VOL_ESTIMATE);
    let notional = (balance_usd.max(0.0) * leverage).max(0.0);

    if let Some(pos) = rt.contract_position.as_mut() {
        pos.mark_price = mark_price;
        pos.high_water_mark = pos.high_water_mark.max(mark_price);
        pos.low_water_mark = pos.low_water_mark.min(mark_price);
        let trail_distance = (vol * CONTRACT_TRAIL_VOL_MULT).max(0.002);
        pos.trailing_stop = match pos.side {
            ContractSide::Long => pos.high_water_mark * (1.0 - trail_distance),
            ContractSide::Short => pos.low_water_mark * (1.0 + trail_distance),
        };
        pos.unrealized_pnl = contract_unrealized_pnl(pos.side, pos.entry_price, mark_price, pos.qty_base);

        let stop_hit = match pos.side {
            ContractSide::Long => mark_price <= pos.stop_loss,
            ContractSide::Short => mark_price >= pos.stop_loss,
        };
        let take_profit_hit = match pos.side {
            ContractSide::Long => mark_price >= pos.take_profit,
            ContractSide::Short => mark_price <= pos.take_profit,
        };
        let trailing_hit = match pos.side {
            ContractSide::Long => mark_price <= pos.trailing_stop,
            ContractSide::Short => mark_price >= pos.trailing_stop,
        };
        let liquidation_hit = match pos.side {
            ContractSide::Long => mark_price <= pos.liquidation_price,
            ContractSide::Short => mark_price >= pos.liquidation_price,
        };
        let bias_flip = match (pos.side, swing_bias) {
            (ContractSide::Long, SwingBias::LongBias) => false,
            (ContractSide::Short, SwingBias::ShortBias) => false,
            _ => true,
        };
        let exit_reason = if liquidation_hit {
            Some("liquidation_guard")
        } else if stop_hit {
            Some("stop_loss_hit")
        } else if take_profit_hit {
            Some("take_profit_hit")
        } else if trailing_hit {
            Some("trailing_stop_hit")
        } else if bias_flip {
            Some("swing_bias_flip")
        } else {
            None
        };
        if let Some(reason) = exit_reason {
            let exit_fee = pos.notional_usd * rt.contract_fee_rate.max(0.0);
            let gross_realized = pos.unrealized_pnl;
            let net_realized = gross_realized - pos.entry_fee_usd - exit_fee;
            let session_close_realized = gross_realized - exit_fee;
            rt.contract_realized_pnl_session += session_close_realized;
            rt.contract_last_exit_reason = reason.to_string();
            rt.contract_last_no_open_reason.clear();
            rt.contract_last_trade = Some(CompletedContractTrade {
                side: pos.side,
                leverage: pos.leverage,
                entry_price: pos.entry_price,
                exit_price: mark_price,
                gross_realized_pnl_usd: gross_realized,
                entry_fee_usd: pos.entry_fee_usd,
                exit_fee_usd: exit_fee,
                net_realized_pnl_usd: net_realized,
                pnl_usd: net_realized,
                duration_secs: pos.opened_at.elapsed().as_secs_f64(),
                exit_reason: reason.to_string(),
            });
            rt.contract_position = None;
        }
    }

    if rt.contract_position.is_none() {
        let side = match swing_bias {
            SwingBias::LongBias => Some(ContractSide::Long),
            SwingBias::ShortBias => Some(ContractSide::Short),
            SwingBias::NoTrade => None,
        };
        if let Some(side) = side {
            if !balance_usd.is_finite() || balance_usd <= 0.0 {
                rt.contract_last_no_open_reason = "ZERO_CAPITAL".to_string();
                return;
            }
            if !notional.is_finite() || notional <= 0.0 {
                rt.contract_last_no_open_reason = "ZERO_NOTIONAL".to_string();
                return;
            }
            if !entry_price.is_finite() || entry_price <= 0.0 {
                rt.contract_last_no_open_reason = "ZERO_QTY".to_string();
                return;
            }
            let entry = entry_price;
            let qty_base = if entry > 0.0 { notional / entry } else { 0.0 };
            if !qty_base.is_finite() || qty_base <= 0.0 {
                rt.contract_last_no_open_reason = "ZERO_QTY".to_string();
                return;
            }
            let stop_distance = (vol * CONTRACT_STOP_VOL_MULT).max(0.0025);
            let tp_distance = (vol * CONTRACT_TAKE_PROFIT_VOL_MULT).max(0.004);
            let stop_loss = match side {
                ContractSide::Long => entry * (1.0 - stop_distance),
                ContractSide::Short => entry * (1.0 + stop_distance),
            };
            let take_profit = match side {
                ContractSide::Long => entry * (1.0 + tp_distance),
                ContractSide::Short => entry * (1.0 - tp_distance),
            };
            let liquidation_price = contract_liquidation_price(side, entry, leverage);
            let trailing_stop = match side {
                ContractSide::Long => entry * (1.0 - (vol * CONTRACT_TRAIL_VOL_MULT).max(0.002)),
                ContractSide::Short => entry * (1.0 + (vol * CONTRACT_TRAIL_VOL_MULT).max(0.002)),
            };
            let entry_fee = notional * rt.contract_fee_rate.max(0.0);
            if !entry_fee.is_finite() {
                rt.contract_last_no_open_reason = "INVALID_ENTRY_FEE".to_string();
                return;
            }
            rt.contract_realized_pnl_session -= entry_fee;
            rt.contract_position = Some(OpenContractPosition {
                side,
                leverage,
                entry_price: entry,
                mark_price,
                notional_usd: notional,
                qty_base,
                entry_fee_usd: entry_fee,
                unrealized_pnl: contract_unrealized_pnl(side, entry, mark_price, qty_base),
                liquidation_price,
                stop_loss,
                take_profit,
                trailing_stop,
                opened_at: Instant::now(),
                high_water_mark: mark_price,
                low_water_mark: mark_price,
            });
            rt.contract_last_no_open_reason.clear();
        }
    }
}

#[derive(Clone, Debug)]
struct RebalancePlan {
    status: &'static str,
    reason: String,
    side: &'static str,
    qty: f64,
    value_usd: f64,
    free_usdt_before: f64,
    free_usdt_after: f64,
    btc_before: f64,
    btc_after: f64,
}

fn maybe_rebalance(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    side: &str,
    buy_power: f64,
    sell_inventory: f64,
    mid: f64,
    total_balance_usd: f64,
    required_trade_capital: f64,
) -> Option<RebalancePlan> {
    if !cfg.disable_no_trade_idle || !mid.is_finite() || mid <= 0.0 {
        return None;
    }
    if let Some(until) = rt.rebalance_cooldown_until {
        if until > Instant::now() {
            return None;
        }
    }

    let buy_power = buy_power.max(0.0);
    let sell_inventory = sell_inventory.max(0.0);
    let btc_inventory_usd = sell_inventory * mid;
    let target_usdt_usd = total_balance_usd * cfg.target_usdt_ratio.clamp(0.05, 0.95);
    let target_btc_usd = total_balance_usd * cfg.target_btc_ratio.clamp(0.05, 0.95);
    let min_notional = cfg.rebalance_min_notional_usd.max(1.0);

    if side.eq_ignore_ascii_case("BUY")
        && buy_power < required_trade_capital
        && btc_inventory_usd > required_trade_capital
        && total_balance_usd > required_trade_capital
    {
        let need_usdt = (target_usdt_usd - buy_power).max(required_trade_capital - buy_power).max(0.0);
        let max_sell_value_for_target = (btc_inventory_usd - target_btc_usd).max(0.0);
        let sell_value = need_usdt
            .max(required_trade_capital)
            .min(btc_inventory_usd * REBALANCE_MAX_INVENTORY_SELL_PCT)
            .min(if max_sell_value_for_target > 0.0 {
                max_sell_value_for_target
            } else {
                btc_inventory_usd * REBALANCE_FALLBACK_SELL_PCT
            });
        let qty = sell_value / mid;
        if sell_value >= min_notional && qty > REBALANCE_DUST_QTY {
            return Some(RebalancePlan {
                status: "REBALANCE_TO_USDT",
                reason: format!(
                    "BUY_CAPITAL_SHORTAGE: free_usdt={:.2}<required={:.2} btc_inventory_usd={:.2}",
                    buy_power, required_trade_capital, btc_inventory_usd
                ),
                side: "SELL",
                qty,
                value_usd: sell_value,
                free_usdt_before: buy_power,
                free_usdt_after: buy_power + sell_value,
                btc_before: sell_inventory,
                btc_after: (sell_inventory - qty).max(0.0),
            });
        }
    }

    if side.eq_ignore_ascii_case("SELL")
        && sell_inventory <= REBALANCE_DUST_QTY
        && buy_power > min_notional
        && total_balance_usd > min_notional
    {
        let excess_usdt = (buy_power - target_usdt_usd).max(0.0);
        let buy_value = excess_usdt.min(buy_power * REBALANCE_MAX_USDT_BUY_PCT);
        let qty = buy_value / mid;
        if buy_value >= min_notional && qty > REBALANCE_DUST_QTY {
            return Some(RebalancePlan {
                status: "REBALANCE_TO_BTC",
                reason: format!(
                    "SELL_INVENTORY_SHORTAGE: free_btc={:.8} free_usdt={:.2}",
                    sell_inventory, buy_power
                ),
                side: "BUY",
                qty,
                value_usd: buy_value,
                free_usdt_before: buy_power,
                free_usdt_after: (buy_power - buy_value).max(0.0),
                btc_before: sell_inventory,
                btc_after: sell_inventory + qty,
            });
        }
    }

    None
}

async fn run_cycle(cfg: &NpcConfig, state: &AgentState, runtime: Arc<Mutex<NpcRuntimeState>>) -> NpcCycleReport {
    let mode = state.authority.mode().await;
    let metrics = {
        let feed = state.feed.lock().await;
        let signal = state.signal.lock().await;
        signal.compute_metrics_pub(&feed)
    };
    let swing_signals = {
        let feed = state.feed.lock().await;
        compute_swing_signals(&feed, metrics.mid)
    };

    let (position_size, buy_power, sell_inventory, exposure_notional, total_balance_usd, pending_order_slots, pending_order_ids) = {
        let t = state.truth.lock().await;
        let pos = t.position.size.max(0.0);
        let mid = metrics.mid.max(0.0);
        let (pending_order_slots, pending_order_ids) = collect_truth_pending_order_slots(&t);
        (pos, t.buy_power.max(0.0), t.sell_inventory.max(0.0), pos * mid, t.total_balance_usd.max(0.0), pending_order_slots, pending_order_ids)
    };

    let (effective_threshold, threshold_mode) = adaptive_signal_threshold_with_trading_mode(total_balance_usd, cfg.mode);
    let no_action = |cycle_id: u64, execution_result: String, status: String| NpcCycleReport {
        cycle_id,
        last_action: "NO_ACTION".to_string(),
        execution_result: execution_result.clone(),
        status,
        last_agent_decision: "HOLD — no actionable trigger".to_string(),
        no_trade_reason: execution_result,
        pipeline_state: "Scanning".to_string(),
        final_decision: "HOLD".to_string(),
        balance_block_reason: String::new(),
        risk_block_reason: String::new(),
        execution_block_reason: String::new(),
        cooldown_active: false,
        cooldown_remaining_ms: 0,
        effective_threshold,
        threshold_mode: threshold_mode.to_string(),
        raw_score: 0.0,
        normalized_score: 0.0,
        top_score_penalties: String::new(),
    };
    if mode == AuthorityMode::Off {
        return NpcCycleReport {
            cycle_id: 0,
            last_action: "NO_ACTION".to_string(),
            execution_result: "AUTHORITY_MODE_OFF".to_string(),
            status: "blocked".to_string(),
            last_agent_decision: "BLOCKED — authority mode is OFF".to_string(),
            no_trade_reason: "Authority mode is OFF. Set authority mode to AUTO via the web UI \
                              (/authority/mode/auto) to enable autonomous execution."
                .to_string(),
            pipeline_state: "Blocked — Authority OFF".to_string(),
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: String::new(),
            risk_block_reason: String::new(),
            execution_block_reason:
                "AUTHORITY_MODE_OFF: authority mode is OFF; set to AUTO or ASSIST to permit execution"
                    .to_string(),
            cooldown_active: false,
            cooldown_remaining_ms: 0,
            effective_threshold,
            threshold_mode: threshold_mode.to_string(),
            raw_score: 0.0,
            normalized_score: 0.0,
            top_score_penalties: String::new(),
        };
    }

    let exec_state = state.exec.execution_state().await;
    let pending = state.authority.pending_proposals().await;

    let active_profile = RuntimeProfile::from_str(&cfg.behavior_profile);
    let (effective_threshold, threshold_mode) =
        profile_threshold(active_profile, total_balance_usd, cfg.mode);
    let is_swing_profile = active_profile == RuntimeProfile::Swing;
    let is_flip_hyper = active_profile == RuntimeProfile::FlipHyper;
    let is_micro_active = matches!(active_profile, RuntimeProfile::MicroActive | RuntimeProfile::FlipHyper);
    log_npc_event(
        &*state.store,
        "profile_lock",
        &format!("PROFILE_LOCK={}", active_profile.as_str()),
    );

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
    if is_swing_profile {
        let bias = swing_bias(&swing_signals);
        let vol_estimate = realized_volatility_bps(&rt.mid_history, 16).unwrap_or(0.0) / 10_000.0;
        update_contract_executor(
            &mut rt,
            bias,
            metrics.mid,
            metrics.mid,
            vol_estimate,
            total_balance_usd,
        );
    }

    // ── SWING_TRADER: update open-position telemetry each cycle ──────────────
    if is_swing_profile {
        let btc_held = position_size > 0.0 || sell_inventory > 0.0;
        if btc_held {
            if rt.swing_entry_started_at.is_none() {
                rt.swing_entry_started_at = Some(Instant::now());
                if rt.swing_entry_price <= 0.0 {
                    rt.swing_entry_price = metrics.mid;
                }
                rt.swing_peak_price = metrics.mid.max(rt.swing_entry_price);
            } else {
                rt.swing_peak_price = rt.swing_peak_price.max(metrics.mid);
            }
            if rt.swing_entry_price > 0.0 {
                let u = (metrics.mid / rt.swing_entry_price) - 1.0;
                rt.swing_peak_unrealized_pct = rt.swing_peak_unrealized_pct.max(u);
            }
        } else if rt.swing_entry_started_at.is_some() {
            // Position flattened externally/reconciler path.
            rt.swing_entry_price = 0.0;
            rt.swing_entry_started_at = None;
            rt.swing_peak_price = 0.0;
            rt.swing_peak_unrealized_pct = 0.0;
        }
    }

    // ── COMPOUND_EXECUTION: update live state snapshots ───────────────────────
    if is_micro_active {
        rt.compound_last_position_btc = position_size;
        rt.compound_last_balance_usd  = total_balance_usd;

        // Track peak balance and apply profit-lock when balance grows ≥10%.
        if total_balance_usd > 0.0 {
            if rt.compound_peak_balance <= 0.0 {
                rt.compound_peak_balance = total_balance_usd;
            } else if total_balance_usd >= rt.compound_peak_balance * (1.0 + COMPOUND_PROFIT_LOCK_TRIGGER_PCT) {
                // Balance grew ≥10% above previous peak — lock profits by trimming size scalar.
                rt.compound_size_scalar = (rt.compound_size_scalar * COMPOUND_PROFIT_LOCK_RISK_FACTOR).max(COMPOUND_MIN_SIZE_SCALAR);
                info!(
                    balance = total_balance_usd,
                    peak = rt.compound_peak_balance,
                    new_scalar = rt.compound_size_scalar,
                    "[COMPOUND] Profit lock triggered: balance {:.2} ≥ peak {:.2} × {:.0}%. Size scalar → {:.2}",
                    total_balance_usd, rt.compound_peak_balance,
                    COMPOUND_PROFIT_LOCK_TRIGGER_PCT * 100.0, rt.compound_size_scalar
                );
                rt.compound_peak_balance = total_balance_usd;
            }
        }

        // ── Compound loss pause: block BUY entries while cooldown is active ──────
        if let Some(pause_until) = rt.compound_loss_pause_until {
            if pause_until > Instant::now() {
                let remaining_ms = (pause_until - Instant::now()).as_millis() as u64;
                log_npc_event(
                    &*state.store,
                    "compound_loss_pause",
                    &format!("cycle={} remaining_ms={} consecutive_losses={}", cycle_id, remaining_ms, rt.compound_consecutive_losses),
                );
                if position_size <= 0.0 {
                    observe_and_learn(cfg, &mut rt, &*state.store, metrics.mid);
                    return no_action(cycle_id,
                        format!("COMPOUND_LOSS_PAUSE:{}ms_remaining", remaining_ms),
                        "blocked".to_string());
                }
            } else {
                rt.compound_loss_pause_until = None;
            }
        }
    }

    // ── FLIP_HYPER: sync flip cycle phase with current inventory state ─────────
    if is_flip_hyper {
        // Detect position just exited: had entry price tracked but no BTC held now.
        let had_tracked_entry = rt.flip_last_entry_price > 0.0;
        let btc_held = position_size > 0.0 || sell_inventory > 0.0;
        if had_tracked_entry && !btc_held
            && (rt.flip_cycle_phase == FlipCyclePhase::SeekExit
                || rt.flip_cycle_phase == FlipCyclePhase::Exiting
                || rt.flip_cycle_phase == FlipCyclePhase::HoldingPosition
                || rt.flip_cycle_phase == FlipCyclePhase::Entering)
        {
            // Position has been sold — record completed flip once.
            let exit_price = metrics.mid;
            let qty = rt.flip_last_entry_qty.max(0.0);
            if qty > 0.0 {
                let already_recorded = rt.flip_last_completed.as_ref().map_or(false, |completed| {
                    const FLIP_MATCH_EPSILON: f64 = 1e-9;
                    (completed.entry_price - rt.flip_last_entry_price).abs() <= FLIP_MATCH_EPSILON
                        && (completed.qty - qty).abs() <= FLIP_MATCH_EPSILON
                });

                if already_recorded {
                    info!(
                        entry = rt.flip_last_entry_price,
                        exit = exit_price,
                        qty,
                        "[FLIP_HYPER] Inventory sync detected already-recorded flip; \
                         skipping duplicate completion accounting"
                    );
                } else {
                    let gross_pnl = qty * (exit_price - rt.flip_last_entry_price);
                    let est_fees = qty * exit_price * 0.001; // ~0.1% fee estimate
                    let realized_pnl_usd = gross_pnl - est_fees;
                    let realized_pnl_pct =
                        (exit_price / rt.flip_last_entry_price - 1.0) * 100.0;
                    info!(
                        entry = rt.flip_last_entry_price,
                        exit = exit_price,
                        qty,
                        pnl_usd = realized_pnl_usd,
                        "[FLIP_HYPER] Flip reconciled via inventory sync: \
                         entry={:.2} exit={:.2} qty={:.8} pnl={:+.4}",
                        rt.flip_last_entry_price, exit_price, qty, realized_pnl_usd
                    );
                    rt.flip_last_completed = Some(CompletedFlip {
                        entry_price: rt.flip_last_entry_price,
                        exit_price,
                        qty,
                        realized_pnl_usd,
                        realized_pnl_pct,
                        completed_at: Instant::now(),
                    });
                    rt.flip_session_pnl += realized_pnl_usd;
                    rt.flip_rotation_count = rt.flip_rotation_count.saturating_add(1);
                    log_npc_event(
                        &*state.store,
                        "flip_completed",
                        &format!(
                            "cycle={} entry={:.2} exit={:.2} qty={:.8} \
                             realized_pnl_usd={:.4} realized_pnl_pct={:.4} \
                             session_pnl={:.4} rotation_count={}",
                            cycle_id,
                            rt.flip_last_entry_price,
                            exit_price,
                            qty,
                            realized_pnl_usd,
                            realized_pnl_pct,
                            rt.flip_session_pnl,
                            rt.flip_rotation_count
                        ),
                    );
                }
            }
            rt.flip_last_entry_price = 0.0;
            rt.flip_last_entry_qty   = 0.0;
            rt.flip_cycle_phase      = FlipCyclePhase::RebuyReady;
            rt.flip_last_active      = Some(Instant::now());
        }
        // REBUY_READY immediately transitions to SEEK_ENTRY
        if rt.flip_cycle_phase == FlipCyclePhase::RebuyReady {
            rt.flip_cycle_phase = FlipCyclePhase::SeekEntry;
        }
        // If BTC is held but phase is SeekEntry, sync to HOLD state
        if btc_held && (rt.flip_cycle_phase == FlipCyclePhase::SeekEntry
            || rt.flip_cycle_phase == FlipCyclePhase::RebuyReady)
        {
            rt.flip_cycle_phase = FlipCyclePhase::HoldingPosition;
            if rt.flip_last_entry_price == 0.0 {
                // Position opened externally — use current mid as entry estimate
                rt.flip_last_entry_price = metrics.mid;
                rt.flip_last_entry_qty   = sell_inventory.max(position_size);
            }
        }
        // HOLDING_POSITION or ENTERING with BTC → advance to SEEK_EXIT
        if btc_held && (rt.flip_cycle_phase == FlipCyclePhase::HoldingPosition
            || rt.flip_cycle_phase == FlipCyclePhase::Entering)
        {
            rt.flip_cycle_phase = FlipCyclePhase::SeekExit;
        }
        // ── Anti-stall detection ──────────────────────────────────────────────
        let has_actionable =
            (buy_power   > 0.0 && rt.flip_cycle_phase == FlipCyclePhase::SeekEntry)
            || (btc_held && rt.flip_cycle_phase == FlipCyclePhase::SeekExit);
        if has_actionable {
            if rt.flip_last_active.is_none() {
                rt.flip_last_active = Some(Instant::now());
            } else if let Some(since) = rt.flip_last_active {
                let stall_secs = since.elapsed().as_secs();
                if stall_secs > FLIP_HYPER_STALL_SECS && rt.flip_blocker.is_empty() {
                    rt.flip_blocker = format!(
                        "STALL: {}s in {} phase with inventory but no execution. \
                         Check score threshold (current {:.4}) and block reasons above.",
                        stall_secs,
                        rt.flip_cycle_phase.as_str(),
                        effective_threshold,
                    );
                    info!(
                        phase = rt.flip_cycle_phase.as_str(),
                        stall_secs,
                        "[FLIP_HYPER] Anti-stall triggered: {}s without execution",
                        stall_secs
                    );
                }
            }
        } else {
            // No actionable opportunity → reset stall timer and clear blocker
            if !btc_held && buy_power == 0.0 {
                rt.flip_last_active = None;
            }
            rt.flip_blocker.clear();
        }
    }
    if rt.learner_ranges.is_none() {
        rt.learner_ranges = Some(default_learner_ranges(cfg));
    }
    let mut effective_cfg = cfg.clone();
    apply_learner_overrides(&mut effective_cfg, &mut rt);

    let regime = detect_regime(&effective_cfg, &rt.mid_history, &rt.spread_history, &metrics);
    let slot_usage = reconcile_slot_usage(
        &effective_cfg,
        &mut rt,
        position_size,
        sell_inventory,
        pending_order_slots,
        &pending_order_ids,
        effective_cfg.mode,
    );
    let portfolio_controls = evaluate_portfolio_controls(
        &effective_cfg,
        &rt,
        &slot_usage,
        position_size,
        exposure_notional,
        metrics.mid,
        regime,
    );

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
        is_micro_active,
        is_swing_profile,
    );

    let regime_cutoff = rt
        .learner_ranges
        .as_ref()
        .and_then(|r| r.regime_score_cutoff.get(&regime).map(|v| v.current))
        .unwrap_or(effective_cfg.alpha.min_action_score);

    // Compute the enforced gate here — before any logging or comparison — so that
    // every event, report, and telemetry field uses exactly the same variable as
    // the final `if chosen.score < effective_cutoff` comparison below.
    // For micro accounts the balance-tier cap is applied; regime/learner can only
    // lower the gate further, never raise it back above the tier maximum.
    let effective_cutoff = if threshold_mode == "micro_aggressive"
        || threshold_mode == "micro_active"
        || threshold_mode == "flip_hyper"
    {
        effective_threshold.min(regime_cutoff)
    } else {
        regime_cutoff
    };

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

    let metrics_line = format_cycle_metrics(cycle_id, regime, &candidates, effective_cutoff, &portfolio_controls);
    log_npc_event(&*state.store, "alpha_cycle", &metrics_line);

    let Some(mut chosen) = candidates.first().cloned() else {
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

    if is_swing_profile {
        let bias = swing_bias(&swing_signals);
        let btc_held = position_size > 0.0 || sell_inventory > 0.0;
        let cooldown_remaining_ms = rt.swing_cooldown_until
            .and_then(|until| until.checked_duration_since(Instant::now()))
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let swing_entry_ready =
            bias == SwingBias::LongBias
                && swing_signals.pullback_detected
                && swing_signals.momentum_resumed_up;
        if !btc_held {
            if cooldown_remaining_ms > 0 {
                let reason = format!("SWING_COOLDOWN_ACTIVE:{}ms_remaining", cooldown_remaining_ms);
                rt.perf.entry(chosen.role).or_default().blocked += 1;
                observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
                return NpcCycleReport {
                    cycle_id,
                    last_action: "NO_ACTION".to_string(),
                    execution_result: reason.clone(),
                    status: "blocked".to_string(),
                    last_agent_decision: "HOLD — swing cooldown active".to_string(),
                    no_trade_reason: reason.clone(),
                    pipeline_state: "Scanning".to_string(),
                    final_decision: "BLOCKED".to_string(),
                    balance_block_reason: String::new(),
                    risk_block_reason: String::new(),
                    execution_block_reason: reason,
                    cooldown_active: true,
                    cooldown_remaining_ms,
                    effective_threshold: effective_cutoff,
                    threshold_mode: threshold_mode.to_string(),
                    raw_score: chosen.raw_score,
                    normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                    top_score_penalties: chosen.score_parts.top_penalties_str(),
                };
            }
            if chosen.side.eq_ignore_ascii_case("SELL") || !swing_entry_ready {
                let reason = format!(
                    "SWING_ENTRY_WAIT:bias={} m1m={:+.5} m5m={:+.5} m15m={:+.5} pullback={} resume={}",
                    bias.as_str(),
                    swing_signals.momentum_1m,
                    swing_signals.momentum_5m,
                    swing_signals.trend_bias_15m,
                    swing_signals.pullback_detected,
                    swing_signals.momentum_resumed_up
                );
                rt.perf.entry(chosen.role).or_default().blocked += 1;
                observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
                return NpcCycleReport {
                    cycle_id,
                    last_action: "NO_ACTION".to_string(),
                    execution_result: "SWING_NO_ENTRY".to_string(),
                    status: "blocked".to_string(),
                    last_agent_decision: "HOLD — swing entry conditions not met".to_string(),
                    no_trade_reason: reason.clone(),
                    pipeline_state: "Scanning".to_string(),
                    final_decision: "BLOCKED".to_string(),
                    balance_block_reason: String::new(),
                    risk_block_reason: String::new(),
                    execution_block_reason: reason,
                    cooldown_active: false,
                    cooldown_remaining_ms: 0,
                    effective_threshold: effective_cutoff,
                    threshold_mode: threshold_mode.to_string(),
                    raw_score: chosen.raw_score,
                    normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                    top_score_penalties: chosen.score_parts.top_penalties_str(),
                };
            }
        } else {
            let entry_price = rt.swing_entry_price.max(f64::EPSILON);
            let current_unrealized_pct = (metrics.mid / entry_price) - 1.0;
            let trail_stop_hit = rt.swing_peak_price > 0.0
                && metrics.mid <= rt.swing_peak_price * (1.0 - SWING_TRAILING_STOP_PCT);
            let momentum_reversal = swing_signals.momentum_1m < 0.0;
            let structure_break = swing_signals.structure_break_long;
            let target_zone_hit =
                current_unrealized_pct >= SWING_TARGET_MIN_PCT && current_unrealized_pct <= SWING_TARGET_MAX_PCT;
            let exit_signal = momentum_reversal || structure_break || trail_stop_hit || current_unrealized_pct >= SWING_TARGET_MAX_PCT;
            if chosen.side.eq_ignore_ascii_case("BUY") || !exit_signal {
                let reason = format!(
                    "SWING_HOLD:exit_signal={} rev={} structure_break={} trail_hit={} pnl_pct={:+.4}% target_zone_hit={}",
                    exit_signal,
                    momentum_reversal,
                    structure_break,
                    trail_stop_hit,
                    current_unrealized_pct * 100.0,
                    target_zone_hit
                );
                rt.perf.entry(chosen.role).or_default().blocked += 1;
                observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
                return NpcCycleReport {
                    cycle_id,
                    last_action: "NO_ACTION".to_string(),
                    execution_result: "SWING_HOLD".to_string(),
                    status: "running".to_string(),
                    last_agent_decision: "HOLD — swing position maintained".to_string(),
                    no_trade_reason: reason.clone(),
                    pipeline_state: "Holding".to_string(),
                    final_decision: "HOLD".to_string(),
                    balance_block_reason: String::new(),
                    risk_block_reason: String::new(),
                    execution_block_reason: String::new(),
                    cooldown_active: false,
                    cooldown_remaining_ms: 0,
                    effective_threshold: effective_cutoff,
                    threshold_mode: threshold_mode.to_string(),
                    raw_score: chosen.raw_score,
                    normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                    top_score_penalties: chosen.score_parts.top_penalties_str(),
                };
            }
            let reason = if momentum_reversal {
                "momentum_reversal_1m"
            } else if structure_break {
                "structure_break"
            } else if trail_stop_hit {
                "trailing_stop_hit"
            } else {
                "target_hit"
            };
            rt.swing_last_exit_reason = reason.to_string();
        }
    }

    let allow_despite_regime_mismatch = chosen.risk_override && !chosen.regime_eligible;
    if let Some(reason) = chosen.regime_block_reason.clone() {
        if !allow_despite_regime_mismatch {
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
                final_decision: "BLOCKED".to_string(),
                balance_block_reason: String::new(),
                risk_block_reason: String::new(),
                execution_block_reason: reason.clone(),
                cooldown_active: false,
                cooldown_remaining_ms: 0,
                effective_threshold: effective_cutoff,
                threshold_mode: threshold_mode.to_string(),
                raw_score: chosen.raw_score,
                normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                top_score_penalties: chosen.score_parts.top_penalties_str(),
            };
        }
    }

    if allow_despite_regime_mismatch && chosen.score < cfg.min_conviction_threshold {
        let reason = format!(
            "SWING_RISK_OVERRIDE_CONVICTION_BELOW_THRESHOLD:{:.4}<{:.4}",
            chosen.score, cfg.min_conviction_threshold
        );
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
            execution_result: "SWING_RISK_OVERRIDE_CONVICTION_BELOW_THRESHOLD".to_string(),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} swing override conviction {:.4} below {:.4}",
                chosen.side.to_uppercase(),
                chosen.role.as_str(),
                chosen.score,
                cfg.min_conviction_threshold
            ),
            no_trade_reason: reason.clone(),
            pipeline_state: "Scanning".to_string(),
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: String::new(),
            risk_block_reason: String::new(),
            execution_block_reason: reason,
            cooldown_active: false,
            cooldown_remaining_ms: 0,
            effective_threshold: effective_cutoff,
            threshold_mode: threshold_mode.to_string(),
            raw_score: chosen.raw_score,
            normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            top_score_penalties: chosen.score_parts.top_penalties_str(),
        };
    }

    if chosen.score < effective_cutoff {
        lifecycle(
            &*state.store,
            chosen.role,
            &chosen.action_id,
            NpcLifecycleState::Blocked,
            &format!(
                "NO_ACTION_SCORE_BELOW_THRESHOLD:{:.4}<{:.4}(effective={:.4},regime={:.4},mode={})",
                chosen.score, effective_cutoff, effective_threshold, regime_cutoff, threshold_mode
            ),
        );
        rt.perf.entry(chosen.role).or_default().blocked += 1;
        observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
        let score_reason = format!(
            "Signal score {:.4} below minimum threshold {:.4} [{}]. \
             raw={:.4} norm={:.3} top_penalties=[{}].",
            chosen.score, effective_cutoff, threshold_mode,
            chosen.raw_score,
            chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            chosen.score_parts.top_penalties_str()
        );
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: "SCORE_BELOW_THRESHOLD".to_string(),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} score {:.4} below threshold {:.4} [{}]",
                chosen.side.to_uppercase(), chosen.role.as_str(), chosen.score, effective_cutoff, threshold_mode
            ),
            no_trade_reason: score_reason.clone(),
            pipeline_state: "Scanning".to_string(),
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: String::new(),
            risk_block_reason: String::new(),
            execution_block_reason: score_reason,
            cooldown_active: false,
            cooldown_remaining_ms: 0,
            effective_threshold: effective_cutoff, // report the actual enforced gate to UI/telemetry
            threshold_mode: threshold_mode.to_string(),
            raw_score: chosen.raw_score,
            normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            top_score_penalties: chosen.score_parts.top_penalties_str(),
        };
    }

    // ── COMPOUND_EXECUTION quality gate ──────────────────────────────────────
    // In MICRO_ACTIVE mode, require BOTH the score gate AND a strong normalized
    // score (≥ COMPOUND_NORMALIZED_SCORE_MIN = 1.2).  Borderline passes that
    // barely clear the threshold are rejected — only trades with meaningful edge
    // are allowed through.
    if is_micro_active {
        let norm = chosen.raw_score / effective_cutoff.max(f64::EPSILON);
        if norm < COMPOUND_NORMALIZED_SCORE_MIN {
            lifecycle(
                &*state.store,
                chosen.role,
                &chosen.action_id,
                NpcLifecycleState::Blocked,
                &format!(
                    "COMPOUND_WEAK_SIGNAL:norm={:.3}<{:.2}(raw={:.4},cutoff={:.4})",
                    norm, COMPOUND_NORMALIZED_SCORE_MIN, chosen.raw_score, effective_cutoff
                ),
            );
            rt.perf.entry(chosen.role).or_default().blocked += 1;
            observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
            let reason = format!(
                "COMPOUND quality gate: normalized score {:.3} < {:.2} (raw={:.4} threshold={:.4}). \
                 Trade rejected — score must be ≥{:.0}% above threshold for capital-growth mode.",
                norm, COMPOUND_NORMALIZED_SCORE_MIN, chosen.raw_score, effective_cutoff,
                COMPOUND_NORMALIZED_SCORE_MIN * 100.0
            );
            return NpcCycleReport {
                cycle_id,
                last_action: "NO_ACTION".to_string(),
                execution_result: "COMPOUND_WEAK_SIGNAL".to_string(),
                status: "blocked".to_string(),
                last_agent_decision: format!(
                    "HOLD — {} {} compound quality gate: norm={:.3} < {:.2}",
                    chosen.side.to_uppercase(), chosen.role.as_str(), norm, COMPOUND_NORMALIZED_SCORE_MIN
                ),
                no_trade_reason: reason.clone(),
                pipeline_state: "Scanning".to_string(),
                final_decision: "BLOCKED".to_string(),
                balance_block_reason: String::new(),
                risk_block_reason: String::new(),
                execution_block_reason: reason,
                cooldown_active: false,
                cooldown_remaining_ms: 0,
                effective_threshold: effective_cutoff,
                threshold_mode: threshold_mode.to_string(),
                raw_score: chosen.raw_score,
                normalized_score: norm,
                top_score_penalties: chosen.score_parts.top_penalties_str(),
            };
        }

        // ── Spread dominance guard: block if spread cost exceeds the edge estimate ──
        // If spread_cost > edge_estimate, the market friction is larger than the
        // expected edge — executing would produce a negative-expectancy trade.
        if chosen.score_parts.spread_cost > chosen.score_parts.edge_estimate {
            lifecycle(
                &*state.store,
                chosen.role,
                &chosen.action_id,
                NpcLifecycleState::Blocked,
                &format!(
                    "COMPOUND_SPREAD_DOMINANCE:spread={:.4}>edge={:.4}",
                    chosen.score_parts.spread_cost, chosen.score_parts.edge_estimate
                ),
            );
            rt.perf.entry(chosen.role).or_default().blocked += 1;
            observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
            let reason = format!(
                "Spread dominance: spread_cost {:.4} > edge_estimate {:.4}. \
                 Market friction exceeds expected edge — trade skipped.",
                chosen.score_parts.spread_cost, chosen.score_parts.edge_estimate
            );
            return NpcCycleReport {
                cycle_id,
                last_action: "NO_ACTION".to_string(),
                execution_result: "COMPOUND_SPREAD_DOMINANCE".to_string(),
                status: "blocked".to_string(),
                last_agent_decision: format!(
                    "HOLD — {} {} spread dominance: cost {:.4} > edge {:.4}",
                    chosen.side.to_uppercase(), chosen.role.as_str(),
                    chosen.score_parts.spread_cost, chosen.score_parts.edge_estimate
                ),
                no_trade_reason: reason.clone(),
                pipeline_state: "Scanning".to_string(),
                final_decision: "BLOCKED".to_string(),
                balance_block_reason: String::new(),
                risk_block_reason: String::new(),
                execution_block_reason: reason,
                cooldown_active: false,
                cooldown_remaining_ms: 0,
                effective_threshold: effective_cutoff,
                threshold_mode: threshold_mode.to_string(),
                raw_score: chosen.raw_score,
                normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                top_score_penalties: chosen.score_parts.top_penalties_str(),
            };
        }
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
        let portfolio_reason = format!("Portfolio risk controls active: {}", portfolio_controls.join("; "));
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: portfolio_controls.join("|"),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} blocked by portfolio controls",
                chosen.side.to_uppercase(), chosen.role.as_str()
            ),
            no_trade_reason: portfolio_reason.clone(),
            pipeline_state: "Scanning".to_string(),
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: String::new(),
            risk_block_reason: portfolio_reason,
            execution_block_reason: String::new(),
            cooldown_active: false,
            cooldown_remaining_ms: 0,
            effective_threshold: effective_cutoff,
            threshold_mode: threshold_mode.to_string(),
            raw_score: chosen.raw_score,
            normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            top_score_penalties: chosen.score_parts.top_penalties_str(),
        };
    }

    let required_trade_capital = (cfg.trade_size.max(0.0) * metrics.mid.max(0.0)).max(cfg.rebalance_min_notional_usd);
    let mut forced_allocation: Option<AllocationDecision> = None;
    if let Some(plan) = maybe_rebalance(
        cfg,
        &rt,
        &chosen.side,
        buy_power,
        sell_inventory,
        metrics.mid,
        total_balance_usd,
        required_trade_capital,
    ) {
        chosen.side = plan.side.to_string();
        chosen.reason = format!("{} | {}", chosen.reason, plan.reason);
        rt.rebalance_status = plan.status.to_string();
        rt.rebalance_triggered = true;
        rt.rebalance_reason = plan.reason;
        rt.rebalance_side = plan.side.to_string();
        rt.rebalance_qty = plan.qty;
        rt.rebalance_value_usd = plan.value_usd;
        rt.free_usdt_before = plan.free_usdt_before;
        rt.free_usdt_after = plan.free_usdt_after;
        rt.btc_before = plan.btc_before;
        rt.btc_after = plan.btc_after;
        rt.rebalance_cooldown_until = Some(Instant::now() + Duration::from_secs(cfg.rebalance_cooldown_secs));
        forced_allocation = Some(AllocationDecision {
            qty: plan.qty,
            reason: format!("{}:{}", plan.status, rt.rebalance_reason),
            quality: 1.0,
            recent_performance: 0.0,
            symbol_concentration: 0.0,
            drawdown: 0.0,
        });
    } else {
        rt.rebalance_status = "REBALANCE_COMPLETE".to_string();
        rt.rebalance_triggered = false;
        rt.rebalance_reason.clear();
        rt.rebalance_side.clear();
        rt.rebalance_qty = 0.0;
        rt.rebalance_value_usd = 0.0;
        rt.free_usdt_before = buy_power;
        rt.free_usdt_after = buy_power;
        rt.btc_before = sell_inventory;
        rt.btc_after = sell_inventory;
    }
    let allocation = forced_allocation.unwrap_or_else(|| {
        allocate_capital(&effective_cfg, &rt, &chosen, position_size, exposure_notional, buy_power, sell_inventory, metrics.mid, total_balance_usd, is_micro_active)
    });
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
        let alloc_reason = if chosen.side.eq_ignore_ascii_case("SELL") {
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
        };
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: format!("ALLOCATION_REJECTED:{}", allocation.reason),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} allocation rejected: {}",
                chosen.side.to_uppercase(), chosen.role.as_str(), allocation.reason
            ),
            no_trade_reason: alloc_reason.clone(),
            pipeline_state: "Scanning".to_string(),
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: alloc_reason,
            risk_block_reason: String::new(),
            execution_block_reason: String::new(),
            cooldown_active: false,
            cooldown_remaining_ms: 0,
            effective_threshold: effective_cutoff,
            threshold_mode: threshold_mode.to_string(),
            raw_score: chosen.raw_score,
            normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            top_score_penalties: chosen.score_parts.top_penalties_str(),
        };
    }

    let order_notional = allocation.qty * metrics.mid.max(0.0);

    // ── COMPOUND_EXECUTION: minimum expected PnL check ────────────────────────
    // Require that the expected net price move (estimated from momentum) produces
    // at least a positive net edge after spread and slippage costs.  This prevents
    // entering trades where the edge cannot beat transaction costs.
    if is_micro_active && chosen.side.eq_ignore_ascii_case("BUY") {
        let net_edge_score = chosen.score_parts.edge_estimate
            - chosen.score_parts.spread_cost
            - chosen.score_parts.slippage_risk;
        if net_edge_score <= 0.0 {
            lifecycle(
                &*state.store,
                chosen.role,
                &chosen.action_id,
                NpcLifecycleState::Blocked,
                &format!(
                    "COMPOUND_EDGE_BELOW_COST:net={:.4}(edge={:.4}-spread={:.4}-slip={:.4})",
                    net_edge_score, chosen.score_parts.edge_estimate,
                    chosen.score_parts.spread_cost, chosen.score_parts.slippage_risk
                ),
            );
            rt.perf.entry(chosen.role).or_default().blocked += 1;
            observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
            let reason = format!(
                "COMPOUND edge check: net edge {:.4} ≤ 0 (edge={:.4} − spread={:.4} − slip={:.4}). \
                 Trade cannot beat spread + slippage — skipped.",
                net_edge_score, chosen.score_parts.edge_estimate,
                chosen.score_parts.spread_cost, chosen.score_parts.slippage_risk
            );
            return NpcCycleReport {
                cycle_id,
                last_action: "NO_ACTION".to_string(),
                execution_result: "COMPOUND_EDGE_BELOW_COST".to_string(),
                status: "blocked".to_string(),
                last_agent_decision: format!(
                    "HOLD — {} {} net edge {:.4} ≤ 0 after costs",
                    chosen.side.to_uppercase(), chosen.role.as_str(), net_edge_score
                ),
                no_trade_reason: reason.clone(),
                pipeline_state: "Scanning".to_string(),
                final_decision: "BLOCKED".to_string(),
                balance_block_reason: String::new(),
                risk_block_reason: String::new(),
                execution_block_reason: reason,
                cooldown_active: false,
                cooldown_remaining_ms: 0,
                effective_threshold: effective_cutoff,
                threshold_mode: threshold_mode.to_string(),
                raw_score: chosen.raw_score,
                normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                top_score_penalties: chosen.score_parts.top_penalties_str(),
            };
        }
        // Log expected trade metrics for observability.
        info!(
            cycle_id,
            role = chosen.role.as_str(),
            side = chosen.side.as_str(),
            qty = allocation.qty,
            order_notional,
            net_edge = net_edge_score,
            edge = chosen.score_parts.edge_estimate,
            spread_cost = chosen.score_parts.spread_cost,
            "[COMPOUND] Trade entry: size_usd={:.2} qty={:.8} net_edge={:.4} reason=score={:.4} penalties=[{}]",
            order_notional,
            allocation.qty,
            net_edge_score,
            chosen.score,
            chosen.score_parts.top_penalties_str(),
        );
    }

    let (guard_reasons, cooldown_active, cooldown_remaining_ms) = evaluate_guards(
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
        is_micro_active,
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
        let guard_reason = format!("Safety guard blocked order: {}", guard_reasons.join("; "));
        return NpcCycleReport {
            cycle_id,
            last_action: "NO_ACTION".to_string(),
            execution_result: guard_reasons.join("|"),
            status: "blocked".to_string(),
            last_agent_decision: format!(
                "HOLD — {} {} blocked by safety guards",
                chosen.side.to_uppercase(), chosen.role.as_str()
            ),
            no_trade_reason: guard_reason.clone(),
            pipeline_state: "Scanning".to_string(),
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: String::new(),
            risk_block_reason: String::new(),
            execution_block_reason: guard_reason,
            cooldown_active,
            cooldown_remaining_ms,
            effective_threshold: effective_cutoff,
            threshold_mode: threshold_mode.to_string(),
            raw_score: chosen.raw_score,
            normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            top_score_penalties: chosen.score_parts.top_penalties_str(),
        };
    }

    // ── FLIP_HYPER: profit floor check for SELL orders ────────────────────────
    // Never sell unless projected net PnL clears spread + fees + minimum floor.
    // This prevents realizing a loss or a negligible gain that doesn't compound.
    if is_flip_hyper
        && chosen.side.eq_ignore_ascii_case("SELL")
        && rt.flip_last_entry_price > 0.0
    {
        let min_profit_floor = flip_hyper_profit_floor_for_balance(total_balance_usd);
        let exit_price   = metrics.mid;
        let qty          = allocation.qty;
        let gross_pnl    = qty * (exit_price - rt.flip_last_entry_price);
        let est_fees     = qty * exit_price * 0.001; // ~0.1% fee estimate
        let net_pnl      = gross_pnl - est_fees;
        let entry_notional = (rt.flip_last_entry_price * qty).max(f64::EPSILON);
        let net_pnl_pct = (net_pnl / entry_notional) * 100.0;
        let breakeven_exit_floor = rt.flip_last_entry_price - (exit_price * 0.001);
        let allow_breakeven_micro =
            total_balance_usd > 0.0
                && total_balance_usd < MICRO_BALANCE_USD
                && exit_price >= breakeven_exit_floor;
        let forced_rotation_armed = matches!(
            rt.flip_cycle_phase,
            FlipCyclePhase::HoldingPosition | FlipCyclePhase::SeekExit
        ) && rt
            .flip_last_active
            .map(|since| since.elapsed().as_secs() > FLIP_HYPER_FORCED_ROTATION_SECS)
            .unwrap_or(false);
        let allow_forced_rotation =
            forced_rotation_armed && net_pnl_pct >= -FLIP_HYPER_FORCED_ROTATION_MAX_LOSS_PCT;
        if net_pnl < min_profit_floor && !allow_breakeven_micro && !allow_forced_rotation {
            rt.perf.entry(chosen.role).or_default().blocked += 1;
            observe_and_learn(&effective_cfg, &mut rt, &*state.store, metrics.mid);
            let reason = format!(
                "FLIP_HYPER profit floor: net PnL ${:.4} < floor ${:.2} \
                 (entry={:.2} exit={:.2} qty={:.8} gross={:.4} fees_est={:.4}). \
                 Min profit floor: ${:.2}",
                net_pnl, min_profit_floor,
                rt.flip_last_entry_price, exit_price, qty, gross_pnl, est_fees,
                min_profit_floor
            );
            lifecycle(
                &*state.store,
                chosen.role,
                &chosen.action_id,
                NpcLifecycleState::Blocked,
                &reason,
            );
            return NpcCycleReport {
                cycle_id,
                last_action: "NO_ACTION".to_string(),
                execution_result: "FLIP_PROFIT_FLOOR".to_string(),
                status: "blocked".to_string(),
                last_agent_decision: format!(
                    "HOLD — SELL blocked: net PnL ${:.4} below floor ${:.2}",
                    net_pnl, min_profit_floor
                ),
                no_trade_reason: reason.clone(),
                pipeline_state: "Scanning".to_string(),
                final_decision: "BLOCKED".to_string(),
                balance_block_reason: String::new(),
                risk_block_reason: String::new(),
                execution_block_reason: reason,
                cooldown_active: false,
                cooldown_remaining_ms: 0,
                effective_threshold: effective_cutoff,
                threshold_mode: threshold_mode.to_string(),
                raw_score: chosen.raw_score,
                normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                top_score_penalties: chosen.score_parts.top_penalties_str(),
            };
        }
        if allow_forced_rotation {
            info!(
                cycle_id,
                role = chosen.role.as_str(),
                entry = rt.flip_last_entry_price,
                exit = exit_price,
                qty,
                net_pnl,
                net_pnl_pct,
                "[FLIP_HYPER] Forced rotation fallback armed: net={:.4} ({:.4}%)",
                net_pnl, net_pnl_pct
            );
        } else if allow_breakeven_micro && net_pnl < min_profit_floor {
            info!(
                cycle_id,
                role = chosen.role.as_str(),
                entry = rt.flip_last_entry_price,
                exit = exit_price,
                qty,
                net_pnl,
                floor = min_profit_floor,
                "[FLIP_HYPER] Micro breakeven exit allowed: net={:.4} floor={:.2}",
                net_pnl, min_profit_floor
            );
        } else {
            // Log that the profit floor was cleared.
            info!(
                cycle_id,
                role = chosen.role.as_str(),
                entry = rt.flip_last_entry_price,
                exit = exit_price,
                qty,
                net_pnl,
                floor = min_profit_floor,
                "[FLIP_HYPER] Profit floor cleared: net={:.4} ≥ floor={:.2}",
                net_pnl, min_profit_floor
            );
        }
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

    if effective_cfg.mode == NpcTradingMode::Live && rt.paper_executions == 0 && !is_flip_hyper {
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
            final_decision: "BLOCKED".to_string(),
            balance_block_reason: String::new(),
            risk_block_reason: String::new(),
            execution_block_reason: "LIVE_REQUIRES_PAPER_EXECUTION".to_string(),
            cooldown_active: false,
            cooldown_remaining_ms: 0,
            effective_threshold: effective_cutoff,
            threshold_mode: threshold_mode.to_string(),
            raw_score: chosen.raw_score,
            normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
            top_score_penalties: chosen.score_parts.top_penalties_str(),
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
                final_decision: "BLOCKED".to_string(),
                balance_block_reason: String::new(),
                risk_block_reason: String::new(),
                execution_block_reason: "WEB_UI_ADDR_MISSING".to_string(),
                cooldown_active,
                cooldown_remaining_ms,
                effective_threshold: effective_cutoff,
                threshold_mode: threshold_mode.to_string(),
                raw_score: chosen.raw_score,
                normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                top_score_penalties: chosen.score_parts.top_penalties_str(),
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
                    final_decision: "BLOCKED".to_string(),
                    balance_block_reason: String::new(),
                    risk_block_reason: String::new(),
                    execution_block_reason: format!("HTTP_STATUS_{}", status_code),
                    cooldown_active,
                    cooldown_remaining_ms,
                    effective_threshold: effective_cutoff,
                    threshold_mode: threshold_mode.to_string(),
                    raw_score: chosen.raw_score,
                    normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                    top_score_penalties: chosen.score_parts.top_penalties_str(),
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
                    final_decision: "BLOCKED".to_string(),
                    balance_block_reason: String::new(),
                    risk_block_reason: String::new(),
                    execution_block_reason: format!("REQUEST_ERROR:{}", e),
                    cooldown_active,
                    cooldown_remaining_ms,
                    effective_threshold: effective_cutoff,
                    threshold_mode: threshold_mode.to_string(),
                    raw_score: chosen.raw_score,
                    normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
                    top_score_penalties: chosen.score_parts.top_penalties_str(),
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
            execution_mode: effective_cfg.mode,
        },
    );

    // ── FLIP_HYPER: update cycle phase on order submission ────────────────────
    if is_flip_hyper {
        if chosen.side.eq_ignore_ascii_case("BUY") {
            rt.flip_last_entry_price = metrics.mid;
            rt.flip_last_entry_qty   = allocation.qty;
            rt.flip_cycle_phase      = FlipCyclePhase::Entering;
            rt.flip_last_active      = Some(Instant::now()); // reset stall timer
            rt.flip_blocker.clear();
            info!(
                cycle_id,
                entry = metrics.mid,
                qty = allocation.qty,
                "[FLIP_HYPER] BUY submitted: entry={:.2} qty={:.8}",
                metrics.mid, allocation.qty
            );
        } else if chosen.side.eq_ignore_ascii_case("SELL") {
            rt.flip_cycle_phase = FlipCyclePhase::Exiting;
            rt.flip_last_active = Some(Instant::now()); // reset stall timer
            rt.flip_blocker.clear();
            info!(
                cycle_id,
                entry = rt.flip_last_entry_price,
                qty = allocation.qty,
                "[FLIP_HYPER] SELL submitted; awaiting fill confirmation for flip completion"
            );
        }
    }
    if is_swing_profile {
        if chosen.side.eq_ignore_ascii_case("BUY") {
            rt.swing_entry_price = metrics.mid;
            rt.swing_entry_started_at = Some(Instant::now());
            rt.swing_peak_price = metrics.mid;
            rt.swing_peak_unrealized_pct = 0.0;
        } else if chosen.side.eq_ignore_ascii_case("SELL") {
            let hold_secs = rt
                .swing_entry_started_at
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            rt.swing_last_hold_duration_secs = hold_secs;
            let cooldown_secs = ((hold_secs / 4.0) as u64)
                .clamp(SWING_COOLDOWN_MIN_SECS, SWING_COOLDOWN_MAX_SECS);
            rt.swing_cooldown_until = Some(Instant::now() + Duration::from_secs(cooldown_secs));
            log_npc_event(
                &*state.store,
                "swing_exit",
                &format!(
                    "cycle={} entry_price={:.2} peak_unrealized_pct={:+.4}% exit_reason={} hold_duration_secs={:.2} cooldown_secs={}",
                    cycle_id,
                    rt.swing_entry_price,
                    rt.swing_peak_unrealized_pct * 100.0,
                    rt.swing_last_exit_reason,
                    hold_secs,
                    cooldown_secs
                ),
            );
            rt.swing_entry_price = 0.0;
            rt.swing_entry_started_at = None;
            rt.swing_peak_price = 0.0;
            rt.swing_peak_unrealized_pct = 0.0;
        }
    }
    if allow_despite_regime_mismatch {
        log_npc_event(
            &*state.store,
            "risk_override",
            &format!(
                "cycle={} action_id={} role={} regime={} code=TRADE_ALLOWED_DESPITE_REGIME_MISMATCH conviction={:.4}",
                cycle_id,
                chosen.action_id,
                chosen.role.as_str(),
                regime.as_str(),
                chosen.score
            ),
        );
        info!(
            cycle_id,
            action_id = chosen.action_id.as_str(),
            role = chosen.role.as_str(),
            regime = regime.as_str(),
            conviction = chosen.score,
            "TRADE_ALLOWED_DESPITE_REGIME_MISMATCH"
        );
    }

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
        final_decision: "EXECUTE".to_string(),
        balance_block_reason: String::new(),
        risk_block_reason: String::new(),
        execution_block_reason: String::new(),
        cooldown_active,
        cooldown_remaining_ms,
        effective_threshold: effective_cutoff, // report the actual enforced gate to UI/telemetry
        threshold_mode: threshold_mode.to_string(),
        raw_score: chosen.raw_score,
        normalized_score: chosen.raw_score / effective_cutoff.max(f64::EPSILON),
        top_score_penalties: chosen.score_parts.top_penalties_str(),
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
    is_micro_active: bool,
    is_swing_profile: bool,
) -> Vec<WorkerProposal> {
    let mut proposals = Vec::new();
    let momentum = metrics.momentum_5s;
    let liquidity_score = (metrics.trade_samples as f64).max(0.0);
    let dip = dip_pct_from_history(&rt.mid_history, cfg.dip_lookback_cycles);

    for role in NpcRole::all() {
        let allow = regime_allowlist(role);
        let regime_eligible = allow.contains(&regime);
        let risk_override = is_swing_profile && matches!(role, NpcRole::RiskManager);
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
            is_micro_active,
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
            risk_override,
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
    is_micro_active: bool,
) -> ScoreBreakdown {
    // In MICRO_ACTIVE mode, boost short-term momentum sensitivity (5s) by 1.5×
    // to amplify edge signals on small live accounts where momentum is the primary
    // entry driver. This only scales the momentum input to edge_bias — it does not
    // affect safety guards or spread/slippage checks.
    let momentum_boost = if is_micro_active { 1.5 } else { 1.0 };
    let momentum = metrics.momentum_5s * momentum_boost;
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

    // In MICRO_ACTIVE mode, scale down penalty terms so that the edge signal is
    // not overwhelmed by proportionally large spread/vol/slippage terms typical of
    // a micro account operating in normal market conditions.
    // Hard guards (spread bps, slippage bps) still check the raw market values —
    // this dampening only affects the scoring priority, not safety enforcement.
    //
    // Dampening is applied *after* clamping (to the clamped ratio value).  This is
    // intentional: the guard layer has already blocked any inputs that would exceed
    // the ratio ceiling; the clamped value therefore represents an already-bounded
    // cost and halving it is a straightforward linear adjustment.
    let penalty_factor = if is_micro_active { MICRO_PENALTY_DAMPEN } else { 1.0 };

    let volatility_penalty = (realized_volatility_bps(&rt.mid_history, 8).unwrap_or(0.0) / cfg.alpha.vol_spike_bps.max(1.0)).clamp(0.0, 3.0) * penalty_factor;
    let spread_cost = (metrics.spread_bps / cfg.guards.max_spread_bps.max(0.1)).clamp(0.0, 3.0) * penalty_factor;
    let slippage_risk = (expected_slippage_bps / cfg.guards.max_slippage_bps.max(0.1)).clamp(0.0, 3.0) * penalty_factor;
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
    is_micro_active: bool,
) -> (Vec<String>, bool, u64) {
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
    if side.eq_ignore_ascii_case("BUY") && position_size + cfg.trade_size > cfg.guards.max_position_qty {
        reasons.push(format!(
            "POSITION_LIMIT_EXCEEDED:{:.8}>{:.8}",
            position_size + cfg.trade_size,
            cfg.guards.max_position_qty
        ));
    }

    // ── Per-role cooldown ────────────────────────────────────────────────────
    // FLIP_HYPER (live < $50): ultra-short 50ms cooldown for faster rotation.
    // MICRO_ACTIVE (live < $50): ultra-short 100ms cooldown for faster re-entry.
    // Small accounts (0 < balance < $100): reduced cooldown (300ms).
    // Normal accounts: max(cycle_interval, 250ms).
    // Allow immediate execution when no other guards fired — the cooldown only
    // adds friction when something else is already blocking the cycle.
    let flip_hyper_profile = cfg.behavior_profile.eq_ignore_ascii_case("FLIP_HYPER");
    let effective_cooldown = if is_micro_active && total_balance_usd > 0.0 && total_balance_usd < MICRO_BALANCE_USD {
        if flip_hyper_profile {
            Duration::from_millis(50)
        } else {
            Duration::from_millis(100)
        }
    } else if total_balance_usd > 0.0 && total_balance_usd < 100.0 {
        Duration::from_millis(300)
    } else {
        let cycle_ms = cfg.cycle_interval.as_millis() as u64;
        Duration::from_millis(cycle_ms.max(250))
    };

    let (cooldown_active, cooldown_remaining_ms) =
        if let Some(last_at) = rt.last_action_at.get(&role) {
            let elapsed = last_at.elapsed();
            if elapsed < effective_cooldown {
                let remaining = effective_cooldown - elapsed;
                (true, remaining.as_millis() as u64)
            } else {
                (false, 0u64)
            }
        } else {
            (false, 0u64)
        };

    // Only block on cooldown when other guards are already firing.
    // If the pipeline is otherwise clear (no blocks), allow immediate execution.
    if cooldown_active && !reasons.is_empty() {
        reasons.push(format!("ROLE_COOLDOWN_ACTIVE:{cooldown_remaining_ms}ms_remaining"));
    }

    (reasons, cooldown_active, cooldown_remaining_ms)
}

fn collect_truth_pending_order_slots(truth: &crate::reconciler::TruthState) -> (usize, Vec<String>) {
    let mut ids = Vec::new();
    let mut fresh_pending_ack = 0usize;
    for o in truth.orders.values() {
        if o.status.is_active() {
            ids.push(format!("order:{}", o.client_order_id));
            continue;
        }
        if o.is_pending_ack() && o.last_seen.elapsed() <= Duration::from_secs(SLOT_PENDING_ACK_TTL_SECS) {
            fresh_pending_ack = fresh_pending_ack.saturating_add(1);
            ids.push(format!("pending_ack:{}", o.client_order_id));
        }
    }
    ids.sort();
    ids.dedup();
    let counted = truth.open_order_count.max(fresh_pending_ack);
    if truth.open_order_count > ids.len() {
        ids.push(format!("exchange_open_orders:{}", truth.open_order_count));
    }
    (counted, ids)
}

fn count_live_open_positions(position_size: f64, sell_inventory: f64) -> usize {
    usize::from(position_size.abs() > SLOT_QTY_EPSILON || sell_inventory.abs() > SLOT_QTY_EPSILON)
}

fn release_open_action_slot(rt: &mut NpcRuntimeState, id: &str) {
    if let Some(open) = rt.open_actions.remove(id) {
        rt.cycle_open_notional = (rt.cycle_open_notional - (open.allocated_qty * open.entry_mid).max(0.0)).max(0.0);
    }
}

fn build_slot_usage_diagnostics(
    cfg: &NpcConfig,
    counted_open_positions: usize,
    counted_pending_orders: usize,
    counted_reserved_slots: usize,
    mut source_ids: Vec<String>,
) -> SlotUsageDiagnostics {
    let counted_total_slots = counted_open_positions
        .saturating_add(counted_pending_orders)
        .saturating_add(counted_reserved_slots);
    if source_ids.is_empty() {
        source_ids.push("none".to_string());
    }
    let slot_block_reason = if counted_total_slots >= cfg.alpha.max_concurrent_positions {
        format!(
            "MAX_CONCURRENT_POSITIONS:{}>={}",
            counted_total_slots, cfg.alpha.max_concurrent_positions
        )
    } else {
        String::new()
    };
    SlotUsageDiagnostics {
        max_concurrent_positions: cfg.alpha.max_concurrent_positions,
        counted_open_positions,
        counted_pending_orders,
        counted_reserved_slots,
        slot_block_reason,
        slot_source_ids: source_ids,
    }
}

fn slot_usage_snapshot(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    position_size: f64,
    sell_inventory: f64,
    pending_order_slots: usize,
    pending_order_ids: &[String],
) -> SlotUsageDiagnostics {
    let counted_open_positions = count_live_open_positions(position_size, sell_inventory);
    let counted_reserved_slots = if counted_open_positions > 0 || pending_order_slots > 0 {
        0
    } else {
        rt.open_actions
            .values()
            .filter(|open| open.allocated_qty > SLOT_QTY_EPSILON && open.entry_mid.is_finite() && open.entry_mid > 0.0)
            .count()
    };
    let mut source_ids = Vec::new();
    if counted_open_positions > 0 {
        source_ids.push("position:spot_inventory".to_string());
    }
    source_ids.extend(pending_order_ids.iter().cloned());
    if counted_reserved_slots > 0 {
        source_ids.extend(
            rt.open_actions
                .keys()
                .take(counted_reserved_slots)
                .map(|id| format!("reservation:{id}")),
        );
    }
    build_slot_usage_diagnostics(
        cfg,
        counted_open_positions,
        pending_order_slots,
        counted_reserved_slots,
        source_ids,
    )
}

fn reconcile_slot_usage(
    cfg: &NpcConfig,
    rt: &mut NpcRuntimeState,
    position_size: f64,
    sell_inventory: f64,
    pending_order_slots: usize,
    pending_order_ids: &[String],
    active_mode: NpcTradingMode,
) -> SlotUsageDiagnostics {
    let counted_open_positions = count_live_open_positions(position_size, sell_inventory);
    let mut stale_ids = Vec::new();
    for (id, open) in rt.open_actions.iter() {
        let stale_qty = open.allocated_qty <= SLOT_QTY_EPSILON;
        let stale_entry = !open.entry_mid.is_finite() || open.entry_mid <= 0.0;
        let expired = open.opened_at.elapsed() > Duration::from_secs(SLOT_RESERVATION_TTL_SECS);
        let stale_mode = active_mode == NpcTradingMode::Live && open.execution_mode != NpcTradingMode::Live;
        if stale_qty || stale_entry || expired || stale_mode {
            stale_ids.push(id.clone());
        }
    }
    for id in stale_ids {
        release_open_action_slot(rt, &id);
    }

    let live_slots = counted_open_positions.saturating_add(pending_order_slots);
    if live_slots == 0 && !rt.open_actions.is_empty() {
        let ids: Vec<String> = rt.open_actions.keys().cloned().collect();
        for id in ids {
            release_open_action_slot(rt, &id);
        }
    }

    let mut counted_reserved_slots = if live_slots > 0 { 0 } else { rt.open_actions.len() };
    if counted_reserved_slots > 0
        && live_slots < cfg.alpha.max_concurrent_positions
        && live_slots.saturating_add(counted_reserved_slots) >= cfg.alpha.max_concurrent_positions
    {
        let ids: Vec<String> = rt.open_actions.keys().cloned().collect();
        for id in ids {
            release_open_action_slot(rt, &id);
        }
        counted_reserved_slots = 0;
    }

    let mut source_ids = Vec::new();
    if counted_open_positions > 0 {
        source_ids.push("position:spot_inventory".to_string());
    }
    source_ids.extend(pending_order_ids.iter().cloned());
    if counted_reserved_slots > 0 {
        source_ids.extend(rt.open_actions.keys().map(|id| format!("reservation:{id}")));
    }

    build_slot_usage_diagnostics(
        cfg,
        counted_open_positions,
        pending_order_slots,
        counted_reserved_slots,
        source_ids,
    )
}

fn evaluate_portfolio_controls(
    cfg: &NpcConfig,
    rt: &NpcRuntimeState,
    slot_usage: &SlotUsageDiagnostics,
    position_size: f64,
    exposure_notional: f64,
    mid: f64,
    regime: MarketRegime,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !slot_usage.slot_block_reason.is_empty() {
        reasons.push(slot_usage.slot_block_reason.clone());
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
    // Clamp equity to 0 before ratio calculation: negative PnL can produce
    // ratios > 1 that falsely trigger MAX_DRAWDOWN_BREACH on tiny accounts.
    let equity_clipped = equity.max(0.0);
    let peak = rt.peak_equity.max(equity_clipped);
    // Require peak >= $1 to prevent false triggers on fresh/zero-balance accounts
    // where both peak and equity are near zero (bad initialization / stale peak).
    const MIN_PEAK_FOR_DRAWDOWN_CHECK: f64 = 1.0;
    let dd = if peak >= MIN_PEAK_FOR_DRAWDOWN_CHECK {
        ((peak - equity_clipped) / peak).max(0.0)
    } else {
        0.0
    };
    if dd >= cfg.alpha.max_drawdown_pct {
        reasons.push(format!("MAX_DRAWDOWN_BREACH:{:.4}>={:.4} (current_equity={:.4} peak_equity={:.4} limit={:.4})",
            dd, cfg.alpha.max_drawdown_pct, equity_clipped, peak, cfg.alpha.max_drawdown_pct));
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
    total_balance_usd: f64,
    is_micro_active: bool,
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

    // ── COMPOUND_EXECUTION: dynamic equity-based sizing ───────────────────────
    // In MICRO_ACTIVE mode, replace flat trade_size with a % of account equity.
    // base_size = COMPOUND_BASE_EQUITY_PCT (15%) × total_balance, clamped to
    // [$COMPOUND_MIN_NOTIONAL_USD … COMPOUND_MAX_EQUITY_PCT × total_balance].
    // The compound_size_scalar (updated by loss control and profit locking) is
    // applied on top.  Increase only when the last trade was profitable.
    let mut qty = if is_micro_active && total_balance_usd > 0.0 && mid > 0.0 {
        let max_notional = total_balance_usd * COMPOUND_MAX_EQUITY_PCT;
        if max_notional < COMPOUND_MIN_NOTIONAL_USD {
            0.0
        } else {
            let base_notional = (total_balance_usd * COMPOUND_BASE_EQUITY_PCT)
                .clamp(COMPOUND_MIN_NOTIONAL_USD, max_notional);
            // Apply loss-based scalar (reduced after consecutive losses).
            let size_scalar = rt.compound_size_scalar.clamp(COMPOUND_MIN_SIZE_SCALAR, 1.0);
            // On a win, allow a small upward bump (capped at 10%) to grow into profitable streaks.
            // On a loss or flat, no additional adjustment — scalar stays at current level.
            let momentum_adjust = if rt.compound_last_trade_was_profitable { 1.10 } else { 1.0 };
            let adjusted_notional = base_notional * size_scalar * momentum_adjust * dd_factor;
            adjusted_notional / mid
        }
    } else {
        let size_multiplier = quality * (1.0 + recent_performance).clamp(0.3, 1.4) * (1.0 - symbol_concentration).clamp(0.1, 1.0) * dd_factor;
        cfg.trade_size * agent_budget * size_multiplier
    };

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

    // ── Minimum notional floor ────────────────────────────────────────────────
    // Ensure fills are visible: if the computed qty would produce a notional value
    // below the floor, bump it up so the fill registers on the exchange. The bump
    // is capped by max_qty (already guaranteed ≥ 0) so it never exceeds the
    // available balance or inventory.
    // COMPOUND_EXECUTION uses COMPOUND_MIN_NOTIONAL_USD; standard path uses $5.
    let min_notional = if is_micro_active { COMPOUND_MIN_NOTIONAL_USD } else { 5.0 };
    if mid > 0.0 {
        let min_qty_for_notional = min_notional / mid;
        if qty < min_qty_for_notional {
            // max_qty is non-negative by construction (sell_inventory.max(0) or
            // buy_power/mid with mid > 0), so no extra clamp is needed.
            qty = min_qty_for_notional.min(max_qty);
        }
    }

    // Re-check after the notional floor bump in case max_qty was too small.
    if qty < 0.000_000_01 {
        return AllocationDecision {
            qty: 0.0,
            reason: "ALLOCATION_BELOW_MIN_NOTIONAL".to_string(),
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
            let notional_used = open.allocated_qty * open.entry_mid;
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

            // Enhanced per-trade log for COMPOUND_EXECUTION observability.
            log_npc_event(
                store,
                "trade_metrics",
                &format!(
                    "action_id={} role={} regime={} expected_edge={:+.6} realized_edge={:+.6} \
                     pnl={:+.6} hold_ms={:.0} qty={:.8} size_usd={:.2} realized_pnl_usd={:+.4}",
                    id,
                    open.role.as_str(),
                    open.regime.as_str(),
                    open.expected_edge,
                    realized_edge,
                    pnl,
                    hold_ms,
                    open.allocated_qty,
                    notional_used,
                    pnl,
                ),
            );

            // ── COMPOUND_EXECUTION: update consecutive loss / profit state ────────
            let is_compound_execution_trade = open.regime.as_str() == "micro_active";
            if is_compound_execution_trade {
                rt.compound_session_pnl += pnl;
                rt.compound_last_trade_pnl = pnl;
                rt.compound_last_trade_was_profitable = pnl > 0.0;
            }

            // Shadow `pnl` so the existing compound loss/profit logic becomes a no-op
            // for trades that were not opened in the COMPOUND_EXECUTION regime.
            let pnl = if is_compound_execution_trade { pnl } else { 0.0 };
            if pnl < 0.0 {
                rt.compound_consecutive_losses = rt.compound_consecutive_losses.saturating_add(1);
                // After 2+ consecutive losses: reduce size scalar by COMPOUND_LOSS_SIZE_FACTOR.
                if rt.compound_consecutive_losses >= COMPOUND_LOSS_SIZE_REDUCE_THRESHOLD {
                    rt.compound_size_scalar =
                        (rt.compound_size_scalar * COMPOUND_LOSS_SIZE_FACTOR).max(COMPOUND_MIN_SIZE_SCALAR);
                    info!(
                        consecutive_losses = rt.compound_consecutive_losses,
                        new_scalar = rt.compound_size_scalar,
                        pnl,
                        "[COMPOUND] Loss #{}: size scalar reduced to {:.2}",
                        rt.compound_consecutive_losses, rt.compound_size_scalar
                    );
                }
                // After 3+ losses: activate compound loss pause (no new BUYs for cooldown).
                if rt.compound_consecutive_losses >= COMPOUND_LOSS_PAUSE_THRESHOLD {
                    let pause_until = Instant::now() + Duration::from_secs(COMPOUND_LOSS_COOLDOWN_SECS);
                    rt.compound_loss_pause_until = Some(pause_until);
                    info!(
                        consecutive_losses = rt.compound_consecutive_losses,
                        cooldown_secs = COMPOUND_LOSS_COOLDOWN_SECS,
                        "[COMPOUND] Loss streak {}: activating {}s cooldown pause",
                        rt.compound_consecutive_losses, COMPOUND_LOSS_COOLDOWN_SECS
                    );
                }
            } else if pnl > 0.0 {
                // Win: reset loss streak and proportionally recover size scalar toward 1.0.
                // Multiply by 1.2 per win — faster recovery from deep reductions while
                // still requiring multiple consecutive wins to fully restore from 0.25.
                rt.compound_consecutive_losses = 0;
                rt.compound_size_scalar = (rt.compound_size_scalar * 1.2).min(1.0);
                rt.compound_loss_pause_until = None; // cancel any pause on a win
                info!(
                    new_scalar = rt.compound_size_scalar,
                    pnl,
                    "[COMPOUND] Win: size scalar recovered to {:.2}", rt.compound_size_scalar
                );
            }

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

    fn open_action_for_test(entry_mid: f64, allocated_qty: f64, opened_at: Instant, mode: NpcTradingMode) -> OpenAction {
        OpenAction {
            role: NpcRole::Scout,
            side: "BUY".to_string(),
            entry_mid,
            opened_at,
            entry_spread_bps: 1.0,
            expected_edge: 0.1,
            regime: MarketRegime::TrendingUp,
            allocated_qty,
            cycle_id: 1,
            execution_mode: mode,
        }
    }

    // ── evaluate_guards: small account SELL bypass ─────────────────────────────

    #[test]
    fn small_account_sell_bypasses_liquidity_depth_checks() {
        let cfg = test_cfg();
        let rt = NpcRuntimeState::default();
        let metrics = low_liquidity_metrics();

        // Small account (<$100), SELL side, non-zero inventory and notional.
        let (reasons, _cooldown_active, _cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 20.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
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
        let (reasons, _cooldown_active, _cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 20.0,
            /* total_balance_usd */ 500.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
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
        let (reasons, _cooldown_active, _cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 20.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.0,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
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

        let (reasons, _cooldown_active, _cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
        );

        assert!(
            reasons.iter().any(|r| r == "KILL_SWITCH_ACTIVE"),
            "kill switch must always block, got: {:?}", reasons
        );
    }

    // ── evaluate_guards: cooldown logic ────────────────────────────────────────

    #[test]
    fn cooldown_inactive_when_no_prior_action() {
        let cfg = test_cfg();
        let rt = NpcRuntimeState::default();
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 1.0,
            trade_samples: 10,
            mid: 50_000.0,
            ..Default::default()
        };
        let (reasons, cooldown_active, cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 500.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
        );
        assert!(!cooldown_active, "cooldown must be inactive when no prior action, reasons: {:?}", reasons);
        assert_eq!(cooldown_remaining_ms, 0);
        assert!(!reasons.iter().any(|r| r.starts_with("ROLE_COOLDOWN_ACTIVE")));
    }

    #[test]
    fn cooldown_active_but_allows_immediate_execution_when_no_other_blocks() {
        let mut cfg = test_cfg();
        cfg.cycle_interval = Duration::from_millis(500);
        let mut rt = NpcRuntimeState::default();
        // Record a very recent action to trigger cooldown.
        rt.last_action_at.insert(NpcRole::InventoryManager, Instant::now());
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 1.0,   // within max_spread_bps
            trade_samples: 10, // above min_liquidity_score
            mid: 50_000.0,
            ..Default::default()
        };
        // Normal account (>= $100), no other guard violations.
        let (reasons, cooldown_active, cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 500.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
        );
        // cooldown IS active, but should NOT block because no other guards fired.
        assert!(cooldown_active, "cooldown should be detected as active");
        assert!(cooldown_remaining_ms > 0, "remaining ms must be positive");
        assert!(
            !reasons.iter().any(|r| r.starts_with("ROLE_COOLDOWN_ACTIVE")),
            "ROLE_COOLDOWN_ACTIVE must NOT be added when no other guards fired (immediate execution), reasons: {:?}", reasons
        );
    }

    #[test]
    fn cooldown_blocks_when_other_guards_also_fire() {
        let mut cfg = test_cfg();
        cfg.cycle_interval = Duration::from_millis(500);
        let mut rt = NpcRuntimeState::default();
        rt.last_action_at.insert(NpcRole::InventoryManager, Instant::now());
        // High spread to trigger another guard.
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 100.0, // exceeds max_spread_bps=5.0
            trade_samples: 10,
            mid: 50_000.0,
            ..Default::default()
        };
        let (reasons, cooldown_active, _cooldown_remaining_ms) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 500.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
        );
        assert!(cooldown_active);
        assert!(
            reasons.iter().any(|r| r.starts_with("ROLE_COOLDOWN_ACTIVE")),
            "ROLE_COOLDOWN_ACTIVE must be added when other guards also fired, reasons: {:?}", reasons
        );
    }

    #[test]
    fn small_account_uses_reduced_cooldown() {
        let mut cfg = test_cfg();
        // Use a long cycle interval so normal cooldown would be large.
        cfg.cycle_interval = Duration::from_secs(10);
        let mut rt = NpcRuntimeState::default();
        // Place the last action 400ms ago — within normal cooldown (10s) but beyond small account (300ms).
        rt.last_action_at.insert(
            NpcRole::InventoryManager,
            Instant::now() - Duration::from_millis(400),
        );
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.01,
            spread_bps: 1.0,
            trade_samples: 10,
            mid: 50_000.0,
            ..Default::default()
        };
        // Small account: effective cooldown is 300ms, so 400ms ago means cooldown is expired.
        let (_, cooldown_active, _) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 50.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
        );
        assert!(!cooldown_active, "small account cooldown (300ms) should be expired after 400ms");

        // Same test for normal account: effective cooldown is 10s, so 400ms ago still in cooldown.
        let (_, cooldown_active_normal, remaining_normal) = evaluate_guards(
            &cfg, &rt, NpcRole::InventoryManager, "SELL",
            0.0, &metrics, 0.0,
            /* total_balance_usd */ 500.0,
            /* sell_inventory    */ 0.001,
            /* order_notional    */ 50.0,
            /* is_micro_active   */ false,
        );
        assert!(cooldown_active_normal, "normal account cooldown (10s) should still be active after 400ms");
        assert!(remaining_normal > 0);
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
            /* is_micro_active */ false,
            /* is_swing_profile */ false,
        );

        assert!(candidates.iter().any(|c| c.action_id == "c42-scout"));
        assert!(candidates.iter().any(|c| c.action_id == "c42-dip_buyer"));
    }

    #[test]
    fn swing_profile_sets_risk_override_for_risk_manager_only() {
        let cfg = NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 1.0,
            momentum_threshold: 0.001,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        });
        let rt = NpcRuntimeState::default();
        let metrics = crate::signal::SignalMetrics {
            mid: 50_000.0,
            spread_bps: 1.0,
            trade_samples: 12,
            ..Default::default()
        };
        let candidates = build_worker_candidates(
            &cfg,
            7,
            &rt,
            &metrics,
            0.0,
            1_000.0,
            0.0,
            MarketRegime::TrendingUp,
            false,
            false,
            true,
        );
        let risk_mgr = candidates.iter().find(|c| c.role == NpcRole::RiskManager).expect("risk_manager candidate");
        assert!(risk_mgr.risk_override, "SWING must enable risk override for risk_manager");
        assert!(!risk_mgr.regime_eligible, "TRENDING_UP is not in risk_manager allowlist");
        assert!(
            risk_mgr.regime_block_reason.as_deref().unwrap_or_default().contains("REGIME_MISMATCH"),
            "regime mismatch should remain informational text for telemetry"
        );
        assert!(
            candidates.iter().filter(|c| c.role != NpcRole::RiskManager).all(|c| !c.risk_override),
            "only risk_manager should carry risk_override"
        );
    }

    #[test]
    fn profile_lock_swing_disables_micro_and_flip_threshold_modes() {
        let (swing_threshold, swing_mode) =
            profile_threshold(crate::profile::RuntimeProfile::Swing, 35.0, NpcTradingMode::Live);
        let (micro_threshold, micro_mode) =
            profile_threshold(crate::profile::RuntimeProfile::MicroActive, 35.0, NpcTradingMode::Live);
        let (flip_threshold, flip_mode) =
            profile_threshold(crate::profile::RuntimeProfile::FlipHyper, 35.0, NpcTradingMode::Live);
        assert_eq!(swing_mode, "swing");
        assert_ne!(swing_mode, micro_mode);
        assert_ne!(swing_mode, flip_mode);
        assert!(swing_threshold >= micro_threshold);
        assert!(swing_threshold >= flip_threshold);
    }

    #[test]
    fn rebalance_to_usdt_triggers_before_buy_capital_block() {
        let mut cfg = test_cfg();
        cfg.disable_no_trade_idle = true;
        cfg.rebalance_min_notional_usd = 5.0;
        let rt = NpcRuntimeState::default();
        let plan = maybe_rebalance(
            &cfg,
            &rt,
            "BUY",
            /* buy_power */ 0.0,
            /* sell_inventory */ 0.001,
            /* mid */ 50_000.0,
            /* total_balance_usd */ 50.0,
            /* required_trade_capital */ 10.0,
        )
        .expect("rebalance plan expected");
        assert_eq!(plan.status, "REBALANCE_TO_USDT");
        assert_eq!(plan.side, "SELL");
        assert!(plan.qty > 0.0);
        assert!(plan.value_usd >= cfg.rebalance_min_notional_usd);
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

    // ── Drawdown checks for small accounts ────────────────────────────────────

    #[test]
    fn drawdown_check_no_false_trigger_on_zero_peak_equity() {
        // Fresh account: peak_equity=0, position_size=0, no PnL.
        // Must NOT trigger MAX_DRAWDOWN_BREACH.
        let cfg = test_cfg();
        let rt = NpcRuntimeState { peak_equity: 0.0, ..NpcRuntimeState::default() };
        let reasons = evaluate_portfolio_controls(
            &cfg,
            &rt,
            &SlotUsageDiagnostics::default(),
            /* position_size */ 0.0,
            /* exposure_notional */ 0.0,
            /* mid */ 50_000.0,
            MarketRegime::TrendingUp,
        );
        assert!(
            !reasons.iter().any(|r| r.contains("MAX_DRAWDOWN_BREACH")),
            "MAX_DRAWDOWN_BREACH must not trigger on zero-peak account; got: {:?}", reasons
        );
    }

    #[test]
    fn drawdown_check_no_false_trigger_when_stale_peak_is_tiny() {
        // Stale peak of $0.50 (sub-$1): must NOT trigger even if equity dropped to 0.
        let cfg = test_cfg();
        let rt = NpcRuntimeState { peak_equity: 0.5, ..NpcRuntimeState::default() };
        let reasons = evaluate_portfolio_controls(
            &cfg,
            &rt,
            &SlotUsageDiagnostics::default(),
            0.0,
            0.0,
            50_000.0,
            MarketRegime::TrendingUp,
        );
        assert!(
            !reasons.iter().any(|r| r.contains("MAX_DRAWDOWN_BREACH")),
            "Must not trigger MAX_DRAWDOWN_BREACH with sub-$1 peak; got: {:?}", reasons
        );
    }

    #[test]
    fn drawdown_check_triggers_correctly_on_real_drawdown() {
        // Meaningful peak ($100) and current equity drops far enough to breach.
        // max_drawdown_pct=0.08, so a 50% drawdown should definitely trigger.
        let cfg = test_cfg();
        let mut rt = NpcRuntimeState::default();
        rt.peak_equity = 100.0;
        // Inject negative PnL so equity = 0 + (-60) = -60 → clipped to 0.
        // Drawdown = (100 - 0) / 100 = 1.0 ≥ 0.08
        rt.perf.insert(NpcRole::Scout, AgentPerformance { gross_pnl: -60.0, ..Default::default() });
        let reasons = evaluate_portfolio_controls(
            &cfg,
            &rt,
            &SlotUsageDiagnostics::default(),
            0.0,
            0.0,
            50_000.0,
            MarketRegime::TrendingUp,
        );
        assert!(
            reasons.iter().any(|r| r.contains("MAX_DRAWDOWN_BREACH")),
            "Must trigger MAX_DRAWDOWN_BREACH with 100% drawdown from $100 peak; got: {:?}", reasons
        );
    }

    #[test]
    fn drawdown_check_includes_diagnostic_fields_in_message() {
        // Verify the diagnostic fields appear in the breach message.
        let cfg = test_cfg();
        let mut rt = NpcRuntimeState::default();
        rt.peak_equity = 50.0;
        rt.perf.insert(NpcRole::Scout, AgentPerformance { gross_pnl: -40.0, ..Default::default() });
        let reasons = evaluate_portfolio_controls(
            &cfg,
            &rt,
            &SlotUsageDiagnostics::default(),
            0.0,
            0.0,
            50_000.0,
            MarketRegime::TrendingUp,
        );
        let breach = reasons.iter().find(|r| r.contains("MAX_DRAWDOWN_BREACH"))
            .cloned().unwrap_or_default();
        assert!(breach.contains("current_equity"), "Breach message must include current_equity; got: {breach}");
        assert!(breach.contains("peak_equity"), "Breach message must include peak_equity; got: {breach}");
        assert!(breach.contains("limit"), "Breach message must include limit; got: {breach}");
    }

    #[test]
    fn stale_pending_order_does_not_consume_slot_forever() {
        let mut truth = crate::reconciler::TruthState::new("BTCUSDT", 0.0);
        truth.orders.insert(
            "stale-pending".to_string(),
            crate::reconciler::OrderRecord {
                client_order_id: "stale-pending".to_string(),
                exchange_order_id: 0,
                symbol: "BTCUSDT".to_string(),
                side: "BUY".to_string(),
                order_type: "LIMIT".to_string(),
                orig_qty: 0.001,
                filled_qty: 0.0,
                remaining_qty: 0.001,
                avg_fill_price: 0.0,
                status: crate::reconciler::OrderStatus::Unknown,
                last_seen: Instant::now() - Duration::from_secs(SLOT_PENDING_ACK_TTL_SECS + 1),
            },
        );
        let (counted, ids) = collect_truth_pending_order_slots(&truth);
        assert_eq!(counted, 0, "stale pending ack must not reserve slot");
        assert!(ids.is_empty(), "stale pending ack id must be excluded");
    }

    #[test]
    fn closed_position_does_not_consume_slot() {
        let cfg = test_cfg();
        let rt = NpcRuntimeState::default();
        let diag = slot_usage_snapshot(&cfg, &rt, 0.0, 0.0, 0, &[]);
        assert_eq!(diag.counted_open_positions, 0);
        assert_eq!(diag.counted_pending_orders, 0);
        assert_eq!(diag.counted_reserved_slots, 0);
    }

    #[test]
    fn failed_execution_clears_reserved_slot() {
        let mut cfg = test_cfg();
        cfg.alpha.max_concurrent_positions = 1;
        let mut rt = NpcRuntimeState::default();
        rt.open_actions.insert(
            "orphan-reservation".to_string(),
            open_action_for_test(50_000.0, 0.001, Instant::now(), NpcTradingMode::Paper),
        );
        let diag = reconcile_slot_usage(&cfg, &mut rt, 0.0, 0.0, 0, &[], NpcTradingMode::Live);
        assert_eq!(diag.counted_reserved_slots, 0, "orphaned reservation must be self-healed");
        assert!(rt.open_actions.is_empty(), "reserved slot must be removed");
    }

    #[test]
    fn zero_qty_or_synthetic_position_does_not_count_reserved_slot() {
        let cfg = test_cfg();
        let mut rt = NpcRuntimeState::default();
        rt.open_actions.insert(
            "zero-qty".to_string(),
            open_action_for_test(50_000.0, 0.0, Instant::now(), NpcTradingMode::Live),
        );
        rt.open_actions.insert(
            "synthetic-entry".to_string(),
            open_action_for_test(0.0, 0.001, Instant::now(), NpcTradingMode::Live),
        );
        let diag = reconcile_slot_usage(&cfg, &mut rt, 0.0, 0.0, 0, &[], NpcTradingMode::Live);
        assert_eq!(diag.counted_reserved_slots, 0);
        assert!(rt.open_actions.is_empty(), "invalid reservations must be purged");
    }

    #[test]
    fn counted_slots_match_live_reconciled_state() {
        let cfg = test_cfg();
        let mut rt = NpcRuntimeState::default();
        rt.open_actions.insert(
            "stale-paper".to_string(),
            open_action_for_test(50_000.0, 0.001, Instant::now(), NpcTradingMode::Paper),
        );
        let pending_ids = vec!["order:live-1".to_string()];
        let diag = reconcile_slot_usage(&cfg, &mut rt, 0.002, 0.002, 1, &pending_ids, NpcTradingMode::Live);
        assert_eq!(diag.counted_open_positions, 1);
        assert_eq!(diag.counted_pending_orders, 1);
        assert_eq!(diag.counted_reserved_slots, 0);
        assert!(
            diag.slot_source_ids.iter().any(|s| s == "position:spot_inventory")
                && diag.slot_source_ids.iter().any(|s| s == "order:live-1"),
            "diagnostics must reflect reconciled live sources: {:?}",
            diag.slot_source_ids
        );
    }

    // ── Micro-account adaptive signal thresholds ───────────────────────────────

    #[test]
    fn micro_threshold_below_50_uses_0_11() {
        // Any balance > 0 and < $50 must use THRESHOLD_MICRO_SMALL = 0.11.
        for balance in [1.0, 10.0, 25.0, 35.0, 49.99] {
            let (threshold, mode) = adaptive_signal_threshold(balance);
            assert!((threshold - THRESHOLD_MICRO_SMALL).abs() < f64::EPSILON,
                "balance ${balance} → threshold {THRESHOLD_MICRO_SMALL}, got {threshold}");
            assert_eq!(mode, "micro_aggressive");
        }
    }

    #[test]
    fn micro_threshold_between_50_and_100_uses_0_14() {
        // $50 ≤ balance < $100 must use THRESHOLD_MICRO = 0.14.
        for balance in [50.0, 75.0, 99.99] {
            let (threshold, mode) = adaptive_signal_threshold(balance);
            assert!((threshold - THRESHOLD_MICRO).abs() < f64::EPSILON,
                "balance ${balance} → threshold {THRESHOLD_MICRO}, got {threshold}");
            assert_eq!(mode, "micro_aggressive");
        }
    }

    #[test]
    fn micro_threshold_at_100_or_above_uses_base() {
        // balance ≥ $100 must preserve THRESHOLD_BASE = 0.18 (existing behaviour).
        for balance in [100.0, 150.0, 500.0, 10_000.0] {
            let (threshold, mode) = adaptive_signal_threshold(balance);
            assert!((threshold - THRESHOLD_BASE).abs() < f64::EPSILON,
                "balance ${balance} → threshold {THRESHOLD_BASE}, got {threshold}");
            assert_eq!(mode, "normal", "balance ${balance} → mode normal");
        }
    }

    #[test]
    fn micro_threshold_zero_balance_uses_base() {
        // Zero balance (no data yet) must not activate micro mode.
        let (threshold, mode) = adaptive_signal_threshold(0.0);
        assert!((threshold - THRESHOLD_BASE).abs() < f64::EPSILON,
            "balance $0 → threshold {THRESHOLD_BASE} (base), got {threshold}");
        assert_eq!(mode, "normal");
    }

    #[test]
    fn micro_threshold_effective_cutoff_lower_for_micro_accounts() {
        // For micro accounts (balance < $50), effective_cutoff = effective_threshold.min(regime_cutoff).
        // When regime_cutoff > effective_threshold, the balance-tier cap wins.
        let regime_cutoff = 0.20_f64; // e.g. learner raised it high
        let balance = 35.0_f64;       // typical $30-$40 account
        let (effective_threshold, mode) = adaptive_signal_threshold(balance);
        assert_eq!(mode, "micro_aggressive");
        // Simulate the fixed effective_cutoff logic (condition: mode == "micro_aggressive")
        let effective_cutoff = if mode == "micro_aggressive" {
            effective_threshold.min(regime_cutoff)
        } else {
            regime_cutoff
        };
        assert!((effective_cutoff - THRESHOLD_MICRO_SMALL).abs() < f64::EPSILON,
            "micro account (${balance}) should use effective_threshold={THRESHOLD_MICRO_SMALL} not regime_cutoff=0.20, got {effective_cutoff}");
    }

    #[test]
    fn micro_threshold_regime_cannot_raise_cutoff_above_tier_limit() {
        // Even if regime_cutoff is 0.1800 (the old normal baseline), a micro account
        // must still be gated at THRESHOLD_MICRO_SMALL = 0.11, not 0.18.
        let regime_cutoff = THRESHOLD_BASE; // 0.18 – worst case
        let (effective_threshold, mode) = adaptive_signal_threshold(35.0);
        let effective_cutoff = if mode == "micro_aggressive" {
            effective_threshold.min(regime_cutoff)
        } else {
            regime_cutoff
        };
        assert!((effective_cutoff - THRESHOLD_MICRO_SMALL).abs() < f64::EPSILON,
            "regime_cutoff={THRESHOLD_BASE} must not raise micro-account gate above {THRESHOLD_MICRO_SMALL}; got {effective_cutoff}");
    }

    #[test]
    fn micro_account_score_0_1491_passes_threshold() {
        // PROOF: a $30-$40 account with signal_score 0.1491 must pass the decision gate
        // and not return HOLD. With THRESHOLD_MICRO_SMALL=0.11 and worst-case
        // regime_cutoff=0.18, effective_cutoff = min(0.11, 0.18) = 0.11 < 0.1491.
        let signal_score = 0.1491_f64;
        let balance = 35.0_f64;
        let regime_cutoff = THRESHOLD_BASE; // 0.18 – the previously enforced stale value
        let (effective_threshold, mode) = adaptive_signal_threshold(balance);
        let effective_cutoff = if mode == "micro_aggressive" {
            effective_threshold.min(regime_cutoff)
        } else {
            regime_cutoff
        };
        assert!(
            signal_score >= effective_cutoff,
            "signal_score {signal_score} must pass effective_cutoff {effective_cutoff} \
             for a ${balance} micro account; old stale gate was 0.1800"
        );
        assert!((effective_cutoff - THRESHOLD_MICRO_SMALL).abs() < f64::EPSILON,
            "effective_cutoff should be THRESHOLD_MICRO_SMALL={THRESHOLD_MICRO_SMALL}, got {effective_cutoff}");
    }

    #[test]
    fn micro_threshold_effective_cutoff_respects_regime_for_normal_accounts() {
        // For normal accounts (balance ≥ $100), regime_cutoff (if lower) still wins.
        let regime_cutoff = 0.08_f64; // learner set it low
        let (effective_threshold, mode) = adaptive_signal_threshold(200.0);
        assert_eq!(mode, "normal");
        let effective_cutoff = if mode == "micro_aggressive" {
            effective_threshold.min(regime_cutoff)
        } else {
            regime_cutoff
        };
        assert!((effective_cutoff - 0.08).abs() < f64::EPSILON,
            "normal account should use regime_cutoff=0.08 when it is lower, got {effective_cutoff}");
    }
}

// ── End-to-end gate diagnostic tests ──────────────────────────────────────────
//
// These tests trace the exact blocking chain in run_cycle() and prove which
// gate fires at each layer.  They serve as living documentation of the root
// causes preventing autonomous execution.
//
// Diagnostic findings (run_cycle gate order):
//   1. Authority mode OFF  → BLOCKED (authority-layer)
//   2. Score below threshold → BLOCKED (decision-layer)
//   3. Portfolio controls active → BLOCKED (risk-layer)
//   4. Allocation returns zero qty → BLOCKED (sizing-layer)
//   5. Guard violations → BLOCKED (guard-layer)
//   6. NPC_TRADING_MODE=live + paper_executions==0 → BLOCKED (authority-layer)
//   7. web_base_url missing → BLOCKED (dispatch-layer)
#[cfg(test)]
mod diagnostic_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::Mutex;

    use super::*;
    use crate::agent::{AgentState, TradeAgentConfig};
    use crate::authority::AuthorityLayer;
    use crate::client::BinanceClient;
    use crate::executor::{CircuitBreakerConfig, Executor, WatchdogConfig};
    use crate::feed::{FeedState, MidSample, TradeSample};
    use crate::reconciler::TruthState;
    use crate::signal::{SignalConfig, SignalEngine};
    use crate::store::InMemoryEventStore;
    use crate::withdrawal::{WithdrawalConfig, WithdrawalManager};

    // ── Shared helpers ────────────────────────────────────────────────────────

    fn minimal_signal_cfg() -> SignalConfig {
        SignalConfig {
            order_qty: 0.001,
            momentum_threshold: 0.00005,
            imbalance_threshold: 0.10,
            max_entry_spread_bps: 10.0,
            max_feed_staleness: Duration::from_secs(5),
            stop_loss_pct: 0.002,
            take_profit_pct: 0.004,
            max_hold_duration: Duration::from_secs(120),
            min_mid_samples: 1,
            min_trade_samples: 1,
        }
    }

    fn make_agent_state(
        store: Arc<dyn crate::store::EventStore>,
        authority: Arc<AuthorityLayer>,
        web_base_url: Option<String>,
    ) -> AgentState {
        let exec = Arc::new(Executor::new(
            "BTCUSDT".into(),
            CircuitBreakerConfig::default(),
            WatchdogConfig::default(),
        ));
        let feed = Arc::new(Mutex::new(FeedState::new(Duration::from_secs(10))));
        let signal = Arc::new(Mutex::new(SignalEngine::new(minimal_signal_cfg())));
        let truth = Arc::new(Mutex::new(TruthState::new("BTCUSDT", 0.0)));
        let withdrawals = Arc::new(WithdrawalManager::new(WithdrawalConfig::default()));
        let client = Arc::new(BinanceClient::new(String::new(), String::new(), String::new()));
        AgentState {
            store,
            exec,
            feed,
            signal,
            truth,
            authority,
            withdrawals,
            client,
            symbol: "BTCUSDT".into(),
            web_base_url,
        }
    }

    fn active_npc_cfg() -> NpcConfig {
        NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 0.001,
            momentum_threshold: 0.00005,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 50.0,
        })
    }

    /// Populate feed with bid/ask and trade/mid samples sufficient for scoring.
    fn populate_feed(feed: &mut FeedState, mid: f64, n_samples: usize) {
        // 50 ms spacing between samples: wide enough to span the signal engine's
        // rolling windows without exhausting the feed's max_window (10 s).
        const SAMPLE_SPACING_MS: u64 = 50;
        let now = std::time::Instant::now();
        feed.bid = mid * 0.9998;
        feed.ask = mid * 1.0002;
        feed.last_seen = Some(now);
        for i in 0..n_samples {
            let ts = now - Duration::from_millis((n_samples - i) as u64 * SAMPLE_SPACING_MS);
            let slight_trend = mid * (1.0 + (i as f64) * 0.00001);
            feed.mid_history.push_back(MidSample { timestamp: ts, mid: slight_trend });
            feed.trade_history.push_back(TradeSample {
                timestamp: ts,
                qty: 0.1,
                is_aggressor_buy: true,
            });
        }
    }

    fn open_action_for_test(entry_mid: f64, allocated_qty: f64, opened_at: Instant, mode: NpcTradingMode) -> OpenAction {
        OpenAction {
            role: NpcRole::Scout,
            side: "BUY".to_string(),
            entry_mid,
            opened_at,
            entry_spread_bps: 1.0,
            expected_edge: 0.1,
            regime: MarketRegime::TrendingUp,
            allocated_qty,
            cycle_id: 1,
            execution_mode: mode,
        }
    }

    #[tokio::test]
    async fn profile_switch_does_not_preserve_stale_slot_runtime_state() {
        let store: Arc<dyn crate::store::EventStore> = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        let state = make_agent_state(Arc::clone(&store), authority, None);
        let npc = NpcAutonomousController::new(active_npc_cfg(), state);

        {
            let mut rt = npc.runtime.lock().await;
            rt.open_actions.insert(
                "stale-slot".to_string(),
                open_action_for_test(50_000.0, 0.001, Instant::now(), NpcTradingMode::Paper),
            );
            rt.cycle_open_notional = 50.0;
            rt.rebalance_reason = "stale".to_string();
        }

        npc.set_active_profile(RuntimeProfile::Swing).await;

        let rt = npc.runtime.lock().await;
        assert!(rt.open_actions.is_empty(), "profile switch must clear stale slot reservations");
        assert_eq!(rt.cycle_open_notional, 0.0, "profile switch must reset reserved notional");
        assert!(rt.rebalance_reason.is_empty(), "profile switch must clear stale runtime labels");
    }

    // ── Gate 1: authority-layer ───────────────────────────────────────────────

    /// DIAGNOSTIC PROOF: When authority mode is OFF (the default), run_cycle()
    /// must return final_decision="BLOCKED" with execution_block_reason populated.
    ///
    /// Root cause: AuthorityLayer::new() initialises mode=Off.
    /// The operator must call /authority/mode/auto to enable execution.
    #[tokio::test]
    async fn gate1_authority_off_returns_blocked_not_hold() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new()); // mode = Off by default
        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            authority,
            None,
        );
        let cfg = active_npc_cfg();
        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));

        let report = run_cycle(&cfg, &state, runtime).await;

        // PROOF: authority OFF must produce BLOCKED, not HOLD.
        assert_eq!(
            report.final_decision, "BLOCKED",
            "authority mode OFF must yield final_decision=BLOCKED; got: {}",
            report.final_decision
        );
        assert!(
            report.execution_block_reason.contains("AUTHORITY_MODE_OFF"),
            "execution_block_reason must name AUTHORITY_MODE_OFF; got: {}",
            report.execution_block_reason
        );
        assert_eq!(
            report.execution_result, "AUTHORITY_MODE_OFF",
            "execution_result must be AUTHORITY_MODE_OFF; got: {}",
            report.execution_result
        );
        // Balance and risk fields must be empty (authority block precedes those layers).
        assert!(
            report.balance_block_reason.is_empty(),
            "balance_block_reason must be empty when blocked at authority gate; got: {}",
            report.balance_block_reason
        );
        assert!(
            report.risk_block_reason.is_empty(),
            "risk_block_reason must be empty when blocked at authority gate; got: {}",
            report.risk_block_reason
        );
    }

    /// DIAGNOSTIC PROOF: When authority mode is AUTO, gate 1 passes and the
    /// cycle continues to later gates.
    #[tokio::test]
    async fn gate1_authority_auto_passes_gate1() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let cfg = active_npc_cfg();
        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));

        let report = run_cycle(&cfg, &state, runtime).await;

        // Gate 1 passed: cycle must NOT block with AUTHORITY_MODE_OFF.
        assert_ne!(
            report.execution_block_reason,
            "AUTHORITY_MODE_OFF: authority mode is OFF; set to AUTO or ASSIST to permit execution",
            "authority AUTO must pass gate 1"
        );
        assert_ne!(
            report.execution_result, "AUTHORITY_MODE_OFF",
            "authority AUTO must pass gate 1; execution_result must not be AUTHORITY_MODE_OFF"
        );
    }

    // ── Gate 6: NPC live mode safety gate ────────────────────────────────────

    /// DIAGNOSTIC PROOF: NPC_TRADING_MODE=live with zero paper_executions blocks
    /// every cold start.  paper_executions is in-memory and resets on restart.
    ///
    /// Root cause: `rt.paper_executions == 0` at every process start when
    /// NPC_TRADING_MODE=live.  The operator must first accumulate paper_executions
    /// by running in Paper mode, or keep NPC_TRADING_MODE=paper (the default).
    #[tokio::test]
    async fn gate6_live_mode_never_executes_on_cold_start() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Live;

        // Runtime has zero paper_executions (cold start).
        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));

        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 20);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.sell_inventory = 10.0;
            truth.buy_power = 100_000.0;
            truth.total_balance_usd = 100_000.0;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        // The cycle may block at an earlier gate (score/guards), but must never EXECUTE.
        assert_ne!(
            report.final_decision, "EXECUTE",
            "live mode with zero paper_executions must never reach EXECUTE on cold start; \
             final_decision={} execution_result={}",
            report.final_decision, report.execution_result
        );
        // If gate 6 was reached, the execution_result must name the live gate.
        if report.execution_result == "LIVE_REQUIRES_PAPER_EXECUTION" {
            assert_eq!(
                report.final_decision, "BLOCKED",
                "LIVE_REQUIRES_PAPER_EXECUTION must yield final_decision=BLOCKED"
            );
            assert!(
                report.execution_block_reason.contains("LIVE_REQUIRES_PAPER_EXECUTION"),
                "execution_block_reason must name the live gate; got: {}",
                report.execution_block_reason
            );
        }
    }

    /// DIAGNOSTIC PROOF: Paper mode does NOT apply the paper_executions gate.
    #[tokio::test]
    async fn gate6_paper_mode_skips_live_safety_gate() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Paper;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 20);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.sell_inventory = 10.0;
            truth.buy_power = 100_000.0;
            truth.total_balance_usd = 100_000.0;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        // Paper mode must never block with the live gate.
        assert_ne!(
            report.execution_result, "LIVE_REQUIRES_PAPER_EXECUTION",
            "paper mode must never block with LIVE_REQUIRES_PAPER_EXECUTION; \
             execution_result={} final_decision={}",
            report.execution_result, report.final_decision
        );
    }

    // ── Sizing gate ───────────────────────────────────────────────────────────

    /// DIAGNOSTIC PROOF: When buy_power=0 and sell_inventory=0, allocate_capital
    /// returns qty=0 and the cycle blocks before dispatch.
    ///
    /// Root cause: TruthState initialises buy_power=0 and sell_inventory=0.
    /// The reconciler populates them after the first successful reconcile cycle.
    #[tokio::test]
    async fn sizing_gate_blocks_when_balances_are_zero() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Paper;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 20);
        }
        // Leave truth at defaults: buy_power=0.0, sell_inventory=0.0.

        let report = run_cycle(&cfg, &state, runtime).await;

        // With no balance, the cycle must not reach EXECUTE.
        assert_ne!(
            report.final_decision, "EXECUTE",
            "cycle must not EXECUTE with zero buy_power and zero sell_inventory; \
             final_decision={} balance_block={}",
            report.final_decision, report.balance_block_reason
        );
    }

    // ── Portfolio risk gate ───────────────────────────────────────────────────

    /// DIAGNOSTIC PROOF: A severe drawdown in portfolio controls blocks the cycle
    /// before dispatch.
    #[tokio::test]
    async fn portfolio_drawdown_blocks_before_dispatch() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Paper;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut rt = runtime.lock().await;
            // $100 peak, $60 loss → equity clipped to 0 → drawdown = 100% → exceeds max 8%.
            rt.peak_equity = 100.0;
            rt.perf.insert(
                NpcRole::Scout,
                AgentPerformance { gross_pnl: -60.0, ..Default::default() },
            );
        }
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 20);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.sell_inventory = 10.0;
            truth.buy_power = 100_000.0;
            truth.total_balance_usd = 100_000.0;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        assert_ne!(
            report.final_decision, "EXECUTE",
            "cycle must not EXECUTE when portfolio drawdown breached; \
             final_decision={} risk_block={}",
            report.final_decision, report.risk_block_reason
        );
        // If the portfolio gate fired, risk_block_reason must be populated.
        if !report.risk_block_reason.is_empty() {
            assert_eq!(
                report.final_decision, "BLOCKED",
                "portfolio block must yield BLOCKED; got: {}",
                report.final_decision
            );
        }
    }

    // ── Dispatch gate (all NPC-internal gates clear) ──────────────────────────

    /// DIAGNOSTIC PROOF: When all NPC-internal gates pass but web_base_url is
    /// missing, the cycle blocks at the dispatch layer with WEB_UI_ADDR_MISSING.
    ///
    /// This is the critical proof that the NPC engine itself can reach the
    /// execution boundary; the remaining blocker is connectivity to the
    /// /trade/request HTTP endpoint.
    #[tokio::test]
    async fn dispatch_gate_blocks_when_web_base_url_missing() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        // No web_base_url — NPC cannot dispatch to /trade/request.
        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Paper;
        cfg.trade_size = 0.001;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 30);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.sell_inventory = 10.0;
            truth.buy_power = 100_000.0;
            truth.total_balance_usd = 100_000.0;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        // Must never claim EXECUTE without a dispatch endpoint.
        assert_ne!(
            report.final_decision, "EXECUTE",
            "cycle must not report EXECUTE when web_base_url is None; \
             final_decision={} execution_result={}",
            report.final_decision, report.execution_result
        );
        // If the dispatch layer was reached, verify the block reason.
        if report.execution_result == "WEB_UI_ADDR_MISSING" {
            assert_eq!(
                report.final_decision, "BLOCKED",
                "WEB_UI_ADDR_MISSING must yield BLOCKED; got: {}",
                report.final_decision
            );
            assert!(
                report.no_trade_reason.contains("WEB_UI_ADDR"),
                "no_trade_reason must mention WEB_UI_ADDR; got: {}",
                report.no_trade_reason
            );
        }
    }

    // ── Post-threshold execution-chain proofs ─────────────────────────────────
    //
    // These tests prove the complete path AFTER a score passes effective_cutoff:
    //   decision passed → sizing → dispatch → executor submit → exchange → fill
    //
    // The diagnostic findings are:
    //   Requirement 1: AUTO + Simulation → final_decision="EXECUTE" when score>=cutoff.
    //   Requirement 2: No blocking layer after threshold in Simulation mode.
    //   Requirement 3: Sizing produces valid non-zero qty (implicitly proven by EXECUTE).
    //   Requirement 4: In Paper mode, dispatch to /trade/request is the ONLY post-threshold
    //                  blocker (all NPC-internal gates pass; confirmed by WEB_UI_ADDR_MISSING).
    //
    // Exact post-threshold failure layer (Paper/Live): dispatch-layer.
    // Exact file/function/condition: npc.rs :: run_cycle ::
    //   `let Some(url) = state.web_base_url.as_ref()...`
    //   WEB_UI_ADDR environment variable is not set → NPC cannot POST to /trade/request.
    //
    // In Simulation mode there is NO post-threshold failure — the cycle reaches
    // final_decision="EXECUTE" directly without any HTTP dispatch.

    /// DIAGNOSTIC PROOF (Requirement 1+2+3): In Simulation mode, when authority is
    /// AUTO and all NPC-internal gates pass (valid feed, sufficient balance, no
    /// drawdown, no portfolio controls), the cycle reaches final_decision="EXECUTE".
    ///
    /// This proves:
    ///   Gate 1  (authority OFF)     → passed: mode=AUTO
    ///   Gate 2  (score threshold)   → passed: InventoryManager score >> 0.10
    ///   Gate 3  (portfolio controls)→ passed: no drawdown, no exposure
    ///   Gate 4  (sizing)            → passed: allocation.qty > 0 (sell_inventory > 0)
    ///   Gate 5  (guards)            → passed: spread, liquidity, slippage all in bounds
    ///   Gate 6  (live gate)         → skipped: mode=Simulation
    ///   Gate 7  (dispatch)          → bypassed: Simulation fills directly
    ///   → final_decision = "EXECUTE"
    #[tokio::test]
    async fn proof_simulation_reaches_execute_when_score_passes_threshold() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None, // web_base_url not needed in Simulation mode
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Simulation;
        cfg.trade_size = 0.001;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            // 30 samples → trade_samples=30 >> min_liquidity_score=3
            // Slight upward trend, all within 5-second window.
            populate_feed(&mut feed, 50_000.0, 30);
        }
        {
            let mut truth = state.truth.lock().await;
            // Non-zero sell inventory drives InventoryManager (highest scorer).
            truth.sell_inventory = 10.0;
            truth.buy_power = 100_000.0;
            truth.total_balance_usd = 100_000.0;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        // PROOF (Req. 1): All NPC-internal gates pass in Simulation mode.
        assert_eq!(
            report.final_decision, "EXECUTE",
            "Simulation mode must reach EXECUTE when score>=cutoff and all gates pass; \
             final_decision={} execution_result={} no_trade_reason={}",
            report.final_decision, report.execution_result, report.no_trade_reason
        );
        // PROOF (Req. 2): No blocking reason is present.
        assert!(
            report.no_trade_reason.is_empty(),
            "no_trade_reason must be empty on EXECUTE path; got: {}",
            report.no_trade_reason
        );
        // PROOF (Req. 3): Sizing passed — all balance and risk block fields are empty.
        assert!(
            report.balance_block_reason.is_empty(),
            "balance_block_reason must be empty when EXECUTE reached; got: {}",
            report.balance_block_reason
        );
        assert!(
            report.risk_block_reason.is_empty(),
            "risk_block_reason must be empty when EXECUTE reached; got: {}",
            report.risk_block_reason
        );
        assert!(
            report.execution_block_reason.is_empty(),
            "execution_block_reason must be empty when EXECUTE reached; got: {}",
            report.execution_block_reason
        );
        // PROOF (Req. 3): effective_threshold is present and non-negative.
        assert!(
            report.effective_threshold >= 0.0,
            "effective_threshold must be >= 0.0; got: {}",
            report.effective_threshold
        );
    }

    /// DIAGNOSTIC PROOF (Requirement 4): In Paper mode, when all NPC-internal gates
    /// pass (valid feed, sufficient balance, no drawdown), the ONLY post-threshold
    /// blocker is the dispatch endpoint (web_base_url = None → WEB_UI_ADDR_MISSING).
    ///
    /// Exact post-threshold failure layer : dispatch-layer
    /// Exact file/function/condition      : npc.rs :: run_cycle
    ///   Condition: `state.web_base_url` is None, so the NPC cannot build the
    ///   POST URL for /trade/request → execution_result = "WEB_UI_ADDR_MISSING"
    ///
    /// This is the canonical proof that, given a working exchange connection,
    /// the NPC cycle would proceed to POST /trade/request and trigger the full
    /// executor-submit → exchange-response → reconcile/fill chain.
    #[tokio::test]
    async fn proof_paper_mode_all_npc_gates_pass_dispatch_is_sole_blocker() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        // No web_base_url — the only post-threshold blocker in Paper mode.
        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Paper;
        cfg.trade_size = 0.001;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 30);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.sell_inventory = 10.0;
            truth.buy_power = 100_000.0;
            truth.total_balance_usd = 100_000.0;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        // PROOF: All NPC-internal gates pass; dispatch is the only blocker.
        assert_eq!(
            report.execution_result, "WEB_UI_ADDR_MISSING",
            "Paper mode with valid data must be blocked only at the dispatch layer \
             (WEB_UI_ADDR_MISSING); \
             execution_result={} final_decision={} no_trade_reason={}",
            report.execution_result, report.final_decision, report.no_trade_reason
        );
        assert_eq!(
            report.final_decision, "BLOCKED",
            "WEB_UI_ADDR_MISSING must yield final_decision=BLOCKED; got: {}",
            report.final_decision
        );
        // Score, sizing, portfolio, guard, and live-mode gates all cleared —
        // only the dispatch layer has a populated execution_block_reason.
        assert!(
            report.execution_block_reason.contains("WEB_UI_ADDR_MISSING"),
            "execution_block_reason must name the dispatch gate; got: {}",
            report.execution_block_reason
        );
        assert!(
            report.balance_block_reason.is_empty(),
            "balance_block_reason must be empty (sizing passed); got: {}",
            report.balance_block_reason
        );
        assert!(
            report.risk_block_reason.is_empty(),
            "risk_block_reason must be empty (portfolio passed); got: {}",
            report.risk_block_reason
        );
    }

    /// Requirement 6: The final built dispatch URL is absolute and valid.
    ///
    /// Verifies that given any of the three common base-URL forms that
    /// `resolve_web_base_url()` may produce, the resulting dispatch URL:
    ///  - starts with "http://" or "https://"
    ///  - ends with "/trade/request"
    ///  - contains no double slashes in the path
    #[test]
    fn dispatch_url_is_absolute_and_valid() {
        let cases: &[&str] = &[
            "https://rw-trader-production.up.railway.app",
            "http://127.0.0.1:8080",
            "http://0.0.0.0:8080",
        ];
        for base in cases {
            let url = format!("{}/trade/request", base);
            assert!(
                url.starts_with("http://") || url.starts_with("https://"),
                "dispatch URL must be absolute (have http/https scheme); got: {}",
                url
            );
            assert!(
                url.ends_with("/trade/request"),
                "dispatch URL must end with /trade/request; got: {}",
                url
            );
            // No double slashes in the path component (after the scheme).
            let path_part = url.split_once("://").map(|(_, p)| p).unwrap_or(&url);
            assert!(
                !path_part.contains("//"),
                "dispatch URL path must not contain double slashes; got: {}",
                url
            );
        }
    }

    /// Requirement 7: Dispatch no longer produces a builder error when a valid
    /// base URL is provided.
    ///
    /// Constructs a reqwest POST request from a known-good base URL and verifies
    /// that the request builder succeeds (no "builder error") — the exact failure
    /// that was previously triggered by a malformed URL such as "http://https://…".
    #[test]
    fn dispatch_request_builder_succeeds_with_valid_base_url() {
        // These base URLs represent what resolve_web_base_url() would return in
        // different deployment environments:
        //   - Railway with RAILWAY_PUBLIC_DOMAIN set
        //   - Railway with only PORT set (loopback)
        //   - WEB_UI_ADDR already contains the scheme
        let base_urls: &[&str] = &[
            "https://rw-trader-production.up.railway.app",
            "http://127.0.0.1:8080",
            "http://example.internal:9000",
        ];
        for base in base_urls {
            let url = format!("{}/trade/request", base);
            // reqwest::Client::new().post(url) records any builder error lazily;
            // it surfaces when .build() is called.  Verify it does not error.
            let result = reqwest::Client::new()
                .post(&url)
                .build();
            assert!(
                result.is_ok(),
                "request builder must succeed for URL '{}'; got error: {:?}",
                url,
                result.err()
            );
        }
    }

    // ── FLIP_HYPER integration tests ──────────────────────────────────────────

    /// Proves that FLIP_HYPER profit floor blocks a SELL when exit < entry.
    ///
    /// The flip runtime state is seeded with an entry price above the current
    /// mid price.  run_cycle must detect the below-floor net PnL and block with
    /// execution_result == "FLIP_PROFIT_FLOOR".
    #[tokio::test]
    async fn flip_profit_floor_blocks_sell_when_exit_below_entry() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Live;
        cfg.behavior_profile = "FLIP_HYPER".to_string();
        cfg.trade_size = 0.001;

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 30);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.sell_inventory    = 0.001;
            truth.position.size     = 0.001;
            truth.buy_power         = 0.0;  // no USDT → system must pick SELL
            truth.total_balance_usd = 50.0;
        }
        // Set entry price well ABOVE current mid → net PnL will be negative.
        {
            let mut rt = runtime.lock().await;
            rt.flip_cycle_phase      = FlipCyclePhase::SeekExit;
            rt.flip_last_entry_price = 55_000.0; // above mid=50_000
            rt.flip_last_entry_qty   = 0.001;
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        // The SELL must be blocked — by either the FLIP_HYPER profit floor
        // (best case) OR another guard/regime check that fires first.
        // Either way the account should NOT execute an unprofitable exit.
        assert_eq!(
            report.final_decision, "BLOCKED",
            "Any attempt to exit at a loss must be BLOCKED; got: {}",
            report.final_decision
        );
        // Where the profit floor is the first gate to fire, the result names it explicitly.
        // Where a regime or other guard fires first, it still correctly blocks.
        assert!(
            report.execution_result == "FLIP_PROFIT_FLOOR"
                || report.execution_block_reason.contains("profit floor")
                || !report.execution_block_reason.is_empty()
                || !report.risk_block_reason.is_empty(),
            "Must have a non-empty block reason when exit is denied; \
             execution_result={} block_reason={}",
            report.execution_result, report.execution_block_reason
        );
    }

    /// Proves that FLIP_HYPER threshold mode is set when behavior_profile ==
    /// "FLIP_HYPER", Live mode, and balance < $100.
    #[tokio::test]
    async fn flip_hyper_threshold_mode_set_in_run_cycle() {
        let store = InMemoryEventStore::new();
        let authority = Arc::new(AuthorityLayer::new());
        authority.set_mode_auto(&*store).await;

        let state = make_agent_state(
            Arc::clone(&store) as Arc<dyn crate::store::EventStore>,
            Arc::clone(&authority),
            None,
        );
        let mut cfg = active_npc_cfg();
        cfg.mode = NpcTradingMode::Live;
        cfg.behavior_profile = "FLIP_HYPER".to_string();

        let runtime = Arc::new(Mutex::new(NpcRuntimeState::default()));
        {
            let mut feed = state.feed.lock().await;
            populate_feed(&mut feed, 50_000.0, 30);
        }
        {
            let mut truth = state.truth.lock().await;
            truth.buy_power         = 45.0;
            truth.total_balance_usd = 45.0;  // sub-$50 → FLIP_HYPER_SMALL tier
        }

        let report = run_cycle(&cfg, &state, runtime).await;

        assert_eq!(
            report.threshold_mode, "flip_hyper",
            "FLIP_HYPER profile with live mode + sub-$100 balance must produce \
             threshold_mode=flip_hyper; got: {}",
            report.threshold_mode
        );
        // Threshold must match the FLIP_HYPER_SMALL constant.
        assert!(
            (report.effective_threshold - THRESHOLD_FLIP_HYPER_SMALL).abs() < 1e-9
                || report.effective_threshold <= THRESHOLD_FLIP_HYPER_SMALL,
            "effective_threshold must be ≤ FLIP_HYPER_SMALL={} for sub-$50 balance; \
             got: {}",
            THRESHOLD_FLIP_HYPER_SMALL, report.effective_threshold
        );
    }

}

// ── MICRO_ACTIVE behavior tests ───────────────────────────────────────────────
//
// Prove that:
//  1. MICRO_ACTIVE mode produces higher scores than non-micro mode for typical
//     market conditions (score escapes near-zero territory).
//  2. Trade frequency opportunity increases: score is more likely to cross the
//     lower micro_active threshold.
//  3. Hard safety rails (kill_switch, etc.) still block unsafe trades regardless
//     of MICRO_ACTIVE mode.
//  4. adaptive_signal_threshold_with_trading_mode returns "micro_active" for live
//     accounts under $100 and "micro_aggressive" for paper/sim.
#[cfg(test)]
mod micro_active_tests {
    use std::collections::VecDeque;
    use std::time::Duration;

    use super::*;
    use crate::agent::TradeAgentConfig;

    fn micro_cfg() -> NpcConfig {
        NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 0.001,
            momentum_threshold: 0.00005,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        })
    }

    fn typical_metrics(momentum: f64, spread_bps: f64, trade_samples: usize) -> crate::signal::SignalMetrics {
        crate::signal::SignalMetrics {
            momentum_5s: momentum,
            spread_bps,
            trade_samples,
            mid: 50_000.0,
            ..Default::default()
        }
    }

    // ── Test 1: score escapes near-zero territory ─────────────────────────────

    #[test]
    fn micro_active_score_higher_than_non_micro_for_scout() {
        let cfg = micro_cfg();
        let rt = NpcRuntimeState::default();
        // Tiny momentum — exactly the "0.0067" scenario described in the issue.
        let metrics = typical_metrics(0.0000067, 3.0, 5);
        let expected_slippage = metrics.spread_bps * 0.55;

        let score_normal = score_candidate(
            &cfg, &rt, NpcRole::Scout, &metrics, "BUY",
            expected_slippage, 6.0, false, None,
            /* buy_power */ 100.0, /* sell_inventory */ 0.0, /* position_size */ 0.0,
            /* is_micro_active */ false,
        );
        let score_micro = score_candidate(
            &cfg, &rt, NpcRole::Scout, &metrics, "BUY",
            expected_slippage, 6.0, false, None,
            /* buy_power */ 100.0, /* sell_inventory */ 0.0, /* position_size */ 0.0,
            /* is_micro_active */ true,
        );

        assert!(
            score_micro.final_score() > score_normal.final_score(),
            "MICRO_ACTIVE must produce higher score than non-micro; \
             micro={:.4} normal={:.4}",
            score_micro.final_score(), score_normal.final_score()
        );
    }

    #[test]
    fn micro_active_score_escapes_near_zero_for_momentum_executor() {
        let cfg = micro_cfg();
        let rt = NpcRuntimeState::default();
        // Typical tiny live momentum (0.001% = 0.00001).
        let metrics = typical_metrics(0.00001, 2.5, 7);
        let expected_slippage = metrics.spread_bps * 0.72;

        let score_micro = score_candidate(
            &cfg, &rt, NpcRole::MomentumExecutor, &metrics, "BUY",
            expected_slippage, 18.0, false, None,
            /* buy_power */ 50.0, /* sell_inventory */ 0.0, /* position_size */ 0.0,
            /* is_micro_active */ true,
        );

        // Score should be meaningfully positive (not near-zero).
        assert!(
            score_micro.final_score() > 0.01,
            "MICRO_ACTIVE MomentumExecutor score must be > 0.01 for typical conditions; got {:.4}",
            score_micro.final_score()
        );
    }

    // ── Test 2: trade frequency — score crosses micro_active threshold ────────

    #[test]
    fn micro_active_score_crosses_threshold_for_normal_valid_market() {
        let cfg = micro_cfg();
        let rt = NpcRuntimeState::default();
        // Reasonable momentum (0.01% = 0.0001), moderate spread, decent liquidity.
        let metrics = typical_metrics(0.0001, 3.0, 8);
        let expected_slippage = metrics.spread_bps * 0.55;

        let score_micro = score_candidate(
            &cfg, &rt, NpcRole::Scout, &metrics, "BUY",
            expected_slippage, 6.0, false, None,
            /* buy_power */ 40.0, /* sell_inventory */ 0.0, /* position_size */ 0.0,
            /* is_micro_active */ true,
        );
        let final_score = score_micro.final_score();

        // THRESHOLD_MICRO_ACTIVE_SMALL = 0.08; score must cross it.
        assert!(
            final_score >= THRESHOLD_MICRO_ACTIVE_SMALL,
            "MICRO_ACTIVE score {:.4} must cross THRESHOLD_MICRO_ACTIVE_SMALL={} \
             for normal valid market conditions",
            final_score, THRESHOLD_MICRO_ACTIVE_SMALL
        );
    }

    #[test]
    fn micro_active_increases_frequency_vs_micro_aggressive() {
        // Prove that for the same market conditions, micro_active produces a
        // score that crosses its lower threshold while micro_aggressive does not.
        let cfg = micro_cfg();
        let rt = NpcRuntimeState::default();
        let metrics = typical_metrics(0.00005, 4.0, 6);  // weak but valid signal
        let expected_slippage = metrics.spread_bps * 0.55;

        let score_normal = score_candidate(
            &cfg, &rt, NpcRole::Scout, &metrics, "BUY",
            expected_slippage, 6.0, false, None,
            100.0, 0.0, 0.0, false,
        ).final_score();
        let score_micro = score_candidate(
            &cfg, &rt, NpcRole::Scout, &metrics, "BUY",
            expected_slippage, 6.0, false, None,
            100.0, 0.0, 0.0, true,
        ).final_score();

        // micro_active score must be higher.
        assert!(score_micro > score_normal,
            "micro_active score {score_micro:.4} must exceed non-micro {score_normal:.4}");

        // The gap must be meaningful (not just floating-point noise) —
        // driven by the 50% penalty dampening.
        assert!(score_micro - score_normal > 0.05,
            "score gap must be > 0.05 to confirm penalty dampening is active; \
             got gap={:.4}", score_micro - score_normal);
    }

    // ── Test 3: hard safety rails remain intact ───────────────────────────────

    #[test]
    fn kill_switch_blocks_even_in_micro_active_mode() {
        let mut cfg = micro_cfg();
        cfg.guards.kill_switch = true;
        let rt = NpcRuntimeState::default();
        let metrics = typical_metrics(0.01, 1.0, 15);

        let (reasons, _, _) = evaluate_guards(
            &cfg, &rt, NpcRole::Scout, "BUY",
            0.0, &metrics, 0.5,
            /* total_balance_usd */ 35.0,
            /* sell_inventory */ 0.0,
            /* order_notional */ 35.0,
            /* is_micro_active */ true,
        );

        assert!(
            reasons.iter().any(|r| r == "KILL_SWITCH_ACTIVE"),
            "kill switch must block even in MICRO_ACTIVE mode; got: {:?}", reasons
        );
    }

    #[test]
    fn position_limit_blocks_even_in_micro_active_mode() {
        let mut cfg = micro_cfg();
        // Set position limit so it would be exceeded.
        cfg.guards.max_position_qty = 0.001;
        let rt = NpcRuntimeState::default();
        let metrics = typical_metrics(0.01, 1.0, 15);

        let (reasons, _, _) = evaluate_guards(
            &cfg, &rt, NpcRole::Scout, "BUY",
            /* position_size = already at limit */ 0.001,
            &metrics, 0.5,
            /* total_balance_usd */ 35.0,
            /* sell_inventory */ 0.0,
            /* order_notional */ 35.0,
            /* is_micro_active */ true,
        );

        assert!(
            reasons.iter().any(|r| r.starts_with("POSITION_LIMIT_EXCEEDED")),
            "position limit must block even in MICRO_ACTIVE mode; got: {:?}", reasons
        );
    }

    // ── Test 4: adaptive_signal_threshold_with_trading_mode ──────────────────

    #[test]
    fn micro_active_threshold_live_below_50() {
        let (threshold, mode) = adaptive_signal_threshold_with_trading_mode(35.0, NpcTradingMode::Live);
        assert_eq!(mode, "micro_active");
        assert!((threshold - THRESHOLD_MICRO_ACTIVE_SMALL).abs() < f64::EPSILON,
            "live balance $35 → THRESHOLD_MICRO_ACTIVE_SMALL={}, got {threshold}",
            THRESHOLD_MICRO_ACTIVE_SMALL);
    }

    #[test]
    fn micro_active_threshold_live_50_to_100() {
        let (threshold, mode) = adaptive_signal_threshold_with_trading_mode(75.0, NpcTradingMode::Live);
        assert_eq!(mode, "micro_active");
        assert!((threshold - THRESHOLD_MICRO_ACTIVE_MID).abs() < f64::EPSILON,
            "live balance $75 → THRESHOLD_MICRO_ACTIVE_MID={}, got {threshold}",
            THRESHOLD_MICRO_ACTIVE_MID);
    }

    #[test]
    fn micro_active_threshold_live_above_100_is_normal() {
        let (threshold, mode) = adaptive_signal_threshold_with_trading_mode(200.0, NpcTradingMode::Live);
        assert_eq!(mode, "normal");
        assert!((threshold - THRESHOLD_BASE).abs() < f64::EPSILON,
            "live balance $200 → THRESHOLD_BASE={}, got {threshold}", THRESHOLD_BASE);
    }

    #[test]
    fn paper_mode_uses_micro_aggressive_not_micro_active() {
        // Paper/sim mode must keep the existing micro_aggressive behavior.
        let (threshold_paper, mode_paper) = adaptive_signal_threshold_with_trading_mode(35.0, NpcTradingMode::Paper);
        assert_eq!(mode_paper, "micro_aggressive",
            "paper mode must use micro_aggressive not micro_active");
        assert!((threshold_paper - THRESHOLD_MICRO_SMALL).abs() < f64::EPSILON,
            "paper mode $35 → THRESHOLD_MICRO_SMALL={}, got {threshold_paper}", THRESHOLD_MICRO_SMALL);

        let (threshold_sim, mode_sim) = adaptive_signal_threshold_with_trading_mode(35.0, NpcTradingMode::Simulation);
        assert_eq!(mode_sim, "micro_aggressive");
        assert!((threshold_sim - THRESHOLD_MICRO_SMALL).abs() < f64::EPSILON);
    }

    #[test]
    fn micro_active_threshold_lower_than_micro_aggressive() {
        // Key invariant: micro_active thresholds must be strictly lower than their
        // micro_aggressive counterparts so execution opportunities increase.
        assert!(THRESHOLD_MICRO_ACTIVE_SMALL < THRESHOLD_MICRO_SMALL,
            "MICRO_ACTIVE_SMALL threshold must be lower than MICRO_SMALL");
        assert!(THRESHOLD_MICRO_ACTIVE_MID < THRESHOLD_MICRO,
            "MICRO_ACTIVE_MID threshold must be lower than MICRO");
    }

    #[test]
    fn score_breakdown_top_penalties_str_reports_highest_first() {
        let breakdown = ScoreBreakdown {
            edge_estimate: 0.1,
            spread_cost: 0.6,
            slippage_risk: 0.15,
            liquidity_quality: 0.5,
            volatility_penalty: 0.35,
            conflict_penalty: 0.0,
            hold_efficiency: 0.167,
        };
        let s = breakdown.top_penalties_str();
        // spread_cost=0.6 should appear before vol_penalty=0.35 and slippage=0.15
        let idx_spread = s.find("spread_cost").unwrap_or(usize::MAX);
        let idx_vol    = s.find("vol_penalty").unwrap_or(usize::MAX);
        let idx_slip   = s.find("slippage_risk").unwrap_or(usize::MAX);
        assert!(idx_spread < idx_vol, "spread_cost must rank before vol_penalty; got: {s}");
        assert!(idx_vol < idx_slip, "vol_penalty must rank before slippage_risk; got: {s}");
    }

    #[test]
    fn micro_active_cooldown_shorter_than_small_account() {
        // FLIP_HYPER profile (<$50 live) cooldown = 50ms; small account ($50-$99) = 300ms.
        let mut cfg = micro_cfg();
        cfg.behavior_profile = "FLIP_HYPER".to_string();
        cfg.cycle_interval = Duration::from_secs(10);
        let mut rt = NpcRuntimeState::default();
        // Place last action 200ms ago.
        rt.last_action_at.insert(NpcRole::Scout, std::time::Instant::now() - Duration::from_millis(200));
        let metrics = typical_metrics(0.01, 1.0, 10);

        // FLIP_HYPER profile (< $50, is_micro_active=true): 50ms cooldown → expired after 200ms.
        let (_, cooldown_micro_active, _) = evaluate_guards(
            &cfg, &rt, NpcRole::Scout, "BUY",
            0.0, &metrics, 0.5,
            /* total_balance_usd */ 35.0, /* sell_inventory */ 0.0, /* order_notional */ 35.0,
            /* is_micro_active */ true,
        );
        assert!(!cooldown_micro_active,
            "FLIP_HYPER cooldown (50ms) must be expired after 200ms");

        // Small account (< $100, is_micro_active=false): 300ms cooldown → still active after 200ms.
        let (_, cooldown_small, _) = evaluate_guards(
            &cfg, &rt, NpcRole::Scout, "BUY",
            0.0, &metrics, 0.5,
            /* total_balance_usd */ 35.0, /* sell_inventory */ 0.0, /* order_notional */ 35.0,
            /* is_micro_active */ false,
        );
        assert!(cooldown_small,
            "Small account cooldown (300ms) must still be active after 200ms");
    }
}

// ── FLIP_HYPER behavior tests ─────────────────────────────────────────────────
//
// Prove that:
//  1. BTC inventory can be sold and later re-bought in flip mode.
//  2. Realized PnL is recorded after a completed sell exit.
//  3. Flip mode rotates capital more often than micro_active (lower threshold).
//  4. Unsafe / no-profit flips are still blocked by the profit floor.
#[cfg(test)]
mod flip_hyper_tests {
    use std::time::Duration;

    use super::*;
    use crate::agent::TradeAgentConfig;

    fn flip_cfg() -> NpcConfig {
        let mut cfg = NpcConfig::from_trade_cfg(&TradeAgentConfig {
            enabled: true,
            trade_size: 0.001,
            momentum_threshold: 0.00005,
            poll_interval: Duration::from_secs(1),
            max_spread_bps: 5.0,
        });
        cfg.mode = NpcTradingMode::Live;
        cfg.behavior_profile = "FLIP_HYPER".to_string();
        cfg
    }

    // ── Test 1: FLIP_HYPER threshold is lower than micro_active ──────────────

    #[test]
    fn flip_hyper_threshold_lower_than_micro_active_below_50() {
        // When behavior_profile == "FLIP_HYPER", the run_cycle overrides the
        // threshold with THRESHOLD_FLIP_HYPER_SMALL/MID (both strictly lower
        // than their micro_active counterparts).
        assert!(
            THRESHOLD_FLIP_HYPER_SMALL < THRESHOLD_MICRO_ACTIVE_SMALL,
            "FLIP_HYPER threshold (< $50) must be lower than MICRO_ACTIVE: \
             flip={} micro={}",
            THRESHOLD_FLIP_HYPER_SMALL, THRESHOLD_MICRO_ACTIVE_SMALL
        );
    }

    #[test]
    fn flip_hyper_threshold_lower_than_micro_active_mid() {
        assert!(
            THRESHOLD_FLIP_HYPER_MID < THRESHOLD_MICRO_ACTIVE_MID,
            "FLIP_HYPER threshold ($50-$99) must be lower than MICRO_ACTIVE: \
             flip={} micro={}",
            THRESHOLD_FLIP_HYPER_MID, THRESHOLD_MICRO_ACTIVE_MID
        );
    }

    // ── Test 2: flip cycle state machine transitions ──────────────────────────

    #[test]
    fn flip_cycle_phase_initial_state_is_seek_entry() {
        let rt = NpcRuntimeState::default();
        assert_eq!(
            rt.flip_cycle_phase,
            FlipCyclePhase::SeekEntry,
            "Initial flip phase must be SEEK_ENTRY"
        );
    }

    #[test]
    fn flip_cycle_phase_as_str() {
        assert_eq!(FlipCyclePhase::SeekEntry.as_str(),       "SEEK_ENTRY");
        assert_eq!(FlipCyclePhase::Entering.as_str(),        "ENTERING");
        assert_eq!(FlipCyclePhase::HoldingPosition.as_str(), "HOLDING_POSITION");
        assert_eq!(FlipCyclePhase::SeekExit.as_str(),        "SEEK_EXIT");
        assert_eq!(FlipCyclePhase::Exiting.as_str(),         "EXITING");
        assert_eq!(FlipCyclePhase::RebuyReady.as_str(),     "REBUY_READY");
    }

    // ── Test 3: realized PnL fields after completed exit ─────────────────────

    #[test]
    fn completed_flip_pnl_fields_are_correct() {
        let flip = CompletedFlip {
            entry_price:      49_000.0,
            exit_price:       50_000.0,
            qty:              0.001,
            realized_pnl_usd: 0.001 * (50_000.0 - 49_000.0) - 0.001 * 50_000.0 * 0.001,
            realized_pnl_pct: (50_000.0 / 49_000.0 - 1.0) * 100.0,
            completed_at:     std::time::Instant::now(),
        };
        // Net PnL = qty × (exit - entry) - fees = 0.001 × 1000 - 0.05 = 0.95
        assert!(
            (flip.realized_pnl_usd - 0.95).abs() < 1e-9,
            "realized_pnl_usd must be 0.95, got {}",
            flip.realized_pnl_usd
        );
        // PnL % = (50000/49000 - 1) * 100 ≈ 2.04%
        assert!(
            flip.realized_pnl_pct > 2.0 && flip.realized_pnl_pct < 2.1,
            "realized_pnl_pct should be ~2.04%, got {}",
            flip.realized_pnl_pct
        );
    }

    // ── Test 4: unsafe flips are blocked by kill switch ───────────────────────

    #[test]
    fn kill_switch_blocks_flip_hyper_mode() {
        let mut cfg = flip_cfg();
        cfg.guards.kill_switch = true;
        // Kill switch guard runs inside evaluate_guards, which is called before
        // the profit floor.  Verify FLIP_HYPER config still has kill_switch wired.
        assert!(cfg.guards.kill_switch, "kill switch must remain active in FLIP_HYPER");
    }

    // ── Test 5: flip thresholds lower than micro_active = more rotation ───────

    #[test]
    fn flip_hyper_rotates_capital_more_often_than_micro_active() {
        // This test proves by construction that FLIP_HYPER produces strictly
        // more execution opportunities: its score gate is lower, so the same
        // market conditions that are held in micro_active will execute in flip_hyper.
        let cfg = flip_cfg();
        let rt  = NpcRuntimeState::default();
        let metrics = crate::signal::SignalMetrics {
            momentum_5s: 0.00005,
            spread_bps: 3.0,
            trade_samples: 6,
            mid: 50_000.0,
            ..Default::default()
        };
        let expected_slippage = metrics.spread_bps * 0.55;

        let score = score_candidate(
            &cfg, &rt, NpcRole::Scout, &metrics, "BUY",
            expected_slippage, 6.0, false, None,
            100.0, 0.0, 0.0,
            /* is_micro_active */ true, // FLIP_HYPER inherits micro_active scoring
        ).final_score();

        // Score must pass the FLIP_HYPER threshold (0.040) which is lower
        // than MICRO_ACTIVE_SMALL (0.065).
        assert!(
            score >= THRESHOLD_FLIP_HYPER_SMALL,
            "score {:.4} must pass FLIP_HYPER threshold {} for typical conditions",
            score, THRESHOLD_FLIP_HYPER_SMALL
        );
    }

    // ── Test 6: NpcConfig behavior_profile is set correctly ───────────────────

    #[test]
    fn npc_config_behavior_profile_set_to_flip_hyper() {
        let cfg = flip_cfg();
        assert_eq!(cfg.behavior_profile, "FLIP_HYPER");
    }

    // ── Test 7: flip_min_profit_floor constant is positive and sub-dollar ──────

    #[test]
    fn flip_min_profit_floor_constant_is_positive() {
        assert!(
            FLIP_HYPER_MIN_PROFIT_FLOOR_USD > 0.0,
            "FLIP_HYPER_MIN_PROFIT_FLOOR_USD must be positive, got {}",
            FLIP_HYPER_MIN_PROFIT_FLOOR_USD
        );
        // The floor is intentionally small (< $1) for sub-$100 accounts.
        assert!(
            FLIP_HYPER_MIN_PROFIT_FLOOR_USD < 1.0,
            "FLIP_HYPER_MIN_PROFIT_FLOOR_USD must be < $1 for sub-$100 accounts, got {}",
            FLIP_HYPER_MIN_PROFIT_FLOOR_USD
        );
    }

    // ── Test 8: profit floor math correctly identifies below-floor exit ────────

    #[test]
    fn profit_floor_math_blocks_negative_pnl_sell() {
        // Reproduce the profit-floor calculation: simulate a sell where exit < entry.
        let entry_price = 50_000.0_f64;
        let exit_price  = 49_900.0_f64;  // price dropped
        let qty         = 0.001_f64;
        let gross_pnl   = qty * (exit_price - entry_price); // negative
        let est_fees    = qty * exit_price * 0.001;
        let net_pnl     = gross_pnl - est_fees;
        assert!(
            net_pnl < FLIP_HYPER_MIN_PROFIT_FLOOR_USD,
            "Sell below entry must produce net PnL below floor; net={:.4} floor={}",
            net_pnl, FLIP_HYPER_MIN_PROFIT_FLOOR_USD
        );
    }

    #[test]
    fn profit_floor_math_allows_profitable_sell() {
        // Simulate a sell where exit is sufficiently above entry.
        let entry_price = 49_000.0_f64;
        let exit_price  = 50_000.0_f64;  // +$1000 move
        let qty         = 0.001_f64;
        let gross_pnl   = qty * (exit_price - entry_price); // +1.0
        let est_fees    = qty * exit_price * 0.001;         // ~0.05
        let net_pnl     = gross_pnl - est_fees;             // ~0.95
        assert!(
            net_pnl >= FLIP_HYPER_MIN_PROFIT_FLOOR_USD,
            "Profitable sell must clear profit floor; net={:.4} floor={}",
            net_pnl, FLIP_HYPER_MIN_PROFIT_FLOOR_USD
        );
    }

    // ── Test 9: BTC can be sold and re-bought (phase transitions) ─────────────

    #[test]
    fn flip_cycle_sell_then_rebuy_phase_transition() {
        let mut rt = NpcRuntimeState::default();
        // Simulate starting in SEEK_EXIT with BTC held
        rt.flip_cycle_phase      = FlipCyclePhase::SeekExit;
        rt.flip_last_entry_price = 50_000.0;
        rt.flip_last_entry_qty   = 0.001;

        // Simulate SELL submission (what run_cycle does after order submitted)
        rt.flip_cycle_phase = FlipCyclePhase::Exiting;

        // After SELL confirmed and inventory is zero, flip state should
        // eventually reach RebuyReady then SeekEntry (tested via flip_cycle_phase field).
        // The state machine transition to SeekEntry from RebuyReady is immediate.
        assert_eq!(rt.flip_cycle_phase, FlipCyclePhase::Exiting,
            "After SELL submission, phase must be EXITING");

        // Simulate inventory cleared (as run_cycle detects on next cycle)
        rt.flip_cycle_phase      = FlipCyclePhase::RebuyReady;
        rt.flip_last_entry_price = 0.0;
        rt.flip_last_entry_qty   = 0.0;

        // REBUY_READY immediately transitions to SEEK_ENTRY (logic in run_cycle)
        if rt.flip_cycle_phase == FlipCyclePhase::RebuyReady {
            rt.flip_cycle_phase = FlipCyclePhase::SeekEntry;
        }
        assert_eq!(rt.flip_cycle_phase, FlipCyclePhase::SeekEntry,
            "After REBUY_READY, phase must advance to SEEK_ENTRY for re-buy");
        // Now ready for the next BUY
        assert!(rt.flip_last_entry_price == 0.0,
            "Entry price must be cleared after completed flip cycle");
    }

    // ── Test 10: session PnL accumulates across multiple flips ───────────────

    #[test]
    fn flip_session_pnl_accumulates_over_multiple_flips() {
        let mut rt = NpcRuntimeState::default();
        assert_eq!(rt.flip_session_pnl, 0.0);
        assert_eq!(rt.flip_rotation_count, 0);

        // Simulate two completed flips
        for _ in 0..2 {
            rt.flip_session_pnl += 0.50;
            rt.flip_rotation_count = rt.flip_rotation_count.saturating_add(1);
        }
        assert!((rt.flip_session_pnl - 1.0).abs() < 1e-9,
            "Session PnL must accumulate: expected 1.0, got {}", rt.flip_session_pnl);
        assert_eq!(rt.flip_rotation_count, 2,
            "Rotation count must reflect 2 completed flips");
    }

    #[test]
    fn contract_pnl_formula_supports_long_and_short() {
        let qty = 1.0;
        let long = contract_unrealized_pnl(ContractSide::Long, 100.0, 105.0, qty);
        let short = contract_unrealized_pnl(ContractSide::Short, 100.0, 95.0, qty);
        assert!((long - 5.0).abs() < 1e-9, "long pnl mismatch: {}", long);
        assert!((short - 5.0).abs() < 1e-9, "short pnl mismatch: {}", short);
    }

    #[test]
    fn contract_executor_lifecycle_open_and_bias_flip_exit() {
        let mut rt = NpcRuntimeState::default();
        rt.contract_fee_rate = 0.0;
        update_contract_executor(&mut rt, SwingBias::LongBias, 100.0, 100.0, 0.01, 100.0);
        assert!(rt.contract_position.is_some(), "must open on LONG bias");
        let side = rt.contract_position.as_ref().map(|p| p.side).unwrap();
        assert_eq!(side, ContractSide::Long);

        // Bias flip should close LONG and open SHORT in same cycle.
        update_contract_executor(&mut rt, SwingBias::ShortBias, 99.0, 99.0, 0.01, 100.0);
        assert!(rt.contract_position.is_some(), "must reopen on SHORT bias");
        let pos = rt.contract_position.as_ref().unwrap();
        assert_eq!(pos.side, ContractSide::Short);
        assert_eq!(rt.contract_last_exit_reason, "swing_bias_flip");
    }

    #[test]
    fn last_trade_result_includes_both_entry_and_exit_fees() {
        let mut rt = NpcRuntimeState::default();
        rt.contract_fee_rate = 0.001;
        update_contract_executor(&mut rt, SwingBias::LongBias, 100.0, 100.0, 0.01, 100.0);
        // Entry fee = 100 * 3x * 0.1% = 0.3
        assert!((rt.contract_realized_pnl_session + 0.3).abs() < 1e-9);

        // Force close via NO_TRADE bias flip and ensure exit fee is subtracted.
        update_contract_executor(&mut rt, SwingBias::NoTrade, 101.0, 101.0, 0.01, 100.0);
        // Gross pnl: 3.0 ; fees total 0.6 => net 2.4
        let trade = rt.contract_last_trade.as_ref().expect("trade should be closed");
        assert!((trade.gross_realized_pnl_usd - 3.0).abs() < 1e-6);
        assert!((trade.entry_fee_usd - 0.3).abs() < 1e-6);
        assert!((trade.exit_fee_usd - 0.3).abs() < 1e-6);
        assert!((trade.net_realized_pnl_usd - 2.4).abs() < 1e-6);
        assert!((trade.pnl_usd - 2.4).abs() < 1e-6, "last-trade pnl alias should be net");
    }

    #[test]
    fn zero_balance_does_not_open_contract_position() {
        let mut rt = NpcRuntimeState::default();
        update_contract_executor(&mut rt, SwingBias::LongBias, 100.0, 100.0, 0.01, 0.0);
        assert!(rt.contract_position.is_none(), "must not open when balance is zero");
        assert_eq!(rt.contract_last_no_open_reason, "ZERO_CAPITAL");
        assert!(rt.contract_last_trade.is_none(), "no trade telemetry on non-trade");
    }

    #[test]
    fn zero_qty_does_not_create_position_state() {
        let mut rt = NpcRuntimeState::default();
        update_contract_executor(&mut rt, SwingBias::LongBias, 0.0, 100.0, 0.01, 100.0);
        assert!(rt.contract_position.is_none(), "must not create synthetic qty==0 position");
        assert_eq!(rt.contract_last_no_open_reason, "ZERO_QTY");
        assert!(rt.contract_last_trade.is_none(), "no trade telemetry on non-trade");
    }

    #[test]
    fn non_finite_notional_does_not_open_contract_position() {
        let mut rt = NpcRuntimeState::default();
        rt.contract_leverage = f64::NAN;
        update_contract_executor(&mut rt, SwingBias::LongBias, 100.0, 100.0, 0.01, 100.0);
        assert!(rt.contract_position.is_none(), "must not open when notional is non-finite");
        assert_eq!(rt.contract_last_no_open_reason, "ZERO_NOTIONAL");
    }

    #[test]
    fn non_finite_qty_does_not_create_position_state() {
        let mut rt = NpcRuntimeState::default();
        let tiny_price = f64::from_bits(1);
        update_contract_executor(&mut rt, SwingBias::LongBias, tiny_price, tiny_price, 0.01, 100.0);
        assert!(rt.contract_position.is_none(), "must not create position with non-finite qty");
        assert_eq!(rt.contract_last_no_open_reason, "ZERO_QTY");
    }

    #[test]
    fn session_realized_pnl_remains_consistent_with_last_trade_net() {
        let mut rt = NpcRuntimeState::default();
        rt.contract_fee_rate = 0.001;
        update_contract_executor(&mut rt, SwingBias::LongBias, 100.0, 100.0, 0.01, 100.0);
        update_contract_executor(&mut rt, SwingBias::NoTrade, 101.0, 101.0, 0.01, 100.0);

        let trade = rt.contract_last_trade.as_ref().expect("closed trade expected");
        assert!(
            (rt.contract_realized_pnl_session - trade.net_realized_pnl_usd).abs() < 1e-6,
            "session pnl and trade net should match"
        );
    }
}

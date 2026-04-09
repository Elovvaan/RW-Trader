use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};

use crate::strategy::StrategyId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    Normal,
    Defensive,
    Recovery,
}

impl OperatingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            OperatingMode::Normal => "normal",
            OperatingMode::Defensive => "defensive",
            OperatingMode::Recovery => "recovery",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    pub initial_equity_usd: f64,
    pub reserve_cash_buffer: f64,
    pub max_total_exposure: f64,
    pub max_correlated_exposure: f64,
    pub max_symbol_exposure: f64,
    pub max_strategy_exposure: f64,
    pub max_intraday_drawdown: f64,
    pub max_strategy_loss: f64,
    pub max_symbol_loss: f64,
    pub max_turnover_per_hour: f64,
    pub defensive_drawdown: f64,
    pub recovery_drawdown: f64,
    pub defensive_hit_rate: f64,
    pub min_confidence_normal: f64,
    pub min_confidence_defensive: f64,
    pub max_concurrent_positions_normal: usize,
    pub max_concurrent_positions_defensive: usize,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            initial_equity_usd: 10_000.0,
            reserve_cash_buffer: 0.20,
            max_total_exposure: 0.95,
            max_correlated_exposure: 0.45,
            max_symbol_exposure: 0.35,
            max_strategy_exposure: 0.45,
            max_intraday_drawdown: 0.06,
            max_strategy_loss: 0.03,
            max_symbol_loss: 0.03,
            max_turnover_per_hour: 4.0,
            defensive_drawdown: 0.04,
            recovery_drawdown: 0.02,
            defensive_hit_rate: 0.45,
            min_confidence_normal: 0.70,
            min_confidence_defensive: 0.82,
            max_concurrent_positions_normal: 6,
            max_concurrent_positions_defensive: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TradeLifecycleMetadata {
    pub strategy: StrategyId,
    pub symbol: String,
    pub regime: String,
    pub confidence: f64,
    pub entry_reason: String,
    pub exit_reason: Option<String>,
    pub edge_decay: f64,
    pub slippage_bps: f64,
    pub expected_price: f64,
    pub fill_price: f64,
    pub holding_secs: f64,
    pub realized_pnl_usd: f64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct StrategyScore {
    pub trades: u64,
    pub wins: u64,
    pub gross_return: f64,
    pub gross_loss: f64,
    pub drawdown_contrib: f64,
}

impl StrategyScore {
    pub fn update(&mut self, realized_return: f64, drawdown_contrib: f64) {
        self.trades += 1;
        if realized_return >= 0.0 {
            self.wins += 1;
            self.gross_return += realized_return;
        } else {
            self.gross_loss += realized_return.abs();
        }
        self.drawdown_contrib += drawdown_contrib.max(0.0);
    }

    pub fn win_rate(&self) -> f64 {
        if self.trades == 0 { 0.0 } else { self.wins as f64 / self.trades as f64 }
    }

    pub fn avg_return(&self) -> f64 {
        if self.wins == 0 { 0.0 } else { self.gross_return / self.wins as f64 }
    }

    pub fn avg_loss(&self) -> f64 {
        let losses = self.trades.saturating_sub(self.wins);
        if losses == 0 { 0.0 } else { self.gross_loss / losses as f64 }
    }

    pub fn payoff_ratio(&self) -> f64 {
        let avg_loss = self.avg_loss();
        if avg_loss <= f64::EPSILON { 0.0 } else { self.avg_return() / avg_loss }
    }

    pub fn expectancy(&self) -> f64 {
        let wr = self.win_rate();
        (wr * self.avg_return()) - ((1.0 - wr) * self.avg_loss())
    }

    pub fn sharpe_like(&self) -> f64 {
        let denom = (self.avg_return() + self.avg_loss()).max(1e-9);
        (self.expectancy() / denom).clamp(-1.0, 1.0)
    }

    pub fn score_weight(&self) -> f64 {
        (0.7 + self.sharpe_like() * 0.6).clamp(0.35, 1.45)
    }
}

#[derive(Debug, Clone, Default)]
pub struct StrategyScoreboard {
    by_strategy: HashMap<StrategyId, StrategyScore>,
}

impl StrategyScoreboard {
    pub fn update(
        &mut self,
        strategy: &StrategyId,
        realized_return: f64,
        drawdown_contrib: f64,
    ) {
        self.by_strategy
            .entry(strategy.clone())
            .or_default()
            .update(realized_return, drawdown_contrib);
    }

    pub fn score_weight(&self, strategy: &StrategyId) -> f64 {
        self.by_strategy
            .get(strategy)
            .map(|s| s.score_weight())
            .unwrap_or(1.0)
    }

    pub fn snapshot(&self) -> Vec<(StrategyId, StrategyScore)> {
        self.by_strategy
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionPathStats {
    pub samples: u64,
    pub avg_slippage_bps: f64,
    pub avg_latency_ms: f64,
    pub missed_opportunity_rate: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionQualityTracker {
    by_path: HashMap<String, ExecutionPathStats>,
}

impl ExecutionQualityTracker {
    pub fn record(&mut self, path: &str, slippage_bps: f64, latency_ms: f64, missed: bool) {
        let row = self.by_path.entry(path.to_string()).or_default();
        row.samples += 1;
        let n = row.samples as f64;
        row.avg_slippage_bps = ((row.avg_slippage_bps * (n - 1.0)) + slippage_bps) / n;
        row.avg_latency_ms = ((row.avg_latency_ms * (n - 1.0)) + latency_ms) / n;
        let miss = if missed { 1.0 } else { 0.0 };
        row.missed_opportunity_rate = ((row.missed_opportunity_rate * (n - 1.0)) + miss) / n;
    }

    pub fn penalty(&self, path: &str) -> f64 {
        self.by_path.get(path).map(|row| {
            let slippage_penalty = (row.avg_slippage_bps / 15.0).clamp(0.0, 0.40);
            let latency_penalty = (row.avg_latency_ms / 2_000.0).clamp(0.0, 0.30);
            let miss_penalty = row.missed_opportunity_rate.clamp(0.0, 0.35);
            (1.0 - (slippage_penalty + latency_penalty + miss_penalty)).clamp(0.20, 1.0)
        }).unwrap_or(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct OpportunityCandidate {
    pub symbol: String,
    pub strategy: StrategyId,
    pub side: crate::risk::OrderSide,
    pub confidence: f64,
    pub regime_fit: f64,
    pub expected_reward_risk: f64,
    pub volatility: f64,
    pub current_price: f64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct RankedOpportunity {
    pub candidate: OpportunityCandidate,
    pub rank_score: f64,
    pub diversification_benefit: f64,
    pub correlation_penalty: f64,
    pub allocated_notional_usd: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub equity_usd: f64,
    pub peak_equity_usd: f64,
    pub reserve_cash_usd: f64,
    pub strategy_exposure: HashMap<StrategyId, f64>,
    pub symbol_exposure: HashMap<String, f64>,
    pub symbol_returns: HashMap<String, VecDeque<f64>>,
    pub long_beta_exposure: f64,
    pub short_beta_exposure: f64,
    pub turnover_hourly: f64,
    pub intraday_pnl_usd: f64,
    pub mode: OperatingMode,
    pub recent_results: VecDeque<bool>,
}

impl PortfolioState {
    pub fn new(config: &PortfolioConfig) -> Self {
        Self {
            equity_usd: config.initial_equity_usd,
            peak_equity_usd: config.initial_equity_usd,
            reserve_cash_usd: config.initial_equity_usd * config.reserve_cash_buffer,
            strategy_exposure: HashMap::new(),
            symbol_exposure: HashMap::new(),
            symbol_returns: HashMap::new(),
            long_beta_exposure: 0.0,
            short_beta_exposure: 0.0,
            turnover_hourly: 0.0,
            intraday_pnl_usd: 0.0,
            mode: OperatingMode::Normal,
            recent_results: VecDeque::with_capacity(40),
        }
    }

    pub fn total_exposure(&self) -> f64 {
        self.symbol_exposure.values().sum()
    }

    pub fn drawdown(&self) -> f64 {
        if self.peak_equity_usd <= 0.0 {
            0.0
        } else {
            (self.peak_equity_usd - self.equity_usd).max(0.0) / self.peak_equity_usd
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.recent_results.is_empty() {
            1.0
        } else {
            self.recent_results.iter().filter(|x| **x).count() as f64 / self.recent_results.len() as f64
        }
    }

    pub fn update_mode(&mut self, cfg: &PortfolioConfig) {
        let dd = self.drawdown();
        let hit_rate = self.hit_rate();
        self.mode = if dd >= cfg.defensive_drawdown || hit_rate < cfg.defensive_hit_rate {
            OperatingMode::Defensive
        } else if dd >= cfg.recovery_drawdown {
            OperatingMode::Recovery
        } else {
            OperatingMode::Normal
        };
    }

    pub fn record_fill(&mut self, symbol: &str, strategy: &StrategyId, notional_usd: f64, is_long: bool) {
        *self.symbol_exposure.entry(symbol.to_string()).or_insert(0.0) += notional_usd.max(0.0);
        *self.strategy_exposure.entry(strategy.clone()).or_insert(0.0) += notional_usd.max(0.0);
        if is_long {
            self.long_beta_exposure += notional_usd.max(0.0);
        } else {
            self.short_beta_exposure += notional_usd.max(0.0);
        }
        let turnover_denom = self.equity_usd.max(1.0);
        self.turnover_hourly += notional_usd.abs() / turnover_denom;
    }

    pub fn record_trade_result(&mut self, pnl_usd: f64) {
        self.intraday_pnl_usd += pnl_usd;
        self.equity_usd += pnl_usd;
        self.peak_equity_usd = self.peak_equity_usd.max(self.equity_usd);
        self.recent_results.push_back(pnl_usd >= 0.0);
        while self.recent_results.len() > 40 {
            self.recent_results.pop_front();
        }
    }

    pub fn update_symbol_return(&mut self, symbol: &str, ret: f64) {
        let q = self.symbol_returns.entry(symbol.to_string()).or_insert_with(|| VecDeque::with_capacity(120));
        q.push_back(ret);
        while q.len() > 120 {
            q.pop_front();
        }
    }
}

#[derive(Debug, Clone)]
pub enum PortfolioRejection {
    MaxTotalExposure,
    MaxCorrelatedExposure,
    MaxIntradayDrawdown,
    MaxStrategyLoss,
    MaxSymbolLoss,
    MaxTurnover,
    MaxSymbolCap,
    MaxStrategyCap,
    MaxConcurrency,
    BelowConfidence,
}

impl std::fmt::Display for PortfolioRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct PortfolioRiskEngine {
    pub config: PortfolioConfig,
}

impl PortfolioRiskEngine {
    pub fn new(config: PortfolioConfig) -> Self {
        Self { config }
    }

    pub fn check(
        &self,
        state: &PortfolioState,
        ranked: &RankedOpportunity,
    ) -> Result<(), PortfolioRejection> {
        let cap = state.equity_usd.max(1.0);
        let next_total = state.total_exposure() + ranked.allocated_notional_usd;
        if (next_total / cap) > self.config.max_total_exposure {
            return Err(PortfolioRejection::MaxTotalExposure);
        }
        if ranked.correlation_penalty < 0.45 {
            return Err(PortfolioRejection::MaxCorrelatedExposure);
        }
        if state.drawdown() > self.config.max_intraday_drawdown {
            return Err(PortfolioRejection::MaxIntradayDrawdown);
        }
        if state.turnover_hourly > self.config.max_turnover_per_hour {
            return Err(PortfolioRejection::MaxTurnover);
        }

        let symbol_expo = state.symbol_exposure.get(&ranked.candidate.symbol).copied().unwrap_or(0.0);
        if ((symbol_expo + ranked.allocated_notional_usd) / cap) > self.config.max_symbol_exposure {
            return Err(PortfolioRejection::MaxSymbolCap);
        }

        let strat_expo = state.strategy_exposure.get(&ranked.candidate.strategy).copied().unwrap_or(0.0);
        if ((strat_expo + ranked.allocated_notional_usd) / cap) > self.config.max_strategy_exposure {
            return Err(PortfolioRejection::MaxStrategyCap);
        }

        let concurrency = state.symbol_exposure.values().filter(|v| **v > 0.0).count();
        let limit = match state.mode {
            OperatingMode::Defensive => self.config.max_concurrent_positions_defensive,
            _ => self.config.max_concurrent_positions_normal,
        };
        if concurrency >= limit {
            return Err(PortfolioRejection::MaxConcurrency);
        }

        let min_conf = match state.mode {
            OperatingMode::Defensive => self.config.min_confidence_defensive,
            _ => self.config.min_confidence_normal,
        };
        if ranked.candidate.confidence < min_conf {
            return Err(PortfolioRejection::BelowConfidence);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PortfolioAllocator {
    pub config: PortfolioConfig,
}

impl PortfolioAllocator {
    pub fn new(config: PortfolioConfig) -> Self {
        Self { config }
    }

    pub fn rank_opportunities(
        &self,
        state: &PortfolioState,
        scoreboard: &StrategyScoreboard,
        execution_quality: &ExecutionQualityTracker,
        candidates: Vec<OpportunityCandidate>,
    ) -> Vec<RankedOpportunity> {
        let mut ranked = Vec::with_capacity(candidates.len());
        for c in candidates {
            let strategy_weight = scoreboard.score_weight(&c.strategy);
            let diversification_benefit = self.diversification_benefit(state, &c.symbol);
            let correlation_penalty = self.correlation_penalty(state, &c.symbol);
            let exec_penalty = execution_quality.penalty("binance:market");
            let rank_score = (c.confidence * 0.30)
                + (c.regime_fit * 0.20)
                + (c.expected_reward_risk.clamp(0.0, 2.0) / 2.0 * 0.20)
                + ((strategy_weight / 1.45).clamp(0.0, 1.0) * 0.15)
                + (diversification_benefit * 0.15);

            let capital_budget = (state.equity_usd - state.reserve_cash_usd).max(0.0);
            let volatility_adjustment = (1.0 / (1.0 + (c.volatility * 120.0))).clamp(0.20, 1.0);
            let mode_adjustment = match state.mode {
                OperatingMode::Normal => 1.0,
                OperatingMode::Recovery => 0.75,
                OperatingMode::Defensive => 0.45,
            };
            let allocated_notional_usd = capital_budget
                * strategy_weight
                * c.regime_fit.clamp(0.0, 1.2)
                * c.confidence.clamp(0.0, 1.0)
                * volatility_adjustment
                * correlation_penalty
                * exec_penalty
                * mode_adjustment
                * 0.05;

            ranked.push(RankedOpportunity {
                candidate: c,
                rank_score,
                diversification_benefit,
                correlation_penalty,
                allocated_notional_usd,
            });
        }

        ranked.sort_by(|a, b| b.rank_score.partial_cmp(&a.rank_score).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    fn diversification_benefit(&self, state: &PortfolioState, symbol: &str) -> f64 {
        let existing = state.symbol_exposure.get(symbol).copied().unwrap_or(0.0);
        let cap = state.equity_usd.max(1.0);
        (1.0 - (existing / cap)).clamp(0.0, 1.0)
    }

    fn correlation_penalty(&self, state: &PortfolioState, symbol: &str) -> f64 {
        let mut worst_corr: f64 = 0.0;
        for other in state.symbol_returns.keys() {
            if other == symbol {
                continue;
            }
            let corr = rolling_corr(state, symbol, other);
            worst_corr = worst_corr.max(corr.abs());
        }
        (1.0 - worst_corr).clamp(0.20, 1.0)
    }
}

fn rolling_corr(state: &PortfolioState, a: &str, b: &str) -> f64 {
    let Some(ra) = state.symbol_returns.get(a) else { return 0.0; };
    let Some(rb) = state.symbol_returns.get(b) else { return 0.0; };
    let n = ra.len().min(rb.len());
    if n < 10 {
        return 0.0;
    }
    let xa: Vec<f64> = ra.iter().rev().take(n).copied().collect();
    let xb: Vec<f64> = rb.iter().rev().take(n).copied().collect();
    let ma = xa.iter().sum::<f64>() / n as f64;
    let mb = xb.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = xa[i] - ma;
        let db = xb[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-12 || vb <= 1e-12 {
        return 0.0;
    }
    (cov / (va.sqrt() * vb.sqrt())).clamp(-1.0, 1.0)
}

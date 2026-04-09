# RW-Trader

RW-Trader is an adaptive trading engine with a hardened execution path, event-sourced audit trail, and portfolio-aware overlays.

## Portfolio Trading Upgrade

The system now layers portfolio intelligence **above** existing strategy logic:

1. Signal + strategy engine still generates opportunities.
2. Portfolio allocator ranks opportunities and computes portfolio-aware budget.
3. Portfolio risk engine enforces non-overridable hard constraints.
4. Existing single-trade risk engine performs order-level safety checks.
5. Authority and executor submit orders only if every gate passes.

Live trading remains disabled unless `LIVE_TRADE=true`.

## New environment / config knobs

### Portfolio universe and capital

- `SYMBOLS` (comma-separated list, default: `SYMBOL`)
- `PORTFOLIO_INITIAL_EQUITY_USD` (default: `10000`)
- `PORTFOLIO_RESERVE_BUFFER` (default: `0.20`)

### Exposure and concentration limits

- `PORTFOLIO_MAX_TOTAL_EXPOSURE` (default: `0.95`)
- `PORTFOLIO_MAX_CORRELATED_EXPOSURE` (default: `0.45`)
- `PORTFOLIO_MAX_SYMBOL_EXPOSURE` (default: `0.35`)
- `PORTFOLIO_MAX_STRATEGY_EXPOSURE` (default: `0.45`)

### Drawdown and loss controls

- `PORTFOLIO_MAX_INTRADAY_DRAWDOWN` (default: `0.06`)
- `PORTFOLIO_MAX_STRATEGY_LOSS` (default: `0.03`)
- `PORTFOLIO_MAX_SYMBOL_LOSS` (default: `0.03`)
- `PORTFOLIO_MAX_TURNOVER_PER_HOUR` (default: `4.0`)

### Drawdown-aware operating modes

- `PORTFOLIO_DEFENSIVE_DRAWDOWN` (default: `0.04`)
- `PORTFOLIO_RECOVERY_DRAWDOWN` (default: `0.02`)
- `PORTFOLIO_DEFENSIVE_HIT_RATE` (default: `0.45`)
- `PORTFOLIO_MIN_CONF_NORMAL` (default: `0.70`)
- `PORTFOLIO_MIN_CONF_DEFENSIVE` (default: `0.82`)
- `PORTFOLIO_MAX_CONCURRENT_NORMAL` (default: `6`)
- `PORTFOLIO_MAX_CONCURRENT_DEFENSIVE` (default: `3`)

## Decision flow

```text
signal metrics
  -> strategy decision
  -> opportunity ranking (confidence + regime fit + R/R + strategy scoreboard + diversification)
  -> portfolio allocation (capital budget with volatility/correlation penalties)
  -> portfolio risk limits
  -> order-level risk limits
  -> authority gate
  -> execution
  -> trade lifecycle + scoreboard + execution quality analytics
```

Portfolio analytics are appended to the event store as `operator_action` events (`portfolio_ranked_opportunity`, `portfolio_rejection`, `trade_lifecycle_analytics`) for audit/review workflows.

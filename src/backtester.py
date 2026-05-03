from __future__ import annotations
"""
Backtesting engine for long-short cross-sectional strategies.

Architecture (no look-ahead bias):
  signal[t]  → weights[t]  → EARN r[t+1]
  port_ret[t+1] = Σ_i  w_i[t] × r_i[t+1]  −  turnover[t] × cost_bps/10 000

The weights are formed at the close of day t.  The portfolio earns the
return from close t to close t+1.  This is the standard "trade-at-close"
model used in academic factor research.

Dollar-neutral constraint: long side sums to +1, short side sums to −1.
(Gross leverage = 2; in other words, $1 long + $1 short per $1 of capital.)
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─── Weight construction ──────────────────────────────────────────────────────

def signal_to_weights(
    signal: pd.DataFrame,
    long_threshold: float  = config.QUINTILE_LONG_THRESHOLD,
    short_threshold: float = config.QUINTILE_SHORT_THRESHOLD,
    min_assets: int        = config.MIN_ASSETS_FOR_SIGNAL,
) -> pd.DataFrame:
    """
    Convert a signal matrix into dollar-neutral portfolio weights.

    Method (quintile):
      1. Cross-sectionally rank the signal at each date (percentile, 0–1).
      2. Long assets with rank ≥ long_threshold (top quintile, ~20%).
         Short assets with rank ≤ short_threshold (bottom quintile, ~20%).
         Neutral everything else.
      3. Normalize long leg to sum to +1, short leg to sum to −1.
      4. Rows with fewer than min_assets valid signals are set to zero
         (no trade on illiquid dates).

    Returns a DataFrame of weights with the same shape as `signal`.
    """
    # Cross-sectional percentile rank at each date
    ranks = signal.rank(axis=1, pct=True, na_option="keep")

    # Binary long/short allocation
    raw_w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    raw_w[ranks >= long_threshold]  =  1.0
    raw_w[ranks <= short_threshold] = -1.0

    # Count valid assets (not NaN in the original signal)
    n_valid = signal.notna().sum(axis=1)
    too_few = n_valid < min_assets

    # Normalize each leg independently
    long_sum  = raw_w.clip(lower=0).sum(axis=1).replace(0, np.nan)
    short_sum = raw_w.clip(upper=0).sum(axis=1).abs().replace(0, np.nan)

    long_part  = raw_w.clip(lower=0).div(long_sum, axis=0).fillna(0.0)
    short_part = raw_w.clip(upper=0).div(short_sum, axis=0).fillna(0.0)

    weights = long_part + short_part
    weights[too_few] = 0.0          # zero out under-populated dates

    return weights


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Daily one-way portfolio turnover.

    turnover[t] = Σ_i |w_i[t] − w_i[t−1]| / 2

    Interpretation: 100% turnover means the entire long-short book was
    replaced once.  The /2 converts two-way to one-way (selling $1 of A
    and buying $1 of B = 1 unit of one-way turnover, not 2).
    """
    return weights.diff().abs().sum(axis=1) / 2.0


# ─── Core backtest ────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = config.MARKET_ORDER_COST_BPS,
) -> pd.Series:
    """
    Compute net daily portfolio returns.

    Formula:
        gross_ret[t+1] = Σ_i  w[t]_i × r[t+1]_i
        TC[t]          = turnover[t] × cost_bps / 10 000
        net_ret[t+1]   = gross_ret[t+1] − TC[t]

    The weights are lagged by 1 period (w[t] → r[t+1]) to prevent
    any look-ahead bias even if the caller forgot to shift.
    Returns and weights are aligned on their common date index.
    """
    # Align on common assets and dates
    common_cols = weights.columns.intersection(returns.columns)
    common_idx  = weights.index.intersection(returns.index)
    w = weights.loc[common_idx, common_cols]
    r = returns.loc[common_idx, common_cols]

    # Forward returns: what w[t] will earn (r[t+1])
    # We shift weights forward by 1: w.shift(1)[t+1] = w[t]
    w_lagged = w.shift(1)

    # Gross return (element-wise product, then sum across assets)
    gross = (w_lagged * r).sum(axis=1)

    # Transaction cost = one-way turnover × cost rate
    turnover = compute_turnover(w)
    tc = turnover * (cost_bps / 10_000)

    net = gross - tc

    # Drop the first row (no lagged weight available → NaN gross return anyway)
    result = pd.DataFrame({
        "gross_return":      gross,
        "transaction_cost":  tc,
        "net_return":        net,
        "turnover":          turnover,
        "n_long":   (w_lagged > 0).sum(axis=1),
        "n_short":  (w_lagged < 0).sum(axis=1),
    })
    return result.iloc[1:]


def run_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    cost_mode: str = "market",
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> dict:
    """
    Full backtest pipeline for a single signal.

    Steps:
      1. Optionally clip to [start_date, end_date]
      2. Compute simple returns from prices
      3. Convert signal to weights (quintile)
      4. Compute net portfolio returns with transaction costs

    cost_mode: "market" → 20 bps, "limit" → 7 bps

    Returns a dict with:
        "returns":  DataFrame (gross, net, cost, turnover per day)
        "weights":  DataFrame (weight per asset per day)
        "signal":   the input signal (aligned to price dates)
    """
    cost_bps = (
        config.MARKET_ORDER_COST_BPS if cost_mode == "market"
        else config.LIMIT_ORDER_COST_BPS
    )

    # Align signal and prices
    common_cols = signal.columns.intersection(prices.columns)
    common_idx  = signal.index.intersection(prices.index)
    sig = signal.loc[common_idx, common_cols]
    px  = prices.loc[common_idx, common_cols]

    if start_date:
        sig = sig.loc[start_date:]
        px  = px.loc[start_date:]
    if end_date:
        sig = sig.loc[:end_date]
        px  = px.loc[:end_date]

    # Simple returns (used for P&L; log returns for signals but simple for P&L)
    simple_ret = px.pct_change()

    weights = signal_to_weights(sig)
    perf_df = compute_portfolio_returns(weights, simple_ret, cost_bps=cost_bps)

    return {
        "returns": perf_df,
        "weights": weights,
        "signal":  sig,
    }


def run_all_backtests(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    cost_mode: str = "market",
) -> dict[str, dict]:
    """
    Run run_backtest() for every signal in the library.

    Returns {signal_name: backtest_result_dict}.
    """
    results = {}
    for name, sig in signals.items():
        print(f"  > {name}")
        results[name] = run_backtest(sig, prices, cost_mode=cost_mode)
    return results

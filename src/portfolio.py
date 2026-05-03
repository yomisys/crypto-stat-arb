from __future__ import annotations
"""
Strategy combination / portfolio construction.

Motivation: individual signals are noisy; combining uncorrelated signals
smooths returns and improves the Sharpe ratio (diversification).

Three combination methods are tested:
  1. Equal-weight          — simple, robust, no in-sample estimation
  2. Inverse-volatility    — down-weights strategies with large swings
  3. Sharpe-weighted       — up-weights strategies that have worked best so far
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluator import sharpe_ratio, annualized_vol


# ─── Signal-level combiners (before backtesting) ──────────────────────────────

def equal_weight_signal(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Average all signals with equal weight after aligning on common dates/assets.

    This is the simplest and often most robust combination.
    """
    df_list = list(signals.values())
    if not df_list:
        raise ValueError("No signals to combine.")
    combined = pd.concat(df_list, axis=0).groupby(level=0).mean()
    return combined.reindex(df_list[0].index)


# ─── Return-level combiners (after backtesting) ───────────────────────────────

def _extract_net_returns(backtest_results: dict[str, dict]) -> pd.DataFrame:
    """Extract net_return Series from each backtest, return as wide DataFrame."""
    series = {
        name: result["returns"]["net_return"]
        for name, result in backtest_results.items()
        if "returns" in result and "net_return" in result["returns"].columns
    }
    return pd.DataFrame(series)


def equal_weight_portfolio(
    backtest_results: dict[str, dict],
    strategy_names: Optional[list[str]] = None,
) -> pd.Series:
    """
    Combine strategy returns with equal weight.

    strategy_names: if provided, only those strategies are combined.
    Returns a daily return Series indexed by date.
    """
    if strategy_names:
        filtered = {k: v for k, v in backtest_results.items() if k in strategy_names}
    else:
        filtered = backtest_results

    ret_df = _extract_net_returns(filtered).dropna(how="all")
    return ret_df.mean(axis=1).rename("equal_weight")


def inverse_vol_portfolio(
    backtest_results: dict[str, dict],
    vol_window: int = 60,
    strategy_names: Optional[list[str]] = None,
) -> pd.Series:
    """
    Inverse-volatility weighted combination.

    At each date t, strategy i gets weight proportional to 1 / σ_i(t),
    where σ_i(t) is the rolling standard deviation over `vol_window` days.

    Finance intuition: target equal risk contribution from each strategy
    (the simplest form of risk parity).
    """
    if strategy_names:
        filtered = {k: v for k, v in backtest_results.items() if k in strategy_names}
    else:
        filtered = backtest_results

    ret_df = _extract_net_returns(filtered).dropna(how="all")

    roll_vol = ret_df.rolling(window=vol_window, min_periods=vol_window // 2).std()
    roll_vol = roll_vol.replace(0, np.nan)

    inv_vol  = 1.0 / roll_vol
    weights  = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    combined = (ret_df * weights).sum(axis=1)
    combined[ret_df.isna().all(axis=1)] = np.nan
    return combined.rename("inv_vol")


def sharpe_weighted_portfolio(
    backtest_results: dict[str, dict],
    min_periods: int = 90,
    strategy_names: Optional[list[str]] = None,
) -> pd.Series:
    """
    Expanding-Sharpe weighted combination.

    At each date t, strategy i's weight is proportional to max(SR_i(t), 0),
    where SR_i(t) is the Sharpe ratio computed over all history up to t.

    Strategies with negative expanding Sharpe get zero weight — we don't
    invest in strategies that have consistently lost money.

    The expanding window (rather than rolling) is more stable but slower
    to adapt to regime changes. This is intentional: a strategy should
    prove itself over years, not just recent weeks.
    """
    if strategy_names:
        filtered = {k: v for k, v in backtest_results.items() if k in strategy_names}
    else:
        filtered = backtest_results

    ret_df = _extract_net_returns(filtered).dropna(how="all")
    n_strats = ret_df.shape[1]
    combined = pd.Series(np.nan, index=ret_df.index, name="sharpe_weighted")

    for t in range(min_periods, len(ret_df)):
        past   = ret_df.iloc[:t]
        sharpes = past.apply(lambda s: sharpe_ratio(s.dropna()))
        sharpes = sharpes.clip(lower=0.0)   # floor at 0 for negative-Sharpe strategies
        total   = sharpes.sum()

        if total > 0:
            w = sharpes / total
            combined.iloc[t] = (ret_df.iloc[t] * w).sum()

    return combined


def build_combinations(
    backtest_results: dict[str, dict],
    strategy_names: Optional[list[str]] = None,
    vol_window: int = 60,
    sharpe_min_periods: int = 90,
) -> dict[str, pd.Series]:
    """
    Build all three combination portfolios and return as a dict.

    Returns:
        {"equal_weight": Series, "inv_vol": Series, "sharpe_weighted": Series}
    """
    return {
        "equal_weight":    equal_weight_portfolio(backtest_results, strategy_names),
        "inv_vol":         inverse_vol_portfolio(backtest_results, vol_window=vol_window, strategy_names=strategy_names),
        "sharpe_weighted": sharpe_weighted_portfolio(backtest_results, min_periods=sharpe_min_periods, strategy_names=strategy_names),
    }

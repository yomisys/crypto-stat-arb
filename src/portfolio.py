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


def _quarter_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Generate quarter-end dates (Mar 31, Jun 30, Sep 30, Dec 31) in [start, end]."""
    month_day = [(3, 31), (6, 30), (9, 30), (12, 31)]
    dates = []
    for year in range(start.year, end.year + 1):
        for m, d in month_day:
            dt = pd.Timestamp(year, m, d)
            if start <= dt <= end:
                dates.append(dt)
    return dates


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


def walk_forward_portfolio(
    backtest_results: dict[str, dict],
    weight_scheme: str = "equal",
    vol_window: int = 60,
    sharpe_window: int = 252,
    strategy_names: Optional[list[str]] = None,
) -> pd.Series:
    """
    Quarterly walk-forward combination portfolio.

    At each quarter-end date (Mar 31 / Jun 30 / Sep 30 / Dec 31), compute
    combination weights using ALL data available up to that date, then apply
    those weights to the NEXT quarter only (out-of-sample).

    weight_scheme:
      "equal"           — 1/N, no estimation needed
      "inv_vol"         — 1/sigma_i, sigma estimated over last `vol_window` days
      "sharpe_weighted" — max(SR_i, 0) / sum, SR estimated over last `sharpe_window` days

    First rebalance: first quarter-end that has at least 30 observations.
    Returns a Series of OOS daily returns; NaN for periods before first rebalance.
    """
    if strategy_names:
        filtered = {k: v for k, v in backtest_results.items() if k in strategy_names}
    else:
        filtered = backtest_results

    ret_df = _extract_net_returns(filtered).dropna(how="all")
    n_strats = ret_df.shape[1]

    rebal_dates = _quarter_ends(ret_df.index.min(), ret_df.index.max())
    combined = pd.Series(np.nan, index=ret_df.index,
                         name=f"wf_{weight_scheme}")

    for k, rebal_dt in enumerate(rebal_dates):
        history = ret_df.loc[:rebal_dt].dropna(how="all")
        if len(history) < 30:
            continue

        # Compute weights on history ending at rebal_dt
        if weight_scheme == "equal":
            w = pd.Series(1.0 / n_strats, index=ret_df.columns)

        elif weight_scheme == "inv_vol":
            recent = history.tail(vol_window)
            rv = recent.std().replace(0, np.nan)
            inv = 1.0 / rv
            total = inv.sum()
            w = (inv / total).fillna(0.0) if total > 0 else pd.Series(
                1.0 / n_strats, index=ret_df.columns)

        elif weight_scheme == "sharpe_weighted":
            recent = history.tail(sharpe_window)
            srs = recent.apply(lambda s: sharpe_ratio(s.dropna())).clip(lower=0.0)
            total = srs.sum()
            w = (srs / total) if total > 0 else pd.Series(
                1.0 / n_strats, index=ret_df.columns)
        else:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme!r}")

        # Apply to next quarter (OOS)
        next_start = rebal_dt + pd.Timedelta(days=1)
        next_end   = (rebal_dates[k + 1] if k + 1 < len(rebal_dates)
                      else ret_df.index.max())
        oos_slice  = ret_df.loc[next_start:next_end]
        if not oos_slice.empty:
            combined.loc[next_start:next_end] = (oos_slice * w).sum(axis=1)

    return combined


def build_walk_forward_combinations(
    backtest_results: dict[str, dict],
    strategy_names: Optional[list[str]] = None,
    vol_window: int = 60,
    sharpe_window: int = 252,
) -> dict[str, pd.Series]:
    """
    Build all three walk-forward combination portfolios.

    Returns OOS-only return series for equal, inv_vol, and sharpe_weighted schemes.
    All returned series are purely out-of-sample — no in-sample contamination.
    """
    return {
        "wf_equal":    walk_forward_portfolio(
            backtest_results, "equal",           strategy_names=strategy_names),
        "wf_inv_vol":  walk_forward_portfolio(
            backtest_results, "inv_vol",          vol_window=vol_window,
            strategy_names=strategy_names),
        "wf_sharpe":   walk_forward_portfolio(
            backtest_results, "sharpe_weighted",  sharpe_window=sharpe_window,
            strategy_names=strategy_names),
    }


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

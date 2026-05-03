from __future__ import annotations
"""
Performance evaluation: compute standard quantitative finance metrics.

All annualization uses 365 periods/year (crypto trades 24 / 7 / 365).
For hourly strategies, use periods_per_year = 365 * 24.

Definitions
───────────
Annualized Return  : geometric CAGR from compound-growth-of-returns
Sharpe Ratio       : (ann_ret − rf) / ann_vol, rf = 0 for crypto
Max Drawdown       : worst peak-to-trough decline (negative number)
Alpha              : intercept of OLS(strategy ~ BTC), annualized
Beta               : slope coefficient, measures BTC exposure
Win Rate           : fraction of days with positive net return
Turnover           : average daily one-way portfolio turnover
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─── Individual metrics ───────────────────────────────────────────────────────

def annualized_return(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Compound annualized growth rate (CAGR)."""
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    total = (1.0 + clean).prod()
    n = len(clean)
    if total <= 0:
        return np.nan
    return float(total ** (periods_per_year / n) - 1.0)


def annualized_vol(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Annualized standard deviation of daily returns."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """
    Annualized Sharpe Ratio.

    For crypto strategies we use rf = 0% (no risk-free rate in DeFi/CeFi
    that is comparable to the strategy risk, and it is conservative).
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_v   = annualized_vol(returns, periods_per_year)
    if pd.isna(ann_ret) or pd.isna(ann_v) or ann_v == 0:
        return np.nan
    return float((ann_ret - rf) / ann_v)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (always ≤ 0).

    Computed from compounded wealth index, not from raw return sum.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    wealth = (1.0 + clean).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def alpha_beta_vs_btc(
    port_returns: pd.Series,
    btc_returns: pd.Series,
    min_obs: int = 30,
) -> dict[str, float]:
    """
    OLS regression: R_port = α + β × R_BTC + ε

    Returns alpha (daily, then annualized), beta, t-stat for alpha, and R².

    Finance interpretation:
      β = 0  → market-neutral strategy (desirable for stat-arb)
      α > 0  → strategy earns positive returns after adjusting for BTC exposure
    """
    aligned = pd.DataFrame({"port": port_returns, "btc": btc_returns}).dropna()
    if len(aligned) < min_obs:
        return {"alpha_ann": np.nan, "beta": np.nan, "alpha_tstat": np.nan, "r_squared": np.nan}

    X = sm.add_constant(aligned["btc"])
    model = sm.OLS(aligned["port"], X).fit()

    alpha_daily = float(model.params["const"])
    alpha_ann   = float((1.0 + alpha_daily) ** 365 - 1.0)
    beta        = float(model.params["btc"])
    alpha_tstat = float(model.tvalues["const"])
    r2          = float(model.rsquared)

    return {
        "alpha_ann":    alpha_ann,
        "beta":         beta,
        "alpha_tstat":  alpha_tstat,
        "r_squared":    r2,
    }


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive net return."""
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    return float((clean > 0).mean())


def avg_turnover(bt_result: dict) -> float:
    """Mean daily one-way turnover from a backtest result dict."""
    df = bt_result.get("returns")
    if df is None or "turnover" not in df.columns:
        return np.nan
    return float(df["turnover"].mean())


def annual_cost_drag(bt_result: dict, periods_per_year: int = 365) -> float:
    """Annualized transaction cost drag (simple sum)."""
    df = bt_result.get("returns")
    if df is None or "transaction_cost" not in df.columns:
        return np.nan
    return float(df["transaction_cost"].mean() * periods_per_year)


# ─── Summary tearsheet ────────────────────────────────────────────────────────

def full_tearsheet(
    bt_result: dict,
    btc_prices: pd.Series,
    label: str = "",
    periods_per_year: int = 365,
) -> pd.Series:
    """
    Compute all performance metrics for one backtest result.

    bt_result : output of backtester.run_backtest()
    btc_prices: BTC close price series (for alpha/beta benchmark)

    Returns a pandas Series of labelled metrics.
    """
    ret_df   = bt_result["returns"]
    net_ret  = ret_df["net_return"].dropna()
    gross_ret= ret_df["gross_return"].dropna()

    btc_ret  = btc_prices.pct_change().reindex(net_ret.index).dropna()
    ab       = alpha_beta_vs_btc(net_ret, btc_ret)

    metrics = {
        "Ann. Return (net) %":   annualized_return(net_ret,   periods_per_year) * 100,
        "Ann. Return (gross) %": annualized_return(gross_ret, periods_per_year) * 100,
        "Ann. Volatility %":     annualized_vol(net_ret,      periods_per_year) * 100,
        "Sharpe Ratio":          sharpe_ratio(net_ret,        periods_per_year=periods_per_year),
        "Sharpe (gross)":        sharpe_ratio(gross_ret,      periods_per_year=periods_per_year),
        "Max Drawdown %":        max_drawdown(net_ret) * 100,
        "Alpha (ann) %":         ab["alpha_ann"]  * 100,
        "Alpha t-stat":          ab["alpha_tstat"],
        "Beta vs BTC":           ab["beta"],
        "R-squared":             ab["r_squared"],
        "Win Rate %":            win_rate(net_ret) * 100,
        "Avg Daily Turnover":    avg_turnover(bt_result),
        "Ann. Cost Drag %":      annual_cost_drag(bt_result, periods_per_year) * 100,
        "N Days":                len(net_ret),
    }

    return pd.Series(metrics, name=label)


def compare_strategies(
    backtest_results: dict[str, dict],
    btc_prices: pd.Series,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """
    Build a comparison table for all strategies, sorted by Sharpe Ratio (desc).

    Returns a DataFrame indexed by strategy name.
    """
    rows = {}
    for name, result in backtest_results.items():
        rows[name] = full_tearsheet(result, btc_prices, label=name, periods_per_year=periods_per_year)

    df = pd.DataFrame(rows).T
    df.index.name = "Strategy"

    # Sort by net Sharpe, descending
    if "Sharpe Ratio" in df.columns:
        df = df.sort_values("Sharpe Ratio", ascending=False)

    return df


# ─── Rolling metrics (for time-series plots) ─────────────────────────────────

def rolling_sharpe(
    returns: pd.Series,
    window: int = 90,
    periods_per_year: int = 365,
) -> pd.Series:
    """Rolling Sharpe ratio with a `window`-day lookback."""
    roll_ret = returns.rolling(window, min_periods=window // 2).mean() * periods_per_year
    roll_vol = returns.rolling(window, min_periods=window // 2).std()  * np.sqrt(periods_per_year)
    return roll_ret / roll_vol.replace(0, np.nan)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full drawdown time series (for plotting)."""
    clean = returns.dropna()
    wealth = (1.0 + clean).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak

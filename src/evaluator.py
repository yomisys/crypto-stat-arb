from __future__ import annotations
"""
Performance evaluation: compute standard quantitative finance metrics.

All annualization uses 365 periods/year (crypto trades 24 / 7 / 365).
For hourly strategies, use periods_per_year = 365 * 24.

Definitions
───────────
Annualized Return  : geometric CAGR from compound-growth-of-returns
Sharpe Ratio       : (ann_ret − rf) / ann_vol, rf = 0 for crypto
Max Drawdown       : worst peak-to-trough decline, log-return method
Alpha              : intercept of OLS(strategy ~ BTC), annualized
Beta               : slope coefficient, measures BTC exposure
Win Rate           : fraction of days with positive net return
Turnover           : average daily one-way portfolio turnover

Drawdown note (Change 4 — mentor feedback):
  Uses log-return cumulation (exp(cumsum(log1p(r)))) for numerical
  stability, especially when compound losses are large (>50% common
  in crypto bear markets).
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


def compute_drawdown(returns_series: pd.Series) -> tuple[pd.Series, float]:
    """
    Compute drawdown series and max drawdown from a returns series.

    Uses log-return cumulation for numerical stability — avoids compounding
    errors when losses are large (e.g. -50% drawdowns common in crypto).

    Method (per mentor feedback):
        log_ret = log1p(r)
        wealth  = exp(cumsum(log_ret))
        dd      = wealth / running_max - 1

    Returns
    -------
    drawdown : pd.Series — per-day drawdown (always <= 0)
    max_dd   : float     — worst (most negative) value
    """
    clean = returns_series.dropna()
    if len(clean) == 0:
        return pd.Series(dtype=float), np.nan
    log_ret     = np.log1p(clean.fillna(0))
    wealth      = np.exp(log_ret.cumsum())
    running_max = wealth.cummax()
    drawdown    = wealth / running_max - 1.0
    return drawdown, float(drawdown.min())


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (always <= 0).

    Drawdown calculation updated per mentor feedback —
    uses log-return cumulation for numerical stability.
    """
    _, mdd = compute_drawdown(returns.dropna())
    return mdd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full drawdown time series (for plotting)."""
    dd, _ = compute_drawdown(returns.dropna())
    return dd


def sig_flag(pvalue: float) -> str:
    """Convert a p-value to significance stars: ***, **, *, or ns."""
    if pd.isna(pvalue):
        return "ns"
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return "ns"


def alpha_beta_vs_btc(
    port_returns: pd.Series,
    btc_returns: pd.Series,
    min_obs: int = 30,
) -> dict[str, float]:
    """
    OLS regression: R_port = alpha + beta * R_BTC + epsilon

    Returns alpha (annualized), beta, t-statistics, p-values, and R-squared.

    Finance interpretation:
      beta = 0  -> market-neutral strategy (desirable for stat-arb)
      alpha > 0 -> positive return after adjusting for BTC exposure
    """
    aligned = pd.DataFrame({"port": port_returns, "btc": btc_returns}).dropna()
    if len(aligned) < min_obs:
        return {
            "alpha_ann":    np.nan,
            "beta":         np.nan,
            "alpha_tstat":  np.nan,
            "alpha_pvalue": np.nan,
            "beta_tstat":   np.nan,
            "r_squared":    np.nan,
        }

    X     = sm.add_constant(aligned["btc"])
    model = sm.OLS(aligned["port"], X).fit()

    alpha_daily  = float(model.params["const"])
    alpha_ann    = float((1.0 + alpha_daily) ** 365 - 1.0)
    beta         = float(model.params["btc"])
    alpha_tstat  = float(model.tvalues["const"])
    alpha_pvalue = float(model.pvalues["const"])
    beta_tstat   = float(model.tvalues["btc"])
    r2           = float(model.rsquared)

    return {
        "alpha_ann":    alpha_ann,
        "beta":         beta,
        "alpha_tstat":  alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "beta_tstat":   beta_tstat,
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


# ─── Period-specific evaluation (Change 1 — train/val split) ─────────────────

def evaluate_period(
    bt_result: dict,
    btc_prices: pd.Series,
    start: Optional[str] = None,
    end:   Optional[str] = None,
    label: str = "",
    periods_per_year: int = 365,
) -> pd.Series:
    """
    Compute performance metrics for a strategy over a specific date window.

    Slices backtest returns to [start, end] before computing — ensures train
    and validation metrics are computed on non-overlapping periods.
    """
    net_full = bt_result["returns"]["net_return"]
    if start:
        net_full = net_full.loc[start:]
    if end:
        net_full = net_full.loc[:end]

    net_ret = net_full.dropna()
    btc_ret = btc_prices.pct_change().reindex(net_ret.index).dropna()
    ab      = alpha_beta_vs_btc(net_ret, btc_ret)

    return pd.Series({
        "Ann. Return (net) %": annualized_return(net_ret, periods_per_year) * 100,
        "Ann. Volatility %":   annualized_vol(net_ret,    periods_per_year) * 100,
        "Sharpe Ratio":        sharpe_ratio(net_ret,      periods_per_year=periods_per_year),
        "Max Drawdown %":      max_drawdown(net_ret) * 100,
        "Alpha (ann) %":       ab["alpha_ann"]    * 100,
        "Alpha t-stat":        ab["alpha_tstat"],
        "Alpha p-value":       ab["alpha_pvalue"],
        "Sig":                 sig_flag(ab["alpha_pvalue"]),
        "Beta vs BTC":         ab["beta"],
        "Beta t-stat":         ab["beta_tstat"],
        "R-squared":           ab["r_squared"],
        "Win Rate %":          win_rate(net_ret) * 100,
        "N Days":              len(net_ret),
    }, name=label)


def train_val_comparison(
    backtest_results: dict[str, dict],
    btc_prices: pd.Series,
    train_start: str = "2021-01-01",
    train_end:   str = "2022-12-31",
    val_start:   str = "2023-01-01",
    val_end:     str = "2025-12-31",
    overfit_threshold: float = 0.50,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """
    Compare in-sample (train) vs out-of-sample (validation) Sharpe for all strategies.

    Strategies where validation Sharpe drops more than `overfit_threshold`
    (default 50%) from training Sharpe are flagged as potential overfits.

    Returns a DataFrame sorted by validation Sharpe (descending).
    """
    rows = []
    for name, result in backtest_results.items():
        tr = evaluate_period(result, btc_prices, train_start, train_end,
                             label=name, periods_per_year=periods_per_year)
        va = evaluate_period(result, btc_prices, val_start, val_end,
                             label=name, periods_per_year=periods_per_year)

        tr_sr = tr["Sharpe Ratio"]
        va_sr = va["Sharpe Ratio"]

        if pd.isna(tr_sr) or tr_sr == 0:
            decay_pct = np.nan
            overfit   = False
        else:
            decay_pct = (tr_sr - va_sr) / abs(tr_sr) * 100
            overfit   = bool(va_sr < tr_sr * (1.0 - overfit_threshold))

        rows.append({
            "Strategy":       name,
            "Train Sharpe":   tr_sr,
            "Val Sharpe":     va_sr,
            "Sharpe Decay %": decay_pct,
            "Overfit Flag":   "OVERFIT" if overfit else "",
            "Train Return %": tr["Ann. Return (net) %"],
            "Val Return %":   va["Ann. Return (net) %"],
            "Train Alpha %":  tr["Alpha (ann) %"],
            "Train Alpha t":  tr["Alpha t-stat"],
            "Val Alpha %":    va["Alpha (ann) %"],
            "Val Alpha t":    va["Alpha t-stat"],
            "Val Sig":        va["Sig"],
        })

    df = pd.DataFrame(rows).set_index("Strategy")

    # Cast numeric columns (string cols: "Overfit Flag", "Val Sig")
    _str_cols = {"Overfit Flag", "Val Sig"}
    for col in df.columns:
        if col not in _str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Val Sharpe", ascending=False)


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
    Includes alpha t-stat, p-value, significance flag, and beta t-stat.
    """
    ret_df    = bt_result["returns"]
    net_ret   = ret_df["net_return"].dropna()
    gross_ret = ret_df["gross_return"].dropna()

    btc_ret = btc_prices.pct_change().reindex(net_ret.index).dropna()
    ab      = alpha_beta_vs_btc(net_ret, btc_ret)

    return pd.Series({
        "Ann. Return (net) %":   annualized_return(net_ret,   periods_per_year) * 100,
        "Ann. Return (gross) %": annualized_return(gross_ret, periods_per_year) * 100,
        "Ann. Volatility %":     annualized_vol(net_ret,      periods_per_year) * 100,
        "Sharpe Ratio":          sharpe_ratio(net_ret,        periods_per_year=periods_per_year),
        "Sharpe (gross)":        sharpe_ratio(gross_ret,      periods_per_year=periods_per_year),
        "Max Drawdown %":        max_drawdown(net_ret) * 100,
        "Alpha (ann) %":         ab["alpha_ann"]    * 100,
        "Alpha t-stat":          ab["alpha_tstat"],
        "Alpha p-value":         ab["alpha_pvalue"],
        "Sig":                   sig_flag(ab["alpha_pvalue"]),
        "Beta vs BTC":           ab["beta"],
        "Beta t-stat":           ab["beta_tstat"],
        "R-squared":             ab["r_squared"],
        "Win Rate %":            win_rate(net_ret) * 100,
        "Avg Daily Turnover":    avg_turnover(bt_result),
        "Ann. Cost Drag %":      annual_cost_drag(bt_result, periods_per_year) * 100,
        "N Days":                len(net_ret),
    }, name=label)


def compare_strategies(
    backtest_results: dict[str, dict],
    btc_prices: pd.Series,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """
    Build a comparison table for all strategies, sorted by Sharpe Ratio (desc).

    Returns a DataFrame indexed by strategy name, including alpha t-stats and
    significance flags.
    """
    rows = {
        name: full_tearsheet(result, btc_prices, label=name,
                              periods_per_year=periods_per_year)
        for name, result in backtest_results.items()
    }
    df = pd.DataFrame(rows).T
    df.index.name = "Strategy"

    # Transpose produces object dtype for all columns when any column is string.
    # Cast everything except the "Sig" flag back to numeric.
    _str_cols = {"Sig"}
    for col in df.columns:
        if col not in _str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

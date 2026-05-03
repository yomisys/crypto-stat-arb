from __future__ import annotations
"""
Momentum signal implementations.

Momentum (Jegadeesh & Titman 1993): assets that have recently outperformed
tend to continue outperforming over the next 3–12 months.
In crypto, this effect is documented at shorter horizons due to higher
retail participation and slower information diffusion.

All signals at time t use only data available at close of day t.
The backtester applies a one-period lag (w[t] → earns r[t+1]).
"""

import numpy as np
import pandas as pd

from .base import (
    compute_log_returns,
    cross_section_rank,
    cross_section_zscore,
    winsorize_signal,
)


def momentum_simple(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Time-series momentum over `window` days.

    Signal at t = ln(P_t / P_{t-window}) = cumulative log-return over window.
    Positive → asset trending up; negative → trending down.

    No shift applied here — the backtester handles the lag.
    """
    log_ret = compute_log_returns(close)
    # Rolling sum of log returns = ln(P_t / P_{t-window})
    raw = log_ret.rolling(window=window, min_periods=min(max(window // 2, 1), window)).sum()
    return cross_section_zscore(raw)


def momentum_cross_section_rank(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Cross-sectional momentum: rank all assets by their past-window return.

    Returns percentile rank in [0, 1] at each date.
    Long top quintile (rank >= 0.8), short bottom quintile (rank <= 0.2).
    """
    log_ret = compute_log_returns(close)
    cum_ret = log_ret.rolling(window=window, min_periods=min(max(window // 2, 1), window)).sum()
    return cross_section_rank(cum_ret)


def momentum_volume_weighted(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Volume-weighted momentum.

    Finance intuition: price moves on heavy volume carry more information
    (conviction behind the move) than low-volume moves. We weight each
    day's return by that day's volume relative to its rolling average.

    signal_t = Σ_{s=t-w}^{t} [ r_s * (vol_s / avg_vol_s) ]

    where avg_vol_s = rolling mean of volume over 'window' days.
    """
    log_ret = compute_log_returns(close)

    # Relative volume (today vs. rolling average)
    avg_vol = volume.rolling(window=window, min_periods=min(max(window // 2, 1), window)).mean()
    rel_vol = volume / avg_vol.replace(0, np.nan)

    # Volume-weighted return, then sum over the window
    weighted = log_ret * rel_vol
    raw = weighted.rolling(window=window, min_periods=min(max(window // 2, 1), window)).sum()
    return cross_section_zscore(raw)


def momentum_weekday_seasonality(close: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    Weekday-vs-weekend seasonality signal.

    Crypto is 24/7, but institutional activity concentrates on weekdays
    (Mon–Fri), while retail-driven momentum tends to extend into weekends.
    This signal computes separate expanding-window mean returns for each
    weekday and uses the per-weekday expected return as a proxy signal.

    Weekday 0 = Monday, 6 = Sunday.

    The signal is weak on its own but can diversify the standard
    time-series momentum when combined in a portfolio.
    """
    log_ret = compute_log_returns(close)
    signal = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    # Compute expanding mean return by weekday for each asset
    for dow in range(7):
        mask = log_ret.index.dayofweek == dow
        day_ret = log_ret[mask]
        # Expanding mean at each occurrence of this weekday
        exp_mean = day_ret.expanding(min_periods=window).mean()
        # Place back into the full signal DataFrame at the correct dates
        signal.loc[mask] = exp_mean.values

    return cross_section_zscore(signal.fillna(0.0))


def all_momentum_signals(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build the full momentum signal library.

    Returns a dict of {name: signal_DataFrame} where each DataFrame has
    shape (n_dates, n_symbols) and is cross-sectionally z-scored.
    """
    signals: dict[str, pd.DataFrame] = {}

    # 1. Time-series momentum across all lookback windows
    from config import MOMENTUM_WINDOWS
    for w in MOMENTUM_WINDOWS:
        signals[f"mom_{w}d"] = momentum_simple(close, w)

    # 2. Cross-sectional rank momentum (quintile-friendly signal)
    for w in [7, 30, 90]:
        raw_rank = momentum_cross_section_rank(close, w)
        # Convert from [0,1] rank to [-1,1] z-score style
        signals[f"cs_mom_{w}d"] = (raw_rank - 0.5) * 2.0

    # 3. Volume-weighted momentum
    for w in [7, 14, 30]:
        signals[f"vw_mom_{w}d"] = momentum_volume_weighted(close, volume, w)

    # 4. Weekday seasonality
    signals["weekday_mom"] = momentum_weekday_seasonality(close)

    return signals

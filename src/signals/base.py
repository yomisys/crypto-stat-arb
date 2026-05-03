from __future__ import annotations
"""
Base utilities shared by all signal modules.

Key design rule (look-ahead prevention):
  signal[t] is computed from data observed at or before close of day t.
  In the backtester, weights derived from signal[t] are applied to the
  NEXT period's return: port_ret[t+1] = w[t] * r[t+1].
  This is the standard "signal-today, trade-at-close, earn-tomorrow" model.
"""

import numpy as np
import pandas as pd


def cross_section_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each asset within the cross-section at every date.

    Returns percentile ranks in [0, 1]:  0 = worst, 1 = best.
    NaN assets are excluded from the ranking (they remain NaN).
    """
    return signal.rank(axis=1, pct=True, na_option="keep")


def cross_section_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Demean and scale the signal cross-sectionally at every date.

    After this transform, at each date:
      - mean across assets  = 0
      - std  across assets  = 1
    This makes the portfolio dollar-neutral by construction.
    """
    mean = signal.mean(axis=1)
    std  = signal.std(axis=1).replace(0, np.nan)
    return signal.sub(mean, axis=0).div(std, axis=0)


def winsorize_signal(signal: pd.DataFrame, clip_z: float = 3.0) -> pd.DataFrame:
    """
    Clip extreme signal values at ±clip_z cross-sectional standard deviations.
    Applied after cross_section_zscore, so clip_z=3 removes the top/bottom ~0.3%.
    """
    return signal.clip(lower=-clip_z, upper=clip_z)


def time_series_zscore(signal: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Normalize each asset's signal using its own rolling mean and std.
    Useful for making signal magnitudes comparable across assets with
    different historical volatility regimes.
    """
    roll_mean = signal.rolling(window=window, min_periods=window // 4).mean()
    roll_std  = signal.rolling(window=window, min_periods=window // 4).std().replace(0, np.nan)
    return (signal - roll_mean) / roll_std


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) returns: (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change()

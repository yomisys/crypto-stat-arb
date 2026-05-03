from __future__ import annotations
"""
Reversal (mean-reversion) signal implementations.

Short-horizon reversal (Jegadeesh 1990): assets that drop sharply tend to
rebound within 1–5 days, and vice versa. The effect is attributed to
overreaction by retail traders and liquidity shocks.

In crypto, this is particularly pronounced because:
  - Retail sentiment amplifies intraday moves
  - Low-volume markets are more susceptible to noise
  - Leverage cascades create exaggerated price swings that partially reverse

All signals at time t use only data available at close of day t.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .base import (
    compute_log_returns,
    cross_section_zscore,
    winsorize_signal,
)


def reversal_short(close: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Short-horizon reversal: bet against recent price movement.

    Signal at t = -1 × cumulative return over [t-window, t].
    A large negative recent return → positive signal (expect bounce).
    A large positive recent return → negative signal (expect pullback).

    Works best at 1-day and 3-day horizons in crypto.
    """
    log_ret = compute_log_returns(close)
    recent_ret = log_ret.rolling(window=window, min_periods=1).sum()
    signal = -recent_ret  # negate for reversal
    return cross_section_zscore(winsorize_signal(signal))


def reversal_low_volume(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 3,
    vol_lookback: int = 30,
    vol_percentile: float = 0.35,
) -> pd.DataFrame:
    """
    Low-volume reversal.

    Finance intuition: price moves on abnormally low volume are more likely
    to be noise (temporary liquidity imbalances) rather than information-driven.
    These moves tend to reverse more reliably than high-volume moves.

    The signal is the standard reversal, but:
      - Amplified on low-volume days (volume < vol_percentile percentile)
      - Attenuated on high-volume days (signal set to 0)
    """
    log_ret = compute_log_returns(close)
    recent_ret = log_ret.rolling(window=window, min_periods=1).sum()

    # Rolling volume percentile threshold
    vol_thresh = volume.rolling(window=vol_lookback, min_periods=vol_lookback // 2).quantile(vol_percentile)
    is_low_vol = (volume < vol_thresh).astype(float)

    # Reversal signal amplified by low-volume indicator
    signal = -recent_ret * is_low_vol
    return cross_section_zscore(winsorize_signal(signal))


def reversal_pairs_spread(
    close: pd.DataFrame,
    zscore_window: int = 60,
    corr_window: int = 90,
    min_corr: float = 0.65,
) -> pd.DataFrame:
    """
    Pairs/correlation reversal (simplified statistical arbitrage).

    For each highly-correlated pair (i, j), compute the rolling z-score of
    the log-price spread: spread = ln(P_i) - ln(P_j).

    When the spread is far above its historical mean:
      - Asset i is expensive relative to j → short i, long j (expect convergence)
    When the spread is far below:
      - Asset i is cheap relative to j → long i, short j

    The per-asset signal aggregates signals from all pairs that include that asset.

    Look-ahead safety: the rolling statistics use a window ending at t,
    so we only use information available at close of t.
    """
    log_px = np.log(close.replace(0, np.nan))
    symbols = list(close.columns)
    n = len(symbols)

    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    count  = pd.DataFrame(0,   index=close.index, columns=close.columns)

    # Compute pairwise signals only for correlated pairs
    log_ret = log_px.diff()
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = symbols[i], symbols[j]

            # Rolling correlation to identify stable pairs (no look-ahead:
            # we use the same rolling window as for the z-score)
            roll_corr = (
                log_ret[[si, sj]]
                .rolling(window=corr_window, min_periods=corr_window // 2)
                .corr()
                .unstack()[si][sj]
            )

            # Only activate the pair when rolling correlation is high enough
            active = roll_corr >= min_corr

            # Log-price spread and its rolling z-score
            spread     = log_px[si] - log_px[sj]
            roll_mean  = spread.rolling(window=zscore_window, min_periods=zscore_window // 2).mean()
            roll_std   = spread.rolling(window=zscore_window, min_periods=zscore_window // 2).std()
            z          = (spread - roll_mean) / roll_std.replace(0, np.nan)
            z_clipped  = z.clip(-3, 3)

            # When spread is high: short i, long j → signal_i < 0, signal_j > 0
            z_active = z_clipped.where(active, other=0.0).fillna(0.0)
            signal[si] += -z_active
            signal[sj] +=  z_active
            pair_active = (active & z.notna()).astype(int)
            count[si]  += pair_active
            count[sj]  += pair_active

    # Normalize by number of active pairs to keep signal magnitudes stable
    count_safe = count.replace(0, np.nan)
    signal = signal / count_safe
    signal = signal.fillna(0.0)

    return cross_section_zscore(signal)


def reversal_macro_regime(
    close: pd.DataFrame,
    btc_col: str = "BTC/USDT",
    reversal_window: int = 3,
    vol_window: int = 20,
    high_vol_amplify: float = 1.5,
) -> pd.DataFrame:
    """
    Macro-regime conditioned reversal.

    Uses BTC realized volatility as a proxy for the crypto risk regime:
      - High BTC vol (risk-off): herding and panic exaggerate moves →
        reversals are stronger and more reliable
      - Low BTC vol (risk-on): trend-following dominates →
        reversals are weaker

    Signal = base_reversal × regime_scalar
    where regime_scalar is a rolling BTC vol percentile scaled to [1, high_vol_amplify].
    """
    # Fall back to first column if BTC not in universe
    if btc_col not in close.columns:
        btc_col = close.columns[0]

    btc_log_ret = np.log(close[btc_col] / close[btc_col].shift(1))
    btc_realvol = btc_log_ret.rolling(window=vol_window, min_periods=vol_window // 2).std() * np.sqrt(365)

    # Rolling percentile rank of realized vol (0 = calmest, 1 = most stressed)
    vol_pct = btc_realvol.rolling(window=252, min_periods=60).rank(pct=True).fillna(0.5)

    # Base reversal signal
    base = reversal_short(close, window=reversal_window)

    # Regime scalar: 1.0 in calm periods, up to high_vol_amplify in stressed periods
    scalar = 1.0 + (high_vol_amplify - 1.0) * vol_pct
    scalar = scalar.reindex(base.index).fillna(1.0)

    return base.mul(scalar, axis=0)


def all_reversal_signals(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    btc_col: str = "BTC/USDT",
) -> dict[str, pd.DataFrame]:
    """
    Build the full reversal signal library.

    Returns {name: signal_DataFrame}, each cross-sectionally z-scored.
    """
    signals: dict[str, pd.DataFrame] = {}

    # 1. Short-horizon reversal at 1-day and 3-day horizons
    from config import REVERSAL_WINDOWS
    for w in REVERSAL_WINDOWS:
        signals[f"rev_{w}d"] = reversal_short(close, w)

    # 2. Low-volume reversal
    signals["low_vol_rev"] = reversal_low_volume(close, volume)

    # 3. Pairs spread reversal
    signals["pairs_rev"] = reversal_pairs_spread(close)

    # 4. Macro-regime conditioned reversal
    if btc_col not in close.columns:
        btc_col = close.columns[0]
    signals["macro_rev"] = reversal_macro_regime(close, btc_col=btc_col)

    return signals

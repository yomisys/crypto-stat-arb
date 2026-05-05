"""
Central configuration for the crypto statistical-arbitrage research project.
All constants live here so every module imports from a single source of truth.
"""

from datetime import datetime
from pathlib import Path

# ─── Universe ─────────────────────────────────────────────────────────────────
# Top ~25 cryptos by market cap available on Binance spot.
# Newer coins (APT, OP, ARB) were listed in 2022-2023, so they will have
# shorter histories; that is fine — the backtester handles partial data.
TOP_CRYPTOS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "LTC/USDT", "BCH/USDT", "UNI/USDT", "ATOM/USDT",
    "XLM/USDT", "ETC/USDT", "ALGO/USDT", "FIL/USDT", "TRX/USDT",
    "NEAR/USDT", "APT/USDT", "OP/USDT", "ARB/USDT", "VET/USDT",
]

# Coins available since at least 2021 (used for long-horizon tests)
CORE_CRYPTOS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "LTC/USDT", "BCH/USDT", "LINK/USDT", "XLM/USDT",
    "ETC/USDT", "ALGO/USDT", "VET/USDT", "ATOM/USDT", "TRX/USDT",
    "DOT/USDT",
]

# ─── Date Ranges ──────────────────────────────────────────────────────────────
DAILY_START  = datetime(2021, 1, 1)   # ~3.5 years of daily data
HOURLY_START = datetime(2023, 11, 1)  # ~18 months of hourly data

# ─── Train / Validation Split ─────────────────────────────────────────────────
# ALL parameter selection uses training data only; evaluation uses both.
TRAIN_START = datetime(2021, 1,  1)
TRAIN_END   = datetime(2022, 12, 31)
VAL_START   = datetime(2023, 1,  1)
VAL_END     = datetime(2025, 12, 31)

# ─── Transaction Costs ────────────────────────────────────────────────────────
MARKET_ORDER_COST_BPS = 20   # 0.20%  — typical taker fee on major exchanges
LIMIT_ORDER_COST_BPS  =  7   # 0.07%  — typical maker fee

# ─── Signal Parameters ────────────────────────────────────────────────────────
MOMENTUM_WINDOWS  = [1, 3, 7, 14, 30, 60, 90]  # lookback windows in days
REVERSAL_WINDOWS  = [1, 3]                       # short-horizon reversal windows
PAIRS_ZSCORE_WINDOW    = 60    # rolling window for spread z-score
PAIRS_CORR_MIN         = 0.65  # min rolling correlation to include a pair
VOL_REGIME_WINDOW      = 20    # days used to estimate BTC realized vol

# ─── Portfolio Construction ───────────────────────────────────────────────────
# Quintile strategy: long top quintile (Q5), short bottom quintile (Q1)
QUINTILE_LONG_THRESHOLD  = 0.80   # percentile rank cut-off for long leg
QUINTILE_SHORT_THRESHOLD = 0.20   # percentile rank cut-off for short leg
MIN_ASSETS_FOR_SIGNAL    = 8      # skip rebalance if fewer assets have valid signal

# ─── Annualization ────────────────────────────────────────────────────────────
# Crypto trades 24 / 7 / 365 — no weekend or holiday closures
PERIODS_PER_YEAR = {"1d": 365, "1h": 365 * 24}

# ─── File Paths ───────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
RAW_DAILY_DIR = DATA_DIR / "raw" / "daily"
RAW_HOURLY_DIR= DATA_DIR / "raw" / "hourly"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories on import (idempotent)
for _d in [RAW_DAILY_DIR, RAW_HOURLY_DIR, PROCESSED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

from __future__ import annotations
"""
Data pipeline: fetch OHLCV from Binance via CCXT, store as Parquet, load as DataFrames.

No API key is required — all endpoints used are public.
Each symbol gets its own parquet file so incremental updates are cheap.

Naming convention for stored files:
    data/raw/daily/BTC_USDT.parquet
    data/raw/hourly/ETH_USDT.parquet

All timestamps are stored tz-naive UTC to avoid pandas alignment issues.
"""

import time
import logging
from pathlib import Path
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)

# Milliseconds per bar for each supported timeframe
_TF_MS = {"1d": 86_400_000, "1h": 3_600_000}


# ─── Exchange ──────────────────────────────────────────────────────────────────

def get_exchange() -> ccxt.Exchange:
    """Return a Binance exchange instance using only public (unauthenticated) endpoints."""
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


# ─── Fetch helpers ────────────────────────────────────────────────────────────

def fetch_ohlcv_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    pause_sec: float = 0.3,
) -> pd.DataFrame:
    """
    Paginate through CCXT fetch_ohlcv (max 1 000 bars per call) and return
    a complete DataFrame for [since_ms, until_ms).

    Columns: open, high, low, close, volume
    Index:   DatetimeIndex, tz-naive UTC
    """
    tf_ms = _TF_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use '1d' or '1h'.")

    all_bars: list = []
    current = since_ms

    while current < until_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        except ccxt.NetworkError as exc:
            logger.warning("Network error for %s (%s): %s — retrying in 10 s", symbol, timeframe, exc)
            time.sleep(10)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error for %s (%s): %s — skipping", symbol, timeframe, exc)
            break

        if not bars:
            break

        all_bars.extend(bars)
        last_ts = bars[-1][0]
        current = last_ts + tf_ms  # advance past the last returned bar

        # Binance returns fewer than 1 000 bars when we've reached the present
        if len(bars) < 1000:
            break

        time.sleep(pause_sec)

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")  # tz-naive UTC
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Clip to requested range
    cutoff = pd.Timestamp(until_ms, unit="ms")
    df = df[df.index < cutoff]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _symbol_to_filename(symbol: str) -> str:
    """'BTC/USDT' → 'BTC_USDT'"""
    return symbol.replace("/", "_")


def _filename_to_symbol(stem: str) -> str:
    """'BTC_USDT' → 'BTC/USDT'  (first underscore only)"""
    return stem.replace("_", "/", 1)


# ─── Fetch + store ────────────────────────────────────────────────────────────

def fetch_and_store_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_dt: datetime,
    save_dir: Path,
    force_refresh: bool = False,
) -> bool:
    """
    Fetch OHLCV for one symbol and persist to Parquet.

    If the file already exists and force_refresh=False, only new bars
    (after the last stored timestamp) are downloaded and appended.

    Returns True on success, False if no data could be retrieved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"{_symbol_to_filename(symbol)}.parquet"

    existing: pd.DataFrame | None = None
    since_ms = int(since_dt.timestamp() * 1000)

    if fname.exists() and not force_refresh:
        try:
            existing = pd.read_parquet(fname, engine="pyarrow")
            if not existing.empty:
                last_ts = existing.index.max()
                # Resume one bar after the last stored bar
                tf_ms = _TF_MS.get(timeframe, 86_400_000)
                since_ms = int(last_ts.timestamp() * 1000) + tf_ms
        except Exception as exc:
            logger.warning("Could not read %s: %s — re-fetching from scratch", fname, exc)
            existing = None

    until_ms = int(datetime.utcnow().timestamp() * 1000)

    if since_ms >= until_ms:
        logger.debug("%s (%s): data is already up to date", symbol, timeframe)
        return True

    new_bars = fetch_ohlcv_symbol(exchange, symbol, timeframe, since_ms, until_ms)

    if new_bars.empty and existing is None:
        logger.warning("%s (%s): no data retrieved", symbol, timeframe)
        return False

    if existing is not None and not new_bars.empty:
        combined = pd.concat([existing, new_bars])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    elif existing is not None:
        combined = existing
    else:
        combined = new_bars

    combined.to_parquet(fname, engine="pyarrow", compression="snappy")
    logger.info("%s (%s): %d bars saved → %s", symbol, timeframe, len(combined), fname)
    return True


def fetch_and_store_universe(
    timeframe: str = "1d",
    symbols: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, bool]:
    """
    Fetch and store OHLCV for the full universe of symbols.

    timeframe: '1d' (daily) or '1h' (hourly)
    symbols:   override the default universe from config.py
    Returns:   {symbol: success} mapping
    """
    if symbols is None:
        symbols = config.TOP_CRYPTOS

    since_dt = config.DAILY_START if timeframe == "1d" else config.HOURLY_START
    save_dir  = config.RAW_DAILY_DIR if timeframe == "1d" else config.RAW_HOURLY_DIR

    exchange = get_exchange()

    # Filter to symbols actually listed on Binance
    try:
        markets = exchange.load_markets()
        valid_symbols = [s for s in symbols if s in markets]
        skipped = set(symbols) - set(valid_symbols)
        if skipped:
            logger.warning("Not on Binance, skipping: %s", skipped)
    except Exception:
        valid_symbols = symbols

    results: dict[str, bool] = {}
    for sym in tqdm(valid_symbols, desc=f"Fetching {timeframe}"):
        ok = fetch_and_store_symbol(exchange, sym, timeframe, since_dt, save_dir, force_refresh)
        results[sym] = ok
        time.sleep(0.2)

    n_ok = sum(results.values())
    print(f"\nOK {timeframe}: {n_ok}/{len(valid_symbols)} symbols fetched successfully")
    return results


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_ohlcv_wide(
    timeframe: str = "1d",
    field: str = "close",
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a single OHLCV field as a wide DataFrame (rows = dates, columns = symbols).

    Example:
        close = load_ohlcv_wide("1d", "close")  # DataFrame[date, BTC/USDT, ETH/USDT, ...]
    """
    save_dir = config.RAW_DAILY_DIR if timeframe == "1d" else config.RAW_HOURLY_DIR
    files = sorted(save_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files in {save_dir}. Run fetch_and_store_universe('{timeframe}') first."
        )

    series_dict: dict[str, pd.Series] = {}
    for fpath in files:
        sym = _filename_to_symbol(fpath.stem)
        if symbols is not None and sym not in symbols:
            continue
        try:
            df = pd.read_parquet(fpath, engine="pyarrow")
            if field in df.columns:
                series_dict[sym] = df[field]
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath, exc)

    if not series_dict:
        raise ValueError(f"No data loaded for field '{field}' (timeframe={timeframe})")

    wide = pd.DataFrame(series_dict)
    wide.index = pd.to_datetime(wide.index)   # ensure DatetimeIndex
    if wide.index.tz is not None:
        wide.index = wide.index.tz_localize(None)  # strip timezone
    return wide.sort_index()


def load_all_fields(
    timeframe: str = "1d",
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all five OHLCV fields.

    Returns:
        {"open": df, "high": df, "low": df, "close": df, "volume": df}
        Each df has shape (n_dates, n_symbols).
    """
    return {
        field: load_ohlcv_wide(timeframe, field, symbols)
        for field in ["open", "high", "low", "close", "volume"]
    }


def compute_and_save_returns(
    timeframe: str = "1d",
    log: bool = True,
) -> pd.DataFrame:
    """
    Compute returns from close prices and save to data/processed/.

    log=True  → log returns:    ln(P_t / P_{t-1})
    log=False → simple returns: (P_t - P_{t-1}) / P_{t-1}

    Returns the wide returns DataFrame.
    """
    close = load_ohlcv_wide(timeframe, "close")
    if log:
        returns = np.log(close / close.shift(1))
    else:
        returns = close.pct_change()

    fname = config.PROCESSED_DIR / f"returns_{timeframe}.parquet"
    returns.to_parquet(fname, engine="pyarrow", compression="snappy")
    logger.info("Returns (%s) saved to %s", timeframe, fname)
    return returns


def load_returns(timeframe: str = "1d") -> pd.DataFrame:
    """
    Load pre-computed returns from data/processed/.
    If the file doesn't exist, build it from raw OHLCV first.
    """
    fname = config.PROCESSED_DIR / f"returns_{timeframe}.parquet"
    if not fname.exists():
        logger.info("Returns file not found — building from raw OHLCV")
        return compute_and_save_returns(timeframe)
    return pd.read_parquet(fname, engine="pyarrow")

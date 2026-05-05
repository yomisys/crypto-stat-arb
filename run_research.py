"""
run_research.py  —  Crypto Statistical Arbitrage Research Pipeline
==================================================================
Run from the classproject directory:

    python run_research.py

What it does (end to end):
  1. Fetch OHLCV data for 25 coins from Binance (cached locally after first run)
  2. Build 19 momentum + reversal signals
  3. Backtest each signal as a long-short quintile strategy (20 bps cost)
  4. Train/Validation split evaluation (2021-2022 train | 2023-2025 val)
  5. Alpha t-statistics and significance flags for all strategies
  6. Walk-forward quarterly combination portfolios (OOS only)
  7. Save six updated charts to outputs/

Flags:
  --fetch     Force re-download even if local cache exists
  --limit N   Only use first N coins (handy for a quick test run)

Methodology note:
  All reported results are out-of-sample (validation: 2023-2025) unless
  explicitly labelled as in-sample. Drawdown uses log-return cumulation
  per mentor feedback.
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
from data_pipeline import fetch_and_store_universe, load_all_fields
from signals.momentum import all_momentum_signals
from signals.reversal import all_reversal_signals
from backtester import run_all_backtests
from evaluator import (
    compare_strategies, train_val_comparison,
    sharpe_ratio, annualized_return, annualized_vol,
    max_drawdown, drawdown_series, rolling_sharpe, sig_flag,
)
from portfolio import build_walk_forward_combinations

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "2021-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2025-12-31"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


def section(title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _idx(series):
    """Convert DatetimeIndex to numpy array for matplotlib compatibility."""
    return series.index.to_numpy()


# ─── Plot functions ───────────────────────────────────────────────────────────

def plot_train_val_sharpe(tv_table, filename, top_n=15, figsize=(13, 7)):
    """Chart 01 — Train vs Validation Sharpe side by side."""
    df = tv_table.head(top_n).copy()
    y  = np.arange(len(df))
    h  = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10.colors

    train_c = ["steelblue" if v >= 0 else "lightcoral" for v in df["Train Sharpe"]]
    val_c   = ["seagreen"  if v >= 0 else "tomato"     for v in df["Val Sharpe"]]

    bars_t = ax.barh(y + h / 2, df["Train Sharpe"].values, h,
                     color=train_c, label="Train (2021-2022)", alpha=0.85)
    bars_v = ax.barh(y - h / 2, df["Val Sharpe"].values, h,
                     color=val_c,   label="Val (2023-2025)",   alpha=0.85)

    for bar, flag in zip(bars_v, df.get("Overfit Flag", pd.Series())):
        if flag == "OVERFIT":
            bar.set_edgecolor("red")
            bar.set_linewidth(2.0)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Sharpe Ratio (annualized, net of 20 bps cost)")
    ax.set_title("Strategy Ranking — Train vs Validation Sharpe",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save(fig, filename)


def plot_equity_curves(returns_dict, title, filename,
                       split_date=None, figsize=(13, 5)):
    """Chart 02 — Equity curves with optional train/val split line."""
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10.colors
    for i, (label, ret) in enumerate(returns_dict.items()):
        clean  = ret.dropna()
        wealth = (1 + clean).cumprod()
        ax.plot(_idx(wealth), wealth.values, label=label,
                color=colors[i % len(colors)], lw=1.8)

    if split_date is not None:
        sd = pd.Timestamp(split_date)
        ax.axvline(sd, color="black", lw=1.5, ls="--", alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(sd, ylim[1] * 0.97, "  Val start",
                fontsize=8, color="black", va="top")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio value ($1 initial)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    save(fig, filename)


def plot_drawdowns(returns_dict, title, filename, figsize=(13, 4)):
    """Chart 03 — Drawdown (fixed log-return formula)."""
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10.colors
    for i, (label, ret) in enumerate(returns_dict.items()):
        dd = drawdown_series(ret.dropna()) * 100
        ax.fill_between(_idx(dd), dd.values, 0,
                        alpha=0.2, color=colors[i % len(colors)])
        ax.plot(_idx(dd), dd.values, label=label,
                color=colors[i % len(colors)], lw=1.4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower left", fontsize=8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    save(fig, filename)


def plot_rolling_sharpe_multi(returns_dict, title, filename,
                              window=90, figsize=(13, 4)):
    """Chart 04 — Rolling 90-day Sharpe."""
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10.colors
    for i, (label, ret) in enumerate(returns_dict.items()):
        rs = rolling_sharpe(ret.dropna(), window=window)
        ax.plot(_idx(rs), rs.values, label=label,
                color=colors[i % len(colors)], lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe (90-day rolling, annualized)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    save(fig, filename)


def plot_wf_combinations(wf_combos, best_individual, best_name,
                         btc_ret, filename, figsize=(13, 5)):
    """Chart 05 — Walk-forward OOS equity curves only."""
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10.colors
    plot_dict = dict(wf_combos)
    plot_dict[f"best_indiv ({best_name})"] = best_individual
    plot_dict["BTC Buy&Hold"] = btc_ret

    for i, (label, ret) in enumerate(plot_dict.items()):
        clean  = ret.dropna()
        wealth = (1 + clean).cumprod()
        ax.plot(_idx(wealth), wealth.values, label=label,
                color=colors[i % len(colors)], lw=1.8)

    ax.set_title("Walk-Forward Combination Portfolios (OOS only)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio value ($1 initial)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    save(fig, filename)


def plot_cost_sensitivity_val(results, top_names, filename,
                              val_start=VAL_START, figsize=(10, 5)):
    """Chart 06 — Cost sensitivity on validation set only."""
    cost_range = [0, 5, 10, 15, 20, 30, 50]
    records = []
    for name in top_names:
        r     = results[name]["returns"]
        gross = (r["net_return"] + r["transaction_cost"]).loc[val_start:]
        turn  = r["turnover"].loc[val_start:]
        for cbps in cost_range:
            net = gross - turn * (cbps / 10_000)
            records.append({"Strategy": name, "Cost (bps)": cbps,
                             "Sharpe": sharpe_ratio(net.dropna())})

    df     = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10.colors
    for i, name in enumerate(top_names):
        sub = df[df["Strategy"] == name]
        ax.plot(sub["Cost (bps)"].values, sub["Sharpe"].values, "o-",
                label=name, lw=2, color=colors[i % len(colors)])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(20, color="grey",  lw=1, ls=":", label="Market order (20 bps)")
    ax.axvline(7,  color="green", lw=1, ls=":", label="Limit order (7 bps)")
    ax.set_xlabel("Transaction cost (bps, one-way)")
    ax.set_ylabel("Net Sharpe Ratio (validation set only)")
    ax.set_title("Sharpe Sensitivity to Transaction Costs (Validation 2023-2025)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, filename)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main(force_fetch=False, coin_limit=None):

    # ── 1. Data ───────────────────────────────────────────────────────────────
    section("STEP 1 / 7  —  Data Collection")

    symbols = config.TOP_CRYPTOS[:coin_limit] if coin_limit else config.TOP_CRYPTOS
    print(f"Universe: {len(symbols)} coins")

    print("Fetching daily OHLCV from Binance (cached after first run) ...")
    fetch_and_store_universe("1d", symbols=symbols, force_refresh=force_fetch)

    print("Fetching hourly OHLCV ...")
    fetch_and_store_universe("1h", symbols=symbols, force_refresh=force_fetch)

    # ── 2. Load ───────────────────────────────────────────────────────────────
    section("STEP 2 / 7  —  Loading Data")

    fields   = load_all_fields("1d", symbols=symbols)
    close_d  = fields["close"]
    volume_d = fields["volume"]

    valid    = close_d.columns[close_d.notna().sum() >= 200]
    close_d  = close_d[valid]
    volume_d = volume_d[valid]

    print(f"Loaded:     {len(close_d.columns)} coins  x  {len(close_d)} days")
    print(f"Date range: {close_d.index.min().date()}  ->  {close_d.index.max().date()}")

    btc_prices = (close_d["BTC/USDT"] if "BTC/USDT" in close_d.columns
                  else close_d.iloc[:, 0])
    btc_ret    = btc_prices.pct_change()

    print(f"\nBTC benchmark  |  Ann. return: "
          f"{annualized_return(btc_ret.dropna())*100:.1f}%  |  "
          f"Ann. vol: {annualized_vol(btc_ret.dropna())*100:.1f}%  |  "
          f"Sharpe: {sharpe_ratio(btc_ret.dropna()):.2f}")

    # ── 3. Signals ────────────────────────────────────────────────────────────
    section("STEP 3 / 7  —  Signal Generation")

    print("Building momentum signals ...")
    mom_sigs = all_momentum_signals(close_d, volume_d)
    print(f"  {len(mom_sigs)} momentum signals")

    print("Building reversal signals (pairs step may take ~30 s) ...")
    rev_sigs = all_reversal_signals(close_d, volume_d)
    print(f"  {len(rev_sigs)} reversal signals")

    all_signals = {**mom_sigs, **rev_sigs}
    print(f"  {len(all_signals)} total signals")

    # ── 4. Backtesting ────────────────────────────────────────────────────────
    section("STEP 4 / 7  —  Backtesting (20 bps market-order cost)")

    print("Running backtests ...")
    results = run_all_backtests(all_signals, close_d, cost_mode="market")

    # ── 5. Full-sample performance + alpha stats ──────────────────────────────
    section("STEP 5 / 7  —  Performance Evaluation (alpha t-stats, corrected drawdown)")

    table = compare_strategies(results, btc_prices)

    display_cols = [
        "Ann. Return (net) %", "Ann. Volatility %", "Sharpe Ratio",
        "Max Drawdown %", "Alpha (ann) %", "Alpha t-stat", "Sig",
        "Beta vs BTC", "Win Rate %", "Avg Daily Turnover",
    ]

    print("\nAll strategies — full sample — ranked by net Sharpe Ratio:")
    print(table[display_cols].round(2).to_string())

    # ── 6. Train / Validation split ───────────────────────────────────────────
    section("STEP 6 / 7  —  Train / Validation Split  (Change 1)")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Val  : {VAL_START}   to {VAL_END}")
    print()

    tv = train_val_comparison(
        results, btc_prices,
        train_start=TRAIN_START, train_end=TRAIN_END,
        val_start=VAL_START,     val_end=VAL_END,
    )

    tv_display = ["Train Sharpe", "Val Sharpe", "Sharpe Decay %", "Overfit Flag",
                  "Train Alpha %", "Train Alpha t", "Val Alpha %", "Val Alpha t", "Val Sig"]
    print("Train vs Validation performance (sorted by Val Sharpe):")
    print(tv[tv_display].round(2).to_string())

    n_overfit = (tv["Overfit Flag"] == "OVERFIT").sum()
    n_val_pos = (tv["Val Sharpe"] > 0).sum()
    print(f"\n{n_val_pos}/{len(tv)} strategies have positive validation Sharpe")
    print(f"{n_overfit}/{len(tv)} strategies flagged as potential overfits "
          "(val Sharpe < 50% of train Sharpe)")

    # Tier 1 = positive val Sharpe + significant alpha in val window
    tier1_mask = (tv["Val Sharpe"] > 0) & (tv["Val Sig"].isin(["***", "**", "*"]))
    tier1 = tv[tier1_mask].index.tolist()
    print(f"\nTier 1 strategies (positive val Sharpe + significant val alpha): "
          f"{len(tier1)}")
    for s in tier1[:10]:
        print(f"  {s:30s}  Val SR={tv.loc[s,'Val Sharpe']:+.2f}  "
              f"Val alpha={tv.loc[s,'Val Alpha %']:+.1f}%  "
              f"({tv.loc[s,'Val Sig']})")

    # ── Walk-forward combinations ─────────────────────────────────────────────
    section("STEP 7 / 7  —  Walk-Forward Combination Portfolios  (Change 2)")

    good = tv[tv["Val Sharpe"] > 0].index.tolist()
    print(f"Combining {len(good)} strategies with positive val Sharpe")

    wf_combos = build_walk_forward_combinations(results, strategy_names=good or None)

    print("\nWalk-forward combination performance (OOS only):")
    hdr = f"  {'Portfolio':<18}  {'Ann.Ret%':>9}  {'Sharpe':>7}  {'MaxDD%':>8}  {'WinRate%':>9}"
    print(hdr)
    print("  " + "-" * 56)
    for name, ret in wf_combos.items():
        clean = ret.dropna()
        if len(clean) < 30:
            print(f"  {name:<18}  (insufficient OOS data)")
            continue
        ar  = annualized_return(clean) * 100
        sr  = sharpe_ratio(clean)
        mdd = max_drawdown(clean) * 100
        wr  = (clean > 0).mean() * 100
        print(f"  {name:<18}  {ar:>+9.2f}  {sr:>+7.2f}  {mdd:>8.2f}  {wr:>9.1f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    section("Saving Charts  ->  outputs/")

    # 01 — Train vs Val Sharpe side-by-side
    plot_train_val_sharpe(tv, "01_sharpe_ranking.png")

    # 02 — Equity curves: top 5 + BTC + train/val split line
    top5      = tv["Val Sharpe"].nlargest(5).index.tolist()
    top5_rets = {n: results[n]["returns"]["net_return"] for n in top5}
    top5_rets["BTC Buy&Hold"] = btc_ret
    plot_equity_curves(
        top5_rets,
        "Top-5 Strategy Equity Curves (net of 20 bps)",
        "02_equity_curves_top5.png",
        split_date=VAL_START,
    )

    # 03 — Drawdown (recomputed with fixed log-return formula)
    plot_drawdowns(
        {n: results[n]["returns"]["net_return"] for n in top5},
        "Drawdown — Top-5 Strategies (log-return method)",
        "03_drawdowns_top5.png",
    )

    # 04 — Rolling Sharpe: top 3 + walk-forward
    best3  = tv["Val Sharpe"].nlargest(3).index.tolist()
    rs_dict = {n: results[n]["returns"]["net_return"] for n in best3}
    rs_dict.update({k: v for k, v in wf_combos.items()})
    plot_rolling_sharpe_multi(
        rs_dict,
        "Rolling 90-Day Sharpe: Top Strategies + Walk-Forward Combinations",
        "04_rolling_sharpe.png",
    )

    # 05 — Walk-forward OOS equity curves (replaces old combo chart)
    best_name = best3[0] if best3 else top5[0]
    plot_wf_combinations(
        wf_combos, results[best_name]["returns"]["net_return"],
        best_name, btc_ret, "05_combinations.png",
    )

    # 06 — Cost sensitivity on validation set only
    plot_cost_sensitivity_val(results, best3, "06_cost_sensitivity.png")

    # ── Generalisability Assessment ───────────────────────────────────────────
    section("Generalisability Assessment")

    print("\nIn-sample vs Out-of-sample Sharpe for Tier-1 strategies:")
    hdr2 = (f"  {'Strategy':30s}  {'Train SR':>9}  {'Val SR':>7}  "
            f"{'Decay%':>8}  {'Val Sig':>7}  Comment")
    print(hdr2)
    print("  " + "-" * 80)
    for s in (tier1[:10] if tier1 else tv.head(10).index.tolist()):
        tr_sr  = tv.loc[s, "Train Sharpe"]
        va_sr  = tv.loc[s, "Val Sharpe"]
        decay  = tv.loc[s, "Sharpe Decay %"]
        sig    = tv.loc[s, "Val Sig"]
        flag   = tv.loc[s, "Overfit Flag"]
        if flag == "OVERFIT":
            comment = "likely overfit"
        elif pd.isna(va_sr) or va_sr < 0:
            comment = "alpha does not survive OOS"
        elif sig in ("***", "**"):
            comment = "alpha survives walk-forward test"
        elif sig == "*":
            comment = "marginal OOS significance"
        else:
            comment = "positive but not significant OOS"
        print(f"  {s:30s}  {tr_sr:>+9.2f}  {va_sr:>+7.2f}  "
              f"{decay:>7.1f}%  {sig:>7s}  {comment}")

    # ── Save updated CSV ──────────────────────────────────────────────────────
    section("Saving Outputs")

    # Full-sample metrics (includes alpha t-stat, p-value, corrected drawdown)
    full_cols = [
        "Ann. Return (net) %", "Ann. Volatility %", "Sharpe Ratio",
        "Max Drawdown %", "Alpha (ann) %", "Alpha t-stat", "Alpha p-value",
        "Sig", "Beta vs BTC", "Beta t-stat", "R-squared",
        "Win Rate %", "Avg Daily Turnover", "Ann. Cost Drag %", "N Days",
    ]
    csv_path = OUT_DIR / "strategy_metrics.csv"
    table[full_cols].round(4).to_csv(csv_path)
    print(f"  strategy_metrics.csv saved  ({len(table)} strategies)")

    # Train/val comparison CSV
    tv_csv = OUT_DIR / "train_val_comparison.csv"
    tv.round(4).to_csv(tv_csv)
    print(f"  train_val_comparison.csv saved")

    print("\n" + "=" * 70)
    print("  DONE — charts in outputs/ | CSVs: strategy_metrics.csv,"
          " train_val_comparison.csv")
    print("=" * 70)

    return table, tv, results, wf_combos


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto stat-arb research pipeline"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Force re-download of market data (ignore local cache)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only use the first N coins (e.g. --limit 10 for a quick test)"
    )
    args = parser.parse_args()
    main(force_fetch=args.fetch, coin_limit=args.limit)

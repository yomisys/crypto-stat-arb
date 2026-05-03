"""
run_research.py  —  Crypto Statistical Arbitrage Research Pipeline
==================================================================
Run from the classproject directory:

    python run_research.py

What it does (end to end):
  1. Fetch OHLCV data for 25 coins from Binance (cached locally after first run)
  2. Build 19 momentum + reversal signals
  3. Backtest each signal as a long-short quintile strategy (20 bps cost)
  4. Print a ranked performance table
  5. Build equal-weight / inverse-vol / Sharpe-weighted combined portfolios
  6. Save six charts to outputs/

Flags:
  --fetch     Force re-download even if local cache exists
  --limit N   Only use first N coins (handy for a quick test run)

Example quick test:
    python run_research.py --limit 10
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
matplotlib.use("Agg")          # headless: save to files, no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
from data_pipeline import fetch_and_store_universe, load_all_fields
from signals.momentum import all_momentum_signals
from signals.reversal import all_reversal_signals
from backtester import run_all_backtests
from evaluator import (
    compare_strategies, sharpe_ratio, annualized_return,
    annualized_vol, max_drawdown, drawdown_series, rolling_sharpe,
)
from portfolio import build_combinations

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


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


# ─── Plot functions (no GUI — all save to PNG) ────────────────────────────────

def _idx(series):
    """Convert DatetimeIndex to numpy array for matplotlib 3.3 compatibility."""
    return series.index.to_numpy()


def plot_equity_curves(returns_dict, title, filename, figsize=(13, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    for i, (label, ret) in enumerate(returns_dict.items()):
        clean = ret.dropna()
        wealth = (1 + clean).cumprod()
        ax.plot(_idx(wealth), wealth.values, label=label,
                color=colors[i % len(colors)], lw=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio value ($1 initial)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    save(fig, filename)


def plot_drawdowns(returns_dict, title, filename, figsize=(13, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
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


def plot_sharpe_bars(table, filename, figsize=(12, 6)):
    top = table.head(15)
    colors = ["steelblue" if v >= 0 else "tomato"
              for v in top["Sharpe Ratio"]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top.index[::-1], top["Sharpe Ratio"].values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Sharpe Ratio (annualized, net of 20 bps cost)")
    ax.set_title("Strategy Ranking by Net Sharpe Ratio", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, filename)


def plot_rolling_sharpe_multi(returns_dict, title, filename,
                               window=90, figsize=(13, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
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


def plot_cost_sensitivity(results, top_names, filename, figsize=(10, 5)):
    from backtester import signal_to_weights, compute_portfolio_returns

    cost_range = [0, 5, 10, 15, 20, 30, 50]
    records = []
    for name in top_names:
        sig = results[name]["signal"]
        px  = results[name]["weights"]      # reuse aligned prices indirectly
        # Re-run with synthetic cost sweep using stored weights
        w = results[name]["weights"]
        simple_ret = results[name]["returns"]["gross_return"] + \
                     results[name]["returns"]["transaction_cost"]  # add back cost
        for cbps in cost_range:
            net = simple_ret - w.diff().abs().sum(axis=1).shift(1).fillna(0) * (cbps / 10_000)
            records.append({"Strategy": name, "Cost (bps)": cbps,
                             "Sharpe": sharpe_ratio(net.dropna())})

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    for i, name in enumerate(top_names):
        sub = df[df["Strategy"] == name]
        ax.plot(sub["Cost (bps)"].values, sub["Sharpe"].values, "o-",
                label=name, lw=2, color=colors[i % len(colors)])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(20, color="grey", lw=1, ls=":", label="Market order (20 bps)")
    ax.axvline(7,  color="green", lw=1, ls=":", label="Limit order (7 bps)")
    ax.set_xlabel("Transaction cost (bps, one-way)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Sharpe Sensitivity to Transaction Costs", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, filename)


def plot_corr_heatmap(returns_df, title, filename, figsize=(13, 11)):
    import seaborn as sns
    corr = returns_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    labels = [c.replace("/USDT", "") for c in corr.columns]
    corr.columns = labels
    corr.index   = labels
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.6})
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, filename)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main(force_fetch=False, coin_limit=None):

    # ── 1. Data ───────────────────────────────────────────────────────────────
    section("STEP 1 / 6  —  Data Collection")

    symbols = config.TOP_CRYPTOS[:coin_limit] if coin_limit else config.TOP_CRYPTOS
    print(f"Universe: {len(symbols)} coins")

    print("Fetching daily OHLCV from Binance (cached after first run) ...")
    fetch_and_store_universe("1d", symbols=symbols, force_refresh=force_fetch)

    print("Fetching hourly OHLCV ...")
    fetch_and_store_universe("1h", symbols=symbols, force_refresh=force_fetch)

    # ── 2. Load ───────────────────────────────────────────────────────────────
    section("STEP 2 / 6  —  Loading Data")

    fields   = load_all_fields("1d", symbols=symbols)
    close_d  = fields["close"]
    volume_d = fields["volume"]

    # Drop coins with fewer than 200 days of history
    valid = close_d.columns[close_d.notna().sum() >= 200]
    close_d  = close_d[valid]
    volume_d = volume_d[valid]

    print(f"Loaded:     {len(close_d.columns)} coins  x  {len(close_d)} days")
    print(f"Date range: {close_d.index.min().date()}  →  {close_d.index.max().date()}")

    miss = (close_d.isna().sum() / len(close_d) * 100)
    miss = miss[miss > 0]
    if not miss.empty:
        print(f"Missing data (% of rows): {miss.round(1).to_dict()}")

    btc_prices = (close_d["BTC/USDT"] if "BTC/USDT" in close_d.columns
                  else close_d.iloc[:, 0])
    btc_ret    = btc_prices.pct_change()

    print(f"\nBTC benchmark  |  Ann. return: "
          f"{annualized_return(btc_ret.dropna())*100:.1f}%  |  "
          f"Ann. vol: {annualized_vol(btc_ret.dropna())*100:.1f}%  |  "
          f"Sharpe: {sharpe_ratio(btc_ret.dropna()):.2f}")

    # ── 3. Signals ────────────────────────────────────────────────────────────
    section("STEP 3 / 6  —  Signal Generation")

    print("Building momentum signals ...")
    mom_sigs = all_momentum_signals(close_d, volume_d)
    print(f"  {len(mom_sigs)} momentum signals")

    print("Building reversal signals (pairs step may take ~30 s) ...")
    rev_sigs = all_reversal_signals(close_d, volume_d)
    print(f"  {len(rev_sigs)} reversal signals")

    all_signals = {**mom_sigs, **rev_sigs}
    print(f"  {len(all_signals)} total signals")

    # ── 4. Backtesting ────────────────────────────────────────────────────────
    section("STEP 4 / 6  —  Backtesting (20 bps market-order cost)")

    print("Running backtests ...")
    results = run_all_backtests(all_signals, close_d, cost_mode="market")

    # ── 5. Performance table ──────────────────────────────────────────────────
    section("STEP 5 / 6  —  Performance Evaluation")

    table = compare_strategies(results, btc_prices)

    display_cols = [
        "Ann. Return (net) %",
        "Ann. Volatility %",
        "Sharpe Ratio",
        "Max Drawdown %",
        "Alpha (ann) %",
        "Beta vs BTC",
        "Win Rate %",
        "Avg Daily Turnover",
    ]

    print("\nAll strategies — ranked by net Sharpe Ratio:")
    print(table[display_cols].round(2).to_string())

    # Signal: momentum vs reversal breakdown
    mom_names = [n for n in table.index if n in mom_sigs]
    rev_names = [n for n in table.index if n in rev_sigs]
    best_mom  = table.loc[mom_names, "Sharpe Ratio"].idxmax() if mom_names else "—"
    best_rev  = table.loc[rev_names, "Sharpe Ratio"].idxmax() if rev_names else "—"
    print(f"\nBest momentum strategy : {best_mom}  "
          f"(Sharpe {table.loc[best_mom,'Sharpe Ratio']:.2f})")
    print(f"Best reversal strategy : {best_rev}  "
          f"(Sharpe {table.loc[best_rev,'Sharpe Ratio']:.2f})")

    # ── 6. Combined portfolios ────────────────────────────────────────────────
    section("STEP 6 / 6  —  Strategy Combination")

    good = table[table["Sharpe Ratio"] > 0].index.tolist()
    print(f"Using {len(good)} profitable strategies for combination")

    combos = build_combinations(results, strategy_names=good if good else None)

    print("\nCombined portfolio performance:")
    combo_hdr = f"  {'Portfolio':<20}  {'Ann.Ret%':>9}  {'Sharpe':>7}  {'MaxDD%':>8}  {'WinRate%':>9}"
    print(combo_hdr)
    print("  " + "-" * 58)
    for name, ret in combos.items():
        clean = ret.dropna()
        ar  = annualized_return(clean) * 100
        sr  = sharpe_ratio(clean)
        mdd = max_drawdown(clean) * 100
        wr  = (clean > 0).mean() * 100
        print(f"  {name:<20}  {ar:>+9.2f}  {sr:>+7.2f}  {mdd:>8.2f}  {wr:>9.1f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    section("Saving Charts  →  outputs/")

    # 1. Sharpe bar chart for all strategies
    plot_sharpe_bars(table, "01_sharpe_ranking.png")

    # 2. Equity curves: top 5 strategies + BTC
    top5 = table["Sharpe Ratio"].nlargest(5).index.tolist()
    top5_rets = {n: results[n]["returns"]["net_return"] for n in top5}
    top5_rets["BTC Buy&Hold"] = btc_ret
    plot_equity_curves(top5_rets,
                       "Top-5 Strategy Equity Curves (net of 20 bps)",
                       "02_equity_curves_top5.png")

    # 3. Drawdown for top 5
    plot_drawdowns({n: results[n]["returns"]["net_return"] for n in top5},
                   "Drawdown — Top-5 Strategies",
                   "03_drawdowns_top5.png")

    # 4. Rolling Sharpe: top 3 + combined
    best3 = table["Sharpe Ratio"].nlargest(3).index.tolist()
    rs_dict = {n: results[n]["returns"]["net_return"] for n in best3}
    rs_dict.update({"equal_weight": combos["equal_weight"],
                    "inv_vol":      combos["inv_vol"]})
    plot_rolling_sharpe_multi(rs_dict,
                              "Rolling 90-Day Sharpe: Top Strategies + Combinations",
                              "04_rolling_sharpe.png")

    # 5. Combined portfolio equity curves
    combo_plot = dict(combos)
    combo_plot[f"best_individual ({best3[0]})"] = results[best3[0]]["returns"]["net_return"]
    combo_plot["BTC Buy&Hold"] = btc_ret
    plot_equity_curves(combo_plot,
                       "Combined Portfolios vs Best Individual vs BTC",
                       "05_combinations.png")

    # 6. Cost sensitivity for top 3
    plot_cost_sensitivity(results, best3, "06_cost_sensitivity.png")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DONE — all results printed above; charts saved to outputs/")
    print("=" * 70)

    # Save metrics table to CSV for easy reference in a report
    csv_path = OUT_DIR / "strategy_metrics.csv"
    table[display_cols].round(4).to_csv(csv_path)
    print(f"  Metrics CSV saved to {csv_path}")

    return table, results, combos


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

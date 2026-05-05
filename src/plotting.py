from __future__ import annotations
"""
Visualization helpers for the crypto stat-arb research notebook.

All functions return matplotlib Axes or Figures so callers can further
customize or embed them in multi-panel layouts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluator import drawdown_series, rolling_sharpe

# ─── Style defaults ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})
COLORS = sns.color_palette("tab10")


# ─── Equity curves ────────────────────────────────────────────────────────────

def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    log_scale: bool = False,
    split_date: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple = (12, 5),
) -> plt.Axes:
    """
    Plot compounded wealth index (starting at $1) for each return series.

    split_date: if provided (e.g. "2023-01-01"), draw a vertical dashed line
                to visually separate the train and validation windows.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, (label, ret) in enumerate(returns_dict.items()):
        clean = ret.dropna()
        wealth = (1 + clean).cumprod()
        ax.plot(wealth.index.to_numpy(), wealth.values,
                label=label, color=COLORS[i % len(COLORS)], lw=1.8)

    if split_date is not None:
        sd = pd.Timestamp(split_date)
        ax.axvline(sd, color="black", lw=1.5, ls="--", alpha=0.7)
        ax.text(sd, ax.get_ylim()[1] * 0.97, "  Val start",
                fontsize=8, color="black", va="top")

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($1 initial)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    return ax


def plot_drawdown(
    returns_dict: dict[str, pd.Series],
    title: str = "Drawdown",
    ax: plt.Axes | None = None,
    figsize: tuple = (12, 4),
) -> plt.Axes:
    """Plot drawdown time series for each strategy."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, (label, ret) in enumerate(returns_dict.items()):
        dd = drawdown_series(ret.dropna())
        ax.fill_between(dd.index, dd.values * 100, 0,
                        alpha=0.25, color=COLORS[i % len(COLORS)])
        ax.plot(dd.index, dd.values * 100,
                label=label, color=COLORS[i % len(COLORS)], lw=1.5)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower left", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    return ax


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    window: int = 90,
    title: str = "Rolling 90-Day Sharpe Ratio",
    ax: plt.Axes | None = None,
    figsize: tuple = (12, 4),
) -> plt.Axes:
    """Plot rolling Sharpe ratio for each strategy."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, (label, ret) in enumerate(returns_dict.items()):
        rs = rolling_sharpe(ret.dropna(), window=window)
        ax.plot(rs.index, rs.values, label=label, color=COLORS[i % len(COLORS)], lw=1.5)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    return ax


# ─── 4-panel tearsheet ────────────────────────────────────────────────────────

def plot_tearsheet(
    bt_result: dict,
    btc_returns: pd.Series,
    label: str = "Strategy",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    4-panel tearsheet for a single backtest result:
      Top-left:     Cumulative returns (net vs gross vs BTC)
      Top-right:    Drawdown
      Bottom-left:  Rolling 90-day Sharpe
      Bottom-right: Daily turnover
    """
    ret_df = bt_result["returns"]
    net    = ret_df["net_return"].dropna()
    gross  = ret_df["gross_return"].dropna()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Strategy Tearsheet: {label}", fontsize=14, fontweight="bold")

    # ── 1. Cumulative returns ──────────────────────────────────────────────
    ax = axes[0, 0]
    for ret, lbl, col in [
        (net,   f"{label} (net)",   COLORS[0]),
        (gross, f"{label} (gross)", COLORS[1]),
        (btc_returns.reindex(net.index).dropna(), "BTC/USDT", "grey"),
    ]:
        w = (1 + ret).cumprod()
        ax.plot(w.index, w, label=lbl, lw=1.8, color=col)
    ax.set_title("Cumulative Return")
    ax.set_ylabel("$1 initial")
    ax.legend(fontsize=8)

    # ── 2. Drawdown ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    dd = drawdown_series(net) * 100
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=COLORS[0])
    ax.plot(dd.index, dd.values, color=COLORS[0], lw=1.5)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")

    # ── 3. Rolling Sharpe ─────────────────────────────────────────────────
    ax = axes[1, 0]
    rs = rolling_sharpe(net, window=90)
    ax.plot(rs.index, rs.values, color=COLORS[0], lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling 90-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe (annualized)")

    # ── 4. Turnover ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    roll_turn = ret_df["turnover"].rolling(30).mean() * 100
    ax.plot(roll_turn.index, roll_turn.values, color=COLORS[2], lw=1.5)
    ax.set_title("30-Day Rolling Average Turnover")
    ax.set_ylabel("One-Way Turnover (%)")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.sca(ax)
        plt.xticks(rotation=25)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─── Cross-asset correlation ──────────────────────────────────────────────────

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Cross-Asset Correlation Matrix",
    figsize: tuple = (12, 10),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Seaborn heatmap of pairwise return correlations."""
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Clean labels: remove '/USDT'
    labels = [c.replace("/USDT", "") for c in corr.columns]
    corr.columns = labels
    corr.index   = labels

    sns.heatmap(
        corr, mask=mask, cmap="RdYlGn", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.6},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax


# ─── Strategy comparison ──────────────────────────────────────────────────────

def plot_strategy_comparison(
    backtest_results: dict[str, dict],
    top_n: int = 10,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Side-by-side bar chart of Sharpe Ratio and Max Drawdown for top-N strategies.
    """
    from src.evaluator import sharpe_ratio, max_drawdown

    names, sharpes, drawdowns = [], [], []
    for name, result in backtest_results.items():
        net = result["returns"]["net_return"].dropna()
        names.append(name)
        sharpes.append(sharpe_ratio(net))
        drawdowns.append(abs(max_drawdown(net)) * 100)

    df = pd.DataFrame({"name": names, "Sharpe": sharpes, "MaxDD%": drawdowns})
    df = df.sort_values("Sharpe", ascending=False).head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors_sr = ["steelblue" if v > 0 else "tomato" for v in df["Sharpe"]]
    ax1.barh(df["name"], df["Sharpe"], color=colors_sr)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("Sharpe Ratio (net, annualized)")
    ax1.set_xlabel("Sharpe")

    ax2.barh(df["name"], df["MaxDD%"], color="tomato", alpha=0.75)
    ax2.set_title("Maximum Drawdown (%)")
    ax2.set_xlabel("Drawdown (%)")

    plt.tight_layout()
    return fig


def plot_train_val_sharpe(
    tv_table: pd.DataFrame,
    top_n: int = 15,
    title: str = "Train vs Validation Sharpe Ratio",
    figsize: tuple = (13, 7),
) -> plt.Figure:
    """
    Side-by-side horizontal bar chart of Train Sharpe vs Validation Sharpe.

    Overfit strategies (Overfit Flag == 'OVERFIT') are highlighted with
    a red border on their validation bar.

    tv_table : output of evaluator.train_val_comparison()
    """
    df = tv_table.head(top_n).copy()
    y  = np.arange(len(df))
    h  = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    train_colors = ["steelblue" if v >= 0 else "lightcoral"
                    for v in df["Train Sharpe"]]
    val_colors   = ["seagreen"  if v >= 0 else "tomato"
                    for v in df["Val Sharpe"]]

    bars_t = ax.barh(y + h / 2, df["Train Sharpe"].values, h,
                     color=train_colors, label="Train (2021-2022)", alpha=0.85)
    bars_v = ax.barh(y - h / 2, df["Val Sharpe"].values, h,
                     color=val_colors,   label="Val (2023-2025)",   alpha=0.85)

    # Mark overfit strategies with a red edge on the val bar
    for bar, flag in zip(bars_v, df.get("Overfit Flag", [])):
        if flag == "OVERFIT":
            bar.set_edgecolor("red")
            bar.set_linewidth(2.0)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Sharpe Ratio (annualized, net of 20 bps cost)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_signal_heatmap(
    signal: pd.DataFrame,
    title: str = "Signal Heatmap (top/bottom = long/short)",
    last_n_days: int = 60,
    figsize: tuple = (14, 6),
) -> plt.Axes:
    """Colour-map of signal strength per coin over the most recent N days."""
    df = signal.tail(last_n_days).copy()
    df.columns = [c.replace("/USDT", "") for c in df.columns]
    df.index   = [str(d.date()) if hasattr(d, "date") else str(d) for d in df.index]

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.T, cmap="RdYlGn", center=0, ax=ax,
                linewidths=0, yticklabels=True, cbar_kws={"label": "Signal z-score"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    step = max(1, last_n_days // 10)
    ax.set_xticks(range(0, last_n_days, step))
    ax.set_xticklabels(df.index[::step], rotation=45, fontsize=8)
    plt.tight_layout()
    return ax

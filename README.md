# crypto-stat-arb

Statistical Arbitrage in Cryptocurrency Markets — Quantitative Finance Research Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yomisys/crypto-stat-arb/blob/main/crypto_stat_arb.ipynb)

---

## What this is

A complete quantitative research system that tests momentum and mean-reversion strategies across the top 25 cryptocurrencies (Binance, no API key required).

| Component | Description |
|-----------|-------------|
| **Data** | 3+ years daily OHLCV via CCXT/Binance, cached as Parquet |
| **Signals** | 14 momentum + 5 reversal signals (cross-sectional, volume-weighted, pairs spread) |
| **Backtest** | Long-short quintile, 20 bps cost model, no look-ahead bias |
| **Evaluation** | Sharpe, alpha t-stat, max drawdown, train/validation split |
| **Combination** | Quarterly walk-forward equal / inv-vol / Sharpe-weighted portfolios |

## How to run

Click the **Open in Colab** badge above, then **Runtime → Run all**.

Data is fetched automatically on first run (~3-5 min). Subsequent runs in the same session are instant (reads from cache).

## Repo structure

```
crypto_stat_arb.ipynb      # Self-contained Colab notebook (run this)
run_research.py            # Standalone headless script (python run_research.py)
create_colab_notebook.py   # Regenerates the notebook from source
config.py                  # Universe, dates, cost model constants
src/
  data_pipeline.py         # CCXT fetch + Parquet cache
  signals/
    momentum.py            # 14 momentum signals
    reversal.py            # 5 reversal signals
  backtester.py            # Long-short quintile engine
  evaluator.py             # Performance metrics + train/val split
  portfolio.py             # Walk-forward combination portfolios
  plotting.py              # Chart helpers
requirements.txt           # pip dependencies
```

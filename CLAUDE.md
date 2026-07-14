# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A quantitative reference report on SK Hynix (000660.KS) stock price: multiple pretrained/statistical/ML
models for price forecasting, multilingual (Korean+English) news sentiment fusion, and market-context
diagnosis (market-wide vs sector vs peer vs idiosyncratic), published as a static dashboard via GitHub
Pages. See `README.md` for the pipeline overview and `ARCHITECTURE.md` for why specific models/methodology
were chosen — in particular, read the "프로세스 분리" section in ARCHITECTURE.md before changing
`main.py`'s subprocess structure; it exists to work around a real crash, not stylistic preference.

## Commands

```
pip install -r requirements.txt
python src/main.py           # runs the full pipeline, regenerates docs/index.html + docs/assets/*.png
```

There is no test suite in this repo (it's a report-generation pipeline, not a library with pure logic
to unit test — each stage does real network I/O or model inference). Verify changes by running
`python src/main.py` end to end and inspecting `docs/index.html` and the generated charts.

## Architecture

`src/main.py` orchestrates two fully separate subprocesses (see "프로세스 분리" in ARCHITECTURE.md for
why they must be separate OS processes, not just separate functions) plus a final in-process report step:

1. **`stage_collect.py`** (own process): `fetch_price.py` (yfinance, SK Hynix), `related_stocks.py`
   (KOSPI/S&P500/NASDAQ market indices, SOX sector index, Samsung/Micron/SanDisk peers — three tiers
   used to diagnose whether a price move is market-wide, sector-wide, or idiosyncratic),
   `index_scan.py` (correlation scan against representative KOSPI200/NASDAQ100/S&P500 constituent
   samples — not the full index, for runtime/rate-limit reasons), `fetch_news.py` (Korean + English
   Google News RSS, 14 queries total, near-duplicate headlines merged), `sentiment.py` (per-language
   finance-tuned classifiers — `snunlp/KR-FinBert-SC` for Korean, `ProsusAI/finbert` for English —
   fused via confidence × query-importance × recency-decay weighting).
2. **`stage_forecast.py`** (own process): `forecast.py` blends `amazon/chronos-bolt-base` (zero-shot
   pretrained time-series foundation model) with a damped-trend exponential smoothing model
   (statsmodels), weighted 70/30 based on backtest results. `technical_model.py` (gradient-boosting
   quantile regression on RSI/MACD/moving-average/volatility features, scikit-learn) is a third model
   shown for comparison but not blended into the ensemble. News sentiment is applied as a multiplicative
   adjustment to the ensemble median, decaying exponentially over the forecast horizon — see
   ARCHITECTURE.md for why this is an overlay rather than a jointly-trained covariate (point-in-time
   historical news isn't practically obtainable from free/no-login sources at scale).
   `backtest.py` validates all of this with walk-forward rolling backtests comparing ensemble vs.
   Chronos-alone vs. a naive (persistence) baseline, plus directional-accuracy (hit rate).
3. **`report.py`** (main process): renders matplotlib charts (Korean labels need
   `font.family = "Malgun Gothic"` — don't remove that rcParam) and a static `docs/index.html` with
   outlook, methodology, forecast/model-comparison tables, sentiment breakdown, market/sector/peer +
   index-correlation comparison, and backtest results (including an actual-vs-predicted-return scatter).

**Why the subprocess split**: loading the two sentiment classifiers (~900MB) and then
Chronos-Bolt-base (~800MB) in the same process caused intermittent segfaults / "paging file too small"
errors in this environment. `ProcessPoolExecutor` alone didn't fix it (the parent process stays alive
and its memory isn't fully reclaimed by the OS). Only having `stage_collect.py` fully exit before
`stage_forecast.py` starts as a brand-new process reliably avoided it. If you see similar crashes when
adding new heavy models, consider whether they need their own stage/process rather than assuming it's a
one-off flake.

**When a newer/better pretrained model becomes available**, prefer swapping to it — this project is
meant to use the best available pretrained/established model, not stay pinned to whatever was chosen
first. Verify it actually loads/runs in the target environment before committing to it, and re-run the
rolling backtest to see whether it actually improves on the current ensemble (don't assume newer = better
without checking `rolling_backtest()`'s output).

**Deployment**: GitHub Pages serves static files only, so there's no build step on GitHub's side —
`docs/index.html` and `docs/assets/*.png` are committed directly (not gitignored) and Pages is
configured to serve `main` branch `/docs`. Regenerating the report means running `python src/main.py`
locally and committing the changed `docs/` files.

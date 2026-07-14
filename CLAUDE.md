# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PoC that predicts SK Hynix (000660.KS) stock price using pretrained models only (no training from scratch), fuses in news sentiment as a simple heuristic adjustment, and publishes the result as a static dashboard via GitHub Pages. See `README.md` for the pipeline overview and `ARCHITECTURE.md` for why specific models were chosen (in particular, a hard per-process memory ceiling encountered during development that ruled out larger models — read that before "upgrading" a model without checking memory feasibility first).

## Commands

```
pip install -r requirements.txt
python src/main.py           # runs the full pipeline, regenerates docs/index.html + docs/assets/chart.png
```

There is no test suite in this repo (it's a PoC/report-generation pipeline, not a library with pure logic to unit test — each stage does real network I/O or model inference).

## Architecture

`src/main.py` runs the stages in sequence, each a separate module with a single responsibility:

1. `fetch_price.py` — `yfinance` history for `000660.KS`.
2. `related_stocks.py` — `yfinance` history for Samsung Electronics (005930.KS), Micron (MU), SanDisk (SNDK); `compute_correlations()` correlates *daily returns* (not raw price) against SK Hynix — raw-price correlation is misleading when both series just trend upward. Reference-only, not fed into the forecast.
3. `fetch_news.py` + `sentiment.py` — Google News RSS (no API key), 4 queries: `"SK하이닉스"`, `"SK하이닉스 목표주가"` (analyst/brokerage price-target coverage), `"반도체 지정학"`, `"반도체 업황"` (macro/industry news) → `monologg/koelectra-small-finetuned-nsmc` pretrained classifier → per-headline sentiment score in [-1, 1].
4. `forecast.py` — `amazon/chronos-bolt-small` (Amazon Chronos-**Bolt**, a pretrained zero-shot time-series foundation model — no training step here; more accurate and faster than the original `chronos-t5` family at the same size, per Amazon's benchmarks). Uses `pipeline.predict_quantiles(..., quantile_levels=[0.1, 0.5, 0.9])`, which returns quantiles directly — don't revert to sampling + `np.quantile` from the old `chronos-t5` code path. The average news sentiment score is then applied as a multiplicative adjustment (`1 + sentiment * SENTIMENT_ADJUSTMENT`) to the median/low/high forecast — this is a hand-picked heuristic, not a jointly-trained model, because Chronos only accepts a univariate price series.
5. `backtest.py` — validates the forecast before it's trusted. `backtest()` hides the most recent 5 trading days and forecasts them from the data before that (single window). `rolling_backtest()` repeats this at 10 historical points (walk-forward, 5-day step) and reports mean/std MAPE — neither ever lets future actuals leak into the context used for that window's forecast.
6. `report.py` — renders matplotlib charts (Korean labels need `font.family = "Malgun Gothic"` — don't remove that rcParam, DejaVu Sans has no Hangul glyphs) and a static `docs/index.html` with an outlook summary, methodology explanation, forecast chart/table, news/sentiment table, related-stock correlation chart/table, and the backtest results (including the rolling-MAPE bar chart).

**When a newer/better pretrained model becomes available** (updated Chronos version, a better Korean financial sentiment model, etc.), prefer swapping to it — this project is explicitly meant to use the best available pretrained model, not stay pinned to whatever was chosen first. Just verify it actually loads/runs in the target environment before committing to it (see the memory-ceiling note below).

**Important environment note**: this was built in a sandbox with a per-process memory ceiling around 500-700MB (confirmed by direct `numpy` allocation tests — see `ARCHITECTURE.md`), which is why the sentiment model is a small ELECTRA (~14M params) instead of a finance-specific BERT (~110M params) that kept failing to load. If you're running in an environment without that constraint, swapping back to a finance-specific model (e.g. `snunlp/KR-FinBert-SC`) in `sentiment.py` would likely improve sentiment accuracy — just verify it loads reliably first.

**Deployment**: GitHub Pages serves static files only, so there's no build step on GitHub's side — `docs/index.html` and `docs/assets/chart.png` are committed directly (not gitignored) and Pages is configured to serve `main` branch `/docs`. Regenerating the report means running `python src/main.py` locally and committing the changed `docs/` files.

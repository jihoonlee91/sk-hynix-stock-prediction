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

`src/main.py` runs four stages in sequence, each a separate module with a single responsibility:

1. `fetch_price.py` — `yfinance` history for `000660.KS`.
2. `fetch_news.py` + `sentiment.py` — Google News RSS (no API key) → `monologg/koelectra-small-finetuned-nsmc` pretrained classifier → per-headline sentiment score in [-1, 1].
3. `forecast.py` — `amazon/chronos-t5-small` (Amazon Chronos, a pretrained zero-shot time-series foundation model — no training step here) forecasts the next `HORIZON` trading days from the last ~90 closes. The average news sentiment score is then applied as a multiplicative adjustment (`1 + sentiment * SENTIMENT_ADJUSTMENT`) to the median/low/high forecast — this is a hand-picked heuristic, not a jointly-trained model, because Chronos only accepts a univariate price series.
4. `report.py` — renders a matplotlib chart (Korean labels need `font.family = "Malgun Gothic"` — don't remove that rcParam, DejaVu Sans has no Hangul glyphs) and a static `docs/index.html` with the chart, forecast table, and news/sentiment table.

**Important environment note**: this was built in a sandbox with a per-process memory ceiling around 500-700MB (confirmed by direct `numpy` allocation tests — see `ARCHITECTURE.md`), which is why the sentiment model is a small ELECTRA (~14M params) instead of a finance-specific BERT (~110M params) that kept failing to load. If you're running in an environment without that constraint, swapping back to a finance-specific model (e.g. `snunlp/KR-FinBert-SC`) in `sentiment.py` would likely improve sentiment accuracy — just verify it loads reliably first.

**Deployment**: GitHub Pages serves static files only, so there's no build step on GitHub's side — `docs/index.html` and `docs/assets/chart.png` are committed directly (not gitignored) and Pages is configured to serve `main` branch `/docs`. Regenerating the report means running `python src/main.py` locally and committing the changed `docs/` files.

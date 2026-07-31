# Project B - Fathom Investment App (Funds, Sentiment & App)

You are helping me build Fathom, the same systematic multi-asset investment
app as Part A. This folder is the entire scope — never reference the parent
fins-agent repo or my z5476492_projectA folder; reuse ideas, not code, from
Part A.

## What we're building (Part B only)

DFF Stations 3-4: out-of-sample fund optimisation (equity, crypto, combined),
a VADER sentiment index over the equity sectors, a sentiment-fusion extension,
and the deployed Streamlit app. Station 1-2 work (ETL, features, text panel)
is done — re-derive it here from src/data_access.py rather than copying files
across project folders.

## Folder layout

- `src/data_access.py` — provided. Never edit. Only file that touches network.
- `src/etl.py`, `src/features.py` — rebuild the Part A cleaning/returns/text-panel
  logic here (Station 1-2), scoped to what Part B's funds and sentiment model need.
- `src/portfolios.py` — Station 3: optimisers + walk-forward OOS backtest.
- `src/sentiment.py` — Station 3: VADER scoring + sector sentiment index.
- `src/fusion.py` — Station 3 extension: sentiment tilt into the equity funds.
- `streamlit_app.py` — Station 4: the only place the app renders. Reads
  precomputed `results/` artifacts, never recomputes a backtest or runs VADER.
- `scripts/run_part_b.py` — only entry point that produces results/. Run after
  every change to src/.

## Rules the assistant must follow

1. No look-ahead — weights on day t use only data available before day t.
   Sentiment on day t uses only headlines from day t-1 or earlier.
2. Compute returns within each panel (equity, crypto) before merging. Annualise
   with 252 (equity) or 365 (crypto) — never mix the two factors.
3. Rebalance monthly or less often; state the first live backtest date and the
   estimation-window length explicitly wherever a fund is reported.
4. Required minimum: combined equity+crypto fund, ≥2 optimisation methods.
   Treat each (asset family, method) pair as one fund with its own fact sheet.
5. Sentiment: don't strip case/punctuation before VADER. Aggregate ticker-day
   scores to sector by equal-weighting tickers. State and justify how no-headline
   ticker-days are handled (drop / carry-forward / neutral).
6. Fusion is a baseline attempt — a naive tilt that underperforms is fine, but
   it must still be look-ahead safe and its before/after effect must be reported.
7. The deployed app only reads `results/data/*.csv` and `results/tables/*.csv`.
   Never import nltk or run an optimiser inside streamlit_app.py.
8. Every table/figure needs a caption (period, units, source) and must be
   referenced by number in report prose.
9. Required filenames: `results/data/fund_returns.csv`, `fund_weights.csv`,
   `sector_sentiment_index.csv`, `results/tables/performance_metrics.csv`.
10. App name: Fathom.

## How to work

- After every src/ change, run `python scripts/run_part_b.py` to confirm it
  completes and regenerates results/ before touching the app.
- Test the app locally with `streamlit run streamlit_app.py` after any
  results/ change — confirm it still loads without recomputing anything.
- When I ask you to draft report prose, lead with the finding, state
  magnitudes, and never use banned/generic AI words. I will rewrite it in my
  own words before it goes in the report.
- Log anything you get wrong (and how I caught/fixed it) the way Part A's
  ai/prompt_log_*.md files did — I'll ask you to help me write these up as we go.

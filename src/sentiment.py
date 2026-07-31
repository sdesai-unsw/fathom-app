"""Station 3 - the sentiment model: score headlines, aggregate to a daily
per-ticker score, then to an equal-weight daily sector index.

No-headline ticker-days are treated as neutral (score 0.0), not dropped or
carried forward. VADER already scores a genuinely neutral headline as 0.0, so
a no-news day gets the same "no signal" value instead of assuming persistence
(carry-forward invents information that was not published that day) or
leaving the sector average undefined on thin-news days (Materials, Utilities,
Real Estate - see context/DATA_GUIDE.md). The trade-off: about half of scored
headlines are false neutrals under plain VADER, so neutral-fill likely mutes
the index further on those days - a stated limitation, not hidden.

This index is deliberately NOT lagged - it is a standalone daily analytic
(results/data/sector_sentiment_index.csv) shown as-is. Lagging only matters
once the signal feeds a trading decision, so that step lives in lag_signal
below and is applied by src/fusion.py, not here.
"""
from __future__ import annotations

import pandas as pd


def _analyzer():
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """VADER compound score for every headline in `panel` (the output of
    features.assemble_headline_panel). Titles are scored raw - unchanged
    case and punctuation - because VADER's lexicon and negation heuristics
    rely on both.
    """
    analyzer = _analyzer()
    compound = panel["title"].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"])
    out = panel[["ticker", "sector", "trading_date"]].copy()
    out["compound"] = compound
    return out


def ticker_day_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean VADER compound score per (ticker, trading_date). Multiple
    headlines on the same ticker-day are averaged, not summed, so a busy
    news day does not mechanically dominate a quiet one."""
    return (
        scored.groupby(["ticker", "sector", "trading_date"], as_index=False)["compound"]
        .mean()
        .rename(columns={"compound": "sentiment"})
    )


def sector_ticker_map(equities: pd.DataFrame) -> dict[str, list[str]]:
    """Sector -> ticker list, read from the clean equity panel rather than
    hardcoded, so it always matches the data actually loaded."""
    return equities.groupby("sector")["ticker"].unique().apply(list).to_dict()


def sector_sentiment_index(
    ticker_day: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    tickers_by_sector: dict[str, list[str]],
) -> pd.DataFrame:
    """Equal-weight daily sentiment index per sector.

    Every (ticker, trading_date) on the full trading calendar gets a value:
    the ticker's mean VADER score for that day if it had headlines,
    otherwise 0.0 (see module docstring). The sector index is the equal-weight
    mean across that sector's tickers for every trading day, so the series has
    no gaps even for thinner-news sectors.
    """
    calendar = pd.DatetimeIndex(sorted(trading_calendar)).normalize()
    wide = ticker_day.pivot(index="trading_date", columns="ticker", values="sentiment")
    wide = wide.reindex(calendar).fillna(0.0)

    sector_cols = {}
    for sector, tickers in tickers_by_sector.items():
        present = [t for t in tickers if t in wide.columns]
        sector_cols[sector] = wide[present].mean(axis=1)
    index = pd.DataFrame(sector_cols).sort_index()
    index.index.name = "date"
    return index


def lag_signal(signal: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Shift a daily signal forward by `lag_days` trading days so a decision
    on day t uses only information available at day t - lag_days or earlier.
    Apply this to sector_sentiment_index immediately before it feeds a fund
    (src/fusion.py) - the saved index itself stays unlagged (see module
    docstring).
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1 to stay look-ahead safe")
    return signal.shift(lag_days)

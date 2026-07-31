"""Station 1 - ETL: load the hosted data and clean it for Part B.

Part A already carries the full integrity-check exhibits (missing-date audit,
outlier screen, data-integrity summary) - that reporting lives there, not
here. This module rebuilds only what Part B's funds and sentiment model need:
deduplicated, calendar-capped panels. Every loader returns (clean_df, report)
so the caller can log what was found without re-deriving it.
"""
from __future__ import annotations

import pandas as pd

from src import data_access

# Crypto data carries 10 stray rows dated 2024-01-01 (see context/DATA_GUIDE.md);
# cap the sample at the data's stated coverage.
CRYPTO_END_DATE = "2023-12-31"


def _duplicate_check(df: pd.DataFrame, subset: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows on `subset`, keeping the first occurrence."""
    dup_mask = df.duplicated(subset=subset, keep="first")
    n_dupes = int(dup_mask.sum())
    return df.loc[~dup_mask].copy(), n_dupes


def _missing_date_summary(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> int:
    """Total (ticker, date) gaps versus `calendar`, summed across tickers -
    a cheap sanity check before the return panels feed the optimiser."""
    have = df.groupby("ticker")["date"].apply(set)
    return int(sum(len(calendar.difference(pd.DatetimeIndex(sorted(d)))) for d in have))


def load_clean_equities() -> tuple[pd.DataFrame, dict]:
    """Load equity prices, deduplicated by (ticker, date).

    The equity trading calendar is the union of all observed equity dates in
    the data (about 252 days/year).
    """
    raw = data_access.load_equity_prices()
    df, n_dupes = _duplicate_check(raw, subset=["ticker", "date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    report = {
        "n_raw_rows": len(raw),
        "n_duplicates_removed": n_dupes,
        "n_clean_rows": len(df),
        "calendar": calendar,
        "n_missing_ticker_dates": _missing_date_summary(df, calendar),
    }
    return df, report


def load_clean_crypto() -> tuple[pd.DataFrame, dict]:
    """Load crypto prices, capped at CRYPTO_END_DATE and deduplicated by
    (ticker, date). Crypto trades 7 days a week, so its calendar is the full
    daily date range, not the equity calendar - combine_equity_crypto_returns
    in features.py handles the calendar alignment.
    """
    raw = data_access.load_crypto_prices()
    raw = raw.loc[raw["date"] <= pd.Timestamp(CRYPTO_END_DATE)].copy()
    df, n_dupes = _duplicate_check(raw, subset=["ticker", "date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    report = {
        "n_raw_rows": len(raw),
        "n_duplicates_removed": n_dupes,
        "n_clean_rows": len(df),
        "calendar": calendar,
        "n_missing_ticker_dates": _missing_date_summary(df, calendar),
    }
    return df, report


def load_clean_news() -> tuple[pd.DataFrame, dict]:
    """Load headlines and dedupe on (ticker, date, title) - the news panel has
    many legitimate rows per ticker-date, so ticker-date alone is not a valid
    duplicate key. Title text is kept raw (case, punctuation) for VADER in
    src/sentiment.py.
    """
    raw = data_access.load_news_headlines()
    df, n_dupes = _duplicate_check(raw, subset=["ticker", "date", "title"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    report = {
        "n_raw_rows": len(raw),
        "n_duplicates_removed": n_dupes,
        "n_clean_rows": len(df),
    }
    return df, report

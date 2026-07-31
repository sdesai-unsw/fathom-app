"""Station 2 - features: return features and headline text assembly.

Assembly only - scoring the headlines into sentiment (and lagging the signal)
is the Station 3 model in src/sentiment.py, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker, long format (ticker, date, ret).

    The one feature the portfolio optimiser needs. Computed within each
    ticker's own price series before any cross-panel merge, so equity and
    crypto returns are never built from mismatched calendars.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")[price_col].pct_change()
    return df.loc[df["ret"].notna(), ["ticker", "date", "ret"]].reset_index(drop=True)


def returns_wide(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long returns (ticker, date, ret) to wide (date index x ticker columns)."""
    return returns_long.pivot(index="date", columns="ticker", values="ret").sort_index()


def combine_equity_crypto_returns(equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading calendar.

    Returns are computed within each panel first (by the caller, via
    daily_returns), then crypto's wide return matrix is reindexed onto the
    equity date index. This intentionally drops weekend-only crypto moves,
    since a fund trading on equity days could not act on them. Do not merge
    price levels across calendars and difference afterwards - that creates
    spurious returns.
    """
    eq_wide = returns_wide(equity_returns)
    cr_wide = returns_wide(crypto_returns)
    cr_on_eq_calendar = cr_wide.reindex(eq_wide.index)
    return eq_wide.join(cr_on_eq_calendar, how="left")


def assemble_headline_panel(headlines: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Assemble headlines into a daily panel per ticker and sector, date-aligned
    to the equity trading calendar.

    Maps every headline to its equity trading day: the same day if the
    headline date is a trading day, otherwise the next trading day (a
    Saturday or Monday headline both land on Monday). The headline `date` is
    timezone-aware UTC while the price calendar is timezone-naive, so the
    timestamp is normalised to a tz-naive calendar day before alignment.
    Raw headline text (title) is kept unmodified - VADER (src/sentiment.py)
    relies on stopwords, casing, and punctuation that a cleaning step would
    strip.
    """
    df = headlines.copy()
    headline_date = pd.to_datetime(df["date"])
    if headline_date.dt.tz is not None:
        headline_date = headline_date.dt.tz_convert("UTC").dt.tz_localize(None)
    df["headline_date"] = headline_date.dt.normalize()

    calendar = pd.DatetimeIndex(sorted(trading_calendar)).normalize()
    cal_values = calendar.values
    positions = np.searchsorted(cal_values, df["headline_date"].values, side="left")
    positions = np.clip(positions, 0, len(cal_values) - 1)
    df["trading_date"] = calendar[positions]

    cols = ["ticker", "sector", "trading_date", "headline_date", "title", "url", "publisher"]
    return df[cols].sort_values(["ticker", "trading_date"]).reset_index(drop=True)

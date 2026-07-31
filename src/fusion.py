"""Station 3 extension - fuse the sentiment signal into an equity fund.

A multiplicative sentiment tilt: at each rebalance date, a ticker's base
weight is scaled by its own lagged sentiment score, then renormalised to stay
long-only and sum to 1. Applies to equity funds only - crypto has no news
coverage in this data (context/DATA_GUIDE.md). This is a baseline attempt: a
naive tilt that underperforms the base fund is an acceptable, reportable
result, not a bug to hide.
"""
from __future__ import annotations

import pandas as pd

from src import portfolios


def apply_sentiment(
    weights: pd.DataFrame,
    ticker_sentiment: pd.DataFrame,
    strength: float = 0.5,
) -> pd.DataFrame:
    """Tilt a fund's rebalance-date x ticker weight matrix by per-ticker
    sentiment.

    `ticker_sentiment` must already be lagged by the caller (see
    sentiment.lag_signal) so a rebalance on date t only uses sentiment known
    before t - this function does not lag anything itself, to keep the
    look-ahead-safety decision visible at the call site rather than buried in
    this helper.

    new_weight_i = base_weight_i * max(1 + strength * sentiment_i, 0), then
    renormalised to sum to 1 (long-only preserved). A ticker with no
    sentiment value on record at that date is left untilted (tilt = 1) rather
    than dropped, so the tilt never changes which names a fund holds, only
    their relative sizes.
    """
    aligned = ticker_sentiment.reindex(index=weights.index, columns=weights.columns)
    tilt = (1.0 + strength * aligned.fillna(0.0)).clip(lower=0.0)
    tilted = weights * tilt
    row_sums = tilted.sum(axis=1)
    renormalised = tilted.div(row_sums, axis=0)
    return renormalised.where(row_sums > 0, weights)


def fuse_fund(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    ticker_sentiment: pd.DataFrame,
    strength: float = 0.5,
) -> dict:
    """Tilt `weights` by lagged sentiment and realise the resulting daily
    fund returns, reusing the same hold-until-next-rebalance mechanic as
    portfolios.oos_backtest (via portfolios.realize_returns)."""
    tilted_weights = apply_sentiment(weights, ticker_sentiment, strength=strength)
    tilted_returns = portfolios.realize_returns(returns, tilted_weights)
    return {"weights": tilted_weights, "returns": tilted_returns}


def fusion_effect(base_metrics: dict, tilted_metrics: dict) -> dict:
    """Before-vs-after deltas (tilted minus base) for the fact-sheet metrics
    the brief requires: annualised return, volatility, Sharpe, max drawdown."""
    keys = ["annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown"]
    return {f"delta_{k}": tilted_metrics[k] - base_metrics[k] for k in keys}

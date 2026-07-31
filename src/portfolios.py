"""Station 3 - funds: optimal portfolios + walk-forward out-of-sample backtest.

Four long-only optimisation methods share one backtest engine:
equal-weight, minimum-variance, maximum-Sharpe (mean-variance tangency), and
risk parity. Apply the engine to the equity-only, crypto-only, and combined
return panels from features.py to get up to twelve (family, method) funds.

Risk-free rate is assumed zero throughout (stated choice, not fitted) - the
brief permits this for the Sharpe ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Daily-return covariances are ~1e-4 in magnitude, small enough that SLSQP can
# mistake a flat gradient for convergence and stall at the starting weights
# (the "solver scaling" trap in context/DATA_GUIDE.md). Optimising on returns
# scaled by RETURN_SCALE keeps the objective well-conditioned; long-only
# mean-variance and risk-parity objectives are scale-invariant in the optimal
# weights (rf = 0), so this does not change the solution, only the numerics.
RETURN_SCALE = 100.0


def _clean(hist: pd.DataFrame) -> pd.DataFrame:
    """Drop any asset with missing history inside the estimation window -
    every asset in this project's data has a complete history, but this keeps
    the optimiser from failing silently if that ever changes."""
    return hist.dropna(axis=1, how="any")


def _equal_weight(hist: pd.DataFrame) -> pd.Series:
    cols = _clean(hist).columns
    return pd.Series(1.0 / len(cols), index=cols)


def _solve(objective, n: int, x0: np.ndarray | None = None) -> np.ndarray:
    x0 = x0 if x0 is not None else np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        return x0
    return np.clip(result.x, 0.0, None) / np.clip(result.x, 0.0, None).sum()


def _min_variance(hist: pd.DataFrame) -> pd.Series:
    cols = _clean(hist).columns
    cov = (hist[cols] * RETURN_SCALE).cov().values
    n = len(cols)

    def objective(w):
        return w @ cov @ w

    w = _solve(objective, n)
    return pd.Series(w, index=cols)


def _max_sharpe(hist: pd.DataFrame) -> pd.Series:
    cols = _clean(hist).columns
    scaled = hist[cols] * RETURN_SCALE
    mu = scaled.mean().values
    cov = scaled.cov().values
    n = len(cols)

    def objective(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(w @ mu) / vol

    w = _solve(objective, n)
    return pd.Series(w, index=cols)


def _risk_parity(hist: pd.DataFrame) -> pd.Series:
    cols = _clean(hist).columns
    cov = (hist[cols] * RETURN_SCALE).cov().values
    n = len(cols)

    def objective(w):
        port_var = w @ cov @ w
        marginal = cov @ w
        contrib = w * marginal
        target = port_var / n
        return float(np.sum((contrib - target) ** 2))

    w = _solve(objective, n)
    return pd.Series(w, index=cols)


METHODS = {
    "equal_weight": _equal_weight,
    "min_variance": _min_variance,
    "max_sharpe": _max_sharpe,
    "risk_parity": _risk_parity,
}


def _rebalance_dates(index: pd.DatetimeIndex) -> set:
    """Last trading day present in `index` for each calendar month - a
    monthly rebalance, satisfying the brief's "monthly or less often" rule."""
    s = pd.Series(index, index=index)
    last_per_month = s.groupby([index.year, index.month]).max()
    return set(last_per_month.values)


def realize_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Apply a rebalance-date x ticker weight matrix to a returns panel: hold
    each row's weights from that date until the next row's date (inclusive of
    the day the weights change - the rebalance itself trades at that day's
    close), and compute the resulting daily fund return.

    This is the "hold weights forward" mechanic oos_backtest uses for its own
    weights; src/fusion.py reuses it to realise the returns of a
    sentiment-tilted weight matrix without re-deriving this logic.
    """
    if weights.empty:
        return pd.Series(dtype=float, name="ret")
    rebal_dates = set(weights.index)
    first_date = weights.index.min()
    daily_returns: dict[pd.Timestamp, float] = {}
    current_weights: pd.Series | None = None

    for date in returns.index:
        if date < first_date:
            continue
        if date in rebal_dates:
            current_weights = weights.loc[date]
        if current_weights is not None:
            r = returns.loc[date].reindex(current_weights.index).fillna(0.0)
            daily_returns[date] = float((current_weights * r).sum())

    return pd.Series(daily_returns, name="ret").sort_index()


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
    window: int = 252,
    min_window: int | None = None,
) -> dict:
    """Walk-forward out-of-sample backtest for one (family, method) fund.

    On each rebalance date (the last trading day of a calendar month), form
    weights from the trailing `window` observations strictly before that date
    (no look-ahead), then hold those weights until the next rebalance date.
    The fund only "goes live" once `min_window` (default: `window`)
    observations of history exist - dates before that are excluded, not
    zero-filled, so the reported series has no fabricated returns.

    Returns a dict: daily fund returns, weights at each rebalance date, the
    method name, and the first live date.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', choose from {list(METHODS)}")
    solve = METHODS[method]
    min_window = min_window or window

    rebal_dates = _rebalance_dates(returns.index)
    weights_over_time: dict[pd.Timestamp, pd.Series] = {}

    for date in returns.index:
        if date in rebal_dates:
            hist = returns.loc[returns.index < date].tail(window)
            if len(hist) >= min_window:
                weights_over_time[date] = solve(hist)

    weights_df = pd.DataFrame(weights_over_time).T.sort_index()
    weights_df.index.name = "date"
    fund_returns = realize_returns(returns, weights_df)
    return {
        "method": method,
        "returns": fund_returns,
        "weights": weights_df,
        "first_live_date": fund_returns.index.min() if len(fund_returns) else None,
    }


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return (CAGR), annualised volatility, Sharpe (rf = 0), and
    maximum drawdown from a daily fund-return series."""
    r = daily_returns.dropna()
    n = len(r)
    growth = (1.0 + r).cumprod()
    cumulative_return = float(growth.iloc[-1] - 1.0) if n else float("nan")
    years = n / periods_per_year if n else float("nan")
    annualised_return = float((1.0 + cumulative_return) ** (1.0 / years) - 1.0) if n and years > 0 else float("nan")
    annualised_vol = float(r.std() * np.sqrt(periods_per_year)) if n else float("nan")
    sharpe = annualised_return / annualised_vol if annualised_vol not in (0.0, float("nan")) else float("nan")
    drawdown = growth / growth.cummax() - 1.0 if n else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if n else float("nan")
    return {
        "n_obs": n,
        "cumulative_return": cumulative_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def growth_of_one_dollar(daily_returns: pd.Series) -> pd.Series:
    """Cumulative growth of a $1 investment - the fact-sheet headline series."""
    return (1.0 + daily_returns.dropna()).cumprod()


def current_holdings(weights: pd.DataFrame) -> pd.Series:
    """Target weights from the most recent rebalance - what the fact sheet
    reports as the fund's current holdings.

    Renamed to "weight" rather than left as the rebalance-date Timestamp that
    `.iloc[-1]` carries as its `.name` - passing a Series named after a
    Timestamp straight into an Altair-backed chart (e.g. st.bar_chart) makes
    Altair's shorthand parser choke on the colons in its string form.
    """
    if weights.empty:
        return pd.Series(dtype=float, name="weight")
    return weights.iloc[-1].sort_values(ascending=False).rename("weight")

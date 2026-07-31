"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import etl, features, fusion, portfolios, sentiment, viz  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SOURCE = "Course-provided equity, crypto, and news headline data (context/DATA_GUIDE.md)"

# (fund family, key into the returns dict below, periods/year, estimation window)
FUND_SPECS = [
    ("Equity", "equity", 252, 252),
    ("Crypto", "crypto", 365, 365),
    ("Combined", "combined", 252, 252),
]
METHODS = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]


def build_funds(returns_by_family: dict[str, pd.DataFrame]):
    """Run the OOS backtest for every (family, method) fund. Returns the
    per-fund daily returns, per-fund weights-over-time, and a metrics table."""
    fund_returns, fund_weights, metrics_rows = {}, {}, []
    for family, key, ppy, window in FUND_SPECS:
        wide = returns_by_family[key]
        for method in METHODS:
            result = portfolios.oos_backtest(wide, method=method, window=window)
            fund_name = f"{family} {method.replace('_', ' ').title()}"
            fund_returns[fund_name] = result["returns"]
            fund_weights[fund_name] = result["weights"]

            m = portfolios.performance_metrics(result["returns"], periods_per_year=ppy)
            m.update({
                "fund": fund_name,
                "family": family,
                "method": method,
                "first_live_date": result["first_live_date"],
            })
            metrics_rows.append(m)
    return fund_returns, fund_weights, pd.DataFrame(metrics_rows)


def save_fund_outputs(fund_returns: dict, fund_weights: dict, metrics: pd.DataFrame):
    returns_wide = pd.DataFrame(fund_returns).sort_index()
    returns_wide.index.name = "date"
    returns_wide.to_csv(RESULTS / "data" / "fund_returns.csv")

    weights_long = []
    for fund_name, w in fund_weights.items():
        melted = w.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
        melted.insert(1, "fund", fund_name)
        weights_long.append(melted)
    pd.concat(weights_long, ignore_index=True).to_csv(
        RESULTS / "data" / "fund_weights.csv", index=False
    )

    cols = ["fund", "family", "method", "first_live_date", "n_obs", "cumulative_return",
            "annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown"]
    metrics[cols].to_csv(RESULTS / "tables" / "performance_metrics.csv", index=False)


def save_figures(fund_returns: dict, fund_weights: dict, metrics: pd.DataFrame,
                  sector_index: pd.DataFrame, fusion_table: pd.DataFrame):
    """Build the required Part B exhibits (results/figures/*.png + .caption.txt),
    using the Fathom design system in src/viz.py."""
    base = metrics[~metrics["fund"].str.contains("Sentiment")]
    all_dates = pd.concat([s.dropna() for s in fund_returns.values()]).index
    sample = f"{all_dates.min().date()} to {all_dates.max().date()}, out-of-sample"

    # 1. Growth of $1 across the Combined fund's methods (the required minimum family).
    # Column names drop the shared "Combined " prefix - every series in this
    # figure is a Combined fund, so the prefix is redundant and only leaves
    # less room for the end-of-line dollar labels.
    combined_funds = base.loc[base["family"] == "Combined", "fund"].tolist()
    best_combined = base.loc[base["family"] == "Combined"].sort_values("sharpe_ratio").iloc[-1]["fund"]
    short = lambda f: f.replace("Combined ", "")
    growth_wide = pd.DataFrame({short(f): fund_returns[f] for f in combined_funds})
    fig, _ = viz.cumulative_return_lines(
        growth_wide,
        title=f"{best_combined} has the best risk-adjusted (Sharpe) return among the Combined "
              "fund's four methods, despite Max Sharpe compounding to more dollars.",
        emphasis_cols=[short(best_combined)], emphasis_label=short(best_combined),
        neutral_label="Other Combined methods", source=SOURCE, sample=sample,
    )
    viz.save(fig, FIGURES, "growth_of_one_dollar_combined", viz.caption(
        "Growth of $1 across the Combined fund's optimisation methods",
        f"{best_combined} has the highest out-of-sample Sharpe ratio among the four Combined methods",
        sample, "Dollars, from a $1 investment at each fund's first live rebalance", SOURCE,
    ))

    # 2 & 3. Drawdown and weights-over-time for the single best-performing base fund.
    best_overall = base.sort_values("sharpe_ratio").iloc[-1]["fund"]
    best_overall_dd = abs(portfolios.performance_metrics(fund_returns[best_overall])["max_drawdown"])
    fig, _ = viz.drawdown_area(
        fund_returns[best_overall],
        title=f"{best_overall} lost {best_overall_dd:.0%} from peak to trough at its worst point.",
        source=SOURCE, sample=sample,
    )
    viz.save(fig, FIGURES, "drawdown_best_fund", viz.caption(
        f"Drawdown from peak, {best_overall}",
        "Percentage decline in growth of $1 from its running maximum",
        sample, "Percent", SOURCE,
    ))

    fig, _ = viz.weights_concentration_area(
        fund_weights[best_overall],
        title=f"{best_overall}'s weight concentration in its top holdings over time.",
        top_n=5, source=SOURCE, sample=sample,
    )
    viz.save(fig, FIGURES, "weights_over_time_best_fund", viz.caption(
        f"Portfolio weight over time, {best_overall}",
        "Top 5 holdings by average weight (ranked teal), all other holdings summed (navy)",
        sample, "Portfolio weight (0-1, sums to 1 each rebalance)", SOURCE,
    ))

    # 4. Sharpe barplot across all base funds and methods.
    fig, _ = viz.sharpe_barplot(
        base, title="Combined Risk Parity and Equity Equal Weight lead on risk-adjusted return; "
                     "Crypto Max Sharpe is the only fund with a negative Sharpe ratio.",
        source=SOURCE, sample=sample,
    )
    viz.save(fig, FIGURES, "sharpe_barplot_all_funds", viz.caption(
        "Sharpe ratio by fund and method",
        "Risk-free rate assumed 0; the highest-Sharpe fund is highlighted",
        sample, "Sharpe ratio (dimensionless)", SOURCE,
    ))

    # 5. Sector sentiment index over time. Uses the full equity calendar, not
    # the funds' out-of-sample window - a separate sample string, not `sample`
    # (which describes the funds' live dates and does not apply here).
    most_positive = sector_index.mean().idxmax()
    thinnest_coverage = (sector_index == 0).mean().idxmax()
    emphasis_sectors = [most_positive, thinnest_coverage]
    sentiment_sample = f"{sector_index.index.min().date()} to {sector_index.index.max().date()}"
    fig, _ = viz.level_lines(
        sector_index,
        title=f"{most_positive} news reads most positive on average; {thinnest_coverage}, "
              "the thinnest-covered sector, stays closest to neutral.",
        ylabel="Sentiment index (VADER compound)",
        emphasis_cols=emphasis_sectors, smooth_window=21,
        emphasis_label="Most positive / thinnest-coverage sector", neutral_label="Other sectors",
        source=SOURCE, sample=sentiment_sample,
    )
    viz.save(fig, FIGURES, "sector_sentiment_index", viz.caption(
        "Equal-weight equity sector sentiment index",
        "21-trading-day rolling mean of the daily index in results/data/sector_sentiment_index.csv; "
        "no-headline ticker-days are scored neutral (0)",
        sentiment_sample, "VADER compound score, -1 (most negative) to +1 (most positive)", SOURCE,
    ))

    # 6. Fusion before-vs-after.
    fig, _ = viz.fusion_before_after_bar(
        fusion_table, title="The sentiment tilt raises Sharpe for three of four equity methods; "
                             "Min Variance is the exception.",
        source=SOURCE, sample=sample,
    )
    viz.save(fig, FIGURES, "fusion_before_after", viz.caption(
        "Sentiment fusion: Sharpe ratio before vs after the tilt",
        "Each equity fund's weights tilted by its holdings' own sentiment, lagged >=1 trading day",
        sample, "Sharpe ratio (dimensionless)", SOURCE,
    ))


def main():
    equities, eq_report = etl.load_clean_equities()
    crypto, cr_report = etl.load_clean_crypto()
    news, news_report = etl.load_clean_news()
    print(f"equities: {equities.shape}, {eq_report['n_duplicates_removed']} dupes removed, "
          f"{eq_report['n_missing_ticker_dates']} missing ticker-dates")
    print(f"crypto:   {crypto.shape}, {cr_report['n_duplicates_removed']} dupes removed, "
          f"{cr_report['n_missing_ticker_dates']} missing ticker-dates")
    print(f"news:     {news.shape}, {news_report['n_duplicates_removed']} dupes removed")

    equity_returns = features.daily_returns(equities)
    crypto_returns = features.daily_returns(crypto)
    combined_returns = features.combine_equity_crypto_returns(equity_returns, crypto_returns)
    print(f"combined returns panel: {combined_returns.shape} "
          f"({combined_returns.index.min().date()} to {combined_returns.index.max().date()})")

    headline_panel = features.assemble_headline_panel(news, eq_report["calendar"])
    print(f"headline panel: {headline_panel.shape}, "
          f"{headline_panel['ticker'].nunique()} tickers")

    returns_by_family = {
        "equity": features.returns_wide(equity_returns),
        "crypto": features.returns_wide(crypto_returns),
        "combined": combined_returns,
    }
    fund_returns, fund_weights, metrics = build_funds(returns_by_family)
    print("\nperformance metrics:")
    print(metrics[["fund", "first_live_date", "annualised_return",
                    "annualised_volatility", "sharpe_ratio", "max_drawdown"]]
          .to_string(index=False))

    # Sanity check (context/DATA_GUIDE.md "solver scaling" trap): confirm the
    # optimisers actually produced different weights, not four copies of the
    # starting equal-weight guess.
    last_date = fund_weights["Combined Min Variance"].index.max()
    combined_last = {
        m: fund_weights[f"Combined {m.replace('_', ' ').title()}"].loc[last_date]
        for m in METHODS
    }
    max_pairwise_diff = max(
        (combined_last[a] - combined_last[b]).abs().max()
        for i, a in enumerate(METHODS) for b in METHODS[i + 1:]
    )
    print(f"\nmax pairwise weight difference across Combined methods on {last_date.date()}: "
          f"{max_pairwise_diff:.4f} (should be well above 0 - methods must diverge)")

    scored = sentiment.score_headlines(headline_panel)
    ticker_day = sentiment.ticker_day_sentiment(scored)
    sector_map = sentiment.sector_ticker_map(equities)
    sector_index = sentiment.sector_sentiment_index(ticker_day, eq_report["calendar"], sector_map)
    sector_index.to_csv(RESULTS / "data" / "sector_sentiment_index.csv")

    n_ticker_days = len(eq_report["calendar"]) * equities["ticker"].nunique()
    n_covered = len(ticker_day)
    print(f"\nsentiment: {len(scored):,} headlines scored, "
          f"{n_covered:,}/{n_ticker_days:,} ticker-days had a headline "
          f"({n_covered / n_ticker_days:.1%} coverage, rest filled neutral)")
    print(f"saved: {RESULTS/'data'/'sector_sentiment_index.csv'}")

    # Fusion extension: tilt each equity fund's weights by its tickers' own
    # lagged sentiment (not the sector index - the fund holds individual
    # names). Lag by 1 trading day here, at the point of use, so a rebalance
    # on date t only sees sentiment known before t.
    ticker_sentiment_wide = ticker_day.pivot(
        index="trading_date", columns="ticker", values="sentiment"
    ).reindex(eq_report["calendar"]).fillna(0.0)
    lagged_ticker_sentiment = sentiment.lag_signal(ticker_sentiment_wide, lag_days=1)

    fusion_rows = []
    for method in METHODS:
        base_name = f"Equity {method.replace('_', ' ').title()}"
        fused_name = f"{base_name} + Sentiment"
        fused = fusion.fuse_fund(
            returns_by_family["equity"], fund_weights[base_name],
            lagged_ticker_sentiment, strength=0.5,
        )
        fund_returns[fused_name] = fused["returns"]
        fund_weights[fused_name] = fused["weights"]

        base_metrics = portfolios.performance_metrics(fund_returns[base_name], periods_per_year=252)
        fused_metrics = portfolios.performance_metrics(fused["returns"], periods_per_year=252)
        effect = fusion.fusion_effect(base_metrics, fused_metrics)
        fusion_rows.append({"base_fund": base_name, **effect,
                             "base_sharpe": base_metrics["sharpe_ratio"],
                             "sentiment_sharpe": fused_metrics["sharpe_ratio"]})

        metrics.loc[len(metrics)] = {
            "fund": fused_name, "family": "Equity", "method": f"{method}+sentiment",
            "first_live_date": fused["returns"].index.min() if len(fused["returns"]) else None,
            **fused_metrics,
        }

    fusion_table = pd.DataFrame(fusion_rows)
    fusion_table.to_csv(RESULTS / "tables" / "fusion_comparison.csv", index=False)
    print("\nfusion before-vs-after (tilted minus base):")
    print(fusion_table.round(4).to_string(index=False))
    print(f"saved: {RESULTS/'tables'/'fusion_comparison.csv'}")

    save_fund_outputs(fund_returns, fund_weights, metrics)
    print(f"\nsaved: {RESULTS/'data'/'fund_returns.csv'}, {RESULTS/'data'/'fund_weights.csv'}, "
          f"{RESULTS/'tables'/'performance_metrics.csv'}")

    save_figures(fund_returns, fund_weights, metrics, sector_index, fusion_table)
    print(f"saved 6 figures to {FIGURES}")


if __name__ == "__main__":
    main()

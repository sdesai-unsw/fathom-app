"""Fathom - systematic multi-asset funds, built on out-of-sample backtests and
a news-sentiment overlay. See PROJECT_BRIEF.md and CLAUDE.md.

The app only reads precomputed artifacts under results/ (written by
scripts/run_part_b.py). It never recomputes a backtest or scores sentiment -
the free tier cannot, and a live investor journey needs to load in seconds.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to a public GitHub repo, then connect it on
               share.streamlit.io with entrypoint streamlit_app.py (see brief App. D).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.portfolios import current_holdings, growth_of_one_dollar, performance_metrics  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Fathom", layout="wide", page_icon="\U0001F4C8")


# ---------------------------------------------------------------- data -----

@st.cache_data(ttl=3_600, show_spinner=False)
def load_fund_returns() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "data" / "fund_returns.csv", index_col="date", parse_dates=True)


@st.cache_data(ttl=3_600, show_spinner=False)
def load_fund_weights() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "data" / "fund_weights.csv", parse_dates=["date"])


@st.cache_data(ttl=3_600, show_spinner=False)
def load_performance_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "tables" / "performance_metrics.csv", parse_dates=["first_live_date"])


@st.cache_data(ttl=3_600, show_spinner=False)
def load_sector_sentiment() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "data" / "sector_sentiment_index.csv", index_col="date", parse_dates=True)


@st.cache_data(ttl=3_600, show_spinner=False)
def load_fusion_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "tables" / "fusion_comparison.csv")


missing = [f for f in (
    "data/fund_returns.csv", "data/fund_weights.csv",
    "tables/performance_metrics.csv", "data/sector_sentiment_index.csv",
) if not (RESULTS / f).exists()]
if missing:
    st.error(
        "Missing precomputed results: " + ", ".join(missing) +
        ". Run `python scripts/run_part_b.py` first, then reload."
    )
    st.stop()

fund_returns = load_fund_returns()
fund_weights = load_fund_weights()
metrics = load_performance_metrics()
sector_sentiment = load_sector_sentiment()
fusion_comparison = load_fusion_comparison() if (RESULTS / "tables" / "fusion_comparison.csv").exists() else None

PCT_COLS = ["annualised_return", "annualised_volatility", "max_drawdown"]


def _display_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PCT_COLS:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["sharpe_ratio"] = out["sharpe_ratio"].round(2)
    return out


st.title("Fathom")
st.caption(
    "Systematically managed multi-asset funds, backed by an out-of-sample "
    "backtest and a news-sentiment overlay on the equity sleeve."
)

tab_compare, tab_factsheet, tab_allocate, tab_sentiment, tab_about = st.tabs(
    ["Compare Funds", "Fund Fact Sheet", "Allocate", "Sentiment", "About"]
)


# ---------------------------------------------------------- compare --------

with tab_compare:
    st.subheader("Compare funds")
    families = sorted(metrics["family"].unique())
    picked_families = st.multiselect("Fund family", families, default=families)
    view = metrics[metrics["family"].isin(picked_families)].sort_values("sharpe_ratio", ascending=False)

    st.dataframe(
        _display_metrics(view)[["fund", "family", "method", "annualised_return",
                                 "annualised_volatility", "sharpe_ratio", "max_drawdown"]],
        width="stretch", hide_index=True,
    )
    st.caption(
        "Annualised return, volatility and Sharpe ratio (risk-free rate assumed 0) "
        "from each fund's out-of-sample walk-forward backtest, 2020-2023 "
        "(equities/combined) or the daily 2020-2023 crypto calendar."
    )

    st.bar_chart(view.set_index("fund")["sharpe_ratio"], horizontal=True)
    st.caption("Sharpe ratio by fund and method (rf = 0). Higher is better risk-adjusted performance.")

    default_growth = view["fund"].head(4).tolist()
    growth_funds = st.multiselect("Growth of $1 - funds to compare", view["fund"].tolist(), default=default_growth)
    if growth_funds:
        growth = pd.DataFrame({f: growth_of_one_dollar(fund_returns[f]) for f in growth_funds})
        st.line_chart(growth)
        st.caption(
            f"Growth of a $1 investment from each fund's first live rebalance date "
            f"through {fund_returns.index.max().date()}. Funds start on different "
            "dates because the estimation window must fill before a fund goes live."
        )


# --------------------------------------------------------- fact sheet ------

with tab_factsheet:
    st.subheader("Fund fact sheet")
    fund_name = st.selectbox("Fund", metrics["fund"].tolist())
    row = metrics.loc[metrics["fund"] == fund_name].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", f"{row['annualised_return']*100:.1f}%")
    c2.metric("Annualised volatility", f"{row['annualised_volatility']*100:.1f}%")
    c3.metric("Sharpe ratio (rf=0)", f"{row['sharpe_ratio']:.2f}")
    c4.metric("Max drawdown", f"{row['max_drawdown']*100:.1f}%")
    st.caption(f"First live rebalance: {row['first_live_date'].date()}. n = {int(row['n_obs'])} trading days.")

    r = fund_returns[fund_name].dropna()
    growth = growth_of_one_dollar(r)
    dd = growth / growth.cummax() - 1.0

    col_growth, col_dd = st.columns(2)
    with col_growth:
        st.line_chart(growth.rename("Growth of $1"))
        st.caption("Cumulative growth of a $1 investment, out-of-sample.")
    with col_dd:
        st.area_chart(dd.rename("Drawdown"))
        st.caption("Drawdown from the running peak of growth of $1.")

    weights = fund_weights.loc[fund_weights["fund"] == fund_name]
    if not weights.empty:
        wide = weights.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
        top_tickers = wide.mean().sort_values(ascending=False).head(10).index
        st.area_chart(wide[top_tickers])
        st.caption(
            "Portfolio weight over time for the 10 holdings with the largest "
            f"average weight (of {wide.shape[1]} total holdings)."
        )

        holdings = current_holdings(wide).head(15)
        st.bar_chart(holdings, horizontal=True)
        st.caption(f"Current holdings - target weights from the fund's most recent rebalance "
                   f"({wide.index.max().date()}), top 15 by weight.")


# ----------------------------------------------------------- allocate ------

with tab_allocate:
    st.subheader("Set your allocation")
    st.caption("Split a hypothetical investment across funds. Allocations are normalised to sum to 100%.")

    chosen = st.multiselect("Funds to invest in", metrics["fund"].tolist(),
                             default=metrics["fund"].head(3).tolist())
    if not chosen:
        st.info("Pick at least one fund.")
    else:
        raw_pcts = {}
        cols = st.columns(len(chosen))
        for col, f in zip(cols, chosen):
            raw_pcts[f] = col.slider(f, 0, 100, round(100 / len(chosen)), key=f"alloc_{f}")

        total = sum(raw_pcts.values())
        if total == 0:
            st.warning("At least one allocation must be above 0%.")
        else:
            weights = {f: p / total for f, p in raw_pcts.items()}
            st.write({f: f"{w*100:.1f}%" for f, w in weights.items()})

            common_returns = pd.DataFrame({f: fund_returns[f] for f in chosen}).dropna()
            if common_returns.empty:
                st.warning("Selected funds have no overlapping live dates.")
            else:
                blended = sum(common_returns[f] * w for f, w in weights.items())
                blended_growth = growth_of_one_dollar(blended)
                blended_metrics = performance_metrics(blended, periods_per_year=252)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Annualised return", f"{blended_metrics['annualised_return']*100:.1f}%")
                m2.metric("Annualised volatility", f"{blended_metrics['annualised_volatility']*100:.1f}%")
                m3.metric("Sharpe ratio (rf=0)", f"{blended_metrics['sharpe_ratio']:.2f}")
                m4.metric("Max drawdown", f"{blended_metrics['max_drawdown']*100:.1f}%")

                st.line_chart(blended_growth.rename("Blended growth of $1"))
                st.caption(
                    f"Blended portfolio over the dates all selected funds were live "
                    f"({common_returns.index.min().date()} to {common_returns.index.max().date()}), "
                    "reweighted daily returns at the chosen allocation, no rebalancing cost."
                )


# ----------------------------------------------------------- sentiment -----

with tab_sentiment:
    st.subheader("Equity sector sentiment")
    sectors = sorted(sector_sentiment.columns)
    picked = st.multiselect("Sectors", sectors, default=sectors[:4])
    if picked:
        st.line_chart(sector_sentiment[picked])
        st.caption(
            "Daily equal-weight VADER compound sentiment index per sector, 2020-2023. "
            "Ticker-days with no headline are scored neutral (0), consistent with "
            "VADER's own neutral score for headlines it cannot read as positive or "
            "negative - see src/sentiment.py."
        )

    if fusion_comparison is not None and not fusion_comparison.empty:
        st.subheader("Sentiment fusion: before vs after")
        st.caption(
            "Each equity fund's weights tilted by its holdings' own sentiment "
            "(lagged at least one trading day, see src/fusion.py), compared to "
            "the untilted base fund. A negative delta means the tilt underperformed."
        )
        show = fusion_comparison.copy()
        show["base_sharpe"] = show["base_sharpe"].round(2)
        show["sentiment_sharpe"] = show["sentiment_sharpe"].round(2)
        show["delta_sharpe_ratio"] = show["delta_sharpe_ratio"].round(3)
        st.dataframe(
            show[["base_fund", "base_sharpe", "sentiment_sharpe", "delta_sharpe_ratio"]],
            width="stretch", hide_index=True,
        )
        st.bar_chart(fusion_comparison.set_index("base_fund")["delta_sharpe_ratio"], horizontal=True)


# --------------------------------------------------------------- about -----

with tab_about:
    st.subheader("About Fathom")
    st.markdown(
        "Fathom offers a small set of systematically managed multi-asset funds "
        "to an investor who wants rules-based, evidence-backed exposure to "
        "equities and crypto without picking stocks themselves.\n\n"
        "Every fund is walk-forward backtested out-of-sample - weights on any "
        "given day use only information available before that day - and rebalanced "
        "monthly. The equity sleeve is additionally informed by a VADER "
        "sentiment index built from daily news headlines across 10 sectors, "
        "lagged at least one trading day before it can affect a fund's weights.\n\n"
        "**Investor journey:** compare funds and their risk-adjusted performance, "
        "open a fund's fact sheet, set a personal allocation across funds, and "
        "read the sentiment analytics behind the equity funds."
    )
    st.caption(
        f"Backtest sample: {fund_returns.index.min().date()} to {fund_returns.index.max().date()}. "
        f"{len(metrics)} funds across {metrics['family'].nunique()} asset families. "
        "All figures are precomputed by scripts/run_part_b.py from src/data_access.py - "
        "this app does not recompute backtests or sentiment scores."
    )

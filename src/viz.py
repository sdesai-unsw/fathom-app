"""Self-contained figure helpers - the Fathom design system, carried over from
Part A (context/project_context.md confirms Part B may reuse a student's own
Part A). Deliberately independent of anything outside this folder (no imports
from the parent fins-agent repo or z5476492_projectA) so the pipeline
reproduces from a clean checkout of this zipped project alone.

Every figure follows the same fixed two-colour role, not a cycled palette:
TEAL is the single emphasis colour that marks the answer to the figure's
question (the series, bar, or holding that matters), and NAVY is the neutral
colour for everything else. Validated in Part A's colour-validator script:
TEAL-vs-NAVY passes both colour-blind separation (Delta-E 35.9, deutan) and
contrast against a white page (>=3:1); SLATE is kept only for gridlines,
ticks, and muted text, never a second encoded data colour. Where a figure
needs to rank several emphasised series (for example the top holdings in a
weights chart), rank is shown by varying TEAL's alpha, not by introducing a
new hue - the two-colour system stays intact.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

NAVY = "#1B2A4A"    # neutral: every non-emphasised series, box, or bar
TEAL = "#0F8C8C"    # emphasis: the one thing that answers the figure's question
SLATE = "#5B6B7C"   # gridlines, ticks, muted text only - never a data colour
GRID = "#D9DEE6"

PALETTE = [NAVY, TEAL]

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": SLATE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#1F2933",
    "axes.titlecolor": NAVY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def style():
    return plt.rc_context(RC)


def _horizontal_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.7)
    ax.grid(False, axis="x")


def _source_line(fig: plt.Figure, source: str | None, sample: str | None = None, extra: str | None = None) -> None:
    """Draw a standalone source-and-sample-window footer directly on the
    figure (not just in the Word caption sidecar) - every figure needs both,
    with no exceptions, so it stands alone if copied out of the report."""
    if source or sample or extra:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + (0.125 if extra else 0.09))
        parts = []
        if source:
            parts.append(f"Source: {source}")
        if sample:
            parts.append(f"Sample: {sample}")
        y = 0.035 if extra else 0.01
        fig.text(0.01, y, "   |   ".join(parts), fontsize=7.5, color=SLATE, ha="left", va="bottom")
        if extra:
            fig.text(0.01, 0.01, extra, fontsize=7.5, color=SLATE, ha="left", va="bottom")


def caption(title: str, note: str, sample: str, units: str, source: str) -> str:
    """Build one self-contained caption string (title, note, sample, units, source)."""
    parts = [title.rstrip("."), note.rstrip(".")]
    parts.append(f"Sample: {sample}")
    parts.append(f"Units: {units}")
    parts.append(f"Source: {source}")
    return ". ".join(p for p in parts if p) + "."


def save(fig: plt.Figure, out_dir: Path, stem: str, cap: str) -> Path:
    """Save a figure as a 300 dpi PNG (A4/Word-ready width) plus a caption sidecar.

    `bbox_inches="tight"` is passed explicitly here rather than relied on via
    the "savefig.bbox" rcParam in RC: by the time this function runs, the
    `with style():` context that activated RC while building the figure has
    already exited, so that rcParam is not in effect for this savefig() call.
    Without it, any label placed outside the fixed subplot margins (a long
    fund name, an end-of-line label) is clipped by the canvas edge rather
    than expanding the saved image to fit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(6.3, fig.get_size_inches()[1], forward=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    (out_dir / f"{stem}.caption.txt").write_text(cap, encoding="utf-8")
    plt.close(fig)
    return path


def _declutter_labels(ends: list) -> list:
    """Spread direct end-of-line label y-positions apart by a minimum gap so
    close endpoints do not overlap, with a reconciliation pass for chains of
    collisions. `ends` is a list of [x, y, label, is_emphasis] sorted by y."""
    if not ends:
        return []
    y_min = min(e[1] for e in ends)
    y_max = max(e[1] for e in ends)
    y_span = (y_max - y_min) or 1.0
    min_gap = 0.10 * y_span
    placed = [ends[0][1]]
    for _, y, _, _ in ends[1:]:
        placed.append(max(y, placed[-1] + min_gap))
    for _ in range(2):
        for i in range(len(placed) - 2, -1, -1):
            if placed[i + 1] - placed[i] < min_gap:
                placed[i] = placed[i + 1] - min_gap
    return placed


def cumulative_return_lines(
    wide_returns: pd.DataFrame,
    title: str,
    emphasis_cols: list[str] = (),
    ylabel: str = "Growth of $1",
    source: str | None = None,
    sample: str | None = None,
    emphasis_label: str = "Best fund",
    neutral_label: str = "Other funds",
) -> tuple[plt.Figure, plt.Axes]:
    """Growth-of-$1 lines with direct end-of-line labels instead of a legend.
    `emphasis_cols` are drawn in TEAL; every other series is a muted NAVY
    line, so the reader's eye goes to the fund that answers the figure's
    question rather than splitting evenly across all of them."""
    with style():
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        ends = []
        for col in wide_returns.columns:
            series = wide_returns[col].dropna()
            if series.empty:
                continue
            w = (1.0 + series).cumprod()
            is_emphasis = col in emphasis_cols
            ax.plot(
                w.index, w.values,
                color=TEAL if is_emphasis else NAVY,
                linewidth=2.1 if is_emphasis else 1.2,
                alpha=1.0 if is_emphasis else 0.55,
                zorder=3 if is_emphasis else 2,
            )
            ends.append([w.index[-1], w.values[-1], col, is_emphasis])

        ends.sort(key=lambda e: e[1])
        placed = _declutter_labels(ends)
        x_min = min(w.index.min() for w in [(1.0 + wide_returns[c].dropna()).cumprod()
                                             for c in wide_returns.columns if wide_returns[c].notna().any()])
        x_max = wide_returns.index.max()
        label_x = x_max + (x_max - x_min) * 0.02
        for (x, y, col, is_emphasis), y_label in zip(ends, placed):
            if abs(y_label - y) > 1e-9:
                ax.plot([x, label_x], [y, y_label], color=SLATE, linewidth=0.6, alpha=0.6, clip_on=False)
            ax.text(
                label_x, y_label, f"{col} ${y:,.2f}",
                va="center", ha="left", fontsize=8.5,
                color=TEAL if is_emphasis else SLATE,
                fontweight="bold" if is_emphasis else "normal",
                clip_on=False,
            )
        ax.margins(x=0.01)
        right_pad = x_min + (x_max - x_min) * 1.32
        ax.set_xlim(x_min, right_pad)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        _horizontal_grid(ax)
        ax.set_title(title, loc="left", wrap=True)
        if emphasis_cols:
            ax.text(
                0.012, 0.97, f"Teal = {emphasis_label}     Grey = {neutral_label}",
                transform=ax.transAxes, fontsize=8.5, color=SLATE, style="italic", va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        _source_line(fig, source, sample)
        return fig, ax


def level_lines(
    wide_levels: pd.DataFrame,
    title: str,
    ylabel: str,
    emphasis_cols: list[str] = (),
    smooth_window: int | None = None,
    source: str | None = None,
    sample: str | None = None,
    emphasis_label: str = "Emphasis",
    neutral_label: str = "Other series",
) -> tuple[plt.Figure, plt.Axes]:
    """Same neutral/emphasis line language as cumulative_return_lines, but for
    a level series (for example a sentiment index) rather than a compounded
    return series - no cumprod, optional rolling-mean smoothing since a daily
    sentiment index is noisy at the ticker-day level."""
    with style():
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        ends = []
        for col in wide_levels.columns:
            series = wide_levels[col].dropna()
            if series.empty:
                continue
            if smooth_window:
                series = series.rolling(smooth_window, min_periods=max(1, smooth_window // 3)).mean()
            is_emphasis = col in emphasis_cols
            ax.plot(
                series.index, series.values,
                color=TEAL if is_emphasis else NAVY,
                linewidth=1.9 if is_emphasis else 1.0,
                alpha=1.0 if is_emphasis else 0.45,
                zorder=3 if is_emphasis else 2,
            )
            ends.append([series.index[-1], series.values[-1], col, is_emphasis])

        ends.sort(key=lambda e: e[1])
        placed = _declutter_labels(ends)
        x_min, x_max = wide_levels.index.min(), wide_levels.index.max()
        label_x = x_max + (x_max - x_min) * 0.02
        for (x, y, col, is_emphasis), y_label in zip(ends, placed):
            if not is_emphasis:
                continue
            if abs(y_label - y) > 1e-9:
                ax.plot([x, label_x], [y, y_label], color=SLATE, linewidth=0.6, alpha=0.6, clip_on=False)
            ax.text(
                label_x, y_label, f"{col} {y:+.3f}",
                va="center", ha="left", fontsize=8.5, color=TEAL, fontweight="bold", clip_on=False,
            )
        ax.axhline(0, color=SLATE, linewidth=0.8, alpha=0.6, zorder=1)
        ax.margins(x=0.01)
        right_pad = x_min + (x_max - x_min) * 1.22
        ax.set_xlim(x_min, right_pad)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        _horizontal_grid(ax)
        ax.set_title(title, loc="left", wrap=True)
        if emphasis_cols:
            ax.text(
                0.012, 0.97, f"Teal = {emphasis_label}     Grey = {neutral_label}",
                transform=ax.transAxes, fontsize=8.5, color=SLATE, style="italic", va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        _source_line(fig, source, sample)
        return fig, ax


def drawdown_area(
    daily_returns: pd.Series,
    title: str,
    source: str | None = None,
    sample: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Drawdown from the running peak of growth of $1, filled in NAVY, with
    the single worst drawdown point marked and labelled in TEAL - the one
    number a fact sheet's "max drawdown" figure needs the reader to find."""
    r = daily_returns.dropna()
    growth = (1.0 + r).cumprod()
    dd = growth / growth.cummax() - 1.0
    trough_date = dd.idxmin()
    trough_val = dd.min()

    with style():
        fig, ax = plt.subplots(figsize=(6.3, 3.6))
        ax.fill_between(dd.index, dd.values, 0, color=NAVY, alpha=0.35, zorder=2)
        ax.plot(dd.index, dd.values, color=NAVY, linewidth=1.1, alpha=0.8, zorder=3)
        ax.scatter([trough_date], [trough_val], color=TEAL, s=36, zorder=5, edgecolor="white", linewidth=0.8)
        ax.annotate(
            f"Max drawdown {trough_val:.1%}\n{trough_date.date()}",
            xy=(trough_date, trough_val), xytext=(0, -12), textcoords="offset points",
            ha="center", va="top", fontsize=8.5, color=TEAL, fontweight="bold",
        )
        ax.set_ylim(min(trough_val * 1.25, -0.05), 0.02)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _horizontal_grid(ax)
        ax.set_title(title, loc="left", wrap=True)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown from peak")
        _source_line(fig, source, sample)
        return fig, ax


def weights_concentration_area(
    weights_wide: pd.DataFrame,
    title: str,
    top_n: int = 5,
    source: str | None = None,
    sample: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Stacked-area weights over time. Rather than one hue per ticker (which
    would break the validated two-colour system past a handful of series),
    the `top_n` holdings by average weight are ranked TEAL bands (darkest =
    largest average holding), and every other ticker is summed into one NAVY
    "other holdings" band - the question this figure answers is how
    concentrated the fund is and whether that concentration shifts over time,
    not which of fifty tickers is which."""
    from matplotlib.colors import to_rgba

    avg_weight = weights_wide.mean().sort_values(ascending=False)
    top = avg_weight.head(top_n).index.tolist()
    other = weights_wide.drop(columns=top).sum(axis=1)
    layers = [weights_wide[c] for c in top] + [other]
    labels = top + [f"Other {weights_wide.shape[1] - top_n} holdings"]
    alphas = np.linspace(1.0, 0.45, len(top))
    colors = [to_rgba(TEAL, a) for a in alphas] + [to_rgba(NAVY, 0.55)]

    with style():
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        polys = ax.stackplot(weights_wide.index, layers, colors=colors, labels=labels,
                              edgecolor="white", linewidth=0.3)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _horizontal_grid(ax)
        ax.set_title(title, loc="left", wrap=True)
        ax.set_xlabel("Rebalance date")
        ax.set_ylabel("Portfolio weight")

        # Legend sits outside the axes (to the right) rather than over the
        # plot: with a fund this concentrated in "Other holdings", any
        # in-plot corner the legend could occupy is also where the data is.
        ax.legend(handles=polys, labels=labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=8)
        _source_line(fig, source, sample)
        return fig, ax


def sharpe_barplot(
    metrics: pd.DataFrame,
    title: str,
    fund_col: str = "fund",
    sharpe_col: str = "sharpe_ratio",
    source: str | None = None,
    sample: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal Sharpe-ratio bars, sorted, with the single best fund
    highlighted in TEAL and labelled directly - the rest stay muted NAVY,
    including any fund with a negative Sharpe (which extends left of zero,
    not hidden or dropped)."""
    ordered = metrics.sort_values(sharpe_col, ascending=True).reset_index(drop=True)
    best_idx = ordered[sharpe_col].idxmax()

    with style():
        fig, ax = plt.subplots(figsize=(6.3, max(3.2, 0.34 * len(ordered))))
        colors = [TEAL if i == best_idx else NAVY for i in range(len(ordered))]
        alphas = [0.95 if i == best_idx else 0.6 for i in range(len(ordered))]
        bars = ax.barh(ordered[fund_col], ordered[sharpe_col], color=colors)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)
        best = ordered.loc[best_idx]
        ax.annotate(
            f"{best[sharpe_col]:.2f}", xy=(best[sharpe_col], best_idx),
            xytext=(6 if best[sharpe_col] >= 0 else -6, 0), textcoords="offset points",
            va="center", ha="left" if best[sharpe_col] >= 0 else "right",
            fontsize=8.5, color=TEAL, fontweight="bold",
        )
        ax.axvline(0, color=SLATE, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.7)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", wrap=True)
        ax.set_xlabel("Sharpe ratio (rf = 0)")
        _source_line(fig, source, sample)
        return fig, ax


def fusion_before_after_bar(
    fusion: pd.DataFrame,
    title: str,
    fund_col: str = "base_fund",
    base_col: str = "base_sharpe",
    tilted_col: str = "sentiment_sharpe",
    source: str | None = None,
    sample: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Paired bars per equity fund: NAVY = base fund, TEAL = the same fund
    with the sentiment tilt applied. The pairing itself is the answer this
    figure gives (did the tilt help or hurt each fund), so both colours carry
    equal visual weight here rather than one dominating."""
    n = len(fusion)
    y = np.arange(n)
    height = 0.36

    with style():
        fig, ax = plt.subplots(figsize=(6.3, max(3.0, 0.7 * n)))
        ax.barh(y + height / 2, fusion[base_col], height=height, color=NAVY, alpha=0.75, label="Base fund")
        ax.barh(y - height / 2, fusion[tilted_col], height=height, color=TEAL, alpha=0.9, label="+ Sentiment tilt")
        for yi, (b, t) in enumerate(zip(fusion[base_col], fusion[tilted_col])):
            delta = t - b
            sign = "+" if delta >= 0 else "−"
            ax.annotate(
                f"{sign}{abs(delta):.3f}", xy=(max(b, t), yi - height / 2),
                xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=TEAL if delta >= 0 else SLATE, fontweight="bold",
            )
        ax.axvline(0, color=SLATE, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(fusion[fund_col])
        ax.grid(True, axis="x", alpha=0.7)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", wrap=True)
        # Legend outside the axes (right side), not above: the title can
        # wrap to two lines depending on its length, and an in-plot annotation
        # anchored just above the axes has no reliably empty spot to sit in
        # once that happens.
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
        ax.set_xlabel("Sharpe ratio (rf = 0)")
        _source_line(fig, source, sample)
        return fig, ax

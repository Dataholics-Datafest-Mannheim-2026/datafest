"""
DataFest 2026 — Step 7: Temporal Visualizations
Topic: Bots as Wikipedia Editors

Reads temporal_daily.ndjson (produced by 06_temporal_aggregation.py) and
creates 5 presentation-ready charts about how bot activity evolves over 2025.

Charts saved to exploration/figures/ as PNG (150 dpi).

Requires: uv add matplotlib
Run:      uv run exploration/07_temporal_visualizations.py
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT       = Path(__file__).parent / "temporal_daily.ndjson"
INPUT_TOPIC = Path(__file__).parent / "temporal_daily_by_topic.ndjson"
OUTPUT_DIR  = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

BOT_COLOR   = "#E05C4B"
HUMAN_COLOR = "#4B8BE0"
ROLLING_WINDOW = 1     # days for rolling average — smooths noise, keeps trends (1= no smoothing)

LANGUAGES = ['ar', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'ru', 'sv']
LANG_LABELS = {
    'ar': 'Arabic', 'de': 'German', 'en': 'English', 'es': 'Spanish',
    'fr': 'French', 'it': 'Italian', 'nl': 'Dutch', 'pl': 'Polish',
    'ru': 'Russian', 'sv': 'Swedish',
}
EDIT_TYPES = ["total_edits", "ref_edits", "tmpl_edits", "link_edits", "revert_edits"]
TOPIC_COLORS = {
    "Geography":           "#2980B9",
    "Culture":             "#8E44AD",
    "STEM":                "#27AE60",
    "History_and_Society": "#E67E22",
}
EDIT_LABELS = {
    "total_edits":  "All edits",
    "ref_edits":    "References",
    "tmpl_edits":   "Templates",
    "link_edits":   "Links",
    "revert_edits": "Reverts",
}

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
})


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> figures/{name}.png")


def rolling_avg(arr: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Simple centred rolling average using convolution."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def to_dates(date_series: pl.Series) -> list[datetime.date]:
    """Convert polars Date series to Python date list for matplotlib."""
    return date_series.to_list()


# ── LOAD ──────────────────────────────────────────────────────────────────────
if not INPUT.exists():
    raise SystemExit(f"ERROR: {INPUT.name} not found — run 06_temporal_aggregation.py first.")

print("Loading temporal_daily.ndjson...")
df = pl.read_ndjson(INPUT)
# date comes back as string from ndjson — cast to proper Date type
df = df.with_columns(pl.col("date").cast(pl.Date))
print(f"  {df.shape[0]:,} rows  |  {df['date'].min()} to {df['date'].max()}\n")


# ── GLOBAL DAILY TOTALS ───────────────────────────────────────────────────────
# Sum across all 10 languages to get one bot row + one human row per day.
global_daily = (
    df.group_by(["date", "is_bot"])
    .agg(
        pl.col("total_edits").sum(),
        pl.col("unique_editors").sum(),
        pl.col("word_changes").sum(),
        pl.col("ref_edits").sum(),
        pl.col("tmpl_edits").sum(),
        pl.col("link_edits").sum(),
        pl.col("revert_edits").sum(),
    )
    .sort(["date", "is_bot"])
)

bot_daily   = global_daily.filter(pl.col("is_bot") == True) .sort("date")
human_daily = global_daily.filter(pl.col("is_bot") == False).sort("date")

dates = to_dates(bot_daily["date"])


# ══════════════════════════════════════════════════════════════════════════════
# FIG T1 — Daily edit volume: bot vs human
# ──────────────────────────────────────────────────────────────────────────────
# Stacked area shows absolute volume. The bot area is small relative to humans
# but visible. Overlaid rolling-average lines cut through daily noise.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG T1: Daily edit volume stacked area...")

bot_edits   = bot_daily["total_edits"].to_numpy().astype(float)
human_edits = human_daily["total_edits"].to_numpy().astype(float)

fig, ax = plt.subplots(figsize=(13, 5))
ax.stackplot(dates, bot_edits, human_edits,
             labels=["Bots", "Humans"],
             colors=[BOT_COLOR, HUMAN_COLOR], alpha=0.35)
ax.plot(dates, rolling_avg(bot_edits + human_edits),
        color="#333333", lw=1.5, label=f"{ROLLING_WINDOW}-day avg (total)")

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Edits per day")
ax.set_title("FIG T1 — Daily Edit Volume Across All Languages (2025)\n"
             "Bots contribute consistently; humans drive the bulk and seasonal swings")
ax.legend(loc="upper right")
save(fig, "T1_daily_volume")


# ══════════════════════════════════════════════════════════════════════════════
# FIG T2 — Bot share of total edits over time (global)
# ──────────────────────────────────────────────════════════════════════════════
# Single line: what % of all edits each day were made by bots?
# Rolling average smooths weekend noise. Trend direction is the key signal.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG T2: Global bot share over time...")

total      = bot_edits + human_edits
bot_share  = np.where(total > 0, bot_edits / total * 100, np.nan)
bot_share_smooth = rolling_avg(np.nan_to_num(bot_share))

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(dates, bot_share, color=BOT_COLOR, alpha=0.2, lw=0.8)
ax.plot(dates, bot_share_smooth, color=BOT_COLOR, lw=2,
        label=f"{ROLLING_WINDOW}-day rolling avg")
ax.axhline(np.nanmean(bot_share), color="#555555", lw=1, ls="--",
           label=f"Year average: {np.nanmean(bot_share):.1f}%")

ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Bot share of all edits")
ax.set_title("FIG T2 — Bot Share of All Edits Over Time (Global, 2025)\n"
             "Is automation's role in Wikipedia growing or stable?")
ax.legend(loc="upper right")
save(fig, "T2_global_bot_share")


# ══════════════════════════════════════════════════════════════════════════════
# FIG T3 — Bot share per language over time
# ──────────────────────────────────────────────────────────────────────────────
# 10 subplots (one per language) with the same y-axis scale so you can
# compare absolute bot-share levels across language editions at a glance.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG T3: Bot share per language...")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes = axes.flatten()

for ax, lang in zip(axes, LANGUAGES):
    lang_df = df.filter(pl.col("language") == lang).sort(["date", "is_bot"])
    b = lang_df.filter(pl.col("is_bot") == True) .sort("date")
    h = lang_df.filter(pl.col("is_bot") == False).sort("date")

    # Align on dates present in both
    merged = b.join(h, on="date", suffix="_h")
    if merged.shape[0] == 0:
        ax.set_title(LANG_LABELS[lang])
        continue

    d         = to_dates(merged["date"])
    b_vals    = merged["total_edits"].to_numpy().astype(float)
    h_vals    = merged["total_edits_h"].to_numpy().astype(float)
    total_l   = b_vals + h_vals
    share     = np.where(total_l > 0, b_vals / total_l * 100, np.nan)
    share_sm  = rolling_avg(np.nan_to_num(share))

    ax.plot(d, share, color=BOT_COLOR, alpha=0.15, lw=0.7)
    ax.plot(d, share_sm, color=BOT_COLOR, lw=1.8)
    ax.axhline(np.nanmean(share), color="#888888", lw=0.8, ls="--")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title(f"{LANG_LABELS[lang]}  (avg {np.nanmean(share):.1f}%)")

fig.suptitle("FIG T3 — Bot Share per Language (2025)\n"
             f"{ROLLING_WINDOW}-day rolling avg  |  dashed = year mean",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "T3_bot_share_per_language")


# ══════════════════════════════════════════════════════════════════════════════
# FIG T4 — Bot share by edit category over time
# ──────────────────────────────────────────────────────────────────────────────
# Does the bot share differ by WHAT is being edited?
# E.g. are bots responsible for >50% of all reference edits but <5% of reverts?
# One line per edit category, global across all languages.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG T4: Bot share by edit category...")

# Build a merged bot/human frame aligned on date
bot_g   = bot_daily.sort("date")
human_g = human_daily.sort("date")
merged  = bot_g.join(human_g, on="date", suffix="_h")

fig, ax = plt.subplots(figsize=(13, 5))
colors_cat = ["#333333", "#2980B9", "#8E44AD", "#27AE60", "#E67E22"]

for col, label, color in zip(EDIT_TYPES, [EDIT_LABELS[e] for e in EDIT_TYPES], colors_cat):
    b_vals = merged[col].to_numpy().astype(float)
    h_vals = merged[f"{col}_h"].to_numpy().astype(float)
    total  = b_vals + h_vals
    share  = np.where(total > 0, b_vals / total * 100, np.nan)
    smooth = rolling_avg(np.nan_to_num(share))
    d      = to_dates(merged["date"])
    ax.plot(d, smooth, lw=2, color=color, label=f"{label}  (avg {np.nanmean(share):.1f}%)")

ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Bot share of that edit type")
ax.set_title("FIG T4 — Bot Share by Edit Category (Global, 2025)\n"
             f"Which types of edits are bots dominating?  ({ROLLING_WINDOW}-day rolling avg)")
ax.legend(loc="right", framealpha=0.9)
save(fig, "T4_bot_share_by_category")


# ══════════════════════════════════════════════════════════════════════════════
# FIG T5 — Heatmap: bot share by language × week
# ──────────────────────────────────────────────────────────────────────────────
# Daily data aggregated to ISO weeks (52 columns) so the heatmap stays readable.
# Each cell = bot share of total edits in that language in that week.
# Reveals which languages see consistent bot presence vs sporadic bursts.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG T5: Heatmap language × week...")

df_week = df.with_columns(
    pl.col("date").dt.week().alias("week")
)

weekly = (
    df_week.group_by(["week", "language", "is_bot"])
    .agg(pl.col("total_edits").sum())
    .sort(["week", "language", "is_bot"])
)

# Build 10 × 52 matrix of bot share per language per week
weeks = sorted(weekly["week"].unique().to_list())
matrix = np.full((len(LANGUAGES), len(weeks)), np.nan)

for i, lang in enumerate(LANGUAGES):
    for j, wk in enumerate(weeks):
        sub = weekly.filter((pl.col("language") == lang) & (pl.col("week") == wk))
        b = sub.filter(pl.col("is_bot") == True)["total_edits"].sum()
        h = sub.filter(pl.col("is_bot") == False)["total_edits"].sum()
        if b + h > 0:
            matrix[i, j] = b / (b + h) * 100

fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r",
               vmin=0, vmax=matrix[~np.isnan(matrix)].max())
plt.colorbar(im, ax=ax, label="Bot share %", shrink=0.8)

ax.set_yticks(range(len(LANGUAGES)))
ax.set_yticklabels([LANG_LABELS[l] for l in LANGUAGES])

# Label x-axis as month names at approximate week positions
month_starts = {1: "Jan", 5: "Feb", 9: "Mar", 14: "Apr", 18: "May", 22: "Jun",
                27: "Jul", 31: "Aug", 36: "Sep", 40: "Oct", 44: "Nov", 49: "Dec"}
ax.set_xticks(list(month_starts.keys()))
ax.set_xticklabels(list(month_starts.values()))

ax.set_xlabel("Week of 2025")
ax.set_title("FIG T5 — Bot Share Heatmap: Language × Week (2025)\n"
             "Darker red = higher bot share that week in that language")
save(fig, "T5_heatmap_language_week")


# ══════════════════════════════════════════════════════════════════════════════
# FIG T6 — Unique editors per language per day
# ──────────────────────────────────────────────────────────────────────────────
# 10 subplots, one per language. Each shows the daily count of distinct editors
# (bots in red, humans in blue) so we can see whether bot activity is driven by
# many occasional bots or a handful of persistent ones.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG T6: Unique editors per language per day...")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes = axes.flatten()

for ax, lang in zip(axes, LANGUAGES):
    lang_df = df.filter(pl.col("language") == lang).sort(["date", "is_bot"])
    b = lang_df.filter(pl.col("is_bot") == True).sort("date")
    h = lang_df.filter(pl.col("is_bot") == False).sort("date")

    if b.shape[0] == 0 and h.shape[0] == 0:
        ax.set_title(LANG_LABELS[lang])
        continue

    if h.shape[0] > 0:
        h_dates = to_dates(h["date"])
        h_vals  = rolling_avg(h["unique_editors"].to_numpy().astype(float))
        ax.plot(h_dates, h_vals, color=HUMAN_COLOR, lw=1.5, label="Humans")

    if b.shape[0] > 0:
        b_dates = to_dates(b["date"])
        b_vals  = rolling_avg(b["unique_editors"].to_numpy().astype(float))
        ax.plot(b_dates, b_vals, color=BOT_COLOR, lw=1.5, label="Bots")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(LANG_LABELS[lang])

axes[0].legend(fontsize=8, loc="upper right")

fig.suptitle("FIG T6 — Unique Editors per Day by Language (2025)\n"
             f"How many distinct bot vs human accounts edit each day?  "
             f"({ROLLING_WINDOW}-day rolling avg)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "T6_unique_editors_per_language")


# ── LOAD TOPIC DATA ───────────────────────────────────────────────────────────
if not INPUT_TOPIC.exists():
    print(f"WARNING: {INPUT_TOPIC.name} not found — skipping T7/T8. Run 06 first.")
else:
    print("Loading temporal_daily_by_topic.ndjson...")
    df_topic = pl.read_ndjson(INPUT_TOPIC)
    df_topic = df_topic.with_columns(pl.col("date").cast(pl.Date))
    topics_present = sorted(df_topic["top_category"].unique().to_list())
    print(f"  {df_topic.shape[0]:,} rows  |  topics: {topics_present}\n")


    # ══════════════════════════════════════════════════════════════════════════
    # FIG T7 — Bot share per topic over time (global)
    # ────────────────────────────────────────────────────────────────────────
    # Same shape as T4 but sliced by topic instead of edit category.
    # One line per topic — which subject areas attract the most bot attention?
    # ══════════════════════════════════════════════════════════════════════════
    print("FIG T7: Global bot share per topic over time...")

    # Sum across all languages
    global_topic = (
        df_topic.group_by(["date", "is_bot", "top_category"])
        .agg(pl.col("total_edits").sum())
        .sort(["date", "top_category", "is_bot"])
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    for topic in topics_present:
        color = TOPIC_COLORS.get(topic, "#888888")
        sub = global_topic.filter(pl.col("top_category") == topic)
        b_sub = sub.filter(pl.col("is_bot") == True).sort("date")
        h_sub = sub.filter(pl.col("is_bot") == False).sort("date")
        merged_t = b_sub.join(h_sub, on="date", suffix="_h")
        if merged_t.shape[0] == 0:
            continue
        b_vals = merged_t["total_edits"].to_numpy().astype(float)
        h_vals = merged_t["total_edits_h"].to_numpy().astype(float)
        total  = b_vals + h_vals
        share  = np.where(total > 0, b_vals / total * 100, np.nan)
        smooth = rolling_avg(np.nan_to_num(share))
        d      = to_dates(merged_t["date"])
        ax.plot(d, smooth, lw=2, color=color,
                label=f"{topic}  (avg {np.nanmean(share):.1f}%)")

    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Bot share of edits in topic")
    ax.set_title("FIG T7 — Bot Share by Topic Focus (Global, 2025)\n"
                 f"Which subject areas do bots dominate?  ({ROLLING_WINDOW}-day rolling avg)")
    ax.legend(loc="right", framealpha=0.9)
    save(fig, "T7_bot_share_by_topic")


    # ══════════════════════════════════════════════════════════════════════════
    # FIG T8 — Bot share per topic per language
    # ────────────────────────────────────────────────────────────────────────
    # 10 subplots (one per language), each with one line per topic.
    # Reveals whether topic-level bot patterns are global or language-specific.
    # ══════════════════════════════════════════════════════════════════════════
    print("FIG T8: Bot share per topic per language...")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.flatten()

    for ax, lang in zip(axes, LANGUAGES):
        lang_df = df_topic.filter(pl.col("language") == lang)

        for topic in topics_present:
            color = TOPIC_COLORS.get(topic, "#888888")
            sub   = lang_df.filter(pl.col("top_category") == topic)
            b_sub = sub.filter(pl.col("is_bot") == True).sort("date")
            h_sub = sub.filter(pl.col("is_bot") == False).sort("date")
            merged_t = b_sub.join(h_sub, on="date", suffix="_h")
            if merged_t.shape[0] < 5:
                continue
            b_vals = merged_t["total_edits"].to_numpy().astype(float)
            h_vals = merged_t["total_edits_h"].to_numpy().astype(float)
            total  = b_vals + h_vals
            share  = np.where(total > 0, b_vals / total * 100, np.nan)
            smooth = rolling_avg(np.nan_to_num(share))
            d      = to_dates(merged_t["date"])
            ax.plot(d, smooth, lw=1.5, color=color, label=topic)

        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_title(LANG_LABELS[lang])

    # Single legend from first subplot that has lines
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, loc="upper right")
            break

    fig.suptitle("FIG T8 — Bot Share by Topic Focus per Language (2025)\n"
                 f"Do bots target different topics across language editions?  "
                 f"({ROLLING_WINDOW}-day rolling avg)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "T8_bot_share_by_topic_per_language")


print(f"\nDone. {len(list(OUTPUT_DIR.glob('T*.png')))} temporal figures saved to: {OUTPUT_DIR}")

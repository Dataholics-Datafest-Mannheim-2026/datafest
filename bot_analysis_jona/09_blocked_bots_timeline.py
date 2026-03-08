"""
DataFest 2026 — Step 9: Blocked Bots Timeline
Topic: Bots as Wikipedia Editors

Shows the daily edit volume of Wikipedia-blocked bots over 2025.
One subplot per unique blocked bot; within each subplot, one line per
language that bot was active in (some bots appear in multiple languages).

Source: data/other_data/blocked_bots_in_edit_types.json
Output: figures/blocked_bots_timeline.png

Run: uv run bot_analysis_jona/09_blocked_bots_timeline.py
"""

import json
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR     = Path(__file__).parent.parent / "data"
EDITS_DIR         = BASE_DATA_DIR / "data_extracted" / "edit_types"
BLOCKED_BOTS_JSON = BASE_DATA_DIR / "other_data" / "blocked_bots_in_edit_types.json"
OUTPUT_DIR        = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP_FMT  = "%Y-%m-%dT%H:%M:%S%.3fZ"
ROLLING_WINDOW = 7   # days

# One color per language (stable across all subplots)
LANG_COLORS = {
    "ar": "#E05C4B", "de": "#4B8BE0", "en": "#27AE60", "es": "#F5A623",
    "fr": "#8E44AD", "it": "#2980B9", "nl": "#E67E22", "pl": "#C0392B",
    "ru": "#16A085", "sv": "#7F8C8D",
}

# ── LOAD BLOCKED BOTS ─────────────────────────────────────────────────────────
with open(BLOCKED_BOTS_JSON, encoding="utf-8") as f:
    raw = json.load(f)

# raw = list of single-key dicts → flatten to {lang: [bot_name, ...]}
blocked: dict[str, list[str]] = {}
for entry in raw:
    for lang, bots in entry.items():
        blocked[lang] = bots

# Build reverse mapping: {bot_name: [lang, ...]}
bot_langs: dict[str, list[str]] = {}
for lang, bots in blocked.items():
    for bot in bots:
        bot_langs.setdefault(bot, []).append(lang)

all_bot_names = sorted(bot_langs.keys())
print(f"{len(all_bot_names)} unique blocked bots across {len(blocked)} languages")
for bot, langs in sorted(bot_langs.items()):
    print(f"  {bot!r:35s}  languages: {', '.join(sorted(langs))}")

# ── LOAD & FILTER EDIT DATA ───────────────────────────────────────────────────
print("\nLoading edit files for languages with blocked bots...")

daily_frames = []

for lang in sorted(blocked.keys()):
    bot_names = blocked[lang]
    edits_file = EDITS_DIR / f"{lang}wiki.json.gz"

    if not edits_file.exists():
        print(f"  {lang.upper()}: file not found, skipping")
        continue

    print(f"  {lang.upper()}: loading...", end=" ", flush=True)
    edits = pl.read_ndjson(edits_file)

    edits = edits.with_columns(
        pl.col("revision_timestamp")
        .str.to_datetime(TIMESTAMP_FMT, time_unit="ms")
        .alias("ts")
    )

    edits = edits.filter(pl.col("user_text").is_in(bot_names))
    n = edits.shape[0]
    print(f"{n:,} edits from blocked bots")

    if n == 0:
        continue

    daily = (
        edits
        .with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by(["date", "user_text"])
        .agg(pl.len().alias("daily_edits"))
        .with_columns(pl.lit(lang).alias("language"))
        .sort("date")
    )
    daily_frames.append(daily)

if not daily_frames:
    print("No edit data found for any blocked bot. Exiting.")
    raise SystemExit(1)

all_daily = pl.concat(daily_frames)

# Keep only bots that actually have data
bots_with_data = sorted(all_daily["user_text"].unique().to_list())
n_bots = len(bots_with_data)
print(f"\n{n_bots} bots have edit data in this dataset: {bots_with_data}")

# ── VISUALIZATION — one subplot per bot ───────────────────────────────────────
NCOLS = 4
NROWS = (n_bots + NCOLS - 1) // NCOLS

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 4 * NROWS))
axes_flat = np.array(axes).flatten()

for ax in axes_flat[n_bots:]:
    ax.set_visible(False)

for idx, bot in enumerate(bots_with_data):
    ax = axes_flat[idx]
    bot_data = all_daily.filter(pl.col("user_text") == bot)
    langs_for_bot = sorted(bot_data["language"].unique().to_list())

    # Compute overall date range for this bot (across all languages)
    all_bot_dates = bot_data["date"].to_numpy()
    d_min = all_bot_dates.min().astype("datetime64[D]").astype(object)
    d_max = all_bot_dates.max().astype("datetime64[D]").astype(object)

    for lang in langs_for_bot:
        series = bot_data.filter(pl.col("language") == lang).sort("date")
        if series.shape[0] == 0:
            continue

        dates  = series["date"].to_numpy()
        values = series["daily_edits"].to_numpy().astype(float)

        window   = min(ROLLING_WINDOW, len(values))
        smoothed = np.convolve(values, np.ones(window) / window, mode="same")

        color = LANG_COLORS.get(lang, "#999999")
        label = lang.upper() if len(langs_for_bot) > 1 else None
        ax.plot(dates, smoothed, color=color, linewidth=1.8, label=label)
        ax.fill_between(dates, smoothed, alpha=0.15, color=color)

    # X-axis: 4 ticks evenly spaced between first and last edit date
    # (linspace on matplotlib date numbers guarantees first+last are always shown)
    n_min = mdates.date2num(d_min)
    n_max = mdates.date2num(d_max)
    tick_nums = np.linspace(n_min, n_max, 4)
    ax.set_xticks(tick_nums)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(d_min, d_max)

    ax.set_title(bot, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Daily edits (7d avg)", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Only show language legend if bot appears in multiple languages
    if len(langs_for_bot) > 1:
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7)

fig.suptitle(
    "Blocked Bots — Daily Edit Volume Over 2025\n"
    "One panel per blocked bot  |  lines = language editions  |  7-day rolling avg",
    fontsize=13, fontweight="bold",
)
fig.autofmt_xdate(rotation=30)
fig.tight_layout()

out = OUTPUT_DIR / "blocked_bots_timeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out.name}")

"""
DataFest 2026 — Step 10: Potential Bots Analysis
Topic: Bots as Wikipedia Editors

Loads potential_bot_usernames_unfiltered.json and extracts
all_bot_cluster_users (5,190 accounts identified as bot-like via
clustering). Compares them against confirmed bots and regular humans
across all visualizations from 05_visualizations.py and
07_temporal_visualizations.py.

Three groups:
  bots           — is_bot == True in accounts_master
  potential_bots — user_text in all_bot_cluster_users AND is_bot == False
  humans         — is_bot == False AND NOT in potential_bot_set

Static figures   → potential_bots_01_*  …  potential_bots_12_*
Temporal figures → potential_bots_T1_*  …  potential_bots_T6_*

NOTE: The temporal section loads all 10 raw edit JSON-GZ files.
      Allow 10–20 minutes on first run.

Run: uv run bot_analysis_jona/10_potential_bots_analysis.py
"""

import json
import time
import datetime

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR       = Path(__file__).parent.parent / "data"
EDITS_DIR           = BASE_DATA_DIR / "data_extracted" / "edit_types"
POTENTIAL_BOTS_JSON = BASE_DATA_DIR / "other_data" / "bot_usernames.json"
ACCOUNTS_INPUT      = Path(__file__).parent / "accounts_master.ndjson"
TEMPORAL_INPUT      = Path(__file__).parent / "temporal_daily.ndjson"
HOURLY_INPUT        = Path(__file__).parent / "temporal_hourly.ndjson"
OUTPUT_DIR          = Path(__file__).parent / "figures" / "potential_bot_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES      = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
LANG_LABELS    = {
    "ar": "Arabic", "de": "German",  "en": "English", "es": "Spanish",
    "fr": "French", "it": "Italian", "nl": "Dutch",   "pl": "Polish",
    "ru": "Russian","sv": "Swedish",
}
TIMESTAMP_FMT  = "%Y-%m-%dT%H:%M:%S%.3fZ"
ROLLING_WINDOW = 3

EDIT_TYPES  = ["total_edits", "ref_edits", "tmpl_edits", "link_edits", "revert_edits"]
EDIT_LABELS = {
    "total_edits":  "All edits",
    "ref_edits":    "References",
    "tmpl_edits":   "Templates",
    "link_edits":   "Links",
    "revert_edits": "Reverts",
}

BOT_COLOR       = "#EE8019"   # Wikipedia orange — confirmed bots
HUMAN_COLOR     = "#71D1B3"   # Wikipedia green  — humans
POTENTIAL_COLOR = "#5748B5"   # Wikipedia purple — potential bots

# Minimum active editing days for the filtered T9 24h-rhythm figure
ACTIVE_DAYS_MIN_FILTER = 200

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "legend.framealpha": 0.8,
})


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> figures/{name}.png")


def rolling_avg(arr: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    return np.convolve(arr, np.ones(window) / window, mode="same")


def to_dates(date_series: pl.Series) -> list[datetime.date]:
    return date_series.to_list()


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── LOAD POTENTIAL BOT SET ────────────────────────────────────────────────────
section("LOAD POTENTIAL BOTS")
with open(POTENTIAL_BOTS_JSON, encoding="utf-8") as f:
    raw_json = json.load(f)

potential_bot_set = set(raw_json["all_bot_cluster_users"])
print(f"  {len(potential_bot_set):,} unique potential bot usernames")

# ── LOAD ACCOUNTS MASTER ──────────────────────────────────────────────────────
section("LOAD ACCOUNTS MASTER")
if not ACCOUNTS_INPUT.exists():
    raise SystemExit(f"ERROR: {ACCOUNTS_INPUT.name} not found — run 03_build_accounts.py first.")

df = pl.read_ndjson(ACCOUNTS_INPUT)
print(f"  {df.shape[0]:,} accounts loaded")

df = df.with_columns(
    pl.col("user_text").is_in(list(potential_bot_set)).alias("_is_potential")
)

bots           = df.filter(pl.col("is_bot") == True)
potential_bots = df.filter((pl.col("is_bot") == False) & (pl.col("_is_potential") == True))
humans         = df.filter((pl.col("is_bot") == False) & (pl.col("_is_potential") == False))

n_bots      = len(bots)
n_potential = len(potential_bots)
n_humans    = len(humans)
print(f"  Confirmed bots:  {n_bots:,}")
print(f"  Potential bots:  {n_potential:,}  ({len(potential_bot_set) - n_potential:,} in JSON but not in accounts_master)")
print(f"  Humans:          {n_humans:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURES  (05 equivalents)
# ══════════════════════════════════════════════════════════════════════════════

# ── FIG 01 — Behavioral Fingerprint ──────────────────────────────────────────
section("FIG 01: Behavioral fingerprint")

active_h  = humans.filter(pl.col("edit_velocity").is_not_null())
active_b  = bots.filter(pl.col("edit_velocity").is_not_null())
active_pb = potential_bots.filter(pl.col("edit_velocity").is_not_null())

hv = np.log1p(active_h["edit_velocity"].to_numpy())
hw = np.log1p(active_h["avg_words_changed"].fill_null(0).to_numpy())
bv = np.log1p(active_b["edit_velocity"].to_numpy())
bw = np.log1p(active_b["avg_words_changed"].fill_null(0).to_numpy())
pv = np.log1p(active_pb["edit_velocity"].to_numpy())
pw = np.log1p(active_pb["avg_words_changed"].fill_null(0).to_numpy())

fig, ax = plt.subplots(figsize=(11, 6))
hb = ax.hexbin(hv, hw, gridsize=60, cmap="Blues", mincnt=1, alpha=0.85,
               linewidths=0, norm=LogNorm())
plt.colorbar(hb, ax=ax, label="Human accounts (log density)")

ax.scatter(pv, pw, color=POTENTIAL_COLOR, s=15, zorder=4, alpha=0.6,
           label=f"Potential bots (n={n_potential:,})", edgecolors="none")
ax.scatter(bv, bw, color=BOT_COLOR, s=50, zorder=5,
           label=f"Confirmed bots (n={n_bots})", edgecolors="white", linewidths=0.4)


tick_vals = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals)); ax.set_xticklabels(tick_vals)
ax.set_yticks(np.log1p(tick_vals)); ax.set_yticklabels(tick_vals)
ax.set_xlabel("Edit velocity  (edits per day)")
ax.set_ylabel("Avg. words changed per edit")
ax.set_title("Potential bots cluster near confirmed bots in edit speed")
ax.legend(loc="upper right")
save(fig, "potential_bots_01_behavioral_fingerprint")


# ── FIG 02 — Primary edit type ────────────────────────────────────────────────
section("FIG 02: Primary edit type breakdown")

EDIT_TYPES_02 = ["text", "reference", "template", "links"]

def type_pcts(frame: pl.DataFrame) -> list[float]:
    counts = frame.group_by("primary_edit_type").agg(pl.len().alias("n"))
    total  = len(frame)
    return [
        counts.filter(pl.col("primary_edit_type") == t)["n"].sum() / total * 100
        if t in counts["primary_edit_type"].to_list() else 0.0
        for t in EDIT_TYPES_02
    ]

bot_pcts   = type_pcts(bots)
pb_pcts    = type_pcts(potential_bots)
human_pcts = type_pcts(humans)

x = np.arange(len(EDIT_TYPES_02))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
bars_b = ax.bar(x - w,  bot_pcts,   w, label="Confirmed bots",  color=BOT_COLOR,       alpha=0.9)
bars_p = ax.bar(x,      pb_pcts,    w, label="Potential bots",  color=POTENTIAL_COLOR,  alpha=0.9)
bars_h = ax.bar(x + w,  human_pcts, w, label="Humans",          color=HUMAN_COLOR,      alpha=0.9)

for bar in (*bars_b, *bars_p, *bars_h):
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in EDIT_TYPES_02])
ax.set_ylabel("% of accounts")
ax.set_title("Potential bots favour the same edit types as confirmed bots")
ax.legend()
ax.set_ylim(0, max(max(bot_pcts), max(pb_pcts), max(human_pcts)) * 1.18)
save(fig, "potential_bots_02_primary_edit_type")


# ── FIG 03 — Edit velocity distribution ───────────────────────────────────────
section("FIG 03: Edit velocity distribution")

bv_vals = bots.filter(pl.col("total_edits") >= 5)["edit_velocity"].drop_nulls().to_numpy()
pb_vals = potential_bots.filter(pl.col("total_edits") >= 5)["edit_velocity"].drop_nulls().to_numpy()
hv_vals = humans.filter(pl.col("total_edits") >= 5)["edit_velocity"].drop_nulls().to_numpy()
bv_vals = bv_vals[bv_vals > 0]
pb_vals = pb_vals[pb_vals > 0]
hv_vals = hv_vals[hv_vals > 0]

bins = np.logspace(
    np.log10(0.01),
    np.log10(max(bv_vals.max(), pb_vals.max(), hv_vals.max()) * 1.1),
    60,
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(hv_vals, bins=bins, color=HUMAN_COLOR,     alpha=0.55, label="Humans",         density=True)
ax.hist(pb_vals, bins=bins, color=POTENTIAL_COLOR, alpha=0.65, label="Potential bots", density=True)
ax.hist(bv_vals, bins=bins, color=BOT_COLOR,       alpha=0.75, label="Confirmed bots", density=True)
ax.set_xscale("log")
ax.set_xlabel("Edit velocity  (edits per day, log scale)")
ax.set_ylabel("Density")
ax.set_title("Potential bots edit significantly faster than humans")
ax.legend()
save(fig, "potential_bots_03_velocity_distribution")


# ── FIG 04 — Lorenz curve ─────────────────────────────────────────────────────
section("FIG 04: Lorenz curve")

def lorenz(values: np.ndarray):
    s   = np.sort(values)
    cum = np.cumsum(s)
    return np.linspace(0, 1, len(s) + 1), np.concatenate([[0], cum / cum[-1]])

bot_x,   bot_y   = lorenz(bots["total_edits"].to_numpy())
pb_x,    pb_y    = lorenz(potential_bots["total_edits"].to_numpy())
human_x, human_y = lorenz(humans["total_edits"].to_numpy())

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect equality")
ax.plot(human_x, human_y, color=HUMAN_COLOR,     lw=2, label="Humans")
ax.plot(pb_x,    pb_y,    color=POTENTIAL_COLOR,  lw=2, label="Potential bots")
ax.plot(bot_x,   bot_y,   color=BOT_COLOR,        lw=2, label="Confirmed bots")
ax.set_xlabel("Cumulative fraction of accounts (least → most active)")
ax.set_ylabel("Cumulative fraction of total edits")
ax.set_title("Edits are most concentrated among bots; potential bots fall between")
ax.legend(loc="upper left")
save(fig, "potential_bots_04_lorenz_curve")


# ── FIG 05 — Multilingual reach ───────────────────────────────────────────────
section("FIG 05: Multilingual reach")

def lang_dist(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df_ = frame.group_by("language_count").agg(pl.len().alias("n")).sort("language_count")
    return df_["language_count"].to_numpy(), df_["n"].to_numpy()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

for ax, frame, color, title, use_pct in [
    (ax1, bots,           BOT_COLOR,       "Confirmed bots\n(absolute count)",      False),
    (ax2, potential_bots, POTENTIAL_COLOR,  "Potential bots\n(absolute count)",      False),
    (ax3, humans,         HUMAN_COLOR,      "Humans\n(% of group, log scale)",       True),
]:
    lc_vals, n_vals = lang_dist(frame)
    if use_pct:
        vals = n_vals / n_vals.sum() * 100
        ax.bar(lc_vals, vals, color=color, alpha=0.9, width=0.7)
        ax.set_yscale("log")
        ax.set_ylabel("% of accounts (log scale)")
    else:
        bars = ax.bar(lc_vals, n_vals, color=color, alpha=0.9, width=0.7)
        for bar, v in zip(bars, n_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Number of accounts")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Languages active in")
    ax.set_title(title)

fig.suptitle("Potential bots are active across more languages than humans",
             fontsize=13, fontweight="bold", y=1.02)
save(fig, "potential_bots_05_multilingual_reach")


# ── FIG 06 — Top 20 potential bots table ──────────────────────────────────────
section("FIG 06: Top 20 potential bots table")

top20 = (
    potential_bots
    .filter(pl.col("user_text").is_not_null())
    .sort("total_edits", descending=True)
    .head(20)
    .select([
        "user_text", "total_edits", "language_count",
        "primary_edit_type", "top_topic",
        "edit_velocity", "avg_words_changed",
        "revert_rate_pct", "reverted_rate_pct",
    ])
)

cols = ["Username", "Edits", "Langs", "Edit type", "Top topic",
        "Vel./day", "Avg words\nchanged", "Revert\n%", "Reverted\n%"]
rows = []
for r in top20.iter_rows(named=True):
    rows.append([
        r["user_text"] or "anon",
        f"{r['total_edits']:,}",
        str(r["language_count"]),
        r["primary_edit_type"] or "—",
        r["top_topic"] or "—",
        f"{r['edit_velocity']:.1f}" if r["edit_velocity"] is not None else "—",
        f"{r['avg_words_changed']:.1f}" if r["avg_words_changed"] is not None else "—",
        f"{r['revert_rate_pct']:.1f}%" if r["revert_rate_pct"] is not None else "—",
        f"{r['reverted_rate_pct']:.1f}%" if r["reverted_rate_pct"] is not None else "—",
    ])

fig, ax = plt.subplots(figsize=(16, 7))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.45)
for col_idx in range(len(cols)):
    tbl[0, col_idx].set_facecolor(POTENTIAL_COLOR)
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")
for row_idx in range(1, len(rows) + 1):
    color = "#EDEAF8" if row_idx % 2 == 0 else "white"
    for col_idx in range(len(cols)):
        tbl[row_idx, col_idx].set_facecolor(color)
ax.set_title("Top 20 Most Active Potential Bots",
             fontsize=13, fontweight="bold", pad=12, loc="left")
save(fig, "potential_bots_06_top20_potential_bots_table")


# ── FIG 07 — Topic distribution ───────────────────────────────────────────────
section("FIG 07: Topic distribution")

def topic_pcts(frame: pl.DataFrame) -> dict[str, float]:
    sub = frame.filter(pl.col("top_topic").is_not_null())
    total = sub.shape[0]
    if total == 0:
        return {}
    counts = sub.group_by("top_topic").agg(pl.len().alias("n")).sort("n", descending=True)
    return {r["top_topic"]: r["n"] / total * 100 for r in counts.iter_rows(named=True)}

bot_topics   = topic_pcts(bots)
pb_topics    = topic_pcts(potential_bots)
human_topics = topic_pcts(humans)

all_topics = sorted(set(bot_topics) | set(pb_topics) | set(human_topics))
b_vals = [bot_topics.get(t, 0)   for t in all_topics]
p_vals = [pb_topics.get(t, 0)    for t in all_topics]
h_vals = [human_topics.get(t, 0) for t in all_topics]
order  = sorted(range(len(all_topics)),
                key=lambda i: (b_vals[i] + p_vals[i] + h_vals[i]) / 3, reverse=True)
all_topics = [all_topics[i] for i in order]
b_vals     = [b_vals[i]     for i in order]
p_vals     = [p_vals[i]     for i in order]
h_vals     = [h_vals[i]     for i in order]

y = np.arange(len(all_topics))
w = 0.28
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(y + w,  b_vals, w, color=BOT_COLOR,       alpha=0.9, label="Confirmed bots")
ax.barh(y,      p_vals, w, color=POTENTIAL_COLOR,  alpha=0.9, label="Potential bots")
ax.barh(y - w,  h_vals, w, color=HUMAN_COLOR,      alpha=0.9, label="Humans")
ax.set_yticks(y)
ax.set_yticklabels(all_topics)
ax.set_xlabel("% of accounts whose most-edited topic is this category")
ax.set_title("Potential bots concentrate on fewer topic categories than humans")
ax.legend()
save(fig, "potential_bots_07_topic_distribution")


# ── FIG 08 — Reverted rate ────────────────────────────────────────────────────
section("FIG 08: Reverted rate comparison")

min_edits = 10
b_rev  = bots.filter(pl.col("total_edits") >= min_edits)["reverted_rate_pct"].drop_nulls().to_numpy()
pb_rev = potential_bots.filter(pl.col("total_edits") >= min_edits)["reverted_rate_pct"].drop_nulls().to_numpy()
h_rev  = humans.filter(pl.col("total_edits") >= min_edits)["reverted_rate_pct"].drop_nulls().to_numpy()

bins_labels = ["0%", "0–5%", "5–20%", ">20%"]

def bin_reverted(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(4)
    return np.array([
        (arr == 0).sum(),
        ((arr > 0) & (arr <= 5)).sum(),
        ((arr > 5) & (arr <= 20)).sum(),
        (arr > 20).sum(),
    ]) / len(arr) * 100

b_binned  = bin_reverted(b_rev)
pb_binned = bin_reverted(pb_rev)
h_binned  = bin_reverted(h_rev)

x = np.arange(len(bins_labels))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
bars_b = ax.bar(x - w,  b_binned,  w, color=BOT_COLOR,       alpha=0.9, label=f"Confirmed bots (n={len(b_rev):,})")
bars_p = ax.bar(x,      pb_binned, w, color=POTENTIAL_COLOR,  alpha=0.9, label=f"Potential bots (n={len(pb_rev):,})")
bars_h = ax.bar(x + w,  h_binned,  w, color=HUMAN_COLOR,      alpha=0.9, label=f"Humans (n={len(h_rev):,})")

for bar in (*bars_b, *bars_p, *bars_h):
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(bins_labels)
ax.set_xlabel("Reverted rate (% of edits later undone by others)")
ax.set_ylabel("% of accounts in this bucket")
ax.set_title("Potential bots are reverted more often than humans")
ax.legend()
save(fig, "potential_bots_08_reverted_rate")


# ── FIG 09 — Where do potential bots fall on the behavioral scatter? ──────────
section("FIG 09: Behavioral scatter — potential bots overlay")

active_h_full = humans.filter(pl.col("edit_velocity").is_not_null())
active_pb9    = potential_bots.filter(pl.col("edit_velocity").is_not_null())
active_b9     = bots.filter(pl.col("edit_velocity").is_not_null())

fig, ax = plt.subplots(figsize=(10, 6))
hb = ax.hexbin(
    np.log1p(active_h_full["edit_velocity"].to_numpy()),
    np.log1p(active_h_full["avg_words_changed"].fill_null(0).to_numpy()),
    gridsize=55, cmap="Blues", mincnt=1, alpha=0.75, linewidths=0, norm=LogNorm(),
)
plt.colorbar(hb, ax=ax, label="Human accounts (log density)")
ax.scatter(
    np.log1p(active_pb9["edit_velocity"].to_numpy()),
    np.log1p(active_pb9["avg_words_changed"].fill_null(0).to_numpy()),
    color=POTENTIAL_COLOR, s=18, zorder=4, alpha=0.7,
    label=f"Potential bots (n={n_potential:,})", edgecolors="none",
)
ax.scatter(
    np.log1p(active_b9["edit_velocity"].to_numpy()),
    np.log1p(active_b9["avg_words_changed"].fill_null(0).to_numpy()),
    color=BOT_COLOR, s=55, zorder=5, marker="^",
    label=f"Confirmed bots (n={n_bots})", edgecolors="white", linewidths=0.4,
)
tick_vals = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals)); ax.set_xticklabels(tick_vals)
ax.set_yticks(np.log1p(tick_vals)); ax.set_yticklabels(tick_vals)
ax.set_xlabel("Edit velocity  (edits per day)")
ax.set_ylabel("Avg. words changed per edit")
ax.set_title("Potential bots sit in high-velocity, low-text territory")
ax.legend(loc="upper right")
save(fig, "potential_bots_09_behavioral_scatter")


# ── FIG 10 — Velocity vs active days ─────────────────────────────────────────
section("FIG 10: Edit velocity vs active days")

_filt = (
    (pl.col("edit_velocity_active_days").is_not_null()) &
    (pl.col("active_days").is_not_null()) &
    (pl.col("edit_velocity_active_days") > 0) &
    (pl.col("active_days") > 0)
)
ph   = humans.filter(_filt)
ppb  = potential_bots.filter(_filt)
pb_b = bots.filter(_filt)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    np.log1p(ph["edit_velocity_active_days"].to_numpy()), ph["active_days"].to_numpy(),
    color=HUMAN_COLOR, s=8, zorder=2, alpha=0.4, linewidths=0,
    label=f"Humans (n={len(ph):,})",
)
ax.scatter(
    np.log1p(ppb["edit_velocity_active_days"].to_numpy()), ppb["active_days"].to_numpy(),
    color=POTENTIAL_COLOR, s=18, zorder=3, alpha=0.6, linewidths=0,
    label=f"Potential bots (n={len(ppb):,})",
)
ax.scatter(
    np.log1p(pb_b["edit_velocity_active_days"].to_numpy()), pb_b["active_days"].to_numpy(),
    color=BOT_COLOR, s=60, zorder=5, marker="^",
    label=f"Confirmed bots (n={len(pb_b):,})",
    edgecolors="white", linewidths=0.5,
)
tick_vals_vel = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals_vel))
ax.set_xticklabels(tick_vals_vel)
ax.set_xlabel("Edit velocity (edits per active day)")
ax.set_ylabel("Active editing days (unique calendar days)")
ax.set_title("Highly active potential bots match confirmed bot patterns")
ax.legend(loc="upper right", markerscale=1.5)
save(fig, "potential_bots_10_velocity_vs_active_days")


# ── FIG 11 — Heavily reverted accounts ───────────────────────────────────────
section("FIG 11: Heavily reverted accounts")

plot_h  = humans.filter(_filt)
plot_pb = potential_bots.filter(_filt)
plot_b  = bots.filter(_filt)

# Handle accounts_master that may not have completely_reverted column
if "completely_reverted" in df.columns:
    _heavily = pl.col("completely_reverted") | (pl.col("reverted_rate_pct") >= 80)
else:
    _heavily = pl.col("reverted_rate_pct") >= 80

heavy_h  = plot_h.filter(_heavily)
heavy_pb = plot_pb.filter(_heavily)
heavy_b  = plot_b.filter(_heavily)

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(
    np.log1p(plot_h["edit_velocity_active_days"].to_numpy()), plot_h["active_days"].to_numpy(),
    color=HUMAN_COLOR, s=8, zorder=1, alpha=0.4, linewidths=0,
    label=f"Humans (n={len(plot_h):,})",
)
ax.scatter(
    np.log1p(plot_pb["edit_velocity_active_days"].to_numpy()), plot_pb["active_days"].to_numpy(),
    color=POTENTIAL_COLOR, s=14, zorder=2, alpha=0.5, linewidths=0,
    label=f"Potential bots (n={len(plot_pb):,})",
)
ax.scatter(
    np.log1p(plot_b["edit_velocity_active_days"].to_numpy()), plot_b["active_days"].to_numpy(),
    color=BOT_COLOR, s=55, zorder=3, alpha=0.6, marker="^",
    edgecolors="white", linewidths=0.4,
    label=f"Confirmed bots (n={len(plot_b):,})",
)
ax.scatter(
    np.log1p(heavy_h["edit_velocity_active_days"].to_numpy()), heavy_h["active_days"].to_numpy(),
    color=HUMAN_COLOR, s=35, zorder=5, edgecolors="#3a9e87", linewidths=0.6,
    label=f"Humans ≥80% reverted (n={len(heavy_h):,})",
)
ax.scatter(
    np.log1p(heavy_pb["edit_velocity_active_days"].to_numpy()), heavy_pb["active_days"].to_numpy(),
    color=POTENTIAL_COLOR, s=35, zorder=6, edgecolors="#3a2f8a", linewidths=0.6,
    label=f"Potential bots ≥80% reverted (n={len(heavy_pb):,})",
)
ax.scatter(
    np.log1p(heavy_b["edit_velocity_active_days"].to_numpy()), heavy_b["active_days"].to_numpy(),
    color=BOT_COLOR, s=80, zorder=7, marker="^",
    edgecolors="white", linewidths=0.6,
    label=f"Confirmed bots ≥80% reverted (n={len(heavy_b):,})",
)

tick_vals_vel = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals_vel))
ax.set_xticklabels(tick_vals_vel)
ax.set_xlabel("Edit velocity (edits per active day)")
ax.set_ylabel("Active editing days")
ax.set_title("Heavily reverted potential bots cluster at high velocity")
ax.legend(loc="upper right", markerscale=1.2, fontsize=8)
save(fig, "potential_bots_11_heavily_reverted")


# ── FIG 14 — Velocity vs Active Days: potential bots as density ───────────────
section("FIG 14: Velocity vs active days — potential bot density")

from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm as _LogNorm

def _kde_contours(ax, x, y, color, levels=(0.50, 0.90), lw=1.4):
    """Overlay KDE contour lines at given probability mass levels."""
    if len(x) < 20:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xi = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
        yi = np.linspace(max(0, y.min() - 5), y.max() + 5, 200)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        # Convert levels from probability mass to density thresholds
        z_flat = np.sort(Zi.ravel())[::-1]
        cumsum  = np.cumsum(z_flat) / z_flat.sum()
        thresholds = [z_flat[np.searchsorted(cumsum, lv)] for lv in levels]
        ax.contour(Xi, Yi, Zi, levels=sorted(thresholds), colors=[color],
                   linewidths=lw, linestyles=["--", "-"], alpha=0.85, zorder=6)
    except Exception:
        pass

_x_pb14  = np.log1p(ppb["edit_velocity_active_days"].to_numpy())
_y_pb14  = ppb["active_days"].to_numpy().astype(float)
_x_h14   = np.log1p(ph["edit_velocity_active_days"].to_numpy())
_y_h14   = ph["active_days"].to_numpy().astype(float)
_x_b14   = np.log1p(pb_b["edit_velocity_active_days"].to_numpy())
_y_b14   = pb_b["active_days"].to_numpy().astype(float)

fig, ax = plt.subplots(figsize=(11, 6))

# Humans — light hexbin background
hb_h = ax.hexbin(_x_h14, _y_h14, gridsize=50, cmap="Greens",
                  norm=_LogNorm(vmin=1), mincnt=1, alpha=0.5,
                  linewidths=0, zorder=1)

# Potential bots — hexbin foreground with LogNorm (outliers stay visible)
hb_pb = ax.hexbin(_x_pb14, _y_pb14, gridsize=50, cmap="Purples",
                   norm=_LogNorm(vmin=1), mincnt=1, alpha=0.85,
                   linewidths=0, zorder=2)
plt.colorbar(hb_pb, ax=ax, label="Potential bots (log count per cell)")

# KDE contour lines — 50% and 90% density mass
_kde_contours(ax, _x_pb14, _y_pb14, color=POTENTIAL_COLOR, levels=(0.50, 0.90))

# Confirmed bots — individual points (few enough to show)
ax.scatter(_x_b14, _y_b14, color=BOT_COLOR, s=60, zorder=7, marker="^",
           edgecolors="white", linewidths=0.5,
           label=f"Confirmed bots (n={len(pb_b):,})")

tick_vals_vel = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals_vel))
ax.set_xticklabels(tick_vals_vel)
ax.set_xlabel("Edit velocity (edits per active day)")
ax.set_ylabel("Active editing days (unique calendar days)")
ax.set_title("Potential bot density peaks at moderate velocity — a long tail of highly active accounts")

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color=POTENTIAL_COLOR, lw=1.5, ls="-",  label="Potential bots — 90% density"),
    Line2D([0], [0], color=POTENTIAL_COLOR, lw=1.5, ls="--", label="Potential bots — 50% density"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=BOT_COLOR,
           markersize=8, label=f"Confirmed bots (n={len(pb_b):,})"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=9)
save(fig, "potential_bots_14_velocity_active_days_density")


# ── FIG 15 — Heavily reverted: potential bots as density ─────────────────────
section("FIG 15: Heavily reverted — potential bot density")

_x_pb15   = np.log1p(plot_pb["edit_velocity_active_days"].to_numpy())
_y_pb15   = plot_pb["active_days"].to_numpy().astype(float)
_x_hpb15  = np.log1p(heavy_pb["edit_velocity_active_days"].to_numpy())
_y_hpb15  = heavy_pb["active_days"].to_numpy().astype(float)
_x_hb15   = np.log1p(heavy_b["edit_velocity_active_days"].to_numpy())
_y_hb15   = heavy_b["active_days"].to_numpy().astype(float)
_x_hh15   = np.log1p(heavy_h["edit_velocity_active_days"].to_numpy())
_y_hh15   = heavy_h["active_days"].to_numpy().astype(float)

fig, ax = plt.subplots(figsize=(11, 6))

# All potential bots — muted hexbin baseline
ax.hexbin(_x_pb15, _y_pb15, gridsize=50, cmap="Purples",
          norm=_LogNorm(vmin=1), mincnt=1, alpha=0.35,
          linewidths=0, zorder=1)

# Heavily reverted potential bots — vivid hexbin on top
hb_hpb = ax.hexbin(_x_hpb15, _y_hpb15, gridsize=40, cmap="RdPu",
                    norm=_LogNorm(vmin=1), mincnt=1, alpha=0.9,
                    linewidths=0, zorder=2)
plt.colorbar(hb_hpb, ax=ax, label="Heavily reverted potential bots (log count)")

# KDE contours for heavily reverted potential bots
_kde_contours(ax, _x_hpb15, _y_hpb15, color=POTENTIAL_COLOR, levels=(0.50, 0.90))

# Heavily reverted humans — small scatter for comparison
if len(_x_hh15) > 0:
    ax.scatter(_x_hh15, _y_hh15, color=HUMAN_COLOR, s=12, zorder=5,
               alpha=0.6, linewidths=0,
               label=f"Humans ≥80% reverted (n={len(heavy_h):,})")

# Heavily reverted confirmed bots — individual scatter
if len(_x_hb15) > 0:
    ax.scatter(_x_hb15, _y_hb15, color=BOT_COLOR, s=70, zorder=6, marker="^",
               edgecolors="white", linewidths=0.5,
               label=f"Confirmed bots ≥80% reverted (n={len(heavy_b):,})")

tick_vals_vel = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals_vel))
ax.set_xticklabels(tick_vals_vel)
ax.set_xlabel("Edit velocity (edits per active day)")
ax.set_ylabel("Active editing days")
ax.set_title("Heavily reverted potential bots concentrate at high velocity")

legend_els15 = [
    Line2D([0], [0], color=POTENTIAL_COLOR, lw=1.5, ls="-",  label="Heavily reverted pb — 90% density"),
    Line2D([0], [0], color=POTENTIAL_COLOR, lw=1.5, ls="--", label="Heavily reverted pb — 50% density"),
]
handles15, labels15 = ax.get_legend_handles_labels()
ax.legend(handles=handles15 + legend_els15, loc="upper right", fontsize=9)
save(fig, "potential_bots_15_heavily_reverted_density")


# ── FIG 12 — Editing rhythm ───────────────────────────────────────────────────
section("FIG 12: Editing rhythm box plot")

b_gap  = bots.filter(pl.col("avg_hours_between_edits").is_not_null())["avg_hours_between_edits"].to_numpy()
pb_gap = potential_bots.filter(pl.col("avg_hours_between_edits").is_not_null())["avg_hours_between_edits"].to_numpy()
h_gap  = humans.filter(pl.col("avg_hours_between_edits").is_not_null())["avg_hours_between_edits"].to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(
    [np.log1p(h_gap), np.log1p(pb_gap), np.log1p(b_gap)],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker=".", markersize=2, alpha=0.25, linestyle="none"),
    widths=0.5,
)
bp["boxes"][0].set_facecolor(HUMAN_COLOR);     bp["boxes"][0].set_alpha(0.85)
bp["boxes"][1].set_facecolor(POTENTIAL_COLOR); bp["boxes"][1].set_alpha(0.85)
bp["boxes"][2].set_facecolor(BOT_COLOR);       bp["boxes"][2].set_alpha(0.85)

tick_vals = [0, 0.5, 2, 12, 48, 168, 720, 4320]
ax.set_yticks(np.log1p(tick_vals))
ax.set_yticklabels(["0", "30 min", "2 h", "12 h", "2 days", "1 week", "1 month", "6 months"])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([
    f"Humans\n(n={len(h_gap):,})",
    f"Potential bots\n(n={len(pb_gap):,})",
    f"Confirmed bots\n(n={len(b_gap):,})",
], fontsize=10)
ax.set_ylabel("Avg. time between consecutive edits (log scale)")
ax.set_title("Potential bots edit on much shorter intervals than humans")

def fmt_hours(h: float) -> str:
    if h < 1:    return f"{h*60:.0f} min"
    if h < 24:   return f"{h:.1f} h"
    if h < 168:  return f"{h/24:.1f} days"
    if h < 720:  return f"{h/168:.1f} weeks"
    return f"{h/720:.1f} months"

for x_pos, arr in [(1, h_gap), (2, pb_gap), (3, b_gap)]:
    if len(arr) == 0:
        continue
    med_val = float(np.median(arr))
    med_log = np.log1p(med_val)
    ax.scatter(x_pos, med_log, marker="D", s=50, color="white",
               edgecolors="#333333", linewidths=1.5, zorder=5)

ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "potential_bots_12_editing_rhythm")


# ── FIG 13 — Edit-rank dominance: composition + cumulative contribution ───────
section("FIG 13: Edit-rank dominance")

# Combine all accounts with a group label, sorted by total_edits descending
all_accounts = pl.concat([
    bots.select(["user_text", "total_edits"]).with_columns(pl.lit("Confirmed bots").alias("group")),
    potential_bots.select(["user_text", "total_edits"]).with_columns(pl.lit("Potential bots").alias("group")),
    humans.select(["user_text", "total_edits"]).with_columns(pl.lit("Humans").alias("group")),
]).sort("total_edits", descending=True)

total_accounts = len(all_accounts)
total_edits_all = float(all_accounts["total_edits"].sum())

# ── Left panel: stacked % composition at top-N thresholds ────────────────────
thresholds = [10, 50, 100, 500, 1_000, 5_000, 10_000, total_accounts]
threshold_labels = ["Top 10", "Top 50", "Top 100", "Top 500",
                    "Top 1k", "Top 5k", "Top 10k", "All"]

comp_h, comp_pb, comp_b = [], [], []
for n in thresholds:
    top_n = all_accounts.head(n)
    counts = top_n.group_by("group").agg(pl.len().alias("cnt"))
    cnt = {row["group"]: row["cnt"] for row in counts.iter_rows(named=True)}
    total_n = len(top_n)
    comp_h.append(cnt.get("Humans", 0) / total_n * 100)
    comp_pb.append(cnt.get("Potential bots", 0) / total_n * 100)
    comp_b.append(cnt.get("Confirmed bots", 0) / total_n * 100)

# ── Right panel: cumulative edit share as we walk down the ranked list ────────
groups_arr = all_accounts["group"].to_list()
edits_arr  = all_accounts["total_edits"].to_numpy().astype(float)

cum_total = np.cumsum(edits_arr)
cum_h  = np.cumsum(np.where(np.array(groups_arr) == "Humans",        edits_arr, 0))
cum_pb = np.cumsum(np.where(np.array(groups_arr) == "Potential bots", edits_arr, 0))
cum_b  = np.cumsum(np.where(np.array(groups_arr) == "Confirmed bots", edits_arr, 0))

pct_total = cum_total / total_edits_all * 100
pct_h     = cum_h  / total_edits_all * 100
pct_pb    = cum_pb / total_edits_all * 100
pct_b     = cum_b  / total_edits_all * 100

x_ranks = np.arange(1, total_accounts + 1)

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))

# Left: stacked bar
x = np.arange(len(thresholds))
bar_w = 0.55
ax_left.bar(x, comp_h,  bar_w, label="Humans",        color=HUMAN_COLOR,     alpha=0.88)
ax_left.bar(x, comp_pb, bar_w, bottom=comp_h,          label="Potential bots", color=POTENTIAL_COLOR, alpha=0.88)
ax_left.bar(x, comp_b,  bar_w,
            bottom=[h + p for h, p in zip(comp_h, comp_pb)],
            label="Confirmed bots", color=BOT_COLOR, alpha=0.88)

# Add % labels inside bars for potential bots (most interesting)
for i, (h, pb, b) in enumerate(zip(comp_h, comp_pb, comp_b)):
    if pb >= 3:
        ax_left.text(i, h + pb / 2, f"{pb:.1f}%", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
    if b >= 3:
        ax_left.text(i, h + pb + b / 2, f"{b:.1f}%", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")

ax_left.set_xticks(x)
ax_left.set_xticklabels(threshold_labels, fontsize=9)
ax_left.set_ylabel("Share of accounts in group (%)")
ax_left.set_ylim(0, 100)
ax_left.set_title("Potential bots are overrepresented among top editors")
ax_left.legend(loc="lower right", fontsize=9)
ax_left.spines[["top", "right"]].set_visible(False)

# Right: cumulative edit contribution (log x-axis for readability)
ax_right.plot(x_ranks, pct_total, color="#aaaaaa", lw=1.2, ls="--", label="All accounts combined")
ax_right.plot(x_ranks, pct_h,     color=HUMAN_COLOR,     lw=2.2, label="Humans")
ax_right.plot(x_ranks, pct_pb,    color=POTENTIAL_COLOR,  lw=2.2, label="Potential bots")
ax_right.plot(x_ranks, pct_b,     color=BOT_COLOR,        lw=2.2, label="Confirmed bots")

ax_right.set_xscale("log")
ax_right.set_xlabel("Account rank (1 = most edits, log scale)")
ax_right.set_ylabel("Cumulative % of all edits")
ax_right.set_ylim(0, 100)
ax_right.set_title("A handful of potential bots drive a disproportionate share of all edits")
ax_right.legend(loc="upper left", fontsize=9)
ax_right.spines[["top", "right"]].set_visible(False)

# Reference lines at 25 / 50 / 75 %
for pct_line in [25, 50, 75]:
    ax_right.axhline(pct_line, color="#dddddd", lw=0.8, ls=":")
    ax_right.text(x_ranks[-1] * 1.02, pct_line, f"{pct_line}%",
                  va="center", fontsize=7, color="#aaaaaa")

fig.suptitle(
    "Potential bots dominate the top of the edit leaderboard",
    fontsize=11,
)
fig.tight_layout()
save(fig, "potential_bots_13_edit_rank_dominance")


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SECTION — load raw edit data for potential bots
# ══════════════════════════════════════════════════════════════════════════════
section("TEMPORAL — loading raw edit data for potential bots")
print("  Loading all 10 language files (may take several minutes)...")

potential_bot_list = list(potential_bot_set)

# Subset of potential bots that meet the active-days threshold (for T9)
active_pb_list = (
    potential_bots
    .filter(pl.col("active_days") > ACTIVE_DAYS_MIN_FILTER)
    ["user_text"].to_list()
)
print(f"  Potential bots with active_days > {ACTIVE_DAYS_MIN_FILTER}: {len(active_pb_list):,}")

pb_daily_frames        = []
pb_hourly_frames       = []
pb_hourly_frames_active = []   # filtered to ACTIVE_DAYS_MIN_FILTER

for lang in LANGUAGES:
    edits_file = EDITS_DIR / f"{lang}wiki.json.gz"
    if not edits_file.exists():
        print(f"  {lang.upper()}: file not found, skipping")
        continue

    print(f"  {lang.upper()}: loading...", end=" ", flush=True)
    t0 = time.time()
    edits = pl.read_ndjson(edits_file)
    edits = edits.filter(pl.col("user_text").is_in(potential_bot_list))
    n = edits.shape[0]
    print(f"{n:,} edits from potential bots  ({time.time() - t0:.1f}s)")
    if n == 0:
        continue

    edits = edits.with_columns(
        pl.col("revision_timestamp")
        .str.to_datetime(TIMESTAMP_FMT, time_unit="ms")
        .alias("_ts")
    )
    edits = edits.with_columns(
        pl.col("_ts").dt.date().alias("date"),
        pl.col("_ts").dt.hour().alias("hour"),
    ).drop("_ts")
    edits = edits.with_columns(
        pl.col("edit_types_json").str.count_matches(r'\["Reference"').cast(pl.Int32).alias("n_ref"),
        pl.col("edit_types_json").str.count_matches(r'\["Template"') .cast(pl.Int32).alias("n_tmpl"),
        (
            pl.col("edit_types_json").str.count_matches(r'\["ExternalLink"') +
            pl.col("edit_types_json").str.count_matches(r'\["Wikilink"')
        ).cast(pl.Int32).alias("n_links"),
        (
            pl.col("revision_tags").list.contains("mw-undo") |
            pl.col("revision_tags").list.contains("mw-manual-revert")
        ).cast(pl.Int32).alias("n_revert"),
    )
    daily = (
        edits.group_by("date")
        .agg(
            pl.len()                            .alias("total_edits"),
            pl.col("user_text").n_unique()      .alias("unique_editors"),
            pl.col("n_ref").sum()               .alias("ref_edits"),
            pl.col("n_tmpl").sum()              .alias("tmpl_edits"),
            pl.col("n_links").sum()             .alias("link_edits"),
            pl.col("n_revert").sum()            .alias("revert_edits"),
        )
        .with_columns(pl.lit(lang).alias("language"))
        .sort("date")
    )
    pb_daily_frames.append(daily)

    hourly_pb = (
        edits.group_by("hour")
        .agg(pl.len().alias("total_edits"))
        .with_columns(pl.lit(lang).alias("language"))
        .sort("hour")
    )
    pb_hourly_frames.append(hourly_pb)

    # Filtered hourly — only potential bots with active_days > ACTIVE_DAYS_MIN_FILTER
    edits_active = edits.filter(pl.col("user_text").is_in(active_pb_list))
    if edits_active.shape[0] > 0:
        hourly_pb_active = (
            edits_active.group_by("hour")
            .agg(pl.len().alias("total_edits"))
            .with_columns(pl.lit(lang).alias("language"))
            .sort("hour")
        )
        pb_hourly_frames_active.append(hourly_pb_active)

if not pb_daily_frames:
    print("  WARNING: No temporal data found for potential bots — skipping temporal figures.")
    raise SystemExit(0)

pb_daily_all = pl.concat(pb_daily_frames).sort(["date", "language"])

# Global daily sum across all languages
pb_global = (
    pb_daily_all.group_by("date")
    .agg(
        pl.col("total_edits").sum(),
        pl.col("unique_editors").sum(),
        pl.col("ref_edits").sum(),
        pl.col("tmpl_edits").sum(),
        pl.col("link_edits").sum(),
        pl.col("revert_edits").sum(),
    )
    .sort("date")
)


# ── LOAD EXISTING TEMPORAL DATA ───────────────────────────────────────────────
section("LOAD EXISTING TEMPORAL DATA (temporal_daily.ndjson)")
if not TEMPORAL_INPUT.exists():
    raise SystemExit(f"ERROR: {TEMPORAL_INPUT.name} not found — run 06_temporal_aggregation.py first.")

df_temp = pl.read_ndjson(TEMPORAL_INPUT)
df_temp = df_temp.with_columns(pl.col("date").cast(pl.Date))
print(f"  {df_temp.shape[0]:,} rows loaded  |  {df_temp['date'].min()} to {df_temp['date'].max()}")

global_daily = (
    df_temp.group_by(["date", "is_bot"])
    .agg(
        pl.col("total_edits").sum(),
        pl.col("unique_editors").sum(),
        pl.col("ref_edits").sum(),
        pl.col("tmpl_edits").sum(),
        pl.col("link_edits").sum(),
        pl.col("revert_edits").sum(),
    )
    .sort(["date", "is_bot"])
)

bot_daily_t   = global_daily.filter(pl.col("is_bot") == True).sort("date")
human_daily_t = global_daily.filter(pl.col("is_bot") == False).sort("date")

# Align all three on shared dates
all_dates = sorted(
    set(bot_daily_t["date"].to_list()) &
    set(human_daily_t["date"].to_list()) &
    set(pb_global["date"].to_list())
)

def align(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("date").is_in(all_dates)).sort("date")

bot_g   = align(bot_daily_t)
human_g = align(human_daily_t)
pb_g    = align(pb_global)

bot_edits   = bot_g["total_edits"].to_numpy().astype(float)
human_edits = human_g["total_edits"].to_numpy().astype(float)
pb_edits    = pb_g["total_edits"].to_numpy().astype(float)
# total includes potential_bots (they are part of "human" in temporal_daily since is_bot=False)
total_edits     = bot_edits + human_edits
dates_t         = to_dates(bot_g["date"])


# ── FIG T1 — Daily edit volume ─────────────────────────────────────────────────
section("FIG T1: Daily edit volume")

# Subtract potential_bots from human to avoid double-counting in stacked area
true_human_edits = np.maximum(human_edits - pb_edits, 0)

fig, ax = plt.subplots(figsize=(13, 5))
ax.stackplot(
    dates_t, bot_edits, pb_edits, true_human_edits,
    labels=["Confirmed bots", "Potential bots", "Humans"],
    colors=[BOT_COLOR, POTENTIAL_COLOR, HUMAN_COLOR], alpha=0.35,
)
ax.plot(dates_t, rolling_avg(total_edits), color="#333333", lw=1.5,
        label=f"{ROLLING_WINDOW}-day avg (total)")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Edits per day")
ax.set_title("Potential bots contribute a consistent share of daily Wikipedia edits")
ax.legend(loc="upper right")
save(fig, "potential_bots_T1_daily_volume")


# ── FIG T2 — Potential bot share of total edits over time ─────────────────────
section("FIG T2: Share of edits over time")

pb_share    = np.where(total_edits > 0, pb_edits  / total_edits * 100, np.nan)
bot_share   = np.where(total_edits > 0, bot_edits / total_edits * 100, np.nan)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(dates_t, pb_share, color=POTENTIAL_COLOR, alpha=0.2, lw=0.8)
ax.plot(dates_t, rolling_avg(np.nan_to_num(pb_share)), color=POTENTIAL_COLOR, lw=2,
        label=f"Potential bots ({ROLLING_WINDOW}-day avg)")
ax.plot(dates_t, rolling_avg(np.nan_to_num(bot_share)), color=BOT_COLOR, lw=2, ls="--",
        label=f"Confirmed bots ({ROLLING_WINDOW}-day avg)")
ax.axhline(np.nanmean(pb_share), color=POTENTIAL_COLOR, lw=1, ls=":",
           label=f"Potential bots year avg: {np.nanmean(pb_share):.1f}%")
ax.axhline(np.nanmean(bot_share), color=BOT_COLOR, lw=1, ls=":",
           label=f"Confirmed bots year avg: {np.nanmean(bot_share):.1f}%")
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Share of all edits")
ax.set_title("Potential bots consistently outpace confirmed bots in edit share")
ax.legend(loc="upper right")
save(fig, "potential_bots_T2_global_share")


# ── FIG T3 — Potential bot share per language ──────────────────────────────────
section("FIG T3: Share per language")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes = axes.flatten()

for ax, lang in zip(axes, LANGUAGES):
    lang_bot   = df_temp.filter((pl.col("language") == lang) & (pl.col("is_bot") == True)).sort("date")
    lang_human = df_temp.filter((pl.col("language") == lang) & (pl.col("is_bot") == False)).sort("date")
    lang_pb    = pb_daily_all.filter(pl.col("language") == lang).sort("date")

    merged_bh = lang_bot.join(lang_human, on="date", suffix="_h")
    if merged_bh.shape[0] == 0:
        ax.set_title(LANG_LABELS[lang])
        continue

    merged = merged_bh.join(
        lang_pb.select(["date", "total_edits"]),
        on="date", how="left", suffix="_pb",
    )

    d         = to_dates(merged["date"])
    b_vals    = merged["total_edits"].to_numpy().astype(float)
    h_vals    = merged["total_edits_h"].to_numpy().astype(float)
    pb_vals_l = merged["total_edits_pb"].fill_null(0).to_numpy().astype(float)
    total_l   = b_vals + h_vals  # total (pb already counted in human)

    bot_share_l = np.where(total_l > 0, b_vals    / total_l * 100, np.nan)
    pb_share_l  = np.where(total_l > 0, pb_vals_l / total_l * 100, np.nan)

    ax.plot(d, rolling_avg(np.nan_to_num(bot_share_l)), color=BOT_COLOR,       lw=1.5, label="Confirmed bots")
    ax.plot(d, rolling_avg(np.nan_to_num(pb_share_l)),  color=POTENTIAL_COLOR,  lw=1.5, label="Potential bots")
    ax.axhline(np.nanmean(pb_share_l), color=POTENTIAL_COLOR, lw=0.8, ls="--")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title(f"{LANG_LABELS[lang]}  (pb avg {np.nanmean(pb_share_l):.1f}%)")

axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Potential bot share varies substantially across languages",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "potential_bots_T3_share_per_language")


# ── FIG T4 — Potential bot share by edit category ─────────────────────────────
section("FIG T4: Share by edit category")

# Join all three global aligned frames
merged_bh_g = bot_g.join(human_g, on="date", suffix="_h")
merged_all  = merged_bh_g.join(
    pb_g.select(["date"] + [c for c in EDIT_TYPES]),
    on="date", suffix="_pb",
)

fig, ax = plt.subplots(figsize=(13, 5))
colors_cat = ["#333333", "#2980B9", "#E74C3C", "#F39C12", "#E67E22"]

for col, label, color in zip(EDIT_TYPES, [EDIT_LABELS[e] for e in EDIT_TYPES], colors_cat):
    b_vals_c  = merged_all[col].to_numpy().astype(float)
    h_vals_c  = merged_all[f"{col}_h"].to_numpy().astype(float)
    pb_vals_c = merged_all[f"{col}_pb"].fill_null(0).to_numpy().astype(float)
    total_c   = b_vals_c + h_vals_c  # total (pb included in human)
    pb_sh     = np.where(total_c > 0, pb_vals_c / total_c * 100, np.nan)
    smooth    = rolling_avg(np.nan_to_num(pb_sh))
    ax.plot(to_dates(merged_all["date"]), smooth, lw=2, color=color,
            label=f"{label}  (avg {np.nanmean(pb_sh):.1f}%)")

ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Potential bot share of that edit type")
ax.set_title("Potential bots dominate template and reference edit categories")
ax.legend(loc="right", framealpha=0.9)
save(fig, "potential_bots_T4_share_by_category")


# ── FIG T5 — Heatmap: potential bot share by language × week ──────────────────
section("FIG T5: Heatmap — potential bot share by language × week")

pb_week = pb_daily_all.with_columns(
    pl.col("date").cast(pl.Date).dt.week().alias("week")
)
pb_weekly = (
    pb_week.group_by(["week", "language"])
    .agg(pl.col("total_edits").sum().alias("pb_edits"))
)

total_weekly = (
    df_temp
    .with_columns(pl.col("date").cast(pl.Date).dt.week().alias("week"))
    .group_by(["week", "language"])
    .agg(pl.col("total_edits").sum().alias("total_edits"))
)

joined_weekly = pb_weekly.join(total_weekly, on=["week", "language"], how="left")

weeks       = sorted(joined_weekly["week"].unique().to_list())
week_to_idx = {wk: idx for idx, wk in enumerate(weeks)}
matrix      = np.full((len(LANGUAGES), len(weeks)), np.nan)

for i, lang in enumerate(LANGUAGES):
    for j, wk in enumerate(weeks):
        sub = joined_weekly.filter(
            (pl.col("language") == lang) & (pl.col("week") == wk)
        )
        if sub.shape[0] == 0:
            continue
        pb_e = float(sub["pb_edits"].sum())
        tot  = float(sub["total_edits"].fill_null(0).sum())
        if tot > 0:
            matrix[i, j] = pb_e / tot * 100

fig, ax = plt.subplots(figsize=(16, 5))
non_nan = matrix[~np.isnan(matrix)]
vmax    = float(non_nan.max()) if len(non_nan) > 0 else 10
im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax, label="Potential bot share %", shrink=0.8)

ax.set_yticks(range(len(LANGUAGES)))
ax.set_yticklabels([LANG_LABELS[l] for l in LANGUAGES])

month_starts = {
    1: "Jan", 5: "Feb",  9: "Mar", 14: "Apr", 18: "May", 22: "Jun",
    27: "Jul", 31: "Aug", 36: "Sep", 40: "Oct", 44: "Nov", 49: "Dec",
}
tick_positions = [week_to_idx[wk] for wk in month_starts if wk in week_to_idx]
tick_labels    = [month_starts[wk] for wk in month_starts if wk in week_to_idx]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_xlabel("Week of 2025")
ax.set_title("Potential bot activity concentrates in specific language-week combinations")
save(fig, "potential_bots_T5_heatmap_language_week")


# ── FIG T6 — Unique editors per language per day ──────────────────────────────
section("FIG T6: Unique editors per language per day")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes = axes.flatten()

for ax, lang in zip(axes, LANGUAGES):
    lang_bot   = df_temp.filter((pl.col("language") == lang) & (pl.col("is_bot") == True)).sort("date")
    lang_human = df_temp.filter((pl.col("language") == lang) & (pl.col("is_bot") == False)).sort("date")
    lang_pb    = pb_daily_all.filter(pl.col("language") == lang).sort("date")

    if lang_human.shape[0] > 0:
        ax.plot(to_dates(lang_human["date"]),
                rolling_avg(lang_human["unique_editors"].to_numpy().astype(float)),
                color=HUMAN_COLOR, lw=1.5, label="Humans")
    if lang_pb.shape[0] > 0:
        ax.plot(to_dates(lang_pb["date"]),
                rolling_avg(lang_pb["unique_editors"].to_numpy().astype(float)),
                color=POTENTIAL_COLOR, lw=1.5, label="Potential bots")
    if lang_bot.shape[0] > 0:
        ax.plot(to_dates(lang_bot["date"]),
                rolling_avg(lang_bot["unique_editors"].to_numpy().astype(float)),
                color=BOT_COLOR, lw=1.5, label="Confirmed bots")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(LANG_LABELS[lang])

axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Potential bots form a stable fraction of daily active editors per language",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "potential_bots_T6_unique_editors_per_language")


# ── FIG T7 — 24-hour editing rhythm: all three groups ─────────────────────────
section("FIG T7: 24-hour editing rhythm")

if not HOURLY_INPUT.exists():
    print(f"  WARNING: {HOURLY_INPUT.name} not found — skipping T7. Re-run 06 first.")
elif pb_hourly_frames:
    df_hourly = pl.read_ndjson(HOURLY_INPUT)

    global_hourly = (
        df_hourly.group_by(["hour", "is_bot"])
        .agg(pl.col("total_edits").sum())
        .sort(["is_bot", "hour"])
    )
    bot_h = global_hourly.filter(pl.col("is_bot") == True).sort("hour")
    hum_h = global_hourly.filter(pl.col("is_bot") == False).sort("hour")

    pb_hourly_by_lang = pl.concat(pb_hourly_frames)   # keeps language column
    pb_hourly_by_lang_active = (
        pl.concat(pb_hourly_frames_active) if pb_hourly_frames_active else pl.DataFrame()
    )

    pb_hourly_all = (
        pb_hourly_by_lang
        .group_by("hour")
        .agg(pl.col("total_edits").sum())
        .sort("hour")
    )

    all_hours = np.arange(24)

    def to_hour_array(frame: pl.DataFrame, col: str = "total_edits") -> np.ndarray:
        hour_map = dict(zip(frame["hour"].to_list(),
                            frame[col].to_numpy().astype(float)))
        return np.array([hour_map.get(int(h), 0.0) for h in all_hours])

    def norm_pct(arr: np.ndarray) -> np.ndarray:
        total = arr.sum()
        return arr / total * 100 if total > 0 else arr

    bot_frac = norm_pct(to_hour_array(bot_h))
    hum_frac = norm_pct(to_hour_array(hum_h))
    pb_frac  = norm_pct(to_hour_array(pb_hourly_all))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade night hours (22:00–06:00 UTC)
    ax.axvspan(-0.5, 5.5,  alpha=0.06, color="navy", zorder=0)
    ax.axvspan(21.5, 23.5, alpha=0.06, color="navy", zorder=0)
    ax.axhline(100 / 24, color="#999999", lw=1.2, ls="--",
               label="Uniform distribution (4.17% / h)")

    ax.plot(all_hours, hum_frac, color=HUMAN_COLOR,     lw=2.5, marker="o", ms=4, label="Humans")
    ax.plot(all_hours, pb_frac,  color=POTENTIAL_COLOR,  lw=2.5, marker="o", ms=4, label="Potential bots")
    ax.plot(all_hours, bot_frac, color=BOT_COLOR,        lw=2.5, marker="o", ms=4, label="Confirmed bots")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("% of group's total annual edits")
    ax.set_title("Potential bots edit more uniformly around the clock than humans")
    ax.legend(loc="upper left")
    save(fig, "potential_bots_T7_24h_distribution")

    # ── T8 — per-language 24h editing rhythm (2×5 grid, 3 groups) ─────────────
    section("FIG T8 — 24h rhythm per language (3 groups)")

    LANG_LABELS_10 = {
        "ar": "Arabic", "de": "German", "en": "English",
        "es": "Spanish", "fr": "French", "it": "Italian",
        "nl": "Dutch", "pl": "Polish", "ru": "Russian", "sv": "Swedish",
    }

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.flatten()

    for ax, lang in zip(axes, LANGUAGES):
        # bot / human from df_hourly (confirmed bots, all humans)
        lang_h = df_hourly.filter(pl.col("language") == lang)
        bot_h_l = lang_h.filter(pl.col("is_bot") == True).sort("hour")
        hum_h_l = lang_h.filter(pl.col("is_bot") == False).sort("hour")

        # potential bots hourly for this language
        pb_h_l = pb_hourly_by_lang.filter(pl.col("language") == lang).sort("hour")

        # shade night
        ax.axvspan(-0.5, 5.5,  alpha=0.06, color="navy", zorder=0)
        ax.axvspan(21.5, 23.5, alpha=0.06, color="navy", zorder=0)
        ax.axhline(100 / 24, color="#cccccc", lw=0.8, ls="--")

        if hum_h_l.shape[0] > 0:
            ax.plot(all_hours, norm_pct(to_hour_array(hum_h_l)),
                    color=HUMAN_COLOR, lw=1.8, label="Humans")
        if pb_h_l.shape[0] > 0:
            ax.plot(all_hours, norm_pct(to_hour_array(pb_h_l)),
                    color=POTENTIAL_COLOR, lw=1.8, label="Potential bots")
        if bot_h_l.shape[0] > 0:
            ax.plot(all_hours, norm_pct(to_hour_array(bot_h_l)),
                    color=BOT_COLOR, lw=1.8, label="Confirmed bots")

        ax.set_xticks(range(0, 24, 6))
        ax.set_xticklabels(["0h", "6h", "12h", "18h"])
        ax.set_xlim(-0.5, 23.5)
        ax.set_title(LANG_LABELS_10.get(lang, lang))
        ax.tick_params(axis="both", labelsize=8)

    axes[0].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.set_ylabel("")
    axes[0].set_ylabel("% of group's edits", fontsize=8)
    axes[5].set_ylabel("% of group's edits", fontsize=8)

    fig.suptitle(
        "24h edit patterns differ clearly between groups across all languages",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "potential_bots_T8_24h_per_language")

    # ── T9 — same as T8 but potential bots filtered to active_days > threshold ─
    section(f"FIG T9 — 24h rhythm per language (potential bots active_days > {ACTIVE_DAYS_MIN_FILTER})")

    if pb_hourly_by_lang_active.is_empty():
        print(f"  No data for potential bots with active_days > {ACTIVE_DAYS_MIN_FILTER} — skipping T9.")
    else:
        n_active_pb = len(active_pb_list)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
        axes = axes.flatten()

        for ax, lang in zip(axes, LANGUAGES):
            lang_h = df_hourly.filter(pl.col("language") == lang)
            bot_h_l = lang_h.filter(pl.col("is_bot") == True).sort("hour")
            hum_h_l = lang_h.filter(pl.col("is_bot") == False).sort("hour")
            pb_h_l  = pb_hourly_by_lang_active.filter(pl.col("language") == lang).sort("hour")

            ax.axvspan(-0.5, 5.5,  alpha=0.06, color="navy", zorder=0)
            ax.axvspan(21.5, 23.5, alpha=0.06, color="navy", zorder=0)
            ax.axhline(100 / 24, color="#cccccc", lw=0.8, ls="--")

            if hum_h_l.shape[0] > 0:
                ax.plot(all_hours, norm_pct(to_hour_array(hum_h_l)),
                        color=HUMAN_COLOR, lw=1.8, label="Humans")
            if pb_h_l.shape[0] > 0:
                ax.plot(all_hours, norm_pct(to_hour_array(pb_h_l)),
                        color=POTENTIAL_COLOR, lw=1.8,
                        label=f"Potential bots (active >{ACTIVE_DAYS_MIN_FILTER}d)")
            if bot_h_l.shape[0] > 0:
                ax.plot(all_hours, norm_pct(to_hour_array(bot_h_l)),
                        color=BOT_COLOR, lw=1.8, label="Confirmed bots")

            ax.set_xticks(range(0, 24, 6))
            ax.set_xticklabels(["0h", "6h", "12h", "18h"])
            ax.set_xlim(-0.5, 23.5)
            ax.set_title(LANG_LABELS_10.get(lang, lang))
            ax.tick_params(axis="both", labelsize=8)

        axes[0].legend(fontsize=8, loc="upper left")
        for ax in axes:
            ax.set_ylabel("")
        axes[0].set_ylabel("% of group's edits", fontsize=8)
        axes[5].set_ylabel("% of group's edits", fontsize=8)

        fig.suptitle(
            f"Highly active potential bots (>{ACTIVE_DAYS_MIN_FILTER} active days) show the most bot-like 24h patterns",
            fontsize=10,
        )
        fig.tight_layout()
        save(fig, "potential_bots_T9_24h_per_language_active")


n_saved = len(list(OUTPUT_DIR.glob("*.png")))
print(f"\n\nAll done. {n_saved} figures saved to: {OUTPUT_DIR}")

"""
DataFest 2026 — Step 5: Analysis & Visualizations
Topic: Bots as Wikipedia Editors

Loads accounts_master.ndjson and produces 8 presentation-ready figures.
Each figure is saved individually to exploration/figures/ as a PNG (150 dpi)
so you can drag-and-drop them straight into your slide deck.

Requires: uv add matplotlib
Run:      uv run exploration/05_visualizations.py
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT      = Path(__file__).parent / "accounts_master.ndjson"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

BOT_COLOR     = "#E05C4B"   # red
HUMAN_COLOR   = "#4B8BE0"   # blue
SUSPECT_COLOR = "#F5A623"   # orange (bot-like humans)

# Global style — clean, presentation-ready
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


# ── LOAD ──────────────────────────────────────────────────────────────────────
if not INPUT.exists():
    raise SystemExit(f"ERROR: {INPUT.name} not found — run 03_build_accounts.py first.")

print("Loading accounts_master.ndjson...")
df = pl.read_ndjson(INPUT)

bots   = df.filter(pl.col("is_bot") == True)
humans = df.filter(pl.col("is_bot") == False)

n_bots   = len(bots)
n_humans = len(humans)
print(f"  {df.shape[0]:,} accounts  |  {n_bots:,} bots  |  {n_humans:,} humans\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Behavioral Fingerprint
# ──────────────────────────────────────────────────────────────────────────────
# The key question: can you tell a bot from a human just by how they edit?
#
# X = edit_velocity (edits per day) — how fast they work
# Y = avg_words_changed per edit   — how much text they touch each time
#
# Bots are fast (high velocity) and surgical (low word changes per edit).
# Humans are slower and write more text per edit.
# The two populations separate clearly into distinct clusters.
#
# Method: humans are shown as a 2D density map (hexbin) because there are
# 700k+ of them and individual dots would be unreadable. Bots are shown as
# individual dots since there are only ~300 of them.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 01: Behavioral fingerprint scatter...")

# No edit-count filter — include all accounts so no artificial velocity wall
# appears. Single-edit accounts naturally land at velocity=1 (one edit in one
# day), which is real data. LogNorm colour scale keeps sparse high-velocity
# regions visible even though the low-velocity mass is orders of magnitude denser.
from matplotlib.colors import LogNorm

active_bots   = bots.filter(pl.col("edit_velocity").is_not_null())
active_humans = humans.filter(pl.col("edit_velocity").is_not_null())

# log1p: log(x+1) so that zero-velocity values don't break the log scale
hv = np.log1p(active_humans["edit_velocity"].to_numpy())
hw = np.log1p(active_humans["avg_words_changed"].fill_null(0).to_numpy())
bv = np.log1p(active_bots["edit_velocity"].to_numpy())
bw = np.log1p(active_bots["avg_words_changed"].fill_null(0).to_numpy())

fig, ax = plt.subplots(figsize=(11, 6))

# Human density map — LogNorm makes both the dense low-velocity mass and the
# sparse high-velocity tail readable on the same colour scale
hb = ax.hexbin(hv, hw, gridsize=60, cmap="Blues", mincnt=1, alpha=0.85,
               linewidths=0, norm=LogNorm())
plt.colorbar(hb, ax=ax, label="Human accounts (log density)")

# Individual bot dots on top
ax.scatter(bv, bw, color=BOT_COLOR, s=50, zorder=5, label=f"Bots (n={n_bots})", edgecolors="white", linewidths=0.4)

# Label the top 15 most active bots by name
top_bots = active_bots.sort("total_edits", descending=True).head(15)
for row in top_bots.iter_rows(named=True):
    x = np.log1p(row["edit_velocity"])
    y = np.log1p(row["avg_words_changed"])
    name = row["user_text"] or "anon"
    ax.annotate(name, (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points",
                color="#333333", clip_on=True)

# Readable tick labels: convert log1p back to original values
tick_vals = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals))
ax.set_xticklabels(tick_vals)
ax.set_yticks(np.log1p(tick_vals))
ax.set_yticklabels(tick_vals)

ax.set_xlabel("Edit velocity  (edits per day)")
ax.set_ylabel("Avg. words changed per edit")
ax.set_title("FIG 01 — Behavioral Fingerprint: Bots vs Humans\n"
             "Bots cluster top-left: fast edits, little text changed each time")
ax.legend(loc="upper right")
save(fig, "01_behavioral_fingerprint")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — What do bots actually do? (primary edit type)
# ──────────────────────────────────────────────────────────────────────────────
# Classifies each account by which type of Wikipedia element they change most:
#   text      — inserting/removing words (writing content)
#   reference — editing citations / sources
#   template  — changing infoboxes, formatting templates
#   links     — adding/removing internal or external links
#
# Bots are highly specialized. Humans are more varied.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 02: Primary edit type breakdown...")

EDIT_TYPES  = ["text", "reference", "template", "links"]
TYPE_COLORS = ["#5B9BD5", "#ED7D31", "#70AD47", "#9E48AA"]

def type_pcts(frame: pl.DataFrame) -> list[float]:
    counts = (
        frame.group_by("primary_edit_type")
        .agg(pl.len().alias("n"))
    )
    total = len(frame)
    return [
        counts.filter(pl.col("primary_edit_type") == t)["n"].sum() / total * 100
        if t in counts["primary_edit_type"].to_list() else 0
        for t in EDIT_TYPES
    ]

bot_pcts   = type_pcts(bots)
human_pcts = type_pcts(humans)

x    = np.arange(len(EDIT_TYPES))
w    = 0.35
fig, ax = plt.subplots(figsize=(9, 5))

bars_b = ax.bar(x - w/2, bot_pcts,   w, label="Bots",   color=BOT_COLOR,   alpha=0.9)
bars_h = ax.bar(x + w/2, human_pcts, w, label="Humans", color=HUMAN_COLOR, alpha=0.9)

for bar in (*bars_b, *bars_h):
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in EDIT_TYPES])
ax.set_ylabel("% of accounts")
ax.set_title("FIG 02 — What Bots Do vs What Humans Do\n"
             "Bots specialise in references & templates; humans write text")
ax.legend()
ax.set_ylim(0, max(max(bot_pcts), max(human_pcts)) * 1.18)
save(fig, "02_primary_edit_type")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Edit velocity distribution
# ──────────────────────────────────────────────────────────────────────────────
# How fast do bots and humans edit?
# Plotted on a log scale because the range spans from <1 to 1000+ edits/day.
# The distributions barely overlap — velocity alone is a strong classifier.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 03: Edit velocity distribution...")

# Filter: >= 5 edits and velocity > 0
bv_vals = bots  .filter(pl.col("total_edits") >= 5)["edit_velocity"].drop_nulls().to_numpy()
hv_vals = humans.filter(pl.col("total_edits") >= 5)["edit_velocity"].drop_nulls().to_numpy()
bv_vals = bv_vals[bv_vals > 0]
hv_vals = hv_vals[hv_vals > 0]

bins = np.logspace(np.log10(0.01), np.log10(max(bv_vals.max(), hv_vals.max()) * 1.1), 60)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(hv_vals, bins=bins, color=HUMAN_COLOR, alpha=0.6, label="Humans", density=True)
ax.hist(bv_vals, bins=bins, color=BOT_COLOR,   alpha=0.7, label="Bots",   density=True)
ax.set_xscale("log")
ax.set_xlabel("Edit velocity  (edits per day, log scale)")
ax.set_ylabel("Density")
ax.set_title("FIG 03 — Edit Velocity Distribution\n"
             "Bots edit orders of magnitude faster than humans")
ax.legend()
save(fig, "03_velocity_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04 — Lorenz curve: edit concentration
# ──────────────────────────────────────────────────────────────────────────────
# A Lorenz curve shows inequality. The further the curve bows below the
# diagonal (= perfect equality), the more concentrated the editing is.
#
# Key question: do a tiny fraction of accounts produce most of the edits?
# Answer is yes — especially for humans (most make 1 edit and disappear).
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 04: Lorenz curve...")

def lorenz(values: np.ndarray):
    s = np.sort(values)
    n = len(s)
    cum = np.cumsum(s)
    return np.linspace(0, 1, n + 1), np.concatenate([[0], cum / cum[-1]])

bot_x,   bot_y   = lorenz(bots["total_edits"].to_numpy())
human_x, human_y = lorenz(humans["total_edits"].to_numpy())

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect equality")
ax.plot(human_x, human_y, color=HUMAN_COLOR, lw=2, label="Humans")
ax.plot(bot_x,   bot_y,   color=BOT_COLOR,   lw=2, label="Bots")

# Annotate the % of accounts that make 50% of edits.
# Human and bot annotations are placed on opposite sides of the 50% line
# so they never overlap: humans above, bots below.
for arr_x, arr_y, color, group_label, text_y, va in [
    (human_x, human_y, HUMAN_COLOR, "humans", 0.68, "bottom"),
    (bot_x,   bot_y,   BOT_COLOR,   "bots",   0.28, "top"),
]:
    idx          = np.searchsorted(arr_y, 0.5)
    pct_accounts = arr_x[idx] * 100
    text_x       = max(arr_x[idx] - 0.30, 0.05)   # keep inside the plot
    ax.annotate(
        f"Top {100 - pct_accounts:.0f}% of {group_label}\nmake 50% of edits",
        xy=(arr_x[idx], 0.5),
        xytext=(text_x, text_y),
        fontsize=8.5,
        color=color,
        va=va,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
    )

ax.set_xlabel("Cumulative fraction of accounts (least → most active)")
ax.set_ylabel("Cumulative fraction of total edits")
ax.set_title("FIG 04 — Edit Concentration (Lorenz Curve)\n"
             "A tiny minority of accounts produce the vast majority of edits")
ax.legend(loc="upper left")
save(fig, "04_lorenz_curve")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — Multilingual bots
# ──────────────────────────────────────────────────────────────────────────────
# How many language editions does each bot operate in?
# Bots that cover 8-10 languages are global Wikipedia infrastructure —
# they maintain consistency across language communities automatically.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 05: Multilingual bot distribution...")

bot_lang_counts = (
    bots.group_by("language_count")
    .agg(pl.len().alias("n_bots"))
    .sort("language_count")
)
human_lang_counts = (
    humans.group_by("language_count")
    .agg(pl.len().alias("n_humans"))
    .sort("language_count")
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bots: absolute count per language_count bucket
lc = bot_lang_counts["language_count"].to_numpy()
nb = bot_lang_counts["n_bots"].to_numpy()
bars = ax1.bar(lc, nb, color=BOT_COLOR, alpha=0.9, width=0.7)
for bar, val in zip(bars, nb):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", fontsize=9)
ax1.set_xticks(range(1, 11))
ax1.set_xlabel("Number of languages active in")
ax1.set_ylabel("Number of unique bots")
ax1.set_title("Bots by language reach")

# Humans: % distribution (log scale to handle the huge 1-language majority)
lc_h = human_lang_counts["language_count"].to_numpy()
nh_h = human_lang_counts["n_humans"].to_numpy()
pct_h = nh_h / nh_h.sum() * 100
ax2.bar(lc_h, pct_h, color=HUMAN_COLOR, alpha=0.9, width=0.7)
ax2.set_yscale("log")
ax2.set_xticks(range(1, 11))
ax2.set_xlabel("Number of languages active in")
ax2.set_ylabel("% of human accounts  (log scale)")
ax2.set_title("Humans by language reach")

fig.suptitle("FIG 05 — Multilingual Reach: How many wikis does each account touch?",
             fontsize=13, fontweight="bold", y=1.02)
save(fig, "05_multilingual_reach")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Top 20 bots: summary table
# ──────────────────────────────────────────────────────────────────────────────
# A formatted table of the 20 most active bots with their key metrics.
# Useful as a reference slide — puts names to the patterns.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 06: Top 20 bots table...")

top20 = (
    bots
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

cols = ["Bot name", "Edits", "Langs", "Edit type", "Top topic",
        "Vel./day", "Avg words\nchanged", "Revert\n%", "Reverted\n%"]

rows = []
for r in top20.iter_rows(named=True):
    rows.append([
        r["user_text"] or "anon",
        f"{r['total_edits']:,}",
        str(r["language_count"]),
        r["primary_edit_type"] or "—",
        r["top_topic"] or "—",
        f"{r['edit_velocity']:.1f}",
        f"{r['avg_words_changed']:.1f}",
        f"{r['revert_rate_pct']:.1f}%",
        f"{r['reverted_rate_pct']:.1f}%",
    ])

fig, ax = plt.subplots(figsize=(16, 7))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.45)

# Header row styling
for col_idx in range(len(cols)):
    tbl[0, col_idx].set_facecolor("#2C3E50")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

# Alternating row shading
for row_idx in range(1, len(rows) + 1):
    color = "#F7F9FB" if row_idx % 2 == 0 else "white"
    for col_idx in range(len(cols)):
        tbl[row_idx, col_idx].set_facecolor(color)

ax.set_title("FIG 06 — Top 20 Most Active Bots", fontsize=13, fontweight="bold",
             pad=12, loc="left")
save(fig, "06_top20_bots_table")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — Topic focus: which domains do bots vs humans edit?
# ──────────────────────────────────────────────────────────────────────────────
# Shows the top-level Wikipedia topic categories for bots vs humans.
# Reveals whether bots are evenly spread or concentrated in certain domains.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 07: Topic distribution...")

def topic_pcts(frame: pl.DataFrame) -> dict[str, float]:
    total = frame.filter(pl.col("top_topic").is_not_null()).shape[0]
    if total == 0:
        return {}
    counts = (
        frame.filter(pl.col("top_topic").is_not_null())
        .group_by("top_topic")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    return {r["top_topic"]: r["n"] / total * 100 for r in counts.iter_rows(named=True)}

bot_topics   = topic_pcts(bots)
human_topics = topic_pcts(humans)

# Use all topics that appear in either group
all_topics = sorted(set(bot_topics) | set(human_topics))
b_vals = [bot_topics.get(t, 0)   for t in all_topics]
h_vals = [human_topics.get(t, 0) for t in all_topics]

# Sort by average presence for a cleaner chart
order = sorted(range(len(all_topics)), key=lambda i: (b_vals[i] + h_vals[i]) / 2, reverse=True)
all_topics = [all_topics[i] for i in order]
b_vals     = [b_vals[i]     for i in order]
h_vals     = [h_vals[i]     for i in order]

y = np.arange(len(all_topics))
w = 0.38
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y + w/2, b_vals, w, color=BOT_COLOR,   alpha=0.9, label="Bots")
ax.barh(y - w/2, h_vals, w, color=HUMAN_COLOR, alpha=0.9, label="Humans")
ax.set_yticks(y)
ax.set_yticklabels(all_topics)
ax.set_xlabel("% of accounts whose most-edited topic is this category")
ax.set_title("FIG 07 — Topic Focus: Bots vs Humans\n"
             "Do bots concentrate in specific knowledge domains?")
ax.legend()
save(fig, "07_topic_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Bot quality: how often do bot edits get reverted?
# ──────────────────────────────────────────────────────────────────────────────
# reverted_rate_pct = % of an account's edits that were later undone by others.
# A high rate means the community disagreed with those edits.
#
# Hypothesis: bots should have LOWER reverted rates than humans because they
# run verified algorithms. Exceptions reveal "bad bots" or contested automation.
#
# Only accounts with >= 10 edits are shown to avoid noise from 1-edit accounts.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 08: Reverted rate comparison...")

min_edits = 10
b_rev = bots  .filter(pl.col("total_edits") >= min_edits)["reverted_rate_pct"].drop_nulls().to_numpy()
h_rev = humans.filter(pl.col("total_edits") >= min_edits)["reverted_rate_pct"].drop_nulls().to_numpy()

# Bin into 0%, 0-5%, 5-20%, >20%
bins_labels = ["0%", "0–5%", "5–20%", ">20%"]
def bin_reverted(arr):
    return np.array([
        (arr == 0).sum(),
        ((arr > 0) & (arr <= 5)).sum(),
        ((arr > 5) & (arr <= 20)).sum(),
        (arr > 20).sum(),
    ]) / len(arr) * 100

b_binned = bin_reverted(b_rev)
h_binned = bin_reverted(h_rev)

x = np.arange(len(bins_labels))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars_b = ax.bar(x - w/2, b_binned, w, color=BOT_COLOR,   alpha=0.9, label=f"Bots (n={len(b_rev):,})")
bars_h = ax.bar(x + w/2, h_binned, w, color=HUMAN_COLOR, alpha=0.9, label=f"Humans (n={len(h_rev):,})")

for bar in (*bars_b, *bars_h):
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(bins_labels)
ax.set_xlabel("Reverted rate (% of edits later undone by others)")
ax.set_ylabel("% of accounts in this bucket")
ax.set_title(f"FIG 08 — Edit Quality: How Often Are Edits Reverted?\n"
             f"Accounts with ≥{min_edits} edits only")
ax.legend()
save(fig, "08_reverted_rate")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — Bot-like humans: unflagged automation?
# ──────────────────────────────────────────────────────────────────────────────
# Some human-flagged accounts behave statistically like bots:
# high velocity + low word changes per edit + low focus_score.
# These could be semi-automated scripts, unflagged bots, or extreme power users.
#
# "Suspicious" threshold: edit_velocity > 20
# Any human-flagged account averaging more than 20 edits per day is unusual
# enough to warrant scrutiny — this is the sole criterion, no secondary filter.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 09: Bot-like humans...")

from matplotlib.colors import LogNorm as _LogNorm

active_humans_f = humans.filter(pl.col("edit_velocity").is_not_null())
suspect = active_humans_f.filter(pl.col("edit_velocity") > 20)

fig, ax = plt.subplots(figsize=(10, 6))

# Full human population in one density map — suspects and bots overlaid on top.
all_human_vel = np.log1p(active_humans_f["edit_velocity"].fill_null(0).to_numpy())
all_human_wrd = np.log1p(active_humans_f["avg_words_changed"].fill_null(0).to_numpy())
hb = ax.hexbin(all_human_vel, all_human_wrd, gridsize=55, cmap="Blues",
               mincnt=1, alpha=0.75, linewidths=0, norm=_LogNorm())
plt.colorbar(hb, ax=ax, label="Human accounts (log density)")

# Suspicious humans as orange dots
ax.scatter(
    np.log1p(suspect["edit_velocity"].to_numpy()),
    np.log1p(suspect["avg_words_changed"].to_numpy()),
    color=SUSPECT_COLOR, s=30, zorder=4, alpha=0.8,
    label=f"Bot-like humans (n={len(suspect):,})",
    edgecolors="white", linewidths=0.3,
)

# All bots as red triangles for reference
ax.scatter(
    np.log1p(active_bots["edit_velocity"].to_numpy()),
    np.log1p(active_bots["avg_words_changed"].to_numpy()),
    color=BOT_COLOR, s=55, zorder=5, marker="^",
    label=f"Flagged bots (n={n_bots})", edgecolors="white", linewidths=0.4,
)

tick_vals = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals))
ax.set_xticklabels(tick_vals)
ax.set_yticks(np.log1p(tick_vals))
ax.set_yticklabels(tick_vals)
ax.set_xlabel("Edit velocity  (edits per day)")
ax.set_ylabel("Avg. words changed per edit")
ax.set_title(f"FIG 09 — Bot-like Humans: Unflagged Automation?\n"
             f"{len(suspect):,} human accounts average >20 edits/day "
             f"({len(suspect)/n_humans*100:.2f}% of all humans)")
ax.legend(loc="upper right")
save(fig, "09_botlike_humans")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Bot-like humans: accounting for account longevity
# ──────────────────────────────────────────────────────────────────────────────
# Figure 9 flagged humans with edit_velocity > 20 as "bot-like", but this
# doesn't account for how long the account has been active. A human who has
# been editing for 10 years at 25 edits/day is different from a script that
# ran for 1 month at 25 edits/day.
#
# New criterion: edit_velocity > 20 AND active_days > 300 (≈1 year)
# This better distinguishes persistent automation from highly active humans.
#
# X = edit_velocity (edits per day) — how fast they work
# Y = active_days (account lifespan) — how long they've been active
#
# Method: humans shown as hexbin density, bot-like humans as orange dots,
# flagged bots as red triangles for reference.
# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Edit Velocity vs. Active Editing Days (Humans vs. Bots)
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 10: Edit velocity vs active days (Scatter)...")

# Filter to human accounts with valid velocity and active days
active_humans_f = humans.filter(
    (pl.col("edit_velocity_active_days").is_not_null()) &
    (pl.col("active_days").is_not_null()) &
    (pl.col("edit_velocity_active_days") > 0) &
    (pl.col("active_days") > 0)
)

fig, ax = plt.subplots(figsize=(10, 6))

# All human accounts (small blue dots in the background)
ax.scatter(
    np.log1p(active_humans_f["edit_velocity_active_days"].to_numpy()),
    active_humans_f["active_days"].to_numpy(),
    color=HUMAN_COLOR,
    s=8,           # Small points for dense plotting
    zorder=2,      # Pushed to the back
    alpha=0.4,     # Semi-transparent to reveal density clusters naturally
    linewidths=0,
    label=f"Human accounts (n={len(active_humans_f):,})"
)

# Bots for reference (red triangles on top)
active_bots_days = bots.filter(
    (pl.col("edit_velocity_active_days").is_not_null()) &
    (pl.col("active_days").is_not_null()) &
    (pl.col("edit_velocity_active_days") > 0) &
    (pl.col("active_days") > 0)
)

ax.scatter(
    np.log1p(active_bots_days["edit_velocity_active_days"].to_numpy()),
    active_bots_days["active_days"].to_numpy(),
    color=BOT_COLOR,
    s=60,
    zorder=5,      # Highest zorder so bots are never hidden
    marker="^",
    label=f"Flagged bots (n={len(active_bots_days):,})",
    edgecolors="white",
    linewidths=0.5,
)

# Velocity ticks (Log scale adjusted back to raw numbers for readability)
tick_vals_vel = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals_vel))
ax.set_xticklabels(tick_vals_vel)

ax.set_xlabel("Edit velocity (edits per active day)")
ax.set_ylabel("Active editing days (unique calendar days)")
ax.set_title(
    "FIG 10 — Editing Behavior: Humans vs. Bots\n"
    "Comparing daily edit velocity against total active days"
)

# Adjust legend to show the categories clearly
ax.legend(loc="upper right", markerscale=1.5)
save(fig, "10_velocity_vs_active_days")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Heavily reverted accounts: velocity vs active days
# ──────────────────────────────────────────────────────────────────────────────
# Highlight accounts where ≥80% of edits were reverted (or completely reverted).
# Same axes as FIG 10 — lets us see whether mass-reverted accounts cluster
# differently from the general population and whether this pattern differs
# between bots and humans.
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 11: Heavily reverted accounts (velocity vs active days)...")

_base_filter = (
    (pl.col("edit_velocity_active_days").is_not_null()) &
    (pl.col("active_days").is_not_null()) &
    (pl.col("edit_velocity_active_days") > 0) &
    (pl.col("active_days") > 0)
)
plot_humans = humans.filter(_base_filter)
plot_bots   = bots.filter(_base_filter)

_heavily_reverted = (
    pl.col("completely_reverted") | (pl.col("reverted_rate_pct") >= 80)
)
heavy_h = plot_humans.filter(_heavily_reverted)
heavy_b = plot_bots.filter(_heavily_reverted)

fig, ax = plt.subplots(figsize=(11, 6))

# Background populations — same style as FIG 10
ax.scatter(
    np.log1p(plot_humans["edit_velocity_active_days"].to_numpy()),
    plot_humans["active_days"].to_numpy(),
    color=HUMAN_COLOR, s=8, zorder=1, alpha=0.4, linewidths=0,
    label=f"Humans (n={len(plot_humans):,})",
)
ax.scatter(
    np.log1p(plot_bots["edit_velocity_active_days"].to_numpy()),
    plot_bots["active_days"].to_numpy(),
    color=BOT_COLOR, s=55, zorder=2, alpha=0.6, marker="^",
    edgecolors="white", linewidths=0.4,
    label=f"Bots (n={len(plot_bots):,})",
)

# Heavily reverted accounts — distinct colours on top
# Humans: yellow-green to contrast with the blue mass
ax.scatter(
    np.log1p(heavy_h["edit_velocity_active_days"].to_numpy()),
    heavy_h["active_days"].to_numpy(),
    color="#F1C40F", s=35, zorder=4,
    edgecolors="#B7950B", linewidths=0.6,
    label=f"Humans ≥80% reverted (n={len(heavy_h):,})",
)
# Bots: purple triangle — clearly different from the red bot mass
ax.scatter(
    np.log1p(heavy_b["edit_velocity_active_days"].to_numpy()),
    heavy_b["active_days"].to_numpy(),
    color="#8E44AD", s=80, zorder=5, marker="^",
    edgecolors="white", linewidths=0.6,
    label=f"Bots ≥80% reverted (n={len(heavy_b):,})",
)

tick_vals_vel = [0, 1, 5, 20, 100, 500]
ax.set_xticks(np.log1p(tick_vals_vel))
ax.set_xticklabels(tick_vals_vel)

ax.set_xlabel("Edit velocity (edits per active day)")
ax.set_ylabel("Active editing days")
ax.set_title(
    f"FIG 11 — Heavily Reverted Accounts (≥80% of edits reverted)\n"
    f"Humans: {len(heavy_h):,} ({len(heavy_h)/len(plot_humans)*100:.2f}% of humans)  |  "
    f"Bots: {len(heavy_b):,} ({len(heavy_b)/len(plot_bots)*100:.2f}% of bots)"
)
ax.legend(loc="upper right", markerscale=1.5)
save(fig, "11_heavily_reverted")
"""
DataFest 2026 — Step 2: Bot vs Human Deep Dive
Topic: Bots as Wikipedia Editors

Builds on Part 4 of 01_exploration_Jonathan.py.
Adds:
  - Edit frequency buckets per account type (1, <5, 5-10, <100, >100)
  - Descriptive statistics on edits per account (mean, median, std, etc.)
"""

import polars as pl
from pathlib import Path
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).parent.parent / "data"
EDITS_DIR = BASE_DATA_DIR / "data_extracted" / "edit_types"

LANGUAGES = ['ar', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'ru', 'sv']


# ── HELPERS ───────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def edit_buckets(df: pl.DataFrame, label: str) -> None:
    """Print edit-count frequency distribution in predefined buckets."""
    # Each row here is one unique account with their total edit count.
    bucketed = df.with_columns(
        pl.when(pl.col("edit_count") == 1).then(pl.lit("1 edit"))
        .when(pl.col("edit_count") < 5).then(pl.lit("2-4 edits"))
        .when(pl.col("edit_count") <= 10).then(pl.lit("5-10 edits"))
        .when(pl.col("edit_count") < 100).then(pl.lit("11-99 edits"))
        .otherwise(pl.lit("100+ edits"))
        .alias("bucket")
    )
    result = (
        bucketed.group_by("bucket")
        .agg(pl.len().alias("accounts"))
        .with_columns(
            (pl.col("accounts") / pl.col("accounts").sum() * 100)
            .round(1).alias("pct")
        )
        # Sort by natural bucket order
        .with_columns(
            pl.col("bucket").replace(
                {
                    "1 edit":     0,
                    "2-4 edits":  1,
                    "5-10 edits": 2,
                    "11-99 edits":3,
                    "100+ edits": 4,
                },
                return_dtype=pl.Int8
            ).alias("_order")
        )
        .sort("_order")
        .select(["bucket", "accounts", "pct"])
    )
    print(f"\n  {label} — edit frequency buckets:")
    print(result)


def edit_stats(df: pl.DataFrame, label: str) -> None:
    """Print descriptive statistics on edit counts per account."""
    counts = df.select("edit_count")
    stats = counts.describe()
    # describe() gives: count, null_count, mean, std, min, 25%, 50%, 75%, max
    print(f"\n  {label} — edits per account (descriptive stats):")
    print(stats)

    # Extra: 90th and 99th percentile (skewness indicator)
    p90 = counts["edit_count"].quantile(0.90)
    p99 = counts["edit_count"].quantile(0.99)
    print(f"    90th percentile: {p90:,.0f} edits")
    print(f"    99th percentile: {p99:,.0f} edits")


# ── 1. GLOBAL BOT vs HUMAN SUMMARY (from Part 4) ─────────────────────────────
# This is a direct copy of Part 4 from 01_exploration_Jonathan.py, kept here
# so this file is self-contained and produces the same baseline output.

section("GLOBAL BOT vs HUMAN SUMMARY")

unique_bots = set()
unique_humans = set()
all_stats = []

# Also collect per-user edit counts across all languages for sections 2 & 3.
# We store (user_text, is_bot, edit_count_in_this_language) per language,
# then aggregate globally at the end.
user_counts_per_lang = []

for lang in LANGUAGES:
    edits_file = EDITS_DIR / f"{lang}wiki.json.gz"
    print(f"\n  Loading {lang.upper()}...")
    start = time.time()
    edits_lang = pl.read_ndjson(edits_file)
    print(f"  -> {edits_lang.shape[0]:,} edits in {time.time() - start:.2f}s")

    bots   = edits_lang.filter(pl.col("is_bot") == True)
    humans = edits_lang.filter(pl.col("is_bot") == False)

    # Per-language bot/human split
    bot_split = (
        edits_lang.group_by("is_bot")
        .agg(pl.len().alias("edit_count"))
        .with_columns(
            (pl.col("edit_count") / pl.col("edit_count").sum() * 100)
            .round(1).alias("pct")
        )
        .sort("is_bot")
    )
    print(f"  {lang.upper()} bot vs human:")
    print(bot_split)

    # Collect per-language stats
    all_stats.append({
        "language":       lang.upper(),
        "bot_edits":      bots.shape[0],
        "human_edits":    humans.shape[0],
        "total_edits":    edits_lang.shape[0],
        "bot_accounts":   bots.select("user_text").unique().shape[0],
        "human_accounts": humans.select("user_text").unique().shape[0],
    })

    # Unique account names across languages
    for user in bots.select("user_text").unique().filter(pl.col("user_text").is_not_null())["user_text"]:
        unique_bots.add(user)
    for user in humans.select("user_text").unique().filter(pl.col("user_text").is_not_null())["user_text"]:
        unique_humans.add(user)

    # Per-user edit counts for this language (used in sections 2 & 3)
    per_user = (
        edits_lang.group_by(["user_text", "is_bot"])
        .agg(pl.len().alias("edit_count"))
    )
    user_counts_per_lang.append(per_user)

# Per-language table
stats_df = pl.DataFrame(all_stats).with_columns(
    (pl.col("bot_accounts") + pl.col("human_accounts")).alias("total_accounts")
)
print("\n  Per-language breakdown:")
print(
    stats_df.with_columns(
        (pl.col("bot_edits")    / pl.col("total_edits")    * 100).round(1).alias("bot_edit_pct"),
        (pl.col("bot_accounts") / pl.col("total_accounts") * 100).round(1).alias("bot_account_pct"),
    )
)

# Global totals
te = stats_df["total_edits"].sum()
be = stats_df["bot_edits"].sum()
he = stats_df["human_edits"].sum()
ta = stats_df["total_accounts"].sum()
ba = stats_df["bot_accounts"].sum()
ha = stats_df["human_accounts"].sum()

section("GLOBAL TOTALS")
print(f"  Edits:      {te:>10,}  (bot: {be:,} [{be/te*100:.1f}%]  |  human: {he:,} [{he/te*100:.1f}%])")
print(f"  Accounts*:  {ta:>10,}  (bot: {ba:,} [{ba/ta*100:.1f}%]  |  human: {ha:,} [{ha/ta*100:.1f}%])")
print(f"  Unique bots across all languages:   {len(unique_bots)}")
print(f"  Unique humans across all languages: {len(unique_humans)}")
print(f"  * Per-language sums; cross-language duplicates not deduplicated")


# ── 2. GLOBAL USER EDIT COUNTS ────────────────────────────────────────────────
# Sum each user's edits across all languages they appear in.
# Example: if "Citation bot" edited EN (19,050×) and DE (500×), their global
# count is 19,550. This gives a truer picture of total activity.

section("EDIT FREQUENCY BUCKETS (global, per account)")
print("""
  Method: edits per account summed across all 10 languages.
  A user active in multiple wikis has their edits combined.
  Buckets show how activity is distributed — most accounts are one-hit
  wonders, while a few bots/power-users dominate total edit volume.
""")

all_user_counts = pl.concat(user_counts_per_lang)

# Aggregate: sum edits per (user_text, is_bot) across all languages
global_user_counts = (
    all_user_counts
    .group_by(["user_text", "is_bot"])
    .agg(pl.col("edit_count").sum())
)

bots_global   = global_user_counts.filter(pl.col("is_bot") == True)
humans_global = global_user_counts.filter(pl.col("is_bot") == False)

# Remove null user_text (anonymous edits) for account-level analysis
bots_global   = bots_global.filter(pl.col("user_text").is_not_null())
humans_global = humans_global.filter(pl.col("user_text").is_not_null())

print(f"  Unique bot accounts (global):   {len(bots_global):,}")
print(f"  Unique human accounts (global): {len(humans_global):,}")

edit_buckets(bots_global,   "Bots")
edit_buckets(humans_global, "Humans")


# ── 3. DESCRIPTIVE STATISTICS PER ACCOUNT TYPE ───────────────────────────────
# "Descriptive statistics" = summary numbers that describe the shape of a
# distribution: how many, the average (mean), the middle value (median/50%),
# how spread out (std deviation), minimum, maximum, and percentiles.
# High std relative to mean means a small number of accounts do vastly more
# edits than the typical account — expected for bots.

section("DESCRIPTIVE STATISTICS — edits per account (global)")
print("""
  mean   = average edits per account
  median = the middle value; robust to extreme outliers (unlike mean)
  std    = standard deviation; how much individual counts vary from the mean
  min/max = lowest and highest edit counts
  25%/75% = quartiles: 25% of accounts are below this threshold
""")

edit_stats(bots_global,   "Bots")
edit_stats(humans_global, "Humans")

# Side-by-side comparison of the key numbers
bot_mean   = bots_global["edit_count"].mean()
human_mean = humans_global["edit_count"].mean()
bot_med    = bots_global["edit_count"].median()
human_med  = humans_global["edit_count"].median()
bot_max    = bots_global["edit_count"].max()
human_max  = humans_global["edit_count"].max()

print(f"\n  Summary comparison:")
print(f"  {'Metric':<12} {'Bots':>12} {'Humans':>12}")
print(f"  {'-'*38}")
print(f"  {'Mean':<12} {bot_mean:>12,.1f} {human_mean:>12,.1f}")
print(f"  {'Median':<12} {bot_med:>12,.1f} {human_med:>12,.1f}")
print(f"  {'Max':<12} {bot_max:>12,} {human_max:>12,}")
print(f"  {'Accounts':<12} {len(bots_global):>12,} {len(humans_global):>12,}")

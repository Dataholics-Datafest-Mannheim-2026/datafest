"""
DataFest 2026 — Step 1: Data Exploration
Topic: Bots as Wikipedia Editors
"""

import polars as pl
import json
from pathlib import Path
import time
import gzip

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).parent.parent / "data"
PAGE_INFO_PATH = BASE_DATA_DIR / "data_extracted" / "page_info.json"
PAGE_VIEWS_PATH = BASE_DATA_DIR / "data_extracted" / "page_views.json"
EDITS_DIR = BASE_DATA_DIR / "data_extracted" / "edit_types"

EDITS_LANGUAGE = "en"  # language for single-language exploration
EDITS_PATH = EDITS_DIR / f"{EDITS_LANGUAGE}wiki.json.gz"

LANGUAGES = ['ar', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'ru', 'sv']
SAMPLE_N = None                 # row limit for quick exploration; None = load all
SKIP_DETAILED_BOT_OVERVIEW = False  # skip per-language bot statistics to save time


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_ndjson(path: Path, n: int | None = SAMPLE_N) -> pl.DataFrame:
    """Load ndjson file, optionally limited to first n rows."""
    label = f"sample ({n:,} rows)" if n else "all rows"
    print(f"\n  Loading {label} from: {path}")
    start = time.time()
    df = pl.read_ndjson(path, n_rows=n) if n else pl.read_ndjson(path)
    print(f"  -> {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time() - start:.2f}s")
    return df


def quick_summary(df: pl.DataFrame, name: str) -> None:
    """Print shape, column types, and first 3 rows."""
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {name}")
    print(f"{'='*60}")
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"    {col:<25} {dtype}")
    print(df.head(3))


def print_first_row(path: Path, compressed: bool = False) -> None:
    """Print first JSON line from a file."""
    opener = gzip.open if compressed else open
    with opener(path, 'rt', encoding='utf-8') as f:
        print(json.dumps(json.loads(f.readline().strip()), indent=2))


# ── 0. DATA SCHEMA OVERVIEW ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  DATA SCHEMA OVERVIEW")
print("="*60)

print("\n-- Page Info --")
print_first_row(PAGE_INFO_PATH)

print("\n-- Page Views --")
print_first_row(PAGE_VIEWS_PATH)

print(f"\n-- Edit Types ({EDITS_LANGUAGE.upper()}) --")
print_first_row(EDITS_PATH, compressed=True)


# ── 1. PAGE INFO (~100k rows) ────────────────────────────────────────────────
page_info = load_ndjson(PAGE_INFO_PATH)
quick_summary(page_info, "Page Info")

print("\n  Articles per language:")
print(
    page_info.group_by("wiki_db")
    .agg(pl.len().alias("article_count"))
    .sort("article_count", descending=True)
)


# ── 2. PAGE VIEWS (~36M rows) ────────────────────────────────────────────────
page_views = load_ndjson(PAGE_VIEWS_PATH, n=None)
quick_summary(page_views, "Page Views")

print("\n  Language distribution:")
print(
    page_views.group_by("wiki_db")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)


# ── 3. EDITS — single language deep dive ─────────────────────────────────────
edits = load_ndjson(EDITS_PATH, n=None)
quick_summary(edits, f"Edits — {EDITS_LANGUAGE.upper()}")

# Bot vs human split
print("\n  Bot vs Human edits:")
print(
    edits.group_by("is_bot")
    .agg(pl.len().alias("edit_count"))
    .with_columns((pl.col("edit_count") / pl.col("edit_count").sum() * 100).round(1).alias("pct"))
    .sort("is_bot")
)

# Top 10 most active editors
print("\n  Top 10 editors:")
print(
    edits.group_by("user_text")
    .agg(pl.len().alias("edit_count"), pl.col("is_bot").first().alias("is_bot"))
    .sort("edit_count", descending=True)
    .head(10)
)

# Example bot edit payload
print("\n  Example edit_types_json (bot edit):")
bot_edit = edits.filter(pl.col("is_bot") == True).head(1)
if len(bot_edit) > 0:
    print(json.dumps(json.loads(bot_edit["edit_types_json"][0]), indent=2))
else:
    print("  No bot edits found.")

# ── 4. GLOBAL BOT vs HUMAN SUMMARY ───────────────────────────────────────────
print("\n" + "="*60)
print("  GLOBAL BOT vs HUMAN SUMMARY")
print("="*60)

if SKIP_DETAILED_BOT_OVERVIEW:
    print("\n  Skipped (set SKIP_DETAILED_BOT_OVERVIEW = False to enable)")
else:
    unique_bots = set()
    unique_humans = set()
    all_stats = []

    for lang in LANGUAGES:
        edits_file = EDITS_DIR / f"{lang}wiki.json.gz"
        print(f"\n  Loading {lang.upper()}...")
        start = time.time()
        edits_lang = pl.read_ndjson(edits_file)
        print(f"  -> {edits_lang.shape[0]:,} edits in {time.time() - start:.2f}s")

        bots = edits_lang.filter(pl.col("is_bot") == True)
        humans = edits_lang.filter(pl.col("is_bot") == False)

        # Per-language bot/human split
        bot_split = (
            edits_lang.group_by("is_bot")
            .agg(pl.len().alias("edit_count"))
            .with_columns((pl.col("edit_count") / pl.col("edit_count").sum() * 100).round(1).alias("pct"))
            .sort("is_bot")
        )
        print(f"  {lang.upper()} bot vs human:")
        print(bot_split)

        # Collect stats
        all_stats.append({
            "language": lang.upper(),
            "bot_edits": bots.shape[0],
            "human_edits": humans.shape[0],
            "total_edits": edits_lang.shape[0],
            "bot_accounts": bots.select("user_text").unique().shape[0],
            "human_accounts": humans.select("user_text").unique().shape[0],
        })

        # Collect unique bot and human names across languages
        for user in bots.select("user_text").unique().filter(pl.col("user_text").is_not_null())["user_text"]:
            unique_bots.add(user)
        for user in humans.select("user_text").unique().filter(pl.col("user_text").is_not_null())["user_text"]:
            unique_humans.add(user)

    # Per-language table with percentages
    stats_df = pl.DataFrame(all_stats).with_columns(
        (pl.col("bot_accounts") + pl.col("human_accounts")).alias("total_accounts")
    )
    print("\n  Per-language breakdown:")
    print(
        stats_df.with_columns(
            (pl.col("bot_edits") / pl.col("total_edits") * 100).round(1).alias("bot_edit_pct"),
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

    print(f"\n{'='*60}")
    print(f"  GLOBAL TOTALS")
    print(f"{'='*60}")
    print(f"  Edits:      {te:>10,}  (bot: {be:,} [{be/te*100:.1f}%]  |  human: {he:,} [{he/te*100:.1f}%])")
    print(f"  Accounts*:  {ta:>10,}  (bot: {ba:,} [{ba/ta*100:.1f}%]  |  human: {ha:,} [{ha/ta*100:.1f}%])")
    print(f"  Unique bots across all languages: {len(unique_bots)}")
    print(f"  Unique humans across all languages: {len(unique_humans)}")
    print(f"  * Per-language sums; cross-language duplicates not deduplicated")
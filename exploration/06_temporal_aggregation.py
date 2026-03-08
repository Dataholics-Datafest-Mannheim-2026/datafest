"""
DataFest 2026 — Step 6: Temporal Aggregation
Topic: Bots as Wikipedia Editors

Loads all 10 language edit files and aggregates to daily counts.
Output: exploration/temporal_daily.ndjson

Each output row = one (date, language, is_bot) combination with:
  total_edits   — total edits made that day
  word_changes  — sum of word insertions + deletions (text volume)
  ref_edits     — Reference node edits (citations)
  tmpl_edits    — Template node edits (infoboxes, formatting)
  link_edits    — ExternalLink + Wikilink node edits
  revert_edits  — edits tagged as mw-undo or mw-manual-revert

Run once; visualizations in 07_temporal_visualizations.py read the output.
Run: uv run exploration/06_temporal_aggregation.py
"""

import polars as pl
from pathlib import Path
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR      = Path(__file__).parent.parent / "data"
EDITS_DIR          = BASE_DATA_DIR / "data_extracted" / "edit_types"
PAGE_INFO_PATH     = BASE_DATA_DIR / "data_extracted" / "page_info.json"
OUTPUT_PATH        = Path(__file__).parent / "temporal_daily.ndjson"
OUTPUT_TOPIC_PATH  = Path(__file__).parent / "temporal_daily_by_topic.ndjson"

LANGUAGES     = ['ar', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'ru', 'sv']
TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S%.3fZ"


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


section("TEMPORAL AGGREGATION — daily edit counts per language")

# Load page_info once to get topic labels (same logic as file 03)
print("  Loading page_info for topic labels...")
page_info_raw = pl.read_ndjson(PAGE_INFO_PATH)
page_lookup = (
    page_info_raw
    .with_columns(
        pl.col("predicted_labels")
        .list.first()
        .struct.field("label")
        .str.split(".")
        .list.first()
        .alias("top_category")
    )
    .select(["wiki_db", "page_id", "top_category"])
)

daily_frames       = []
topic_daily_frames = []

for lang in LANGUAGES:
    edits_file = EDITS_DIR / f"{lang}wiki.json.gz"
    print(f"\n  Loading {lang.upper()}...")
    start = time.time()
    edits = pl.read_ndjson(edits_file)
    print(f"  -> {edits.shape[0]:,} rows in {time.time() - start:.2f}s")

    # Strip timestamp to date only — we don't need sub-day precision
    edits = edits.with_columns(
        pl.col("revision_timestamp")
        .str.to_datetime(TIMESTAMP_FMT, time_unit="ms")
        .dt.date()
        .alias("date")
    )

    # Edit type counts via regex on raw JSON — same patterns as file 03
    edits = edits.with_columns(
        pl.col("edit_types_json").str.count_matches(r'\["Word", "insert"').cast(pl.Int32).alias("word_inserts"),
        pl.col("edit_types_json").str.count_matches(r'\["Word", "remove"').cast(pl.Int32).alias("word_removes"),
        pl.col("edit_types_json").str.count_matches(r'\["Reference"')     .cast(pl.Int32).alias("n_ref"),
        pl.col("edit_types_json").str.count_matches(r'\["Template"')      .cast(pl.Int32).alias("n_tmpl"),
        (
            pl.col("edit_types_json").str.count_matches(r'\["ExternalLink"') +
            pl.col("edit_types_json").str.count_matches(r'\["Wikilink"')
        ).cast(pl.Int32).alias("n_links"),
        # Revert: this edit actively undoes someone else's work
        (
            pl.col("revision_tags").list.contains("mw-undo") |
            pl.col("revision_tags").list.contains("mw-manual-revert")
        ).cast(pl.Int32).alias("n_revert"),
    )

    # Collapse to one row per (date, is_bot) for this language
    daily = (
        edits.group_by(["date", "is_bot"])
        .agg(
            pl.len()                                                   .alias("total_edits"),
            pl.col("user_text").n_unique()                             .alias("unique_editors"),
            (pl.col("word_inserts") + pl.col("word_removes")).sum()   .alias("word_changes"),
            pl.col("n_ref").sum()                                      .alias("ref_edits"),
            pl.col("n_tmpl").sum()                                     .alias("tmpl_edits"),
            pl.col("n_links").sum()                                    .alias("link_edits"),
            pl.col("n_revert").sum()                                   .alias("revert_edits"),
        )
        .with_columns(pl.lit(lang).alias("language"))
        .sort("date")
    )
    daily_frames.append(daily)
    print(f"  -> {daily.shape[0]:,} daily rows for {lang.upper()}")

    # Collapse to one row per (date, is_bot, top_category) for this language
    topic_daily = (
        edits
        .join(
            page_lookup
            .filter(pl.col("wiki_db") == f"{lang}wiki")
            .select(["page_id", "top_category"]),
            on="page_id",
            how="left",
        )
        .filter(pl.col("top_category").is_not_null())
        .group_by(["date", "is_bot", "top_category"])
        .agg(pl.len().alias("total_edits"))
        .with_columns(pl.lit(lang).alias("language"))
        .sort("date")
    )
    topic_daily_frames.append(topic_daily)
    print(f"  -> {topic_daily.shape[0]:,} topic-daily rows for {lang.upper()}")


section("WRITE OUTPUT")

all_daily = (
    pl.concat(daily_frames)
    .sort(["date", "language", "is_bot"])
)

print(f"\n  Rows:       {all_daily.shape[0]:,}")
print(f"  Date range: {all_daily['date'].min()}  to  {all_daily['date'].max()}")
print(f"  Languages:  {sorted(all_daily['language'].unique().to_list())}")
print(f"\n  Sample:")
print(all_daily.head(6))

print(f"\n  Writing to: {OUTPUT_PATH}")
all_daily.write_ndjson(OUTPUT_PATH)
size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
print(f"  -> {all_daily.shape[0]:,} rows, {size_mb:.2f} MB")
print(f"\n  Done. Load in 07 with:")
print(f"    pl.read_ndjson('temporal_daily.ndjson')")

all_topic_daily = (
    pl.concat(topic_daily_frames)
    .sort(["date", "language", "is_bot", "top_category"])
)

print(f"\n  Topic rows: {all_topic_daily.shape[0]:,}")
print(f"  Topics:     {sorted(all_topic_daily['top_category'].unique().to_list())}")
print(f"\n  Writing to: {OUTPUT_TOPIC_PATH}")
all_topic_daily.write_ndjson(OUTPUT_TOPIC_PATH)
size_mb = OUTPUT_TOPIC_PATH.stat().st_size / 1_048_576
print(f"  -> {all_topic_daily.shape[0]:,} rows, {size_mb:.2f} MB")
"""
DataFest 2026 — Step 3: Build Master Accounts Dataset
Topic: Bots as Wikipedia Editors

Aggregates all edit data into one enriched record per unique account.
Output: bot_analysis_jona/accounts_master.ndjson — one JSON object per line.
Load in later steps with: pl.read_ndjson("accounts_master.ndjson")

── FIELDS ────────────────────────────────────────────────────────────────────
Identity
  user_text              account name (null = anonymous IP edit)
  is_bot                 Wikipedia bot flag

Volume
  total_edits            edits across all 10 languages
  language_count         distinct language editions edited in
  languages_active       list of language codes, sorted by edit count desc
  edits_by_language      list of {language, edits} per active language
  unique_pages_touched   distinct page_ids across all languages

Time
  first_edit             timestamp of earliest edit
  last_edit              timestamp of latest edit
  activity_span_days     days between first and last edit (0 = one-day account)
  edit_velocity          edits per day across activity span (burst indicator)

Edit character — from parsing edit_types_json with regex
  total_word_inserts     word-level text insertions across all edits
  total_word_removes     word-level text deletions across all edits
  total_word_changes     inserts + removes (total text volume changed)
  total_ref_edits        Reference node edits (citation work)
  total_tmpl_edits       Template node edits (infoboxes, formatting templates)
  total_extlink_edits    ExternalLink node edits
  total_wikilink_edits   Wikilink node edits (internal links between articles)
  total_link_edits       extlink + wikilink combined
  primary_edit_type      dominant edit mode: text | reference | template | links
  avg_words_changed      (word_inserts + removes) / total_edits

Revert behavior — from revision_tags
  revert_count           edits where this account undid someone else's edit
  was_reverted_count     this account's edits that were later undone by others
  revert_rate_pct        revert_count / total_edits * 100
  reverted_rate_pct      was_reverted_count / total_edits * 100
                         (high = edits are contested or low quality)

Topics — joined from page_info predicted_labels
  top_topic              topic category this account edits most (e.g. Geography)
  unique_topic_count     distinct top-level topic categories edited in
  topics_active          list of all topic categories edited in
  avg_pageviews_edited   mean pageview count of pages this account touches
                         (high = targets popular articles; low = niche editor)

Derived
  tagged_edits           edits carrying at least one revision_tag
  tag_rate_pct           tagged_edits / total_edits * 100
  avg_edits_per_language total_edits / language_count
  is_multilingual        edited in more than one language edition
  focus_score            unique_pages_touched / total_edits
                         (low = edits same pages repeatedly; high = broad)
  edit_bucket            activity tier: "1" | "2-4" | "5-10" | "11-99" | "100+"
"""
import polars as pl
from pathlib import Path
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR  = Path(__file__).parent.parent / "data"
EDITS_DIR      = BASE_DATA_DIR / "data_extracted" / "edit_types"
PAGE_INFO_PATH = BASE_DATA_DIR / "data_extracted" / "page_info.json"
OUTPUT_PATH    = Path(__file__).parent / "accounts_master.ndjson"

LANGUAGES     = ['ar', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'ru', 'sv']
TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S%.3fZ"


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── STEP 0: PAGE INFO ─────────────────────────────────────────────────────────
# Load the page catalogue to get topic labels and pageview counts.

section("STEP 0: load page info")

start = time.time()
page_info_raw = pl.read_ndjson(PAGE_INFO_PATH)
print(f"  {page_info_raw.shape[0]:,} pages in {time.time() - start:.2f}s")

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
    .select(["wiki_db", "page_id", "top_category", "pageviews"])
)


# ── STEP 1: PER-LANGUAGE AGGREGATION ─────────────────────────────────────────

section("STEP 1: aggregate per language")

per_lang_frames    = []  # one row per (user, language)
topic_frames       = []  # one row per (user, topic, language) for top_topic
active_days_frames = []  # NEW: one row per (user, unique_date) per language

for lang in LANGUAGES:
    edits_file = EDITS_DIR / f"{lang}wiki.json.gz"
    print(f"\n  Loading {lang.upper()}...")
    start = time.time()
    edits = pl.read_ndjson(edits_file)
    print(f"  -> {edits.shape[0]:,} rows in {time.time() - start:.2f}s")

    # Parse timestamp string to datetime once
    edits = edits.with_columns(
        pl.col("revision_timestamp")
        .str.to_datetime(TIMESTAMP_FMT, time_unit="ms")
        .alias("ts")
    )

    # ── NEW: Extract unique editing dates for this language ───────────────────
    # We strip the time component and keep only unique dates per user to save memory.
    user_dates = (
        edits
        .select([
            "user_text",
            "is_bot",
            pl.col("ts").dt.date().alias("edit_date")
        ])
        .unique()
    )
    active_days_frames.append(user_dates)
    # ──────────────────────────────────────────────────────────────────────────

    # ── Edit type features (regex on raw JSON string) ─────────────────────────
    edits = edits.with_columns(
        pl.col("edit_types_json").str.count_matches(r'\["Word", "insert"') .cast(pl.Int32).alias("word_inserts"),
        pl.col("edit_types_json").str.count_matches(r'\["Word", "remove"') .cast(pl.Int32).alias("word_removes"),
        pl.col("edit_types_json").str.count_matches(r'\["Reference"')      .cast(pl.Int32).alias("n_ref_edits"),
        pl.col("edit_types_json").str.count_matches(r'\["Template"')       .cast(pl.Int32).alias("n_tmpl_edits"),
        pl.col("edit_types_json").str.count_matches(r'\["ExternalLink"')   .cast(pl.Int32).alias("n_extlink_edits"),
        pl.col("edit_types_json").str.count_matches(r'\["Wikilink"')       .cast(pl.Int32).alias("n_wikilink_edits"),
    )

    # ── Revert flags (from revision_tags list) ────────────────────────────────
    edits = edits.with_columns(
        (
            pl.col("revision_tags").list.contains("mw-undo") |
            pl.col("revision_tags").list.contains("mw-manual-revert")
        ).cast(pl.Int32).alias("is_revert"),
        pl.col("revision_tags").list.contains("mw-reverted")
        .cast(pl.Int32).alias("was_reverted"),
    )

    # ── Join with page_info for topic + pageviews ─────────────────────────────
    wiki_db = f"{lang}wiki"
    edits = edits.join(
        page_lookup
        .filter(pl.col("wiki_db") == wiki_db)
        .select(["page_id", "top_category", "pageviews"]),
        on="page_id",
        how="left",
    )

    # ── Per-user aggregation for this language ────────────────────────────────
    agg = (
        edits.group_by(["user_text", "is_bot"])
        .agg(
            pl.len()                                         .alias("edit_count"),
            pl.col("page_id").n_unique()                    .alias("unique_pages"),
            pl.col("ts").min()                               .alias("first_edit"),
            pl.col("ts").max()                               .alias("last_edit"),
            pl.col("revision_tags").list.len().gt(0).sum()  .alias("tagged_edits"),
            pl.col("word_inserts").sum(),
            pl.col("word_removes").sum(),
            pl.col("n_ref_edits").sum(),
            pl.col("n_tmpl_edits").sum(),
            pl.col("n_extlink_edits").sum(),
            pl.col("n_wikilink_edits").sum(),
            pl.col("is_revert").sum()                        .alias("revert_count"),
            pl.col("was_reverted").sum()                     .alias("was_reverted_count"),
            pl.col("pageviews").mean().round(0).cast(pl.Int64).alias("avg_pageviews"),
        )
        .with_columns(pl.lit(lang).alias("language"))
    )
    per_lang_frames.append(agg)

    # Separate topic frame
    topic_agg = (
        edits
        .filter(pl.col("top_category").is_not_null())
        .group_by(["user_text", "top_category"])
        .agg(pl.len().alias("topic_edits"))
    )
    topic_frames.append(topic_agg)

    print(f"  -> {agg.shape[0]:,} unique accounts in {lang.upper()}")


# ── STEP 2: GLOBAL AGGREGATION ────────────────────────────────────────────────

section("STEP 2: global aggregation")

all_lang = pl.concat(per_lang_frames)
print(f"\n  Combined rows: {all_lang.shape[0]:,}")

# Sort so languages_active / edits_by_language list comes out ranked by activity
all_lang = all_lang.sort(["user_text", "edit_count"], descending=[False, True])

# ── NEW: Calculate Global Active Days ─────────────────────────────────────────
# Combine all unique date frames across languages, deduplicate globally, and count.
print("  Calculating global active days...")
global_unique_dates = pl.concat(active_days_frames).unique()

global_active_days = (
    global_unique_dates
    .group_by(["user_text", "is_bot"])
    .agg(pl.len().alias("active_days"))
)
# ──────────────────────────────────────────────────────────────────────────────

accounts = (
    all_lang.group_by(["user_text", "is_bot"])
    .agg(
        pl.col("edit_count").sum()                              .alias("total_edits"),
        pl.col("language").count()                              .alias("language_count"),
        pl.col("language")                                      .alias("languages_active"),
        pl.struct(pl.col("language"), pl.col("edit_count").alias("edits"))
                                                                .alias("edits_by_language"),
        pl.col("unique_pages").sum()                            .alias("unique_pages_touched"),
        pl.col("first_edit").min()                              .alias("first_edit"),
        pl.col("last_edit").max()                               .alias("last_edit"),
        pl.col("tagged_edits").sum()                            .alias("tagged_edits"),
        pl.col("word_inserts").sum()                            .alias("total_word_inserts"),
        pl.col("word_removes").sum()                            .alias("total_word_removes"),
        pl.col("n_ref_edits").sum()                             .alias("total_ref_edits"),
        pl.col("n_tmpl_edits").sum()                            .alias("total_tmpl_edits"),
        pl.col("n_extlink_edits").sum()                         .alias("total_extlink_edits"),
        pl.col("n_wikilink_edits").sum()                        .alias("total_wikilink_edits"),
        pl.col("revert_count").sum()                            .alias("revert_count"),
        pl.col("was_reverted_count").sum()                      .alias("was_reverted_count"),
        pl.col("avg_pageviews").mean().round(0).cast(pl.Int64)  .alias("avg_pageviews_edited"),
    )
)

# ── NEW: Merge Active Days into Global Accounts ───────────────────────────────
accounts = accounts.join(global_active_days, on=["user_text", "is_bot"], how="left")
# ──────────────────────────────────────────────────────────────────────────────

print(f"  Unique accounts (global): {accounts.shape[0]:,}")


# ── STEP 2.5: TOP TOPIC PER ACCOUNT ──────────────────────────────────────────

all_topics = pl.concat(topic_frames)

global_topic_counts = (
    all_topics
    .group_by(["user_text", "top_category"])
    .agg(pl.col("topic_edits").sum())
    .sort(["user_text", "topic_edits"], descending=[False, True])
)

topic_summary = (
    global_topic_counts
    .group_by("user_text")
    .agg(
        pl.col("top_category").first()          .alias("top_topic"),
        pl.col("top_category").count()          .alias("unique_topic_count"),
        pl.col("top_category")                  .alias("topics_active"),
    )
)

accounts = accounts.join(topic_summary, on="user_text", how="left")


# ── STEP 3: DERIVED COLUMNS ───────────────────────────────────────────────────

# Pass 1 — base metrics
accounts = accounts.with_columns(
    (
        (pl.col("last_edit") - pl.col("first_edit")).dt.total_milliseconds() / 86_400_000
    ).floor().cast(pl.Int64)                                    .alias("activity_span_days"),

    (pl.col("tagged_edits") / pl.col("total_edits") * 100)
    .round(1)                                                   .alias("tag_rate_pct"),

    (pl.col("total_edits") / pl.col("language_count"))
    .round(1)                                                   .alias("avg_edits_per_language"),

    (pl.col("language_count") > 1)                             .alias("is_multilingual"),

    (pl.col("revert_count") / pl.col("total_edits") * 100)
    .round(1)                                                   .alias("revert_rate_pct"),

    (pl.col("was_reverted_count") / pl.col("total_edits") * 100)
    .round(1)                                                   .alias("reverted_rate_pct"),

    (pl.col("was_reverted_count") >= pl.col("total_edits"))    .alias("completely_reverted"),

    (pl.col("total_word_inserts") + pl.col("total_word_removes"))
                                                                .alias("total_word_changes"),

    (pl.col("total_extlink_edits") + pl.col("total_wikilink_edits"))
                                                                .alias("total_link_edits"),

    (pl.col("unique_pages_touched") / pl.col("total_edits"))
    .round(3)                                                   .alias("focus_score"),
)

# Pass 2 — dependent metrics
accounts = accounts.with_columns(
    (pl.col("total_edits") / (pl.col("activity_span_days") + 1).cast(pl.Float64))
    .round(2)                                                   .alias("edit_velocity"),

    # ── NEW: Edit velocity based on active days ───────────────────────────────
    (pl.col("total_edits") / pl.col("active_days").cast(pl.Float64))
    .round(2)                                                   .alias("edit_velocity_active_days"),
    # ──────────────────────────────────────────────────────────────────────────

    (pl.col("total_word_changes") / pl.col("total_edits"))
    .round(1)                                                   .alias("avg_words_changed"),

    pl.when(pl.col("total_edits") == 1).then(pl.lit("1"))
    .when(pl.col("total_edits") < 5)  .then(pl.lit("2-4"))
    .when(pl.col("total_edits") <= 10).then(pl.lit("5-10"))
    .when(pl.col("total_edits") < 100).then(pl.lit("11-99"))
    .otherwise(pl.lit("100+"))                                  .alias("edit_bucket"),

    pl.when(
        pl.col("total_word_changes") >= pl.max_horizontal(
            pl.col("total_ref_edits"),
            pl.col("total_tmpl_edits"),
            pl.col("total_link_edits"),
        )
    ).then(pl.lit("text"))
    .when(
        pl.col("total_ref_edits") >= pl.max_horizontal(
            pl.col("total_tmpl_edits"),
            pl.col("total_link_edits"),
        )
    ).then(pl.lit("reference"))
    .when(pl.col("total_tmpl_edits") >= pl.col("total_link_edits"))
    .then(pl.lit("template"))
    .otherwise(pl.lit("links"))                                 .alias("primary_edit_type"),
)


# ── STEP 4: SANITY CHECK ──────────────────────────────────────────────────────

section("STEP 4: sanity check")

bots   = accounts.filter(pl.col("is_bot") == True)
humans = accounts.filter(pl.col("is_bot") == False)

print(f"\n  Total: {accounts.shape[0]:,}  |  Bots: {len(bots):,}  |  Humans: {len(humans):,}")
print(f"  Anonymous (null user_text): {accounts.filter(pl.col('user_text').is_null()).shape[0]:,}")

print("\n  Top 10 accounts by total edits:")
print(
    accounts
    .sort("total_edits", descending=True)
    .select([
        "user_text", "is_bot", "total_edits", "active_days", "edit_velocity_active_days", # NEW: Added active days to print block
        "primary_edit_type", "top_topic", "avg_words_changed", "revert_rate_pct",
        "reverted_rate_pct", "edit_velocity", "focus_score",
    ])
    .head(10)
)

print("\n  Primary edit type — bots vs humans:")
print(
    accounts
    .group_by(["is_bot", "primary_edit_type"])
    .agg(pl.len().alias("count"))
    .sort(["is_bot", "primary_edit_type"])
)

print("\n  Top 5 topics per group:")
print(
    accounts
    .filter(pl.col("top_topic").is_not_null())
    .group_by(["is_bot", "top_topic"])
    .agg(pl.len().alias("count"))
    .sort(["is_bot", "count"], descending=[False, True])
    .group_by("is_bot", maintain_order=True)
    .head(5)
)

print("\n  Edit bucket distribution:")
print(
    accounts
    .group_by(["is_bot", "edit_bucket"])
    .agg(pl.len().alias("count"))
    .with_columns(
        pl.col("edit_bucket")
        .replace({"1": 0, "2-4": 1, "5-10": 2, "11-99": 3, "100+": 4}, return_dtype=pl.Int8)
        .alias("_order")
    )
    .sort(["is_bot", "_order"])
    .select(["is_bot", "edit_bucket", "count"])
)


# ── STEP 5: WRITE OUTPUT ──────────────────────────────────────────────────────

section("STEP 5: write to disk")

accounts_out = accounts.with_columns(
    pl.col("first_edit").dt.to_string("%Y-%m-%dT%H:%M:%SZ"),
    pl.col("last_edit").dt.to_string("%Y-%m-%dT%H:%M:%SZ"),
)

print(f"\n  Writing to: {OUTPUT_PATH}")
start = time.time()
accounts_out.write_ndjson(OUTPUT_PATH)
size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
print(f"  -> {accounts.shape[0]:,} records, {size_mb:.1f} MB in {time.time() - start:.2f}s")
print(f"\n  Done. Load in future steps with:")
print(f"    pl.read_ndjson('{OUTPUT_PATH.name}')")
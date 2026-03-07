"""
DataFest 2026 — Export accounts_master to CSV for manual inspection in Excel.

Run: uv run bot_analysis_jona/04_export_accounts_excel.py
"""

import polars as pl
from pathlib import Path

INPUT  = Path(__file__).parent / "accounts_master.ndjson"
OUTPUT = Path(__file__).parent / "accounts_master_all_accounts_new.csv"

# Cap: export only the TOP_N most active accounts. None = export all.
TOP_N = None

# ── LOAD ──────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT.name}...")
df = pl.read_ndjson(INPUT)
print(f"-> {df.shape[0]:,} accounts total")

# ── FLATTEN NESTED COLUMNS ────────────────────────────────────────────────────
# CSV can't hold lists or structs, so convert every nested column to a string.
# languages_active:  ["en", "de"]                  -> "en, de"
# edits_by_language: [{language:"en", edits:100}]  -> "en:100, de:50"
# topics_active:     ["Geography", "Culture"]       -> "Geography, Culture"
df_flat = df.with_columns(
    pl.col("languages_active")
    .list.join(", ")
    .alias("languages_active"),

    pl.col("edits_by_language")
    .list.eval(
        pl.element().struct.rename_fields(["lang", "edits"])
        .struct.field("lang") + ":" +
        pl.element().struct.rename_fields(["lang", "edits"])
        .struct.field("edits").cast(pl.String)
    )
    .list.join(", ")
    .alias("edits_by_language"),

    pl.col("topics_active")
    .list.join(", ")
    .alias("topics_active"),
)

# ── SORT & CAP ────────────────────────────────────────────────────────────────
df_flat = df_flat.sort("total_edits", descending=True)

if TOP_N is not None:
    df_flat = df_flat.head(TOP_N)
    print(f"Capped to top {TOP_N:,} accounts by total_edits")

# ── WRITE CSV ─────────────────────────────────────────────────────────────────
print(f"Writing to {OUTPUT.name}...")
df_flat.write_csv(OUTPUT)
print(f"Done -> {OUTPUT} ({OUTPUT.stat().st_size / 1_048_576:.1f} MB)")

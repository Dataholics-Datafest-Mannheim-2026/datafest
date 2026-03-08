"""
Resolve page_id → page_title from page_info and attach edit counts for Rubinbot.
Reads rubinbot_edits_summary.json and writes rubinbot_pages_with_titles.json in this folder.
"""
import json
from pathlib import Path

import polars as pl
import schemas.edit_types as sety
import schemas.page_info as spi


def main() -> None:
    base = Path(__file__).parent
    repo_root = base.parent.parent
    summary_path = base / "rubinbot_edits_summary.json"
    edit_types_path = repo_root / "data" / "edit_types" / "ruwiki.json.gz"
    page_info_path = repo_root / "data" / "page_info.json.gz"
    out_path = base / "rubinbot_pages_with_titles.json"

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    page_ids_summary = summary["page_ids"]

    # Edit counts per page for Rubinbot (ruwiki)
    df_edits = pl.read_ndjson(edit_types_path).filter(
        pl.col(sety.USER_TEXT) == "Rubinbot"
    )
    df_count = (
        df_edits.group_by(pl.col(sety.PAGE_ID))
        .agg(pl.len().alias("edit_count"))
    )

    # Page info: page_id → page_title (filter ruwiki if present)
    df_page_info = pl.read_ndjson(page_info_path)
    if spi.WIKI_DB in df_page_info.columns:
        df_page_info = df_page_info.filter(pl.col(spi.WIKI_DB) == "ruwiki")
    df_page_info = df_page_info.select([spi.PAGE_ID, spi.PAGE_TITLE, spi.QID])

    # Merge: counts + titles
    result = df_count.join(
        df_page_info,
        on=sety.PAGE_ID,
        how="left",
    ).sort("edit_count", descending=True)

    # Export as list of { page_id, page_title, qid, edit_count }
    rows = result.to_dicts()
    payload = [
        {
            "page_id": r[sety.PAGE_ID],
            "page_title": r[spi.PAGE_TITLE],
            "qid": r.get(spi.QID),
            "edit_count": r["edit_count"],
        }
        for r in rows
    ]

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(payload)} pages to {out_path}")


if __name__ == "__main__":
    main()

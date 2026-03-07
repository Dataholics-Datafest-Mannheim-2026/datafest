import json
from pathlib import Path

import polars as pl
import schemas.edit_types as sety


def main() -> None:
    # Load ru-wiki edit types
    df = pl.read_ndjson("../data/edit_types/ruwiki.json.gz")

    # Filter for the specific user
    user = "Rubinbot"
    df_user = df.filter(pl.col(sety.USER_TEXT) == user)

    if df_user.is_empty():
        print(f"No edits found for user: {user}")
        return

    # Overall number of edits
    total_edits = df_user.height

    # First and last time the user edited something
    ts_series = df_user.select(
        pl.col(sety.REVISION_TIMESTAMP)
        .str.to_datetime(time_zone="UTC", strict=False)
        .alias("ts")
    )["ts"]
    first_edit_ts = ts_series.min()
    last_edit_ts = ts_series.max()

    # Page IDs of the pages the user edited
    page_ids = (
        df_user.select(pl.col(sety.PAGE_ID).unique().sort())
        .to_series()
        .to_list()
    )

    print(f"User: {user}")
    print(f"Total edits: {total_edits}")
    print(f"First edit timestamp (UTC): {first_edit_ts}")
    print(f"Last edit timestamp (UTC): {last_edit_ts}")
    print(f"Number of distinct pages edited: {len(page_ids)}")
    print("Page IDs:")
    print(page_ids)

    # Export all data to JSON file in this folder
    out_path = Path(__file__).parent / "rubinbot_edits_summary.json"
    payload = {
        "user": user,
        "total_edits": total_edits,
        "first_edit_utc": first_edit_ts.isoformat() if first_edit_ts is not None else None,
        "last_edit_utc": last_edit_ts.isoformat() if last_edit_ts is not None else None,
        "page_ids": page_ids,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()


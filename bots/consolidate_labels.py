"""
Consolidate bot edits per user with label filtering.

Reads the pre-computed edits_per_user_per_label.json and produces a per-user
summary table.  Labels can be filtered by exact name, prefix, or regex.

Usage:
    python consolidate_labels.py                          # all labels
    python consolidate_labels.py --labels Culture.Sports STEM.Biology
    python consolidate_labels.py --prefix Culture         # all Culture.* labels
    python consolidate_labels.py --prefix Culture STEM    # Culture.* + STEM.*
    python consolidate_labels.py --regex "Geography.*Europe"
    python consolidate_labels.py --top-label              # add column with each user's most-edited label
    python consolidate_labels.py -o my_output.csv         # custom output path
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parent
INPUT = BOTS_DIR / "edits_per_user_per_label.json"


def load_data() -> dict:
    with open(INPUT) as f:
        return json.load(f)


def match_labels(all_labels: list[str], *, exact: list[str] | None,
                 prefixes: list[str] | None, pattern: str | None) -> set[str]:
    """Return the set of labels that pass any of the given filters."""
    if not exact and not prefixes and not pattern:
        return set(all_labels)

    kept: set[str] = set()
    if exact:
        for lbl in exact:
            if lbl in all_labels:
                kept.add(lbl)
            else:
                print(f"  warning: label '{lbl}' not found in data", file=sys.stderr)
    if prefixes:
        for pfx in prefixes:
            kept |= {l for l in all_labels if l.startswith(pfx)}
    if pattern:
        rx = re.compile(pattern)
        kept |= {l for l in all_labels if rx.search(l)}
    return kept


def build_table(per_user: dict, labels: list[str], add_top: bool) -> list[dict]:
    """Build one row per user with a column per label + total."""
    rows = []
    for user, label_counts in per_user.items():
        row: dict = {"user_text": user}
        total = 0
        best_label = ""
        best_count = 0
        for lbl in labels:
            c = label_counts.get(lbl, 0)
            row[lbl] = c
            total += c
            if c > best_count:
                best_count = c
                best_label = lbl
        if total == 0:
            continue
        row["total"] = total
        if add_top:
            row["top_label"] = best_label
        rows.append(row)
    rows.sort(key=lambda r: -r["total"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Consolidate bot edits per label per user.")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="Exact label names to include")
    parser.add_argument("--prefix", nargs="+", metavar="PFX",
                        help="Include labels starting with these prefixes")
    parser.add_argument("--regex", metavar="PATTERN",
                        help="Include labels matching this regex")
    parser.add_argument("--top-label", action="store_true",
                        help="Add a column with each user's most-edited label")
    parser.add_argument("--list-labels", action="store_true",
                        help="Print all available labels and exit")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    data = load_data()
    all_labels = sorted(data["per_label"].keys())

    if args.list_labels:
        for lbl in all_labels:
            print(f"  {data['per_label'][lbl]:>8,}  {lbl}")
        return

    kept = match_labels(all_labels, exact=args.labels,
                        prefixes=args.prefix, pattern=args.regex)
    labels = sorted(kept)

    if not labels:
        print("No labels matched the filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Selected {len(labels)} label(s):", file=sys.stderr)
    for lbl in labels:
        print(f"  {lbl}", file=sys.stderr)

    rows = build_table(data["per_user"], labels, args.top_label)
    print(f"{len(rows):,} users with at least 1 edit in selected labels.", file=sys.stderr)

    # Write CSV
    fieldnames = ["user_text"] + labels + ["total"]
    if args.top_label:
        fieldnames.append("top_label")

    if args.output:
        out = open(args.output, "w", newline="")
    else:
        out = sys.stdout

    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.output:
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

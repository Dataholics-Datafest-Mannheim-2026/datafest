"""
Count edits per bot user per predicted page label.

Joins:
  - bot_usernames_unfiltered.json  (all_bot_cluster_users)
  - page_info.json                 (page_id -> top predicted_label)
  - consolidated.json              (edit records: user_text, page_id)

Output: CSV with columns (user_text, predicted_label, edit_count)
"""

import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "data"
BOTS_DIR = BASE / "bots"

# ── 1. Load bot usernames ────────────────────────────────────────────────────
print("Loading bot usernames…", flush=True)
with open(BOTS_DIR / "clustering" / "bot_usernames_unfiltered.json") as f:
    bot_data = json.load(f)
bot_users = set(bot_data["all_bot_cluster_users"])
print(f"  {len(bot_users):,} bot users loaded")

# ── 2. Build page_id → top predicted label mapping ──────────────────────────
print("Loading page_info.json…", flush=True)
page_label: dict[int, str] = {}
with open(DATA / "page_info.json") as f:
    for line in f:
        rec = json.loads(line)
        pid = rec["page_id"]
        labels = rec.get("predicted_labels", [])
        if labels:
            top = max(labels, key=lambda x: x["probability"])
            page_label[pid] = top["label"]
print(f"  {len(page_label):,} page → label mappings")

# ── 3. Stream consolidated.json – count bot edits per (user, label) ─────────
print("Streaming consolidated.json (this may take a while)…", flush=True)
counts: Counter[tuple[str, str]] = Counter()
n_total = 0
n_bot = 0
n_matched = 0

with open(DATA / "edit_types" / "consolidated.json") as f:
    for line in f:
        n_total += 1
        if n_total % 5_000_000 == 0:
            print(f"  processed {n_total:>12,} lines  (bot edits matched: {n_matched:,})", flush=True)

        # Fast pre-filter: skip lines whose user_text is definitely not a bot.
        # json.loads is expensive; extract user_text with a quick string search.
        idx = line.find('"user_text":"')
        if idx == -1:
            idx = line.find('"user_text": "')
            if idx == -1:
                continue
            ustart = idx + len('"user_text": "')
        else:
            ustart = idx + len('"user_text":"')
        uend = line.index('"', ustart)
        user_text = line[ustart:uend]

        # Decode JSON unicode escapes (e.g. \u00ea) if present
        if "\\" in user_text:
            user_text = json.loads(f'"{user_text}"')

        if user_text not in bot_users:
            continue
        n_bot += 1

        # Only do full parse for bot edits (rare → fast overall)
        rec = json.loads(line)
        pid = rec["page_id"]
        label = page_label.get(pid)
        if label is None:
            label = "UNKNOWN"
        counts[(user_text, label)] += 1
        n_matched += 1

print(f"\nDone. {n_total:,} total edits, {n_bot:,} bot edits, {n_matched:,} matched to labels.")

# ── 4. Write output ─────────────────────────────────────────────────────────
out_path = BOTS_DIR / "edits_per_user_per_label.csv"
print(f"Writing results to {out_path}…", flush=True)

rows = sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
with open(out_path, "w") as f:
    f.write("user_text,predicted_label,edit_count\n")
    for (user, label), count in rows:
        # Escape commas/quotes in usernames
        safe_user = user.replace('"', '""')
        if "," in safe_user or '"' in safe_user:
            safe_user = f'"{safe_user}"'
        safe_label = label.replace('"', '""')
        if "," in safe_label or '"' in safe_label:
            safe_label = f'"{safe_label}"'
        f.write(f"{safe_user},{safe_label},{count}\n")

print(f"  {len(rows):,} rows written.")

# Also write a summary JSON with per-user totals and per-label totals
summary = {
    "per_user": {},
    "per_label": Counter(),
}
for (user, label), count in counts.items():
    summary["per_user"].setdefault(user, {})[label] = count
    summary["per_label"][label] += count

summary["per_label"] = dict(sorted(summary["per_label"].items(), key=lambda x: -x[1]))

summary_path = BOTS_DIR / "edits_per_user_per_label.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  Summary JSON written to {summary_path}")

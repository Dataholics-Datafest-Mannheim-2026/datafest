import polars as pl
import json
from pathlib import Path

# Data
page_info = pl.read_ndjson('../data/page_info.json')
page_views = pl.read_ndjson('../data/page_views.json')

language_files = {
    "ar": "../data/arwiki.json.gz",
    "de": "../data/dewiki.json.gz",
    "en": "../data/enwiki.json.gz",
    "es": "../data/eswiki.json.gz",
    "fr": "../data/frwiki.json.gz",
    "it": "../data/itwiki.json.gz",
    "nl": "../data/nlwiki.json.gz",
    "pl": "../data/plwiki.json.gz",
    "ru": "../data/ruwiki.json.gz",
    "sv": "../data/svwiki.json.gz",
}

results = []

for lang, path in language_files.items():
    df = pl.read_ndjson(path)
    total = len(df)
    bot_count = df.filter(pl.col("is_bot") == True).shape[0]
    bot_share = bot_count / total * 100
    results.append({
        "language": f"{lang}wiki",
        "total_edits": total,
        "bot_edits": bot_count,
        "human_edits": total - bot_count,
        "bot_share_%": round(bot_share, 2)
    })

result_df = pl.DataFrame(results).sort("bot_share_%", descending=True)
print(result_df)
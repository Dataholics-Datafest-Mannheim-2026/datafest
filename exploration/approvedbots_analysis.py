import polars as pl
import requests
import time
from pathlib import Path


# ── 1. Fetch approved bots from each language's Wikipedia ──────────────────
def fetch_approved_bots(lang: str) -> set[str]:
    """Fetch all approved bot usernames from a Wikipedia language edition."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "allusers",
        "augroup": "bot",
        "aulimit": "500",
        "format": "json"
    }
    headers = {"User-Agent": "DataFestBot/1.0 (research project; contact@example.com)"}

    bots = set()
    while True:
        for attempt in range(3):  # retry up to 3 times
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}")
                data = response.json()
                break
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {lang}: {e}")
                time.sleep(2)
        else:
            print(f"  Giving up on {lang} after 3 attempts")
            return bots

        for user in data["query"]["allusers"]:
            bots.add(user["name"])

        if "continue" in data:
            params["aufrom"] = data["continue"]["aufrom"]
        else:
            break
        time.sleep(0.5)
    return bots


all_languages = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]

print("Fetching approved bot lists...")
approved_bots_per_lang = {}
for lang in all_languages:
    bots = fetch_approved_bots(lang)
    approved_bots_per_lang[lang] = bots
    print(f"  {lang}wiki: {len(bots)} approved bots found")

all_approved_bots = set().union(*approved_bots_per_lang.values())
print(f"\nTotal unique approved bots across all languages: {len(all_approved_bots)}")

# ── 2. Add 'approved_bot' column to each DataFrame ────────────────────────

language_files = {
    "ar": "../data/arwiki.json.gz",
    "de": "../data/dewiki.json.gz",
    "es": "../data/eswiki.json.gz",
    "fr": "../data/frwiki.json.gz",
    "it": "../data/itwiki.json.gz",
    "nl": "../data/nlwiki.json.gz",
    "pl": "../data/plwiki.json.gz",
    "ru": "../data/ruwiki.json.gz",
    "sv": "../data/svwiki.json.gz",
}

edit_dfs = {}
for lang, path in language_files.items():
    if not Path(path).exists():
        print(f"Skipping {lang} — file not found")
        continue

    df = pl.read_ndjson(path)
    df = df.with_columns(
        pl.col("user_text")
        .is_in(list(all_approved_bots))
        .alias("approved_bot")
    )
    edit_dfs[lang] = df
    print(f"{lang}wiki: {df['approved_bot'].sum()} approved bot edits")

# ── 3. Sanity check ───────────────────────────────────────────────────────

# ── 3. Sanity check ───────────────────────────────────────────────────────

for lang, df in edit_dfs.items():
    total = len(df)
    is_bot = df["is_bot"].sum()
    approved = df["approved_bot"].sum()
    rogue = df.filter((pl.col("is_bot") == True) & (pl.col("approved_bot") == False)).shape[0]
    approved_of_bots = df.filter(
        (pl.col("is_bot") == True) & (pl.col("approved_bot") == True)
    ).shape[0]

    print(f"{lang}wiki | "
          f"is_bot: {is_bot} ({is_bot / total * 100:.1f}%) | "
          f"approved_bot: {approved} ({approved / total * 100:.1f}%) | "
          f"rogue bots: {rogue} | "
          f"approved among all bots: {approved_of_bots / is_bot * 100:.1f}%")
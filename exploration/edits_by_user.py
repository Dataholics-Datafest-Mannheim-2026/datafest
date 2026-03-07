import json

import polars as pl
import schemas.edit_types as sety

# Part of Username to match in user_text column
user_name = "Rubinbot"

# load blocked bot usernames
with open("../data/outside/blocked_bots.json", encoding="utf-8") as f:
    blocked_bots = json.load(f)  # list[str]

languages = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
users_by_lang = []
rows = []
for lang in languages:
    df = pl.read_ndjson(f"../data/edit_types/{lang}wiki.json.gz")
    edits_by_user = df.filter(pl.col("user_text").str.contains(user_name)).select(pl.col("user_text")).unique()
    users_by_lang.append({"lang": lang, "edits": edits_by_user})

    present = (
        df.filter(pl.col(sety.USER_TEXT).is_in(blocked_bots))
        .filter(pl.col(sety.IS_BOT) == False) # Filter whether Wikipedia spotted the bot as such.
        .select(pl.col(sety.USER_TEXT)).unique()
        .with_columns(pl.lit(lang))
        .sort("user_text")
        .head(14)
    )
    if present.height > 0:
        rows.append(present)

if rows:
    result = pl.concat(rows)
    print(result)
else:
    print("No blocked bots found in dataset.")
# {lang_tag : dataFrame}
#print("-- Rubinbot: --")
#print(users_by_lang)

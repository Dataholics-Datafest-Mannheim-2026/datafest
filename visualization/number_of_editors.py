import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import schemas.edit_types as sety

LANGUAGE_CODES = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
# LANGUAGE_CODES = ["de"]

rows = []
for lang in LANGUAGE_CODES:
    df = pl.read_ndjson("../data/edit_types/" + lang + "wiki.json.gz")
    n_bot = (
        df
        .filter(pl.col("is_bot") == True)
        .select(pl.col(sety.USER_TEXT).n_unique())
        .item()
    )
    n_human = (
        df
        .filter(pl.col("is_bot") == False)
        .select(pl.col(sety.USER_TEXT).n_unique())
        .item()
    )
    rows.append({"lang": lang, "bot": n_bot, "human": n_human})
print(rows)

summary = pl.DataFrame(rows)
print(summary)


langs = [r["lang"] for r in rows]
bots = [r["bot"] for r in rows]
humans = [r["human"] for r in rows]

x = np.arange(len(langs))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, humans, width, label="non-bot")
ax.bar(x + width/2, bots,   width, label="bot")
ax.set_xticks(x)
ax.set_xticklabels(langs)
ax.set_ylabel("Unique editors")
ax.set_title("Unique editors per language (bot vs non-bot)")
ax.legend()

plt.show()

import polars as pl
import matplotlib.pyplot as plt
import schemas.edit_types as sety

LANGUAGE_CODES = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
# LANGUAGE_CODES = ["de"]

rows = []
for lang in LANGUAGE_CODES:
    df = pl.read_ndjson("../data/edit_types/" + lang + "wiki.json.gz")
    n_editors = df.select(pl.col(sety.USER_TEXT).n_unique()).item()
    print(n_editors)
    rows.append({"lang": lang, "n_editors": n_editors})
print(rows)

summary = pl.DataFrame(rows)
print(summary)


langs = [r["lang"] for r in rows]
n_editors = [r["n_editors"] for r in rows]
plt.bar(langs, n_editors)
plt.xlabel("Language")
plt.ylabel("Number of editors")
plt.title("Unique editors per language")
plt.show()

import polars as pl

# languages_to_look_at = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
# languages_to_look_at = ["ru", "de"]
languages_to_look_at = ["de"]
for wiki_db in languages_to_look_at:
    df = pl.read_ndjson("../data/edit_types/"+wiki_db+"wiki.json.gz")
    print(df.head(5))

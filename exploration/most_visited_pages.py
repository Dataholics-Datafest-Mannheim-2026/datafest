import polars as pl


RAW_PAGE_VIEWS_JSON_GZ = "../data/page_views.json.gz"

# Columns we have: wiki_db | page_id | day | pageviews
# visits.info()

views_de = pl.read_ndjson(RAW_PAGE_VIEWS_JSON_GZ).filter(pl.col("wiki_db") == "dewiki")
print(len(views_de))

top_n = views_de.sort("pageviews", descending=True).head(10)

print(top_n)

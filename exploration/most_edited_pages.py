import polars as pl

# Number of results
top_k = 10

# TODO: Iterate over the different files to get result for every language
df = pl.read_ndjson("../data/edit_types/dewiki.json.gz")

df_count = df.group_by(pl.col("page_id")).len()
df_count_topN = df_count.top_k(k=top_k, by="len").sort(by="len", descending=True)

print(df_count_topN)

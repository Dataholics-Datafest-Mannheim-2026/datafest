import polars as pl

COLUMN_NAME_NUM_EDITS = "num_edits"

# table<wiki_db: string, items: table<num_edits: int, page_id: int, page_title: string>>

# Number of results
top_k = 10

# TODO: Iterate over the different files to get result for every language
df_edits = pl.read_ndjson("../data/edit_types/dewiki.json.gz")

df_count = df_edits.group_by(pl.col("page_id")).len(COLUMN_NAME_NUM_EDITS)
df_count_topN = df_count.top_k(k=top_k, by=COLUMN_NAME_NUM_EDITS).sort(by=COLUMN_NAME_NUM_EDITS, descending=True)

df_page_info = pl.read_ndjson("../data/page_info.json.gz").select(["page_id", "page_title"])

result = df_count_topN.join(df_page_info, on="page_id", how="left").top_k(k=top_k, by=COLUMN_NAME_NUM_EDITS)

print(result)
print(result.dtypes)
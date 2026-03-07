import polars as pl
import schemas.page_info as spi
import schemas.custom as sc
import schemas.edit_types as sety

# table<wiki_db: string, items: table<num_edits: int, page_id: int, page_title: string>>

# Number of results
top_k = 10

# TODO: Iterate over the different files to get result for every language
df_edits = (pl.read_ndjson("../data/edit_types/dewiki.json.gz")
            .filter(pl.col(sety.IS_BOT) == True))
df_count = df_edits.group_by(pl.col(sety.PAGE_ID)).len(sc.NUM_EDITS)
df_count_topN = df_count.top_k(k=top_k, by=sc.NUM_EDITS).sort(by=sc.NUM_EDITS, descending=True)

# Join on page_id for page_title
df_page_info = pl.read_ndjson("../data/page_info.json.gz").select([spi.PAGE_ID, spi.PAGE_TITLE, spi.PREDICTED_LABELS])
result = df_count_topN.join(df_page_info, on=sety.PAGE_ID, how="left").top_k(k=top_k, by=sc.NUM_EDITS)

# The Pages, which were edited by is_bot == True most & how often they were edited by bots
print("-- Most edited pages by Bots --")
print(result)

print("-- Most edited topics by Bots --")
# TODO

print("-- Most edited pages by Humans --")
# TODO

print("-- Most edited topics by Humans --")
# TODO

print("-- Comparision Bots & Humans --")
# TODO: Wie schneiden die Top Seiten der Bots bei den Humans ab?
# TODO: Wie schneiden die Top Seiten der Humans bei den Bots ab?
# TODO: Wie schneiden die Top Topics der Bots bei den Humans ab?
# TODO: Wie schneiden die Top Topics der Humans bei den Bots ab?

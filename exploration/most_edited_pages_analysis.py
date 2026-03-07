import polars as pl
import schemas.page_info as spi
import schemas.custom as sc
import schemas.edit_types as sety

# table<wiki_db: string, items: table<num_edits: int, page_id: int, page_title: string>>

# Number of results
top_k = 10

# TODO: Iterate over the different files to get result for every language
df_edits_is_bot = (pl.read_ndjson("../data/edit_types/dewiki.json.gz")
                   .filter(pl.col(sety.IS_BOT) == True))
df_count_is_bot = df_edits_is_bot.group_by(pl.col(sety.PAGE_ID)).len(sc.NUM_EDITS)
df_count_topN_is_bot = df_count_is_bot.top_k(k=top_k, by=sc.NUM_EDITS).sort(by=sc.NUM_EDITS, descending=True)

# Join on page_id for page_title
df_page_info = pl.read_ndjson("../data/page_info.json.gz").select([spi.PAGE_ID, spi.PAGE_TITLE, spi.PREDICTED_LABELS])
result_is_bot = df_count_topN_is_bot.join(df_page_info, on=sety.PAGE_ID, how="left")

# The Pages, which were edited by is_bot == True most & how often they were edited by bots
print("-- Most edited pages by Bots --")
print(result_is_bot.top_k(k=top_k, by=sc.NUM_EDITS))

print("-- Most edited topics by Bots --")
# We want: top 10 topics of bot edits and how many bot edits they have.
# Default: assign each page to its TOP-1 predicted label (highest probability),
# then sum that page's bot edit count into the topic.
df_pages_is_bot = df_count_is_bot.join(
    df_page_info.select([spi.PAGE_ID, spi.PREDICTED_LABELS]),
    on=sety.PAGE_ID,
    how="left",
)

df_page_topics_is_bot = (
    df_pages_is_bot
    .select([sety.PAGE_ID, sc.NUM_EDITS, spi.PREDICTED_LABELS])
    .filter(pl.col(spi.PREDICTED_LABELS).is_not_null())
    .explode(spi.PREDICTED_LABELS)
    .unnest(spi.PREDICTED_LABELS)  # -> columns: 'label', 'probability'
    .sort([sety.PAGE_ID, "probability"], descending=[False, True])
    .unique(subset=[sety.PAGE_ID], keep="first")
    .select(
        [
            pl.col("label").alias("topic"),
            pl.col(sc.NUM_EDITS).alias("num_bot_edits"),
        ]
    )
)

df_top_topics_is_bot = (
    df_page_topics_is_bot
    .group_by("topic")
    .agg(pl.col("num_bot_edits").sum().alias("num_bot_edits"))
    .sort("num_bot_edits", descending=True)
    .head(top_k)
)

print(df_top_topics_is_bot)

print("-- Most edited pages by Humans --")
df_edits_is_human = (
    pl.read_ndjson("../data/edit_types/dewiki.json.gz")
    .filter(pl.col(sety.IS_BOT) == False)
)
df_count_is_human = df_edits_is_human.group_by(pl.col(sety.PAGE_ID)).len(sc.NUM_EDITS)
df_count_topN_is_human = (
    df_count_is_human.top_k(k=top_k, by=sc.NUM_EDITS).sort(by=sc.NUM_EDITS, descending=True)
)

# Join on page_id for page_title / predicted labels
result_is_human = df_count_topN_is_human.join(df_page_info, on=sety.PAGE_ID, how="left")

print(result_is_human.top_k(k=top_k, by=sc.NUM_EDITS))

print("-- Most edited topics by Humans --")
df_pages_is_human = df_count_is_human.join(
    df_page_info.select([spi.PAGE_ID, spi.PREDICTED_LABELS]),
    on=sety.PAGE_ID,
    how="left",
)

df_page_topics_is_human = (
    df_pages_is_human
    .select([sety.PAGE_ID, sc.NUM_EDITS, spi.PREDICTED_LABELS])
    .filter(pl.col(spi.PREDICTED_LABELS).is_not_null())
    .explode(spi.PREDICTED_LABELS)
    .unnest(spi.PREDICTED_LABELS)  # -> columns: 'label', 'probability'
    .sort([sety.PAGE_ID, "probability"], descending=[False, True])
    .unique(subset=[sety.PAGE_ID], keep="first")  # pick TOP-1 topic per page
    .select(
        [
            pl.col("label").alias("topic"),
            pl.col(sc.NUM_EDITS).alias("num_human_edits"),
        ]
    )
)

df_top_topics_is_human = (
    df_page_topics_is_human
    .group_by("topic")
    .agg(pl.col("num_human_edits").sum().alias("num_human_edits"))
    .sort("num_human_edits", descending=True)
    .head(top_k)
)

print(df_top_topics_is_human)

print("-- Comparision Bots & Humans --")
# TODO: Wie schneiden die Top Seiten der Bots bei den Humans ab?
# TODO: Wie schneiden die Top Seiten der Humans bei den Bots ab?
# TODO: Wie schneiden die Top Topics der Bots bei den Humans ab?
# TODO: Wie schneiden die Top Topics der Humans bei den Bots ab?

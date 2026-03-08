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
#
# 1) For the top-10 human-edited pages:
#    - share of each page in ALL human edits
#    - share of those same pages in ALL bot edits
#
# 2) For the top-10 bot-edited pages:
#    - share of each page in ALL bot edits
#    - share of those same pages in ALL human edits

total_bot_edits = df_count_is_bot.select(
    pl.col(sc.NUM_EDITS).sum().alias("total_bot_edits")
).to_series()[0]

total_human_edits = df_count_is_human.select(
    pl.col(sc.NUM_EDITS).sum().alias("total_human_edits")
).to_series()[0]

# 1) Human-centric view: top human pages with human & bot shares
df_human_pages_shares = (
    result_is_human
    # bring in bot edit counts for these pages
    .join(df_count_is_bot, on=sety.PAGE_ID, how="left", suffix="_bot")
    .with_columns(
        pl.col(f"{sc.NUM_EDITS}_bot").fill_null(0)
    )
    .with_columns(
        (pl.col(sc.NUM_EDITS) / total_human_edits).alias("human_share_all_human_edits"),
        (pl.col(f"{sc.NUM_EDITS}_bot") / total_bot_edits).alias("bot_share_all_bot_edits"),
    )
    .select(
        [
            pl.col(sety.PAGE_ID),
            pl.col(spi.PAGE_TITLE),
            pl.col(sc.NUM_EDITS).alias("num_human_edits"),
            pl.col("human_share_all_human_edits"),
            pl.col(f"{sc.NUM_EDITS}_bot").alias("num_bot_edits_on_human_top_pages"),
            pl.col("bot_share_all_bot_edits"),
        ]
    )
)

print("-- Share comparison on top human-edited pages --")
print(df_human_pages_shares)

# 2) Bot-centric view: top bot pages with bot & human shares
df_bot_pages_shares = (
    result_is_bot
    # bring in human edit counts for these pages
    .join(df_count_is_human, on=sety.PAGE_ID, how="left", suffix="_human")
    .with_columns(
        pl.col(f"{sc.NUM_EDITS}_human").fill_null(0)
    )
    .with_columns(
        (pl.col(sc.NUM_EDITS) / total_bot_edits).alias("bot_share_all_bot_edits"),
        (pl.col(f"{sc.NUM_EDITS}_human") / total_human_edits).alias("human_share_all_human_edits"),
    )
    .select(
        [
            pl.col(sety.PAGE_ID),
            pl.col(spi.PAGE_TITLE),
            pl.col(sc.NUM_EDITS).alias("num_bot_edits"),
            pl.col("bot_share_all_bot_edits"),
            pl.col(f"{sc.NUM_EDITS}_human").alias("num_human_edits_on_bot_top_pages"),
            pl.col("human_share_all_human_edits"),
        ]
    )
)

print("-- Share comparison on top bot-edited pages --")
print(df_bot_pages_shares)

# Topic-level comparison (analogous to pages)
# 3) For the top-10 human topics:
#    - share of all human edits that fall on pages with that top topic
#    - share of all bot edits that fall on pages with that top topic
#
# 4) For the top-10 bot topics:
#    - share of all bot edits that fall on pages with that top topic
#    - share of all human edits that fall on pages with that top topic

# Reuse df_page_topics_is_human / df_page_topics_is_bot (page -> top-1 topic + edits)

# First, get total edits per topic for humans and bots for ALL topics
df_topic_totals_human = (
    df_page_topics_is_human
    .group_by("topic")
    .agg(pl.col("num_human_edits").sum().alias("num_human_edits_total"))
)

df_topic_totals_bot = (
    df_page_topics_is_bot
    .group_by("topic")
    .agg(pl.col("num_bot_edits").sum().alias("num_bot_edits_total"))
)

total_human_edits_topics = (
    df_topic_totals_human.select(pl.col("num_human_edits_total").sum())
    .to_series()[0]
)

total_bot_edits_topics = (
    df_topic_totals_bot.select(pl.col("num_bot_edits_total").sum())
    .to_series()[0]
)

# 3) Human-centric: top human topics with human & bot shares
df_human_topics_shares = (
    df_top_topics_is_human
    .join(df_topic_totals_bot, on="topic", how="left")
    .with_columns(
        pl.col("num_bot_edits_total").fill_null(0)
    )
    .with_columns(
        (pl.col("num_human_edits") / total_human_edits_topics).alias("human_topic_share_all_human_edits"),
        (pl.col("num_bot_edits_total") / total_bot_edits_topics).alias("bot_topic_share_all_bot_edits"),
    )
    .select(
        [
            pl.col("topic"),
            pl.col("num_human_edits"),
            pl.col("human_topic_share_all_human_edits"),
            pl.col("num_bot_edits_total").alias("num_bot_edits_on_human_top_topics"),
            pl.col("bot_topic_share_all_bot_edits"),
        ]
    )
)

print("-- Share comparison on top human topics --")
print(df_human_topics_shares)

# 4) Bot-centric: top bot topics with bot & human shares
df_bot_topics_shares = (
    df_top_topics_is_bot
    .join(df_topic_totals_human, on="topic", how="left")
    .with_columns(
        pl.col("num_human_edits_total").fill_null(0)
    )
    .with_columns(
        (pl.col("num_bot_edits") / total_bot_edits_topics).alias("bot_topic_share_all_bot_edits"),
        (pl.col("num_human_edits_total") / total_human_edits_topics).alias("human_topic_share_all_human_edits"),
    )
    .select(
        [
            pl.col("topic"),
            pl.col("num_bot_edits"),
            pl.col("bot_topic_share_all_bot_edits"),
            pl.col("num_human_edits_total").alias("num_human_edits_on_bot_top_topics"),
            pl.col("human_topic_share_all_human_edits"),
        ]
    )
)

print("-- Share comparison on top bot topics --")
print(df_bot_topics_shares)

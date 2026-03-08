import polars as pl
import matplotlib.pyplot as plt
import schemas.edit_types as sety

LANGUAGE_CODES = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
# LANGUAGE_CODES = ["de"]


def main() -> None:
    daily_frames: list[pl.DataFrame] = []

    for lang in LANGUAGE_CODES:
        df = pl.read_ndjson(f"../data/edit_types/{lang}wiki.json.gz")

        df = df.with_columns(
            pl.col("revision_timestamp")
            .str.to_datetime(time_zone="UTC", strict=False)
            .alias("ts")
        ).with_columns(pl.col("ts").dt.date().alias("date"))

        per_day = (
            df.group_by("date")
            .agg(pl.col(sety.USER_TEXT).n_unique().alias("n_editors"))
            .with_columns(pl.lit(lang).alias("lang"))
            .sort("date")
        )

        daily_frames.append(per_day)

    daily = pl.concat(daily_frames)

    fig, ax = plt.subplots()

    for lang in LANGUAGE_CODES:
        sub = daily.filter(pl.col("lang") == lang).sort("date")
        if sub.is_empty():
            continue

        dates = sub["date"].to_list()
        counts = sub["n_editors"].to_list()

        ax.plot(dates, counts, label=lang)

    ax.set_xlabel("Date")
    ax.set_ylabel("Unique editors per day")
    ax.set_title("Daily unique editors per language")
    ax.legend()
    fig.autofmt_xdate()

    plt.show()


if __name__ == "__main__":
    main()

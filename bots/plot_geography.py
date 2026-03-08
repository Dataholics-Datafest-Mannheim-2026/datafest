import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

labels = [
    "Geography.Geographical",
    "Geography.Regions.Africa.Africa*",
    "Geography.Regions.Africa.Central_Africa",
    "Geography.Regions.Africa.Eastern_Africa",
    "Geography.Regions.Africa.Northern_Africa",
    "Geography.Regions.Africa.Southern_Africa",
    "Geography.Regions.Africa.Western_Africa",
    "Geography.Regions.Americas.Central_America",
    "Geography.Regions.Americas.North_America",
    "Geography.Regions.Americas.South_America",
    "Geography.Regions.Asia.Asia*",
    "Geography.Regions.Asia.Central_Asia",
    "Geography.Regions.Asia.East_Asia",
    "Geography.Regions.Asia.North_Asia",
    "Geography.Regions.Asia.South_Asia",
    "Geography.Regions.Asia.Southeast_Asia",
    "Geography.Regions.Asia.West_Asia",
    "Geography.Regions.Europe.Eastern_Europe",
    "Geography.Regions.Europe.Europe*",
    "Geography.Regions.Europe.Northern_Europe",
    "Geography.Regions.Europe.Southern_Europe",
    "Geography.Regions.Europe.Western_Europe",
    "Geography.Regions.Oceania",
]
values = [
    5588, 15033, 5, 47, 4787, 166, 46,
    7795, 44544, 49759,
    40210, 450, 5928, 17647, 3322, 1736, 36947,
    45253, 88513, 21034, 42114, 59865,
    3416,
]

# Shorten labels for display
short = [l.replace("Geography.Regions.", "").replace("Geography.", "") for l in labels]

# Color by region group
colors = []
for l in labels:
    if "Africa" in l:
        colors.append("#e6994d")
    elif "Americas" in l:
        colors.append("#4daf4a")
    elif "Asia" in l:
        colors.append("#e55964")
    elif "Europe" in l:
        colors.append("#377eb8")
    elif "Oceania" in l:
        colors.append("#984ea3")
    else:
        colors.append("#999999")

# Sort by value
paired = sorted(zip(values, short, colors), reverse=False)
values_s, short_s, colors_s = zip(*paired)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(values_s)), values_s, color=colors_s, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(short_s)))
ax.set_yticklabels(short_s, fontsize=9)
ax.set_xlabel("Bot edits")
ax.set_title("Bot Edits by Geography Label")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Add value annotations
for bar, val in zip(bars, values_s):
    ax.text(bar.get_width() + 800, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8)

# Legend for region groups
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e6994d", label="Africa"),
    Patch(facecolor="#4daf4a", label="Americas"),
    Patch(facecolor="#e55964", label="Asia"),
    Patch(facecolor="#377eb8", label="Europe"),
    Patch(facecolor="#984ea3", label="Oceania"),
    Patch(facecolor="#999999", label="Other"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("bots/geography_edits.png", dpi=150)
plt.savefig("bots/geography_edits.svg")
print("Saved bots/geography_edits.png and .svg")

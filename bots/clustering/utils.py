import polars as pl
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ALL_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_entity_list(path: str) -> list:
    """Load the precomputed feature JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_is_bot_all_languages(data_dir: str) -> pl.DataFrame:
    """Load is_bot ground truth from all available language editions."""
    dfs = []
    for lang in ALL_LANGUAGES:
        path = Path(data_dir) / f"{lang}wiki.json.gz"
        if not path.exists():
            print(f"  Skipping {lang}wiki – file not found")
            continue
        print(f"  Loading {lang}wiki...")
        df = pl.read_ndjson(path).select(["user_text", "is_bot"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No language files found in {data_dir}")

    return (
        pl.concat(dfs)
        .group_by("user_text")
        .agg(pl.col("is_bot").max().alias("is_bot"))
    )


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def aggregate_edits_per_day(edits_per_day: list) -> dict:
    """Flatten list of {day: count} dicts into aggregate stats."""
    counts = []
    for d in edits_per_day:
        if d is not None:
            counts.extend(d.values())
    counts = [c for c in counts if c is not None]
    if not counts:
        return {"avg_edits_per_day": 0, "std_edits_per_day": 0,
                "max_edits_per_day": 0, "active_days": 0}
    return {
        "avg_edits_per_day": float(np.mean(counts)),
        "std_edits_per_day": float(np.std(counts)),
        "max_edits_per_day": float(np.max(counts)),
        "active_days":       int(len(counts)),
    }


def aggregate_revision_tags(revision_tags: list, all_tags: set) -> dict:
    """Convert tag counts to percentages, one column per tag."""
    merged = {}
    for d in revision_tags:
        if d:
            for k, v in d.items():
                merged[k] = merged.get(k, 0) + v
    total = sum(merged.values()) or 1
    return {f"pct_tag_{tag}": merged.get(tag, 0) / total for tag in all_tags}


def collect_all_tags(entity_list: list) -> set:
    """Find every unique revision tag across all users."""
    tags = set()
    for user in entity_list:
        for d in user.get("revision_tags", []):
            if d:
                tags.update(d.keys())
    return tags


def build_feature_matrix(entity_list: list, is_bot_df: pl.DataFrame) -> pl.DataFrame:
    """Build a flat feature DataFrame ready for PCA + clustering."""
    all_tags      = collect_all_tags(entity_list)
    is_bot_lookup = dict(zip(is_bot_df["user_text"].to_list(),
                             is_bot_df["is_bot"].to_list()))
    rows = []
    for user in entity_list:
        row = {
            "user_text": user["user_text"],
            "is_bot":    bool(is_bot_lookup.get(user["user_text"], False)),
            "avg_revision_comment_length":  user.get("avg_revision_comment_length", 0) or 0,
            "avg_edit_hours":               user.get("avg_edit_hours", 0) or 0,
            "total_edits":                  user.get("total_edits", 0) or 0,
            "pct_of_reverted_edits":        user.get("pct_of_reverted_edits", 0) or 0,
            "pct_of_reverting_edits":       user.get("pct_of_reverting_edits", 0) or 0,
            "unique_edited_articles":       user.get("unique_edited_articels", 0) or 0,
            "unique_edited_languages":      user.get("unique_edited_languages", 0) or 0,
            "median_seconds_between_edits": user.get("median_seconds_between_edits", 0) or 0,
        }
        row.update(aggregate_edits_per_day(user.get("edits_per_day", [])))
        row.update(aggregate_revision_tags(user.get("revision_tags", []), all_tags))
        rows.append(row)
    return pl.DataFrame(rows)


# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────

def run_pca(feature_df: pl.DataFrame):
    """Scale features and reduce to 2D via PCA. Returns X_pca, pca, feature_cols."""
    meta_cols    = ["user_text", "is_bot"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    X = feature_df.select(feature_cols).to_numpy().astype(float)
    X = np.nan_to_num(X)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    return X_pca, pca, feature_cols


# ─────────────────────────────────────────────
# CLUSTER LABELLING
# ─────────────────────────────────────────────

def label_clusters(feature_df: pl.DataFrame, cluster_labels: np.ndarray) -> int:
    """Use is_bot ground truth to decide which cluster = bot cluster."""
    is_bot    = feature_df["is_bot"].to_numpy()
    bot_ratio = {}
    for c in [0, 1]:
        mask         = cluster_labels == c
        ratio        = is_bot[mask].mean() if mask.sum() > 0 else 0
        bot_ratio[c] = ratio
        print(f"  Cluster {c}: {mask.sum()} users, {ratio:.1%} known bots")

    bot_cluster  = max(bot_ratio, key=bot_ratio.get)
    user_cluster = 1 - bot_cluster
    print(f"\n→ Cluster {bot_cluster} = BOT,  Cluster {user_cluster} = USER")
    return bot_cluster


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_pca_loadings(pca, feature_cols, top_n=10, output_path="pca_loadings.png"):
    """Feature importance via PCA loadings – shared across all models."""
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1", "PC2"]
    )
    loadings["importance"] = (
        abs(loadings["PC1"]) * pca.explained_variance_ratio_[0] +
        abs(loadings["PC2"]) * pca.explained_variance_ratio_[1]
    )
    top_features = loadings.sort_values("importance", ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    top_features["PC1"].sort_values().plot(kind="barh", ax=axes[0], color="#4C9BE8")
    axes[0].set_title("PC1 Loadings")
    axes[0].axvline(0, color="black", linewidth=0.8)

    top_features["PC2"].sort_values().plot(kind="barh", ax=axes[1], color="#E8624C")
    axes[1].set_title("PC2 Loadings")
    axes[1].axvline(0, color="black", linewidth=0.8)

    top_features["importance"].sort_values().plot(kind="barh", ax=axes[2], color="#6BBF6B")
    axes[2].set_title(f"Gesamt-Wichtigkeit (Top {top_n})")

    plt.suptitle("PCA Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Loadings plot saved to {output_path}")
    plt.show()

    print("\nTop Features:")
    print(top_features.sort_values("importance", ascending=False))
    return top_features
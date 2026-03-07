import glob
import os
import pydantic
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
    """Load the precomputed feature JSON.
    Handles both a list of dicts and a dict-of-dicts (key = username).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If it's a dict of dicts, convert to list
    if isinstance(data, dict):
        return list(data.values())
    return data

def load_is_bot_all_languages(data_dir: str) -> pl.DataFrame:
    """Load is_bot ground truth from all available language editions (files named ??wiki.json.gz),
    searching recursively under data_dir and accepting any two-character language prefix."""
    data_dir = Path(data_dir)
    dfs = []

    for path in data_dir.rglob("??wiki.json.gz"):
        try:
            print(f"  Loading {path.name}...")
            df = pl.read_ndjson(path).select(["user_text", "is_bot"])
            dfs.append(df)
        except Exception as e:
            print(f"  Failed to load {path.name}: {e}")

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
    """Convert tag counts to percentages, one column per tag.
    Handles format: [{"tag_name": "mobile_editing", "times_used": 1}, ...]
    """
    merged = {}
    for d in revision_tags:
        if d and isinstance(d, dict):
            tag  = d.get("tag_name")
            count = d.get("times_used", 1)
            if tag:
                merged[tag] = merged.get(tag, 0) + count
    total = sum(merged.values()) or 1
    return {f"pct_tag_{tag}": merged.get(tag, 0) / total for tag in all_tags}


def collect_all_tags(entity_list: list) -> set:
    """Find every unique revision tag across all users.
    Handles format: [{"tag_name": "mobile_editing", "times_used": 1}, ...]
    """
    tags = set()
    for user in entity_list:
        for d in user.get("revision_tags", []):
            if d and isinstance(d, dict):
                tag = d.get("tag_name")
                if tag:
                    tags.add(tag)
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

def find_optimal_components(X_scaled, target_variance=0.80,
                             output_path="pca_variance.png"):
    """
    Fit full PCA, plot cumulative explained variance, and return the number
    of components needed to reach target_variance.
    """
    pca_full  = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cum_var   = np.cumsum(pca_full.explained_variance_ratio_)
    n_optimal = int(np.searchsorted(cum_var, target_variance)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=4, color="#4C9BE8")
    ax.axhline(target_variance, color="red",    linestyle="--", label=f"{target_variance:.0%} threshold")
    ax.axvline(n_optimal,       color="orange",  linestyle="--", label=f"{n_optimal} components")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA - Cumulative Explained Variance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Variance plot saved to {output_path}")
    plt.show()

    print(f"  -> {n_optimal} components explain {cum_var[n_optimal - 1]:.1%} of variance "
          f"(target: {target_variance:.0%})")
    return n_optimal


def run_pca(feature_df, n_components=None, target_variance=0.80):
    """
    Scale features and run PCA.
    - If n_components is None, automatically finds how many components
      are needed to reach target_variance (default 80%).
    - Always also computes a 2D projection for visualisation.
    - Returns: X_pca (nD, for clustering), X_pca_2d (2D, for plotting),
               pca (nD model), pca_2d (2D model), feature_cols.
    """
    meta_cols    = ["user_text", "is_bot"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    X        = feature_df.select(feature_cols).to_numpy().astype(float)
    X        = np.nan_to_num(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine n_components automatically if not given
    if n_components is None:
        print(f"  Finding optimal number of components (target: {target_variance:.0%} variance)...")
        n_components = find_optimal_components(X_scaled, target_variance)

    # High-dimensional PCA for clustering
    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    total_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA for clustering: {n_components} components -> {total_var:.1%} variance explained")

    # 2D PCA for visualisation only
    pca_2d   = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    print(f"  PCA for plotting:   PC1={pca_2d.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca_2d.explained_variance_ratio_[1]:.1%}")

    return X_pca, X_pca_2d, pca, pca_2d, feature_cols


# ─────────────────────────────────────────────
# CLUSTER LABELLING
# ─────────────────────────────────────────────

def label_clusters(feature_df: pl.DataFrame, cluster_labels: np.ndarray,
                   bot_threshold: float = 0.001) -> set:
    """
    Label each cluster by its known-bot ratio and feature profile.
    Returns a set of cluster IDs considered bot clusters.
    bot_threshold: minimum known-bot ratio to flag a cluster as bot-related.
    """
    is_bot          = feature_df["is_bot"].to_numpy()
    meta_cols       = ["user_text", "is_bot"]
    feature_cols    = [c for c in feature_df.columns if c not in meta_cols]
    unique_clusters = sorted(set(cluster_labels) - {-1})

    cluster_profiles = []
    bot_clusters     = set()

    print(f"\n{'Cluster':>8}  {'Users':>8}  {'Known Bots':>10}  {'Top Feature'}")
    print("─" * 70)

    for c in unique_clusters:
        mask       = cluster_labels == c
        n_users    = mask.sum()
        bot_ratio  = is_bot[mask].mean() if n_users > 0 else 0

        # Feature means for this cluster
        cluster_data = feature_df.filter(pl.Series(mask)).select(feature_cols)
        means        = cluster_data.mean().row(0)
        top_idx      = int(np.argmax(np.abs(means)))
        top_feature  = feature_cols[top_idx]
        top_value    = means[top_idx]

        is_bot_cluster = bot_ratio >= bot_threshold
        if is_bot_cluster:
            bot_clusters.add(c)

        flag = " ← BOT" if is_bot_cluster else ""
        print(f"  {c:>6}  {n_users:>8}  {bot_ratio:>9.1%}  {top_feature} ({top_value:.2f}){flag}")

        cluster_profiles.append({
            "cluster":        c,
            "n_users":        n_users,
            "bot_ratio":      bot_ratio,
            "top_feature":    top_feature,
            "top_value":      top_value,
            "is_bot_cluster": is_bot_cluster,
        })

    print(f"\n→ Bot clusters (≥{bot_threshold:.1%} known bots): {bot_clusters}")
    return bot_clusters, cluster_profiles


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

def plot_cluster_profiles(feature_df: pl.DataFrame, cluster_labels: np.ndarray,
                           bot_clusters: set, top_n: int = 10,
                           output_path: str = "cluster_profiles.png"):
    """
    For every cluster: bar chart of the top_n most distinctive features
    (ranked by absolute deviation from the global mean).
    Bot clusters are highlighted in red.
    """
    meta_cols    = ["user_text", "is_bot"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    X            = feature_df.select(feature_cols).to_numpy().astype(float)
    X            = np.nan_to_num(X)
    global_means = X.mean(axis=0)

    unique_clusters = sorted(set(cluster_labels) - {-1})
    n_clusters      = len(unique_clusters)

    # Grid layout: up to 4 columns
    ncols = min(4, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 4),
                              squeeze=False)

    for idx, c in enumerate(unique_clusters):
        ax        = axes[idx // ncols][idx % ncols]
        mask      = cluster_labels == c
        c_means   = X[mask].mean(axis=0)
        deviation = c_means - global_means  # how much does this cluster differ?

        # Pick top_n features by absolute deviation
        top_idx    = np.argsort(np.abs(deviation))[::-1][:top_n]
        feat_names = [feature_cols[i] for i in top_idx]
        feat_vals  = deviation[top_idx]

        colors = ["#E8624C" if v > 0 else "#4C9BE8" for v in feat_vals]
        ax.barh(range(len(feat_names)), feat_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names[::-1], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)

        is_bot_cl   = c in bot_clusters
        title_color = "red" if is_bot_cl else "black"
        ax.set_title(f"{'[BOT] ' if is_bot_cl else ''}Cluster {c}  (n={mask.sum():,})",
                     fontsize=9, color=title_color, fontweight="bold")
        ax.set_xlabel("Deviation from global mean", fontsize=7)

    # Hide unused subplots
    for idx in range(n_clusters, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle(f"Cluster Feature Profiles (top {top_n} deviations from global mean)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Cluster profiles plot saved to {output_path}")
    plt.show()

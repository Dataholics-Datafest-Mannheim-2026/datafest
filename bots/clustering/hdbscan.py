import polars as pl
import numpy as np
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

from utils import (
    load_entity_list,
    load_is_bot_all_languages,
    build_feature_matrix,
    run_pca,
    plot_pca_loadings,
)

# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────

def run_hdbscan(X_pca: np.ndarray, min_cluster_size: int = 50, sample_size: int = 50000):
    """
    HDBSCAN on a representative sample, then assigns all users via nearest neighbour.
    This avoids memory issues with large datasets.

    min_cluster_size: minimum users per cluster in the sample
    sample_size: how many users to sample for clustering (default 50k)
    """
    from sklearn.neighbors import KNeighborsClassifier

    # 1. Draw sample
    np.random.seed(42)
    idx      = np.random.choice(len(X_pca), size=min(sample_size, len(X_pca)), replace=False)
    X_sample = X_pca[idx]
    print(f"  Clustering {len(X_sample)} sampled users (min_cluster_size={min_cluster_size})...")

    # 2. HDBSCAN on sample
    hdb           = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    sample_labels = hdb.fit_predict(X_sample)

    n_clusters = len(set(sample_labels) - {-1})
    n_outliers = (sample_labels == -1).sum()
    print(f"  Clusters found in sample: {n_clusters}")
    print(f"  Outliers in sample: {n_outliers}")

    # 3. Assign all users to nearest cluster via KNN
    print(f"  Assigning all {len(X_pca)} users to nearest cluster...")
    knn        = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_sample, sample_labels)
    all_labels = knn.predict(X_pca)

    # Recompute stats on full dataset
    n_clusters_full = len(set(all_labels) - {-1})
    n_outliers_full = (all_labels == -1).sum()
    print(f"  Final clusters: {n_clusters_full}, Outliers: {n_outliers_full}")

    return all_labels, hdb


def label_clusters_hdbscan(feature_df: pl.DataFrame,
                            cluster_labels: np.ndarray) -> dict:
    """
    HDBSCAN can produce more than 2 clusters, so we label each cluster
    individually by its known-bot ratio.
    Returns a dict: cluster_id → 'bot_cluster' | 'user_cluster' | 'outlier'
    """
    is_bot          = feature_df["is_bot"].to_numpy()
    unique_clusters = sorted(set(cluster_labels) - {-1})
    bot_ratios      = {}

    print("\nCluster composition:")
    for c in unique_clusters:
        mask          = cluster_labels == c
        ratio         = is_bot[mask].mean() if mask.sum() > 0 else 0
        bot_ratios[c] = ratio
        print(f"  Cluster {c}: {mask.sum()} users, {ratio:.1%} known bots")

    # Outliers (-1)
    outlier_mask = cluster_labels == -1
    if outlier_mask.sum() > 0:
        outlier_bot_ratio = is_bot[outlier_mask].mean()
        print(f"  Outliers (-1): {outlier_mask.sum()} users, "
              f"{outlier_bot_ratio:.1%} known bots")

    # Cluster with highest bot ratio = bot cluster
    bot_cluster = max(bot_ratios, key=bot_ratios.get)
    print(f"\n→ Cluster {bot_cluster} = BOT CLUSTER "
          f"({bot_ratios[bot_cluster]:.1%} known bots)")
    print("  All other clusters + outliers → USER side")

    return bot_cluster


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_results(X_pca, cluster_labels, is_bot, bot_cluster, pca,
                 output_path="bot_detection_hdbscan_plot.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    known_mask   = np.array(is_bot, dtype=bool)
    unique_clusters = sorted(set(cluster_labels) - {-1})

    # ── Left: All HDBSCAN clusters with individual colors ──
    ax     = axes[0]
    cmap   = plt.cm.get_cmap("tab10", max(len(unique_clusters), 1))

    for i, c in enumerate(unique_clusters):
        mask  = cluster_labels == c
        label = f"Cluster {c} ({'BOT' if c == bot_cluster else 'USER'})"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=cmap(i), alpha=0.5, s=20, label=label)

    # Outliers
    outlier_mask = cluster_labels == -1
    if outlier_mask.sum() > 0:
        ax.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
                   c="#aaaaaa", alpha=0.4, s=15, label="Outlier (-1)")

    # Known bots overlay
    ax.scatter(X_pca[known_mask, 0], X_pca[known_mask, 1],
               c="#1a1a1a", marker="x", s=60, linewidths=1.5,
               label="Known Bot (is_bot=True)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("HDBSCAN – All Clusters")
    ax.legend(loc="upper right", fontsize=8)

    # ── Right: Simplified bot vs user view ──
    ax       = axes[1]
    bot_mask = cluster_labels == bot_cluster
    user_mask = ~bot_mask & ~outlier_mask

    ax.scatter(X_pca[user_mask, 0],  X_pca[user_mask, 1],
               c="#4C9BE8", alpha=0.5, s=20, label="Predicted: User")
    ax.scatter(X_pca[bot_mask, 0],   X_pca[bot_mask, 1],
               c="#E8624C", alpha=0.5, s=20, label="Predicted: Bot")
    ax.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
               c="#aaaaaa", alpha=0.4, s=15, label="Outlier")
    ax.scatter(X_pca[known_mask, 0], X_pca[known_mask, 1],
               c="#1a1a1a", marker="x", s=60, linewidths=1.5,
               label="Known Bot (is_bot=True)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("HDBSCAN – Bot vs User (simplified)")
    ax.legend(loc="upper right")

    plt.suptitle("Bot Detection via PCA + HDBSCAN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def export_results(feature_df: pl.DataFrame, cluster_labels: np.ndarray,
                   bot_cluster: int, output_path="bot_detection_hdbscan_results.csv"):
    result = feature_df.select(["user_text", "is_bot"]).with_columns([
        pl.Series("cluster", cluster_labels.tolist()),
        pl.Series("predicted_label", [
            "outlier"        if c == -1
            else "undetected_bot" if (c == bot_cluster and not b)
            else "known_bot" if b
            else "user"
            for c, b in zip(cluster_labels, feature_df["is_bot"].to_list())
        ])
    ])
    result.write_csv(output_path)
    print(f"Results saved to {output_path}")
    print("\n── Summary ──────────────────────────────")
    print(result.group_by("predicted_label").len().sort("predicted_label"))
    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ENTITY_LIST_PATH = r"C:\Users\lmlin\Desktop\DataFest\datafest\bots\entity_list.json"
    DATA_DIR         = r"C:\Users\lmlin\Desktop\DataFest\datafest\data"

    print("Loading entity list...")
    entity_list = load_entity_list(ENTITY_LIST_PATH)
    print(f"  {len(entity_list)} users loaded")

    print("\nLoading is_bot from all language editions...")
    is_bot_df = load_is_bot_all_languages(DATA_DIR)
    print(f"  {len(is_bot_df)} unique users found")
    print(f"  Known bots: {is_bot_df['is_bot'].sum()}")

    print("\nBuilding feature matrix...")
    feature_df = build_feature_matrix(entity_list, is_bot_df)
    print(f"  {len(feature_df)} users, {len(feature_df.columns) - 2} features")

    print("\nRunning PCA...")
    X_pca, pca, feature_cols = run_pca(feature_df)

    print("\nRunning HDBSCAN...")
    cluster_labels, hdb = run_hdbscan(X_pca, min_cluster_size=50, sample_size=50000)

    print("\nLabelling clusters via ground truth:")
    bot_cluster = label_clusters_hdbscan(feature_df, cluster_labels)

    print("\nPlotting...")
    plot_results(X_pca, cluster_labels, feature_df["is_bot"].to_list(),
                 bot_cluster, pca)

    print("\nFeature Importance...")
    plot_pca_loadings(pca, feature_cols, top_n=10,
                      output_path="hdbscan_pca_loadings.png")

    print("\nExporting results...")
    export_results(feature_df, cluster_labels, bot_cluster)
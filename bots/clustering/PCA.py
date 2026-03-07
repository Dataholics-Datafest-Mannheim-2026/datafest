import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import (
    load_entity_list,
    load_is_bot_all_languages,
    build_feature_matrix,
    run_pca,
    label_clusters,
    plot_pca_loadings,
    plot_cluster_profiles,
)

# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────

def run_kmeans(X_pca: np.ndarray, n_clusters: int = 12):
    """Run KMeans. Change n_clusters freely."""
    kmeans         = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    print(f"  KMeans with {n_clusters} clusters done.")
    return cluster_labels


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_results(X_pca, cluster_labels, is_bot, bot_clusters, pca,
                 output_path="bot_detection_kmeans_plot.png"):
    """
    Scatter plot with one colour per cluster, bot clusters highlighted,
    and known bots marked with X.
    """
    unique_clusters = sorted(set(cluster_labels))
    cmap            = plt.cm.get_cmap("tab20", len(unique_clusters))
    known_mask      = np.array(is_bot, dtype=bool)

    fig, ax = plt.subplots(figsize=(13, 8))

    legend_handles = []
    for i, c in enumerate(unique_clusters):
        mask      = cluster_labels == c
        color     = cmap(i)
        is_bot_cl = c in bot_clusters
        label     = f"{'[BOT] ' if is_bot_cl else ''}Cluster {c} (n={mask.sum():,})"

        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], alpha=0.4, s=8, edgecolors="none")
        legend_handles.append(mpatches.Patch(color=color, label=label))

    ax.scatter(X_pca[known_mask, 0], X_pca[known_mask, 1],
               c="black", marker="x", s=60, linewidths=1.5, zorder=5)
    legend_handles.append(
        mpatches.Patch(color="black", label=f"Known Bots (n={known_mask.sum()})")
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Wikipedia Editor Clusters (PCA + KMeans)\n[BOT] = bot-related cluster")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def export_results(feature_df: pl.DataFrame, cluster_labels: np.ndarray,
                   bot_clusters: set,
                   output_path="bot_detection_kmeans_results.csv"):

    is_bot_list = feature_df["is_bot"].to_list()

    predicted_labels = []
    for c, b in zip(cluster_labels, is_bot_list):
        if b:
            predicted_labels.append("known_bot")
        elif c in bot_clusters:
            predicted_labels.append("undetected_bot")
        else:
            predicted_labels.append("user")

    result = feature_df.select(["user_text", "is_bot"]).with_columns([
        pl.Series("cluster",         cluster_labels),
        pl.Series("predicted_label", predicted_labels),
    ])
    result.write_csv(output_path)
    print(f"Results saved to {output_path}")

    print("\n── Summary by cluster ───────────────────")
    print(result.group_by(["cluster", "predicted_label"]).len().sort(["cluster", "predicted_label"]))
    return result


# ─────────────────────────────────────────────
# MAIN  –  change N_CLUSTERS here
# ─────────────────────────────────────────────

if __name__ == "__main__":
    N_CLUSTERS = 12   # <- adjust freely

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
    X_pca, X_pca_2d, pca, pca_2d, feature_cols = run_pca(feature_df)

    print(f"\nRunning KMeans (n_clusters={N_CLUSTERS})...")
    cluster_labels = run_kmeans(X_pca, n_clusters=N_CLUSTERS)

    print("\nLabelling clusters via ground truth:")
    bot_clusters, cluster_profiles = label_clusters(feature_df, cluster_labels)

    print("\nPlotting...")
    plot_results(X_pca_2d, cluster_labels, feature_df["is_bot"].to_list(),
                 bot_clusters, pca_2d)

    print("\nFeature Importance...")
    plot_pca_loadings(pca_2d, feature_cols, top_n=10,
                      output_path="kmeans_pca_loadings.png")

    print("\nCluster Feature Profiles...")
    plot_cluster_profiles(feature_df, cluster_labels, bot_clusters, top_n=10,
                          output_path="kmeans_cluster_profiles.png")

    print("\nExporting results...")
    export_results(feature_df, cluster_labels, bot_clusters)
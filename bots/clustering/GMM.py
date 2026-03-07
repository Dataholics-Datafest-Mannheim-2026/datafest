import polars as pl
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from utils import (
    load_entity_list,
    load_is_bot_all_languages,
    build_feature_matrix,
    run_pca,
    label_clusters,
    plot_pca_loadings,
)

# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────

def run_gmm(X_pca: np.ndarray):
    gmm            = GaussianMixture(n_components=2, random_state=42, n_init=10)
    gmm.fit(X_pca)
    cluster_labels = gmm.predict(X_pca)
    proba          = gmm.predict_proba(X_pca)  # shape: (n_users, 2)
    return cluster_labels, proba


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_results(X_pca, cluster_labels, proba, is_bot, bot_cluster, pca,
                 output_path="bot_detection_gmm_plot.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    known_mask = np.array(is_bot, dtype=bool)
    user_mask  = cluster_labels != bot_cluster
    bot_mask   = cluster_labels == bot_cluster
    bot_proba  = proba[:, bot_cluster]

    # ── Left: Hard cluster assignments ──
    ax = axes[0]
    ax.scatter(X_pca[user_mask, 0], X_pca[user_mask, 1],
               c="#4C9BE8", alpha=0.5, s=20, label="Predicted: User")
    ax.scatter(X_pca[bot_mask, 0],  X_pca[bot_mask, 1],
               c="#E8624C", alpha=0.5, s=20, label="Predicted: Bot")
    ax.scatter(X_pca[known_mask, 0], X_pca[known_mask, 1],
               c="#1a1a1a", marker="x", s=60, linewidths=1.5,
               label="Known Bot (is_bot=True)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("GMM – Hard Cluster Assignments")
    ax.legend(loc="upper right")

    # ── Right: Bot probability heatmap (GMM unique advantage) ──
    ax = axes[1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=bot_proba, cmap="RdYlBu_r", alpha=0.6, s=20,
                    vmin=0, vmax=1)
    ax.scatter(X_pca[known_mask, 0], X_pca[known_mask, 1],
               c="#1a1a1a", marker="x", s=60, linewidths=1.5,
               label="Known Bot (is_bot=True)")
    plt.colorbar(sc, ax=ax, label="P(Bot)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("GMM – Bot Probability per User\n(rot = hohe Bot-Wahrscheinlichkeit)")
    ax.legend(loc="upper right")

    plt.suptitle("Bot Detection via PCA + Gaussian Mixture Model",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def export_results(feature_df: pl.DataFrame, cluster_labels: np.ndarray,
                   proba: np.ndarray, bot_cluster: int,
                   output_path="bot_detection_gmm_results.csv"):
    bot_proba = proba[:, bot_cluster]
    result = feature_df.select(["user_text", "is_bot"]).with_columns([
        pl.Series("cluster", cluster_labels),
        pl.Series("bot_probability", bot_proba.tolist()),
        pl.Series("predicted_label", [
            "undetected_bot" if (c == bot_cluster and not b)
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

    print("\nRunning GMM...")
    cluster_labels, proba = run_gmm(X_pca)

    print("\nLabelling clusters via ground truth:")
    bot_cluster = label_clusters(feature_df, cluster_labels)

    print("\nPlotting...")
    plot_results(X_pca, cluster_labels, proba, feature_df["is_bot"].to_list(),
                 bot_cluster, pca)

    print("\nFeature Importance...")
    plot_pca_loadings(pca, feature_cols, top_n=10,
                      output_path="gmm_pca_loadings.png")

    print("\nExporting results...")
    export_results(feature_df, cluster_labels, proba, bot_cluster)
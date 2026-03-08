"""
DataFest 2026 — Step 8: Predicting Unproductive Accounts
Topic: Bots as Wikipedia Editors

Trains a LightGBM classifier to find behavioral predictors of "unproductive"
accounts (reverted_rate_pct >= 80%). Runs the full pipeline twice:

  variant "all"  — full dataset, all edit counts included
  variant "min5" — only accounts with total_edits >= 5

Per variant, two figures are saved:
  figures/12_feature_importance_{variant}.png  — gain-based feature importance
  figures/13_shap_summary_{variant}.png        — SHAP beeswarm (shows direction)

Requires: uv add lightgbm shap
Run:      uv run bot_analysis_jona/08_unproductive_predictor.py
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT      = Path(__file__).parent / "accounts_master.ndjson"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE     = 42
TEST_SIZE        = 0.20
REVERT_THRESHOLD = 80.0   # reverted_rate_pct >= this → is_unproductive = 1

NUMERICAL_FEATURES = [
    "total_edits_log",            # log1p(total_edits) — controls for scale
    "edit_velocity_active_days",
    "avg_hours_between_edits",    # inter-edit rhythm
    "focus_score",
    "avg_pageviews_edited",
    "active_days",
    "unique_pages_touched",
]

CATEGORICAL_FEATURES = [
    "is_bot",
    "is_multilingual",
    "primary_edit_type",
    "top_topic",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Variants: (label, min_total_edits or None)
VARIANTS = [
    ("all",  None),
    ("min5", 5),
]

BOT_COLOR   = "#E05C4B"
HUMAN_COLOR = "#4B8BE0"
BAR_COLOR   = "#4B8BE0"


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── LOAD & BASE PREP ──────────────────────────────────────────────────────────
section("LOAD")

df = pl.read_ndjson(INPUT)
print(f"  {df.shape[0]:,} accounts loaded")

# Drop rows with nulls in any feature or target column
required = ["reverted_rate_pct", "edit_velocity_active_days", "active_days",
            "focus_score", "unique_pages_touched", "avg_pageviews_edited",
            "primary_edit_type"]
null_filter = pl.all_horizontal([pl.col(c).is_not_null() for c in required])
df = df.filter(null_filter)
print(f"  {df.shape[0]:,} accounts after dropping rows with nulls in required columns")

# Engineer target variable
df = df.with_columns(
    (pl.col("reverted_rate_pct") >= REVERT_THRESHOLD).cast(pl.Int8).alias("is_unproductive"),
)

print(f"\n  Class balance (full dataset):")
bal = df.group_by("is_unproductive").agg(pl.len().alias("count")).sort("is_unproductive")
print(bal)


# ── PER-VARIANT PIPELINE ──────────────────────────────────────────────────────
for variant_label, min_edits in VARIANTS:
    section(f"VARIANT: {variant_label.upper()}")

    # Apply edit-count filter if requested
    vdf = df if min_edits is None else df.filter(pl.col("total_edits") >= min_edits)
    n_total      = vdf.shape[0]
    n_unprod     = int(vdf["is_unproductive"].sum())
    print(f"  {n_total:,} accounts  |  unproductive: {n_unprod:,} ({n_unprod/n_total*100:.1f}%)")

    # Convert to pandas
    polars_cols = [f for f in ALL_FEATURES if f != "total_edits_log"]
    pdf = vdf.select(polars_cols + ["is_unproductive", "total_edits"]).to_pandas()

    # Log-transform total_edits (in pandas, avoids Polars log edge cases)
    pdf["total_edits_log"] = np.log1p(pdf["total_edits"])

    # Encode booleans → int, categoricals → pandas Categorical (required by LightGBM)
    pdf["is_bot"]          = pdf["is_bot"].astype(int)
    pdf["is_multilingual"] = pdf["is_multilingual"].astype(int)
    pdf["primary_edit_type"] = pdf["primary_edit_type"].astype("category")
    pdf["top_topic"]         = pdf["top_topic"].fillna("Unknown").astype("category")

    X = pdf[ALL_FEATURES]
    y = pdf["is_unproductive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",   # handles imbalanced classes automatically
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=["primary_edit_type", "top_topic"],
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    print(f"\n  ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["productive", "unproductive"]))

    # ── FIG 12: Feature Importance (gain) ────────────────────────────────────
    importance_df = (
        pd.DataFrame({"feature": model.feature_name_, "importance": model.feature_importances_})
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"],
            color=BAR_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    filter_note = "all accounts" if min_edits is None else f"accounts with ≥{min_edits} edits"
    ax.set_title(
        f"Predictors of Unproductive Accounts\n({filter_note}  |  ROC-AUC = {roc:.3f})",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = OUTPUT_DIR / f"unproductive_predictor_feature_importance_{variant_label}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out.name}")

    # ── FIG 13: SHAP Summary (directional) ───────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values is a list [neg_class, pos_class] for binary LightGBM
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(sv, X_test, show=False)
    plt.title(
        f"SHAP Feature Impact — Unproductive Accounts\n({filter_note})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out2 = OUTPUT_DIR / f"unproductive_predictor_shap_summary_{variant_label}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2.name}")


print("\n\nAll done.")

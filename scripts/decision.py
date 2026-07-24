#!/usr/bin/env python3
"""
Decision tree analysis for FastSDPCertification hyperparameter sweep results.

Fits a regression tree that predicts `optimal_value_avg` from the combo
"settings" (strategy, l, u_idx, k, j_idx, neuron bounds), then reports:
  - the tree as human-readable text rules
  - feature importances
  - every leaf ranked by mean optimal_value_avg
  - the exact split conditions of the leaves with the HIGHEST optimal_value_avg
    (i.e. the answer to "which combos maximize optimal_value")

Usage:
    python decision_tree_optimal_value.py results.csv
    python decision_tree_optimal_value.py results.csv --max-depth 5 --min-leaf 5
    python decision_tree_optimal_value.py results.csv --top-k 5 --out tree.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # no display needed, we just save a PNG
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree, _tree

TARGET_COL = "optimal_value_avg"

# Columns treated as "settings" one could pick before launching a run.
# certification_rate and n_runs are left OUT of the default feature list on
# purpose: n_runs is basically constant, and certification_rate is a *result*
# of a combo (like optimal_value_avg itself), not a setting you choose in
# advance. Add either back to FEATURE_COLS below if you want a purely
# exploratory/diagnostic tree instead of a "which settings should I pick" tree.
FEATURE_COLS = [
    "strategy", "l", "u_idx", "k", "j_idx",
    "LB_neuron1", "UB_neuron1", "LB_neuron2", "UB_neuron2",
]

# Sentinel used to fill missing numeric values (e.g. l/u_idx/k/j_idx/bounds
# are empty when strategy == "none", since there is no cross-term). -1 is
# safe here because none of these columns take negative values in practice.
MISSING_SENTINEL = -1


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected column(s) in {csv_path}: {missing_cols}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    if "strategy" in X.columns:
        X = pd.get_dummies(X, columns=["strategy"], prefix="strategy")
    X = X.fillna(MISSING_SENTINEL)
    return X


def fit_tree(X: pd.DataFrame, y: pd.Series, max_depth: int, min_leaf: int) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42)
    tree.fit(X, y)
    return tree


def leaf_rules(tree: DecisionTreeRegressor, feature_names: list[str]) -> dict[int, list[str]]:
    """Map each leaf id to the list of split conditions that define it."""
    t = tree.tree_
    rules: dict[int, list[str]] = {}

    def recurse(node: int, path: list[str]) -> None:
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[t.feature[node]]
            thr = t.threshold[node]
            recurse(t.children_left[node], path + [f"{name} <= {thr:.3f}"])
            recurse(t.children_right[node], path + [f"{name} > {thr:.3f}"])
        else:
            rules[node] = path

    recurse(0, [])

    
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default="combo_summary.csv", help="path to the results CSV")
    parser.add_argument("--max-depth", type=int, default=4, help="tree depth (readability vs. detail)")
    parser.add_argument("--min-leaf", type=int, default=3, help="min samples per leaf (avoids overfitting)")
    parser.add_argument("--top-k", type=int, default=3, help="number of best leaves to detail")
    parser.add_argument("--out", default="decision_tree.png", help="output path for the tree plot")
    args = parser.parse_args()

    df = load_data(args.csv_path)
    X = build_features(df)
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    print(f"Loaded {len(df)} rows from {args.csv_path}\n")

    # Informational held-out score: how much the tree actually generalizes,
    # as opposed to just describing the exact rows we already have.
    if len(df) >= 10:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        probe = fit_tree(X_tr, y_tr, args.max_depth, args.min_leaf)
        r2 = r2_score(y_te, probe.predict(X_te))
        print(f"[info] held-out R^2 (max_depth={args.max_depth}): {r2:.3f}")
        print("       (rough guide only: negative/near-zero => splits below are likely noise)\n")

    # Final tree, fit on ALL rows, used for interpretation.
    tree = fit_tree(X, y, args.max_depth, args.min_leaf)

    print("=== Tree rules ===")
    print(export_text(tree, feature_names=feature_names))

    importances = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("=== Feature importance ===")
    print(importances[importances > 0].to_string(), "\n")

    df = df.copy()
    df["leaf_id"] = tree.apply(X)
    leaf_stats = (
        df.groupby("leaf_id")[TARGET_COL]
        .agg(mean="mean", count="count")
        .sort_values("mean", ascending=False)
    )
    print("=== Leaves ranked by optimal_value_avg (highest first) ===")
    print(leaf_stats.to_string(), "\n")

    rules = leaf_rules(tree, feature_names)
    print(f"=== Top {args.top_k} leaves: conditions that MAXIMIZE {TARGET_COL} ===")
    for leaf_id in leaf_stats.index[: args.top_k]:
        mean_val = leaf_stats.loc[leaf_id, "mean"]
        count = int(leaf_stats.loc[leaf_id, "count"])
        print(f"\nLeaf {leaf_id}  ->  mean {TARGET_COL} = {mean_val:.4f}  (n={count} combos)")
        for cond in rules[leaf_id]:
            print(f"   - {cond}")
        example_combos = df.loc[df["leaf_id"] == leaf_id, "combo"].head(5).tolist()
        print(f"   example combos: {example_combos}")

    plt.figure(figsize=(22, 10))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, fontsize=7, precision=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n[info] tree plot saved to {args.out}")


if __name__ == "__main__":
    main()

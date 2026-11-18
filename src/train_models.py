import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS
from src.hmm_viterbi import compute_route_continuity


def get_models():
    s = config.SEED
    return {
        "LogisticRegression": (
            "Linear / Probabilistic",
            "StandardScaler + C=1.0, max_iter=300",
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=s)),
        ),
        "RandomForest": (
            "Ensemble (Bagging)",
            "n_estimators=80, max_depth=12, min_samples_leaf=15",
            lambda: RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=15, n_jobs=-1, random_state=s),
        ),
        "HistGradientBoosting": (
            "Boosting (Histogram)",
            "max_iter=150, max_depth=8, lr=0.08",
            lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.08, random_state=s),
        ),
        "LightGBM": (
            "Boosting (GBDT)",
            "n_estimators=200, num_leaves=31, lr=0.08",
            lambda: LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.08, n_jobs=-1, random_state=s, verbose=-1),
        ),
        "XGBoost": (
            "Boosting (GBDT)",
            "n_estimators=200, max_depth=6, lr=0.08, hist",
            lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, tree_method="hist", n_jobs=-1, random_state=s, eval_metric="logloss"),
        ),
        "MLPClassifier": (
            "Neural Network",
            "StandardScaler + (64, 32), early_stopping=True",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, early_stopping=True, n_iter_no_change=5, random_state=s)),
        ),
    }


def eval_observation_top1_and_continuity(test_df: pd.DataFrame, probs: np.ndarray, arcs_map: dict) -> tuple[float, float]:
    w = test_df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    w["prob"] = probs
    best = w.loc[w.groupby(["trip_id", "obs_idx"], sort=False)["prob"].idxmax()]
    conts = [compute_route_continuity(g["arc_id"].to_numpy(dtype=np.int64), arcs_map[tid]) for tid, g in best.groupby("trip_id", sort=False)]
    return float(best["label"].mean()), float(np.mean(conts))


def main():
    tr_df = pd.read_pickle(config.OUTPUTS_DIR / "train_features.pkl")
    te_df = pd.read_pickle(config.OUTPUTS_DIR / "test_features.pkl")

    X_tr, y_tr, grps = tr_df[FEATURE_COLS].to_numpy(dtype=np.float32), tr_df["label"].to_numpy(dtype=np.int8), tr_df["trip_id"].to_numpy()
    X_te, y_te = te_df[FEATURE_COLS].to_numpy(dtype=np.float32), te_df["label"].to_numpy(dtype=np.int8)
    arcs_map = {tid: load_trip(config.TEST_DIR, tid)["arcs"] for tid in list_trips(config.TEST_DIR)}

    splits = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups=grps))
    res, cms, mlp_lc = [], {}, None

    for name, (fam, cfg, build) in get_models().items():
        cv_f1s = []
        for tr_i, val_i in splits:
            m = build()
            m.fit(X_tr[tr_i], y_tr[tr_i])
            cv_f1s.append(f1_score(y_tr[val_i], m.predict(X_tr[val_i]), average="macro"))

        clf = build()
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        t_tr = time.perf_counter() - t0

        t1 = time.perf_counter()
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(np.int8)
        t_inf = time.perf_counter() - t1

        top1, cont = eval_observation_top1_and_continuity(te_df, probs, arcs_map)
        cms[name] = confusion_matrix(y_te, preds).tolist()
        if name == "MLPClassifier":
            mlp_lc = clf.named_steps["mlpclassifier"].loss_curve_

        row = {
            "Model": name,
            "Algorithm Family": fam,
            "Key Configuration": cfg,
            "CV Macro F1": round(float(np.mean(cv_f1s)), 4),
            "Top-1 Accuracy": round(top1, 4),
            "Route Continuity": round(cont, 4),
            "Student-Test Macro F1": round(float(f1_score(y_te, preds, average="macro")), 4),
            "Log Loss": round(float(log_loss(y_te, np.clip(probs, 1e-6, 1 - 1e-6))), 4),
            "Train Time (s)": round(t_tr, 2),
            "Inference Time (s)": round(t_inf, 4),
        }
        res.append(row)
        print(row)

    rdf = pd.DataFrame(res)
    rdf.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)
    with open(config.OUTPUTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"results": res, "confusion_matrices": cms}, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(rdf))
    ax[0].bar(x, rdf["CV Macro F1"], color="#3b6998")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(rdf["Model"], rotation=30, ha="right")
    ax[0].set_ylim(0.5, 1.0)
    ax[0].set_title("5-Fold GroupCV Macro F1")

    w = 0.35
    ax[1].bar(x - w / 2, rdf["Top-1 Accuracy"], width=w, label="Top-1 Acc", color="#2ca02c")
    ax[1].bar(x + w / 2, rdf["Student-Test Macro F1"], width=w, label="Macro F1", color="#ff7f0e")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(rdf["Model"], rotation=30, ha="right")
    ax[1].set_ylim(0.5, 1.0)
    ax[1].set_title("Student-Test Top-1 & Macro F1")
    ax[1].legend()

    ax[2].bar(x, rdf["Log Loss"], color="#d62728")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(rdf["Model"], rotation=30, ha="right")
    ax[2].set_title("Student-Test Log Loss (lower is better)")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    if mlp_lc is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(mlp_lc) + 1), mlp_lc, marker="o", color="#1f77b4")
        plt.xlabel("Epoch")
        plt.ylabel("Training Log Loss")
        plt.title("MLPClassifier Training Loss Curve")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "mlp_loss_curve.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()

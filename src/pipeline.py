"""
Radyomik Ozellikler Kullanilarak Papilodem Siniflandirmasi
Yapay Zeka Dersi - Final Odevi (Secenek A)
Tam, veri-sizintisiz makine ogrenmesi pipeline'i.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve)
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SUB_ROOT", os.path.dirname(HERE))

def _resolve_data_dir():
    """Veri klasorunu otomatik bul: once DATA_DIR ortam degiskeni, yoksa proje
    kokundeki data/ klasoru, yoksa eski uskudar_finalodevi_A klasoru."""
    env = os.environ.get("DATA_DIR")
    if env:
        return env
    base = os.path.dirname(HERE)  # proje koku (src'nin bir ust klasoru)
    for c in [os.path.join(base, "data"),
              os.path.join(base, "uskudar_finalodevi_A"),
              os.path.join(base, "..", "uskudar_finalodevi_A")]:
        if os.path.isdir(c):
            return c
    return os.path.join(base, "data")

DATA_DIR = _resolve_data_dir()
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
N_TRIALS = int(os.environ.get("N_TRIALS", "50"))
N_FEATURES = int(os.environ.get("N_FEATURES", "30"))
INNER_SPLITS = 3
sns.set_theme(style="whitegrid", context="talk")


def load_data():
    normal = pd.read_csv(os.path.join(DATA_DIR, "normal_radiomics.csv"))
    papil = pd.read_csv(os.path.join(DATA_DIR, "papilodem_radiomics.csv"))
    normal["label"] = 0; papil["label"] = 1
    normal["group"] = "N_" + normal["PatientIndex"].astype(str)
    papil["group"] = "P_" + papil["PatientIndex"].astype(str)
    df = pd.concat([normal, papil], ignore_index=True)
    feature_cols = [c for c in df.columns if c.startswith("Feature_")]
    return df[feature_cols].copy(), df["label"].values, df["group"].values, feature_cols


def mrmr_select(X, y, k):
    cols = list(X.columns); Xv = X.values
    rel = pd.Series(mutual_info_classif(Xv, y, random_state=RANDOM_STATE), index=cols)
    corr = np.nan_to_num(np.abs(np.corrcoef(Xv, rowvar=False)))
    col_idx = {c: i for i, c in enumerate(cols)}
    selected, remaining = [], cols.copy()
    first = rel.idxmax(); selected.append(first); remaining.remove(first)
    while len(selected) < min(k, len(cols)) and remaining:
        best_score, best_feat = -np.inf, None
        sel_idx = [col_idx[s] for s in selected]
        for f in remaining:
            score = rel[f] - corr[col_idx[f], sel_idx].mean()
            if score > best_score:
                best_score, best_feat = score, f
        selected.append(best_feat); remaining.remove(best_feat)
    return selected, rel


class Preprocessor:
    def __init__(self, corr_threshold=0.95, var_threshold=1e-8):
        self.corr_threshold = corr_threshold; self.var_threshold = var_threshold
    def fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        self.imputer = SimpleImputer(strategy="median").fit(X)
        Xi = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        self.vt = VarianceThreshold(self.var_threshold).fit(Xi)
        keep_var = Xi.columns[self.vt.get_support()]
        Xv = Xi[keep_var]
        corr = Xv.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        self.kept_cols = [c for c in keep_var if c not in drop]
        self.scaler = RobustScaler().fit(Xv[self.kept_cols])
        return self
    def transform(self, X):
        X = X.replace([np.inf, -np.inf], np.nan)
        Xi = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        Xs = self.scaler.transform(Xi[self.kept_cols])
        return pd.DataFrame(Xs, columns=self.kept_cols, index=X.index)


def make_model(name, params):
    if name == "LR": return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)
    if name == "SVM": return SVC(probability=True, random_state=RANDOM_STATE, **params)
    if name == "RF": return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
    if name == "ET": return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
    if name == "GB": return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    if name == "KNN": return KNeighborsClassifier(**params)
    raise ValueError(name)


def suggest_params(name, trial):
    if name == "LR":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "SVM":
        return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name in ("RF", "ET"):
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "GB":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0)}
    if name == "KNN":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2)}
    raise ValueError(name)


def optimize_model(name, X_tr, y_tr, g_tr, n_trials):
    sgkf = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # Fold bolunmesi ve MRMR ozellik secimi hiperparametreden BAGIMSIZ oldugu icin
    # fold basina BIR KEZ hesaplanip onbellege alinir (sizinti yok, ~50x hizli).
    folds = []
    for tr_idx, va_idx in sgkf.split(X_tr, y_tr, groups=g_tr):
        feats, _ = mrmr_select(X_tr.iloc[tr_idx], y_tr[tr_idx], N_FEATURES)
        folds.append((tr_idx, va_idx, feats))
    def objective(trial):
        params = suggest_params(name, trial); scores = []
        for tr_idx, va_idx, feats in folds:
            m = make_model(name, params); m.fit(X_tr.iloc[tr_idx][feats], y_tr[tr_idx])
            scores.append(f1_score(y_tr[va_idx], m.predict(X_tr.iloc[va_idx][feats]), average="macro"))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, n_jobs=int(os.environ.get("OPTUNA_JOBS","2")), show_progress_bar=False)
    return study.best_params, study.best_value, study


def compute_metrics(y_true, y_pred, y_proba):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_proba),
            "PR-AUC": average_precision_score(y_true, y_proba),
            "Balanced-Acc": balanced_accuracy_score(y_true, y_pred),
            "Brier": brier_score_loss(y_true, y_proba)}


def main():
    print(">> Veri yukleniyor...")
    X, y, groups, feat_cols = load_data()
    print(f"   Toplam: {X.shape[0]} ornek, {X.shape[1]} ham ozellik")
    print(f"   Normal={int((y==0).sum())}, Papilodem={int((y==1).sum())}, hasta={len(np.unique(groups))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr_raw, X_te_raw = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]; g_tr = groups[tr_idx]
    assert len(set(groups[tr_idx]) & set(groups[te_idx])) == 0, "HASTA SIZINTISI!"
    print(f"   Train {len(tr_idx)} ornek/{len(np.unique(g_tr))} hasta | Test {len(te_idx)} ornek/{len(np.unique(groups[te_idx]))} hasta")

    pre = Preprocessor(0.95).fit(X_tr_raw, y_tr)
    X_tr = pre.transform(X_tr_raw); X_te = pre.transform(X_te_raw)
    print(f"   Ham {X.shape[1]} -> on isleme sonrasi {X_tr.shape[1]} ozellik")

    final_feats, relevance = mrmr_select(X_tr, y_tr, N_FEATURES)
    print(f"   MRMR secilen ozellik: {len(final_feats)}")
    json.dump(final_feats, open(os.path.join(RES_DIR, "selected_features.json"), "w"), indent=2)

    model_names = ["LR", "SVM", "RF", "ET", "GB", "KNN"]
    best_params, best_cv, fitted = {}, {}, {}
    test_metrics, cv_fold_scores, roc_data, pr_data = {}, {}, {}, {}
    sgkf_eval = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for name in model_names:
        print(f">> [{name}] Optuna HPO ({N_TRIALS} trial)...")
        bp, bv, _ = optimize_model(name, X_tr, y_tr, g_tr, N_TRIALS)
        best_params[name] = bp; best_cv[name] = bv
        print(f"   CV macro-F1={bv:.4f}")
        fold_scores = []
        for tr_i, va_i in sgkf_eval.split(X_tr, y_tr, groups=g_tr):
            m = make_model(name, bp); m.fit(X_tr.iloc[tr_i][final_feats], y_tr[tr_i])
            fold_scores.append(f1_score(y_tr[va_i], m.predict(X_tr.iloc[va_i][final_feats]), average="macro"))
        cv_fold_scores[name] = fold_scores
        base = make_model(name, bp); base.fit(X_tr[final_feats], y_tr); fitted[name] = base
        cal = CalibratedClassifierCV(make_model(name, bp), method="sigmoid", cv=3)
        cal.fit(X_tr[final_feats], y_tr)
        proba = cal.predict_proba(X_te[final_feats])[:, 1]
        pred = (proba >= 0.5).astype(int)
        test_metrics[name] = compute_metrics(y_te, pred, proba)
        fpr, tpr, _ = roc_curve(y_te, proba); prec, rec, _ = precision_recall_curve(y_te, proba)
        roc_data[name] = (fpr, tpr, test_metrics[name]["ROC-AUC"])
        pr_data[name] = (rec, prec, test_metrics[name]["PR-AUC"])

    print(">> Soft-voting ensemble (RF+ET+GB)...")
    estimators = [(n, CalibratedClassifierCV(make_model(n, best_params[n]), method="sigmoid", cv=3))
                  for n in ["RF", "ET", "GB"]]
    ens = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    ens.fit(X_tr[final_feats], y_tr)
    proba = ens.predict_proba(X_te[final_feats])[:, 1]; pred = (proba >= 0.5).astype(int)
    test_metrics["Ensemble"] = compute_metrics(y_te, pred, proba)
    fpr, tpr, _ = roc_curve(y_te, proba); prec, rec, _ = precision_recall_curve(y_te, proba)
    roc_data["Ensemble"] = (fpr, tpr, test_metrics["Ensemble"]["ROC-AUC"])
    pr_data["Ensemble"] = (rec, prec, test_metrics["Ensemble"]["PR-AUC"])
    fold_scores = []
    for tr_i, va_i in sgkf_eval.split(X_tr, y_tr, groups=g_tr):
        e = clone(ens); e.fit(X_tr.iloc[tr_i][final_feats], y_tr[tr_i])
        fold_scores.append(f1_score(y_tr[va_i], e.predict(X_tr.iloc[va_i][final_feats]), average="macro"))
    cv_fold_scores["Ensemble"] = fold_scores

    results_df = pd.DataFrame(test_metrics).T.round(4)
    results_df.to_csv(os.path.join(RES_DIR, "test_metrics.csv"))
    print("\n=== TEST SONUCLARI ===\n" + results_df.to_string())
    json.dump(best_params, open(os.path.join(RES_DIR, "best_params.json"), "w"), indent=2, default=str)
    json.dump({k: float(v) for k, v in best_cv.items()},
              open(os.path.join(RES_DIR, "cv_macro_f1.json"), "w"), indent=2)

    print(">> Istatistiksel testler...")
    stat_models = model_names + ["Ensemble"]
    score_matrix = [cv_fold_scores[m] for m in stat_models]
    stats_out = {}
    try:
        fr_stat, fr_p = stats.friedmanchisquare(*score_matrix)
        stats_out["friedman"] = {"statistic": float(fr_stat), "p_value": float(fr_p)}
    except Exception as e:
        stats_out["friedman"] = {"error": str(e)}
    mean_scores = {m: float(np.mean(cv_fold_scores[m])) for m in stat_models}
    best_model = max(mean_scores, key=mean_scores.get)
    comparisons, pvals = [], []
    for m in stat_models:
        if m == best_model: continue
        a, b = cv_fold_scores[best_model], cv_fold_scores[m]
        try:
            p = 1.0 if np.allclose(a, b) else stats.wilcoxon(a, b)[1]
        except Exception:
            p = float("nan")
        comparisons.append(m); pvals.append(p)
    n_comp = len(pvals)
    bonf = [min(p * n_comp, 1.0) if p == p else p for p in pvals]
    stats_out["wilcoxon_vs_best"] = {"reference": best_model, "comparisons": [
        {"vs": comparisons[i], "p_raw": float(pvals[i]) if pvals[i] == pvals[i] else None,
         "p_bonferroni": float(bonf[i]) if bonf[i] == bonf[i] else None} for i in range(n_comp)]}
    stats_out["mean_cv_macro_f1"] = mean_scores; stats_out["best_model"] = best_model
    json.dump(stats_out, open(os.path.join(RES_DIR, "statistical_tests.json"), "w"), indent=2)
    print(f"   Friedman p={stats_out['friedman'].get('p_value')}, en iyi(CV)={best_model}")

    print(">> Grafikler...")
    order = model_names + ["Ensemble"]
    palette = sns.color_palette("tab10", len(order))
    plt.figure(figsize=(9, 7))
    for n, c in zip(order, palette):
        fpr, tpr, auc = roc_data[n]; plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})", color=c, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Egrileri - Test Seti"); plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(9, 7))
    for n, c in zip(order, palette):
        rec, prec, ap = pr_data[n]; plt.plot(rec, prec, label=f"{n} (AP={ap:.3f})", color=c, lw=2)
    br = float((y_te == 1).mean())
    plt.axhline(br, color="k", ls="--", lw=1, label=f"Baseline={br:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Egrileri - Test Seti"); plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "pr_curve.png"), dpi=150); plt.close()

    proba_ens = ens.predict_proba(X_te[final_feats])[:, 1]
    cm = confusion_matrix(y_te, (proba_ens >= 0.5).astype(int))
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Papilodem"], yticklabels=["Normal", "Papilodem"])
    plt.xlabel("Tahmin"); plt.ylabel("Gercek"); plt.title("Confusion Matrix - Ensemble")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    imp = pd.Series(fitted["RF"].feature_importances_, index=final_feats).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 8)); sns.barplot(x=imp.values, y=imp.index, palette="viridis")
    plt.xlabel("Onem (Random Forest)"); plt.ylabel("Ozellik")
    plt.title("En Onemli 15 Radyomik Ozellik"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150); plt.close()
    imp.to_csv(os.path.join(RES_DIR, "top_features.csv"))

    plt.figure(figsize=(8, 7))
    uncal = VotingClassifier(estimators=[(n, make_model(n, best_params[n])) for n in ["RF", "ET", "GB"]],
                             voting="soft", n_jobs=1).fit(X_tr[final_feats], y_tr)
    for label, p in [("Kalibrasyonsuz", uncal.predict_proba(X_te[final_feats])[:, 1]),
                     ("Sigmoid kalibrasyonlu", proba_ens)]:
        fp, mp = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
        plt.plot(mp, fp, "o-", label=label, lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Mukemmel kalibrasyon")
    plt.xlabel("Ortalama tahmin olasiligi"); plt.ylabel("Gercek pozitif orani")
    plt.title("Calibration Curve - Ensemble"); plt.legend(fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "calibration_curve.png"), dpi=150); plt.close()

    comp = results_df[["Macro-F1", "ROC-AUC", "PR-AUC"]].copy()
    ax = comp.plot(kind="bar", figsize=(12, 7), width=0.8)
    ax.set_ylabel("Skor"); ax.set_title("Model Karsilastirmasi - Test Seti")
    ax.set_ylim(0, 1.0); ax.legend(loc="lower right")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150); plt.close()

    summary = {"n_samples": int(X.shape[0]), "n_raw_features": int(X.shape[1]),
               "n_features_after_preprocess": int(X_tr.shape[1]), "n_selected_features": len(final_feats),
               "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
               "n_patients_total": int(len(np.unique(groups))), "n_trials": N_TRIALS,
               "best_model_cv": best_model, "best_model_test_by_macro_f1": results_df["Macro-F1"].idxmax(),
               "test_class_balance": {"Normal": int((y_te == 0).sum()), "Papilodem": int((y_te == 1).sum())}}
    json.dump(summary, open(os.path.join(RES_DIR, "summary.json"), "w"), indent=2)
    print("\n>> TAMAMLANDI.")


if __name__ == "__main__":
    main()

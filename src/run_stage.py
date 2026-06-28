"""Asamali calistirici: her model ayri cagri (44s limiti icin). Durum state.pkl'de."""
import os, sys, json, joblib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import pipeline as P
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve

STATE = os.path.join(P.ROOT, "state.pkl")
RES = os.path.join(P.ROOT, "results"); os.makedirs(RES, exist_ok=True)
stage = sys.argv[1]

if stage == "prep":
    X, y, groups, _ = P.load_data()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=P.RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups))
    pre = P.Preprocessor(0.95).fit(X.iloc[tr], y[tr])
    X_tr, X_te = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
    feats, _ = P.mrmr_select(X_tr, y[tr], P.N_FEATURES)
    st = dict(X_tr=X_tr, X_te=X_te, y_tr=y[tr], y_te=y[te], g_tr=groups[tr],
              feats=feats, n_all=len(y), n_raw=X.shape[1], n_pre=X_tr.shape[1],
              n_tr=len(tr), n_te=len(te), n_pat=len(np.unique(groups)),
              n_pat_tr=len(np.unique(groups[tr])), n_pat_te=len(np.unique(groups[te])),
              best_params={}, best_cv={}, cv_fold={}, test_metrics={},
              roc={}, pr={}, rf_imp=None)
    joblib.dump(st, STATE)
    json.dump(feats, open(os.path.join(RES, "selected_features.json"), "w"), indent=2)
    print(f"PREP ok: {len(y)} ornek, on-isleme {X_tr.shape[1]} ozellik, MRMR {len(feats)}")

elif stage.startswith("model:"):
    name = stage.split(":")[1]
    st = joblib.load(STATE)
    X_tr, y_tr, g_tr, feats = st["X_tr"], st["y_tr"], st["g_tr"], st["feats"]
    X_te, y_te = st["X_te"], st["y_te"]
    bp, bv, _ = P.optimize_model(name, X_tr, y_tr, g_tr, P.N_TRIALS)
    st["best_params"][name] = bp; st["best_cv"][name] = float(bv)
    sgkf = StratifiedGroupKFold(n_splits=P.INNER_SPLITS, shuffle=True, random_state=P.RANDOM_STATE)
    fs = []
    for tr_i, va_i in sgkf.split(X_tr, y_tr, groups=g_tr):
        m = P.make_model(name, bp); m.fit(X_tr.iloc[tr_i][feats], y_tr[tr_i])
        fs.append(f1_score(y_tr[va_i], m.predict(X_tr.iloc[va_i][feats]), average="macro"))
    st["cv_fold"][name] = fs
    cal = CalibratedClassifierCV(P.make_model(name, bp), method="sigmoid", cv=3)
    cal.fit(X_tr[feats], y_tr)
    proba = cal.predict_proba(X_te[feats])[:, 1]; pred = (proba >= 0.5).astype(int)
    st["test_metrics"][name] = P.compute_metrics(y_te, pred, proba)
    fpr, tpr, _ = roc_curve(y_te, proba); prec, rec, _ = precision_recall_curve(y_te, proba)
    st["roc"][name] = (fpr, tpr, st["test_metrics"][name]["ROC-AUC"])
    st["pr"][name] = (rec, prec, st["test_metrics"][name]["PR-AUC"])
    if name == "RF":
        rf = P.make_model("RF", bp); rf.fit(X_tr[feats], y_tr)
        st["rf_imp"] = rf.feature_importances_
    joblib.dump(st, STATE)
    print(f"{name} ok: CV={bv:.4f}, test Macro-F1={st['test_metrics'][name]['Macro-F1']:.4f}")

print("STAGE DONE", stage)

def finalize():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    from scipy import stats as sstats
    from sklearn.base import clone
    from sklearn.ensemble import VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import confusion_matrix
    sns.set_theme(style="whitegrid", context="talk")
    FIG = os.path.join(P.ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
    st = joblib.load(STATE)
    X_tr, X_te, y_tr, y_te = st["X_tr"], st["X_te"], st["y_tr"], st["y_te"]
    g_tr, feats, bp = st["g_tr"], st["feats"], st["best_params"]
    model_names = ["LR","SVM","RF","ET","GB","KNN"]

    # Ensemble (RF+ET+GB, sigmoid kalibrasyonlu soft-voting)
    est = [(n, CalibratedClassifierCV(P.make_model(n, bp[n]), method="sigmoid", cv=3)) for n in ["RF","ET","GB"]]
    ens = VotingClassifier(estimators=est, voting="soft", n_jobs=1).fit(X_tr[feats], y_tr)
    proba = ens.predict_proba(X_te[feats])[:,1]; pred=(proba>=0.5).astype(int)
    st["test_metrics"]["Ensemble"] = P.compute_metrics(y_te, pred, proba)
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr,tpr,_=roc_curve(y_te,proba); prec,rec,_=precision_recall_curve(y_te,proba)
    st["roc"]["Ensemble"]=(fpr,tpr,st["test_metrics"]["Ensemble"]["ROC-AUC"])
    st["pr"]["Ensemble"]=(rec,prec,st["test_metrics"]["Ensemble"]["PR-AUC"])
    sgkf=StratifiedGroupKFold(n_splits=P.INNER_SPLITS, shuffle=True, random_state=P.RANDOM_STATE)
    fold=[]
    for tr_i,va_i in sgkf.split(X_tr,y_tr,groups=g_tr):
        e=clone(ens); e.fit(X_tr.iloc[tr_i][feats], y_tr[tr_i])
        fold.append(f1_score(y_tr[va_i], e.predict(X_tr.iloc[va_i][feats]), average="macro"))
    st["cv_fold"]["Ensemble"]=fold

    order = model_names+["Ensemble"]
    results_df = pd.DataFrame(st["test_metrics"]).T.loc[order].round(4)
    results_df.to_csv(os.path.join(RES,"test_metrics.csv"))
    json.dump(bp, open(os.path.join(RES,"best_params.json"),"w"), indent=2, default=str)
    json.dump(st["best_cv"], open(os.path.join(RES,"cv_macro_f1.json"),"w"), indent=2)

    # Istatistiksel testler
    sm = order
    so = {}
    try:
        fr=sstats.friedmanchisquare(*[st["cv_fold"][m] for m in sm])
        so["friedman"]={"statistic":float(fr[0]),"p_value":float(fr[1])}
    except Exception as e: so["friedman"]={"error":str(e)}
    mean_scores={m:float(np.mean(st["cv_fold"][m])) for m in sm}
    best=max(mean_scores,key=mean_scores.get)
    comps,pv=[],[]
    for m in sm:
        if m==best: continue
        a,b=st["cv_fold"][best],st["cv_fold"][m]
        try: p=1.0 if np.allclose(a,b) else sstats.wilcoxon(a,b)[1]
        except Exception: p=float("nan")
        comps.append(m); pv.append(p)
    nc=len(pv); bonf=[min(p*nc,1.0) if p==p else p for p in pv]
    so["wilcoxon_vs_best"]={"reference":best,"comparisons":[
        {"vs":comps[i],"p_raw":float(pv[i]) if pv[i]==pv[i] else None,
         "p_bonferroni":float(bonf[i]) if bonf[i]==bonf[i] else None} for i in range(nc)]}
    so["mean_cv_macro_f1"]=mean_scores; so["best_model"]=best
    json.dump(so, open(os.path.join(RES,"statistical_tests.json"),"w"), indent=2)

    pal=sns.color_palette("tab10",len(order))
    # ROC
    plt.figure(figsize=(9,7))
    for n,c in zip(order,pal):
        fpr,tpr,auc=st["roc"][n]; plt.plot(fpr,tpr,label=f"{n} (AUC={auc:.3f})",color=c,lw=2)
    plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Egrileri - Test Seti"); plt.legend(fontsize=11,loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"roc_curve.png"),dpi=150); plt.close()
    # PR
    plt.figure(figsize=(9,7))
    for n,c in zip(order,pal):
        rec,prec,ap=st["pr"][n]; plt.plot(rec,prec,label=f"{n} (AP={ap:.3f})",color=c,lw=2)
    br=float((y_te==1).mean()); plt.axhline(br,color="k",ls="--",lw=1,label=f"Baseline={br:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Egrileri - Test Seti")
    plt.legend(fontsize=11,loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"pr_curve.png"),dpi=150); plt.close()
    # Confusion matrix (Ensemble)
    cm=confusion_matrix(y_te,(proba>=0.5).astype(int))
    plt.figure(figsize=(6.5,5.5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Normal","Papilodem"],yticklabels=["Normal","Papilodem"])
    plt.xlabel("Tahmin"); plt.ylabel("Gercek"); plt.title("Confusion Matrix - Ensemble (Test)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"confusion_matrix.png"),dpi=150); plt.close()
    # Feature importance
    imp=pd.Series(st["rf_imp"],index=feats).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9,8)); sns.barplot(x=imp.values,y=imp.index,palette="viridis")
    plt.xlabel("Onem (Random Forest)"); plt.ylabel("Ozellik"); plt.title("En Onemli 15 Radyomik Ozellik")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"feature_importance.png"),dpi=150); plt.close()
    imp.to_csv(os.path.join(RES,"top_features.csv"))
    # Calibration
    plt.figure(figsize=(8,7))
    uncal=VotingClassifier(estimators=[(n,P.make_model(n,bp[n])) for n in ["RF","ET","GB"]],voting="soft",n_jobs=1).fit(X_tr[feats],y_tr)
    for label,p in [("Kalibrasyonsuz",uncal.predict_proba(X_te[feats])[:,1]),("Sigmoid kalibrasyonlu",proba)]:
        fp,mp=calibration_curve(y_te,p,n_bins=8,strategy="quantile"); plt.plot(mp,fp,"o-",label=label,lw=2)
    plt.plot([0,1],[0,1],"k--",label="Mukemmel kalibrasyon")
    plt.xlabel("Ortalama tahmin olasiligi"); plt.ylabel("Gercek pozitif orani")
    plt.title("Calibration Curve - Ensemble"); plt.legend(fontsize=12); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"calibration_curve.png"),dpi=150); plt.close()
    # Model comparison
    comp=results_df[["Macro-F1","ROC-AUC","PR-AUC"]].copy()
    ax=comp.plot(kind="bar",figsize=(12,7),width=0.8); ax.set_ylabel("Skor")
    ax.set_title("Model Karsilastirmasi - Test Seti"); ax.set_ylim(0,1.0); ax.legend(loc="lower right")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(FIG,"model_comparison.png"),dpi=150); plt.close()

    summary={"n_samples":int(st["n_all"]),"n_raw_features":int(st["n_raw"]),
             "n_features_after_preprocess":int(st["n_pre"]),"n_selected_features":len(feats),
             "n_train":int(st["n_tr"]),"n_test":int(st["n_te"]),"n_patients_total":int(st["n_pat"]),
             "n_patients_train":int(st["n_pat_tr"]),"n_patients_test":int(st["n_pat_te"]),
             "n_trials":P.N_TRIALS,"best_model_cv":best,
             "best_model_test_by_macro_f1":results_df["Macro-F1"].idxmax(),
             "test_class_balance":{"Normal":int((y_te==0).sum()),"Papilodem":int((y_te==1).sum())}}
    json.dump(summary, open(os.path.join(RES,"summary.json"),"w"), indent=2)
    joblib.dump(st, STATE)
    print("FINALIZE ok. En iyi(CV)=",best,"| en iyi test Macro-F1=",results_df["Macro-F1"].idxmax())
    print(results_df.to_string())

if stage == "finalize":
    finalize()
    print("STAGE DONE finalize")

"""Allena e valuta i modelli di churn sulla tabella `dataset` (creata da build_dataset.py).

Modelli confrontati (tutti allenati sullo split "train", scelti sullo split "val",
valutati UNA volta sullo split "test", che è il periodo più recente):

  1. Baseline: regressione logistica con la sola recency
  2. Regressione logistica con tutte le feature
  3. LightGBM (tuning leggero con Optuna)
  4. LightGBM + stagionalità (aggiunge il mese del taglio)
  5. LightGBM + calibrazione isotonica (calibratore addestrato su "val")
  6. LightGBM + calibrazione Platt / Sigmoid (calibratore addestrato su "val")

Output (cartella reports/):
  results.csv, calibration.png, test_predictions.csv

Uso: tasto Play di VS Code, oppure da terminale:
    py src/train_models.py
    py src/train_models.py --n-trials 40
"""
import argparse
import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent  # cartella del progetto (quella che contiene src, sql, data)
SEED = 42
NON_FEATURES = ["snapshot_date", "customer_id", "churn", "split", "snapshot_month"]


def load_dataset(db: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    df = con.execute("SELECT * FROM dataset").df()
    con.close()
    df["snapshot_month"] = pd.to_datetime(df["snapshot_date"]).dt.month
    return df


def make_logreg(C: float):
    """Imputazione mediana (+ indicatori di mancanza), arcsinh per le code lunghe, scaling, logistica."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(np.arcsinh),   # come log, ma ok anche con valori negativi (trend)
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    )

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(p_va: np.ndarray, y_va) -> LogisticRegression:
    """Calibrazione Sigmoid (Platt scaling) sui logit delle probabilità di validazione."""
    return LogisticRegression(C=1e9, max_iter=1000).fit(_logit(p_va), y_va)


def apply_platt(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return cal.predict_proba(_logit(p))[:, 1]


def metrics(name: str, y, p) -> dict:
    return {
        "modello": name,
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


def fit_lgb(params, X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(
        **params, n_estimators=2000, random_state=SEED, verbose=-1, n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def tune_lgb(tr: pd.DataFrame, va: pd.DataFrame, features: list[str], n_trials: int) -> dict:
    """Tuning LightGBM con validazione temporale expanding-window sui tagli di train + val."""
    df_cv = pd.concat([tr, va], ignore_index=True)
    dates = sorted(df_cv["snapshot_date"].unique())
    # Costruisce i fold progressivi (es. [T1,T2]->T3, [T1..T3]->T4, [T1..T4]->val)
    folds = []
    for cutoff in dates[2:]:
        f_tr = df_cv[df_cv["snapshot_date"] < cutoff]
        f_va = df_cv[df_cv["snapshot_date"] == cutoff]
        folds.append((f_tr[features], f_tr["churn"], f_va[features], f_va["churn"]))

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        scores = []
        for X_ftr, y_ftr, X_fva, y_fva in folds:
            model = fit_lgb(params, X_ftr, y_ftr, X_fva, y_fva)
            scores.append(model.best_score_["valid_0"]["binary_logloss"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    best["subsample_freq"] = 1
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "retail.duckdb")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.db)
    features = [c for c in df.columns if c not in NON_FEATURES]
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    y_tr, y_va, y_te = tr["churn"], va["churn"], te["churn"]
    print(f"Feature usate ({len(features)}): {', '.join(features)}")
    print(f"Righe: train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    print(f"Quota di churn: train={y_tr.mean():.1%}  val={y_va.mean():.1%}  test={y_te.mean():.1%}\n")

    probs = {}  # probabilità sul test, per modello

    # 1. Baseline: solo recency
    base = make_logreg(C=1.0).fit(tr[["recency_days"]], y_tr)
    probs["Baseline (solo recency)"] = base.predict_proba(te[["recency_days"]])[:, 1]

    # 2. Regressione logistica: scelgo C su val
    best_C, best_ll = None, np.inf
    for C in [0.001, 0.01, 0.1, 1, 10]:
        m = make_logreg(C).fit(tr[features], y_tr)
        ll = log_loss(y_va, m.predict_proba(va[features])[:, 1])
        if ll < best_ll:
            best_C, best_ll = C, ll
    print(f"Regressione logistica: C scelto su val = {best_C}")
    logreg = make_logreg(best_C).fit(tr[features], y_tr)
    probs["Regressione logistica"] = logreg.predict_proba(te[features])[:, 1]

    # 3. LightGBM con tuning
    print(f"LightGBM: tuning con Optuna ({args.n_trials} prove) ...")
    best_params = tune_lgb(tr, va, features, args.n_trials)
    print("  migliori parametri:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                               for k, v in best_params.items()}))
    (args.out / "best_params.json").write_text(json.dumps(best_params, indent=2))  # serve a analyze_results.py
    gbm = fit_lgb(best_params, tr[features], y_tr, va[features], y_va)
    print(f"  alberi usati (early stopping): {gbm.best_iteration_}")
    p_va_gbm = gbm.predict_proba(va[features])[:, 1]
    probs["LightGBM"] = gbm.predict_proba(te[features])[:, 1]

    # 4. LightGBM + mese del taglio (stagionalità), stessi iperparametri
    feats_s = features + ["snapshot_month"]
    gbm_s = fit_lgb(best_params, tr[feats_s], y_tr, va[feats_s], y_va)
    probs["LightGBM + stagionalità"] = gbm_s.predict_proba(te[feats_s])[:, 1]

    # 5. LightGBM + calibrazione isotonica, addestrata sulle previsioni di val
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va_gbm, y_va)
    probs["LightGBM + isotonica"] = iso.predict(probs["LightGBM"])

    # 6. LightGBM + calibrazione Platt (Sigmoid), addestrata su val
    platt = fit_platt(p_va_gbm, y_va)
    probs["LightGBM + Platt"] = apply_platt(platt, probs["LightGBM"])

    # Risultati sul test
    results = pd.DataFrame([metrics(name, y_te, p) for name, p in probs.items()]).set_index("modello")
    results.round(4).to_csv(args.out / "results.csv")
    print(f"\n=== RISULTATI SUL TEST (quota di churn = {y_te.mean():.1%}) ===")
    print(results.round(4).to_string())
    print("\nLettura: ROC-AUC e PR-AUC più alti = meglio; Brier e LogLoss più bassi = meglio.")
    print("PR-AUC va confrontata con la quota di churn: un modello inutile avrebbe PR-AUC ~ quota.")

    # Importanza delle feature (gain)
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=features).sort_values(ascending=False)
    print("\nTop 10 feature LightGBM (importanza 'gain', in % sul totale):")
    print((100 * imp / imp.sum()).head(10).round(1).to_string())

    # Grafico di calibrazione
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="calibrazione perfetta")
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Probabilità prevista di churn")
    ax.set_ylabel("Frequenza osservata di churn")
    ax.set_title("Curva di calibrazione sul test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "calibration.png", dpi=150)

    # Previsioni di test (serviranno per l'analisi della soglia)
    pred = te[["customer_id", "snapshot_date", "churn"]].copy()
    for name, p in probs.items():
        pred[name] = p
    pred.to_csv(args.out / "test_predictions.csv", index=False)
    print(f"\nSalvati in {args.out}/: results.csv, calibration.png, test_predictions.csv")


if __name__ == "__main__":
    main()

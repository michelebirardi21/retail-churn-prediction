"""Intervalli di confidenza (bootstrap) sulle metriche del test.

Legge reports/test_predictions.csv (creato da train_models.py) e ricampiona i clienti del
test con reinserimento 1000 volte. Per ogni modello calcola ROC-AUC, PR-AUC, Brier e Log-Loss; per
alcune coppie di modelli calcola anche la DIFFERENZA sugli stessi campioni (bootstrap
appaiato), che è il modo corretto per dire se un modello è davvero meglio di un altro.

Limite: il bootstrap misura solo la variabilità dovuta ai clienti del test. Non include la
variabilità del periodo scelto per il test né quella del riaddestramento dei modelli.

Output: reports/bootstrap_ci.csv, reports/bootstrap_diffs.csv

Uso: tasto Play di VS Code, oppure da terminale:
    py src/bootstrap_ci.py
    py src/bootstrap_ci.py --n-boot 2000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["customer_id", "snapshot_date", "churn"]
def _logloss(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

METRICS = {
    "ROC-AUC": roc_auc_score,
    "PR-AUC": average_precision_score,
    "Brier": brier_score_loss,
    "LogLoss": _logloss,
}
LOWER_IS_BETTER = {"Brier", "LogLoss"}
COMPARISONS = [
    ("LightGBM", "Regressione logistica"),
    ("LightGBM", "Baseline (solo recency)"),
    ("Regressione logistica", "Baseline (solo recency)"),
    ("LightGBM + stagionalità", "LightGBM"),
    ("LightGBM + Platt", "LightGBM"),
    ("LightGBM + isotonica", "LightGBM"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    pred = pd.read_csv(args.out / "test_predictions.csv")
    y = pred["churn"].to_numpy()
    models = [c for c in pred.columns if c not in ID_COLS]
    P = {m: pred[m].to_numpy() for m in models}
    n = len(y)

    # Stime puntuali sul test intero
    point = {(m, k): f(y, P[m]) for m in models for k, f in METRICS.items()}

    # Bootstrap: stessi indici per tutti i modelli (serve per le differenze appaiate)
    rng = np.random.default_rng(42)
    boot = {key: np.empty(args.n_boot) for key in point}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for m in models:
            pb = P[m][idx]
            for k, f in METRICS.items():
                boot[(m, k)][b] = f(yb, pb)

    rows = []
    for (m, k), est in point.items():
        lo, hi = np.percentile(boot[(m, k)], [2.5, 97.5])
        rows.append({"modello": m, "metrica": k, "stima": est, "IC95% basso": lo, "IC95% alto": hi})
    ci = pd.DataFrame(rows)
    ci.round(4).to_csv(args.out / "bootstrap_ci.csv", index=False)

    print(f"=== METRICHE SUL TEST CON INTERVALLO BOOTSTRAP AL 95% ({n:,} clienti, {args.n_boot} ricampionamenti) ===")
    for m in models:
        parts = []
        for k in METRICS:
            r = ci[(ci.modello == m) & (ci.metrica == k)].iloc[0]
            parts.append(f"{k} {r['stima']:.3f} [{r['IC95% basso']:.3f}, {r['IC95% alto']:.3f}]")
        print(f"{m:26s} " + "   ".join(parts))

    drows = []
    for a, b in COMPARISONS:
        if a not in models or b not in models:
            continue
        for k in METRICS:
            d = boot[(a, k)] - boot[(b, k)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            better = (d < 0) if k in LOWER_IS_BETTER else (d > 0)
            drows.append({"A": a, "B": b, "metrica": k, "differenza A-B": point[(a, k)] - point[(b, k)],
                          "IC95% basso": lo, "IC95% alto": hi, "A migliore di B (% campioni)": 100 * better.mean()})
    diffs = pd.DataFrame(drows)
    diffs.round(4).to_csv(args.out / "bootstrap_diffs.csv", index=False)

    print("\n=== DIFFERENZE TRA MODELLI (stessi campioni) ===")
    print("Se l'intervallo contiene 0, la differenza non è distinguibile dal caso su questo test.")
    print("Per Brier e Log-Loss valori più bassi sono meglio, quindi una differenza negativa favorisce A.\n")
    for r in drows:
        print(f"{r['A']} - {r['B']} | {r['metrica']:8s} {r['differenza A-B']:+.4f} "
              f"[{r['IC95% basso']:+.4f}, {r['IC95% alto']:+.4f}]  A migliore nel {r['A migliore di B (% campioni)']:.0f}% dei campioni")
    print(f"\nSalvati in {args.out}/: bootstrap_ci.csv, bootstrap_diffs.csv")


if __name__ == "__main__":
    main()

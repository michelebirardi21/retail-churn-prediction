"""Interpretabilità (SHAP) e analisi economica della soglia di intervento.

Prerequisito: aver eseguito build_dataset.py e train_models.py
(train_models salva reports/best_params.json con gli iperparametri di LightGBM).

Cosa fa
  1. Riallena LightGBM con gli stessi iperparametri e calcola i valori SHAP sul test:
     quali comportamenti spingono il rischio di churn in su o in giù.
  2. Analisi della soglia, con costi e benefici IPOTETICI (modificabili da riga di comando):
       - contattare un cliente (sconto, telefonata...) costa  --cost  sterline
       - se il cliente stava per andarsene, la campagna lo trattiene con probabilità --save-rate
       - un cliente trattenuto vale  --margin  x (spesa media per 90 giorni del cliente)
     Profitto di un cliente contattato = (churn reale) x save_rate x valore - costo.
  3. La soglia viene scelta sul periodo di VALIDAZIONE e poi misurata sul TEST,
     quindi il risultato finale non è ottimistico.

Output (cartella reports/): shap_summary.png, shap_bar.png, profit_curve.png, profit_summary.csv

Uso: tasto Play di VS Code, oppure da terminale:
    py src/analyze_results.py
    py src/analyze_results.py --margin 0.25 --save-rate 0.15 --cost 8
    py src/analyze_results.py --seasonal        # usa la variante con il mese del taglio
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

from sklearn.metrics import log_loss
from train_models import NON_FEATURES, ROOT, apply_platt, fit_lgb, fit_platt, load_dataset, make_logreg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def customer_value(df: pd.DataFrame, margin: float) -> pd.Series:
    """Margine atteso in 90 giorni basato sulla spesa NETTA storica fino al taglio."""
    return margin * df["net_spend"] / df["tenure_days"].clip(lower=90) * 90


def profit(contact: np.ndarray, y: np.ndarray, value: np.ndarray, save_rate: float, cost: float) -> float:
    """Profitto totale se si contattano i clienti con contact=True."""
    return float(np.sum(contact * (y * save_rate * value - cost)))


def profit_curve(p, y, value, grid, save_rate, cost) -> np.ndarray:
    return np.array([profit(p >= t, y, value, save_rate, cost) for t in grid])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "retail.duckdb")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--seasonal", action="store_true", help="aggiunge il mese del taglio tra le feature")
    parser.add_argument("--margin", type=float, default=0.30, help="margine lordo (ipotesi)")
    parser.add_argument("--save-rate", type=float, default=0.20, help="prob. di trattenere un cliente a rischio (ipotesi)")
    parser.add_argument("--cost", type=float, default=5.0, help="costo di un contatto in sterline (ipotesi)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    params_file = args.out / "best_params.json"
    if not params_file.exists():
        raise SystemExit(f"Non trovo {params_file}. Esegui prima src/train_models.py.")
    params = json.loads(params_file.read_text())

    df = load_dataset(args.db)
    features = [c for c in df.columns if c not in NON_FEATURES]
    if args.seasonal:
        features = features + ["snapshot_month"]
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    model_name = "LightGBM + stagionalità" if args.seasonal else "LightGBM"

    model = fit_lgb(params, tr[features], tr["churn"], va[features], va["churn"])
    p_va = model.predict_proba(va[features])[:, 1]
    p_te = model.predict_proba(te[features])[:, 1]

    # ------------------------------------------------------------------ SHAP
    print(f"=== SHAP ({model_name}, calcolato sul test) ===")
    sv = shap.TreeExplainer(model).shap_values(te[features])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, te[features], max_display=15, show=False)
    plt.title("Effetto delle feature sul rischio di churn (SHAP, test)")
    plt.tight_layout()
    plt.savefig(args.out / "shap_summary.png", dpi=150)
    plt.close()
    shap.summary_plot(sv, te[features], plot_type="bar", max_display=15, show=False)
    plt.title("Importanza media assoluta (SHAP, test)")
    plt.tight_layout()
    plt.savefig(args.out / "shap_bar.png", dpi=150)
    plt.close()

    imp = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)
    print("Top 10 feature per importanza SHAP media (in log-odds di churn):")
    print(imp.head(10).round(3).to_string())
    print("Nel grafico shap_summary.png: punti a DESTRA = aumentano il rischio di churn; "
          "colore = valore della feature (rosso alto, blu basso, grigio = mancante).\n")

    # ------------------------------------------------- Analisi della soglia
    s, c = args.save_rate, args.cost
    cap = customer_value(tr, args.margin).quantile(0.99)  # limito i grandissimi clienti (percentile 99 del train)
    v_va = customer_value(va, args.margin).clip(upper=cap).to_numpy()
    v_te = customer_value(te, args.margin).clip(upper=cap).to_numpy()
    y_va, y_te = va["churn"].to_numpy(), te["churn"].to_numpy()

    base = make_logreg(1.0).fit(tr[["recency_days"]], tr["churn"])
    pb_va = base.predict_proba(va[["recency_days"]])[:, 1]
    pb_te = base.predict_proba(te[["recency_days"]])[:, 1]

    # Calibrazione Platt (Sigmoid) su val per la regola di valore atteso
    platt = fit_platt(p_va, y_va)
    p_te_cal = apply_platt(platt, p_te)

    # Regressione logistica con tutte le feature (C scelto su val)
    best_C, best_ll = None, np.inf
    for C in [0.001, 0.01, 0.1, 1, 10]:
        m = make_logreg(C).fit(tr[features], tr["churn"])
        ll = log_loss(y_va, m.predict_proba(va[features])[:, 1])
        if ll < best_ll:
            best_C, best_ll = C, ll
    logreg = make_logreg(best_C).fit(tr[features], tr["churn"])
    plr_va = logreg.predict_proba(va[features])[:, 1]
    plr_te = logreg.predict_proba(te[features])[:, 1]

    grid = np.linspace(0, 1, 101)
    curves = {
        model_name: (profit_curve(p_va, y_va, v_va, grid, s, c), profit_curve(p_te, y_te, v_te, grid, s, c)),
        "Regressione logistica": (profit_curve(plr_va, y_va, v_va, grid, s, c), profit_curve(plr_te, y_te, v_te, grid, s, c)),
        "Baseline (solo recency)": (profit_curve(pb_va, y_va, v_va, grid, s, c), profit_curve(pb_te, y_te, v_te, grid, s, c)),
    }
    t_star = {name: grid[int(np.argmax(cv[0]))] for name, cv in curves.items()}  # scelta su VAL

    n_te, n_churn = len(te), int(y_te.sum())
    q_train = float(tr["churn"].mean())

    def row(policy: str, mask: np.ndarray) -> dict:
        return {
            "strategia": policy,
            "clienti contattati (%)": 100 * mask.mean(),
            "churner raggiunti (%)": 100 * (mask & (y_te == 1)).sum() / n_churn,
            "profitto test (£)": profit(mask, y_te, v_te, s, c),
            "profitto per cliente (£)": profit(mask, y_te, v_te, s, c) / n_te,
        }

    policies = [
        ("Non fare nulla", np.zeros(n_te, dtype=bool)),
        ("Contatta tutti", np.ones(n_te, dtype=bool)),
        (f"Baseline recency, soglia {t_star['Baseline (solo recency)']:.2f} (scelta su val)",
         pb_te >= t_star["Baseline (solo recency)"]),
        (f"{model_name}, soglia {t_star[model_name]:.2f} (scelta su val)", p_te >= t_star[model_name]),
        (f"{model_name}, regola di valore atteso (p x {s:g} x valore > costo)", p_te * s * v_te > c),
        (f"Regressione logistica, soglia {t_star['Regressione logistica']:.2f} (scelta su val)",
         plr_te >= t_star["Regressione logistica"]),
        ("Regressione logistica, regola di valore atteso", plr_te * s * v_te > c),
        (f"{model_name} + Platt, regola di valore atteso", p_te_cal * s * v_te > c),
        # Controllo: stessa regola ma senza modello, con la quota media di churn del train al posto di p.
        # Serve a separare quanto guadagno viene dal VALORE del cliente e quanto dal MODELLO.
        (f"Solo valore, senza modello ({q_train:.0%} x {s:g} x valore > costo)", q_train * s * v_te > c),
    ]
    rows = [row(name, mask) for name, mask in policies]
    summary = pd.DataFrame(rows).set_index("strategia")
    perfect = float(np.sum(y_te * np.maximum(s * v_te - c, 0)))
    best_hindsight = grid[int(np.argmax(curves[model_name][1]))]

    print("=== ANALISI DELLA SOGLIA (ipotesi, non dati reali) ===")
    print(f"Ipotesi: margine {args.margin:.0%}, probabilità di trattenere {s:.0%}, costo per contatto £{c:g}")
    print(f"Test: {n_te:,} clienti, di cui {n_churn:,} ({n_churn / n_te:.1%}) in churn\n")
    pd.set_option("display.width", 200)
    print(summary.round(1).to_string())
    print(f"\nLimite teorico con informazione perfetta (si contatta solo chi sta per andarsene): £{perfect:,.0f}")
    print(f"Soglia ottima a posteriori sul test (non utilizzabile in pratica): {best_hindsight:.2f}")
    summary.round(2).to_csv(args.out / "profit_summary.csv")

    # Bootstrap sui clienti del test: quanto sono affidabili le differenze di profitto tra strategie?
    # (le strategie e le soglie sono fisse: si ricampiona solo l'insieme dei clienti)
    names = [n for n, _ in policies]
    contrib = {n: m * (y_te * s * v_te - c) for n, m in policies}   # contributo di ogni cliente
    idx = np.random.default_rng(42).integers(0, n_te, size=(1000, n_te))
    pairs = [
        (names[4], names[1]),   # LightGBM valore atteso vs Contatta tutti
        (names[4], names[-1]),  # LightGBM valore atteso vs Solo valore senza modello
        (names[7], names[4]),   # LightGBM + Platt vs LightGBM grezzo (valore atteso)
        (names[4], names[6]),   # LightGBM vs Regressione logistica (valore atteso)
        (names[-1], names[1]),  # Solo valore senza modello vs Contatta tutti
    ]
    boot_rows = []
    for a, b in pairs:
        d = contrib[a] - contrib[b]
        boot = d[idx].sum(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        boot_rows.append({"A": a, "meno B": b, "differenza (£)": d.sum(), "IC95% basso": lo, "IC95% alto": hi})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.round(1).to_csv(args.out / "profit_bootstrap.csv", index=False)
    print("\nDifferenze di profitto sul test (A meno B), con intervallo bootstrap al 95% (1000 ricampionamenti):")
    for r in boot_rows:
        print(f"  A = {r['A']}\n  B = {r['meno B']}\n  -> {r['differenza (£)']:+,.0f} £  [{r['IC95% basso']:+,.0f}, {r['IC95% alto']:+,.0f}]\n")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, (_, cv_te) in curves.items():
        ax.plot(grid, cv_te, label=name)
    ax.axhline(0, color="black", lw=0.8, label="Non fare nulla")
    ax.axhline(rows[1]["profitto test (£)"], color="gray", ls=":", label="Contatta tutti")
    ax.axhline(perfect, color="green", ls="--", lw=1, label="Informazione perfetta")
    ax.axvline(t_star[model_name], color="C0", ls="--", alpha=0.5, label=f"soglia scelta su val ({t_star[model_name]:.2f})")
    ax.set_xlabel("Soglia di probabilità di churn (si contattano i clienti con p >= soglia)")
    ax.set_ylabel("Profitto sul test (£)")
    ax.set_title(f"Profitto per soglia - ipotesi: margine {args.margin:.0%}, successo {s:.0%}, costo £{c:g}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "profit_curve.png", dpi=150)

    # ------------------------------------------------- Analisi di sensitività
    sens_rows = []
    for sr in [0.10, 0.20, 0.30]:
        for co in [2.0, 5.0, 10.0]:
            mask_mod = (p_te_cal * sr * v_te) > co
            mask_val = (q_train * sr * v_te) > co
            mask_all = np.ones(n_te, dtype=bool)

            prof_mod = profit(mask_mod, y_te, v_te, sr, co)
            prof_val = profit(mask_val, y_te, v_te, sr, co)
            prof_all = profit(mask_all, y_te, v_te, sr, co)

            sens_rows.append({
                "save_rate": f"{sr:.0%}",
                "costo (£)": co,
                "contattati (%)": 100 * mask_mod.mean(),
                "profitto modello (£)": prof_mod,
                "vs Contatta tutti (£)": prof_mod - prof_all,
                "vs Solo valore (£)": prof_mod - prof_val,
            })

    sens_df = pd.DataFrame(sens_rows)
    sens_df.round(1).to_csv(args.out / "sensitivity.csv", index=False)
    print(f"\n=== ANALISI DI SENSITIVITÀ ({model_name} + Platt, regola valore atteso) ===")
    print(sens_df.round(1).to_string(index=False))
    print(f"\nSalvati in {args.out}/: shap_summary.png, shap_bar.png, profit_curve.png, "
          f"profit_summary.csv, profit_bootstrap.csv, sensitivity.csv")


if __name__ == "__main__":
    main()

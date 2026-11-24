"""Costruisce il dataset per il modello: una riga per (cliente, data di taglio).

Per ogni data di taglio esegue sql/02_features.sql, poi unisce i risultati
e assegna lo split temporale (train / val / test) nella tabella `dataset` di DuckDB.

Schema temporale (orizzonte = 90 giorni, tagli distanziati di 90 giorni):

    ... train ... | train | train | train | VAL | TEST
                                              ^     ^
                        il test usa l'ultimo taglio possibile: la sua finestra
                        di 90 giorni finisce dove finiscono i dati.

Poiché la finestra dell'etichetta di ogni taglio finisce al taglio successivo,
nessuna etichetta di train/val "guarda" dentro il periodo di test.

Uso: tasto Play di VS Code, oppure da terminale:
    py src/build_dataset.py
    py src/build_dataset.py --n-train 4
"""
import argparse
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # cartella del progetto (quella che contiene src, sql, data)
SQL_FEATURES = ROOT / "sql" / "02_features.sql"
HORIZON_DAYS = 90


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "retail.duckdb")
    parser.add_argument("--n-train", type=int, default=4, help="numero di date di taglio per il train")
    args = parser.parse_args()

    con = duckdb.connect(str(args.db))
    first_day, last_day = con.execute(
        "SELECT MIN(invoice_date), MAX(invoice_date) FROM purchases"
    ).fetchone()
    print(f"Dati disponibili: {first_day} -> {last_day}")

    # Ultimo taglio possibile: la finestra [taglio, taglio+90) deve stare dentro i dati.
    test_cutoff = last_day - timedelta(days=HORIZON_DAYS)
    val_cutoff = test_cutoff - timedelta(days=HORIZON_DAYS)
    train_cutoffs = [val_cutoff - timedelta(days=HORIZON_DAYS * k) for k in range(1, args.n_train + 1)]

    snapshots = [("train", c) for c in sorted(train_cutoffs)] + [("val", val_cutoff), ("test", test_cutoff)]
    if snapshots[0][1] <= first_day + timedelta(days=90):
        raise SystemExit("Troppi tagli di train: il primo non ha abbastanza storico. Riduci --n-train.")

    sql = SQL_FEATURES.read_text(encoding="utf-8")
    parts = []
    for split, cutoff in snapshots:
        df = con.execute(sql, {"cutoff": cutoff}).df()
        df["split"] = split
        parts.append(df)
        print(f"  taglio {cutoff} ({split:5s}): {len(df):5d} clienti, churn = {df['churn'].mean():.1%}")

    dataset = pd.concat(parts, ignore_index=True)
    con.register("dataset_df", dataset)
    con.execute("CREATE OR REPLACE TABLE dataset AS SELECT * FROM dataset_df")

    print(f"\nTabella `dataset`: {len(dataset):,} righe, {dataset.shape[1]} colonne")
    print("\nRiepilogo per split:")
    print(dataset.groupby("split").agg(righe=("churn", "size"), churn_rate=("churn", "mean")).round(3).to_string())
    print("\nValori mancanti per colonna (solo quelle con NULL):")
    na = dataset.isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "  nessuno")
    con.close()


if __name__ == "__main__":
    main()

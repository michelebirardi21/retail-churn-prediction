"""Carica Online Retail II (xlsx con due fogli) in DuckDB e lancia la pulizia SQL.

Uso: basta premere il tasto Play di VS Code (il file Excel deve stare in data/online_retail_II.xlsx).
Oppure da terminale:
    py src/load_data.py                      # usa il file di default
    py src/load_data.py "data/altro_nome.xlsx"
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # cartella del progetto (quella che contiene src, sql, data)
SQL_CLEAN = ROOT / "sql" / "01_clean.sql"
DEFAULT_XLSX = ROOT / "data" / "online_retail_II.xlsx"


def read_sheets(xlsx_path: Path) -> pd.DataFrame:
    """Legge tutti i fogli, li mette in ordine cronologico e li unisce."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)  # lento (xlsx): ~1-2 minuti
    frames = []
    for name, df in sheets.items():
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df["source_sheet"] = name
        df["_min_date"] = df["InvoiceDate"].min()
        print(f"  foglio '{name}': {len(df):,} righe, "
              f"{df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
        frames.append(df)
    frames.sort(key=lambda d: d["_min_date"].iloc[0])

    # I due fogli si sovrappongono in dicembre 2010: tengo il foglio più recente
    # per il periodo in comune, così non duplico le transazioni.
    cleaned = [frames[0]]
    for prev, nxt in zip(frames, frames[1:]):
        cutoff = nxt["InvoiceDate"].min()
        overlap = (cleaned[-1]["InvoiceDate"] >= cutoff).sum()
        if overlap:
            print(f"  sovrapposizione: rimuovo {overlap:,} righe dal foglio più vecchio "
                  f"(InvoiceDate >= {cutoff})")
            cleaned[-1] = cleaned[-1][cleaned[-1]["InvoiceDate"] < cutoff]
        cleaned.append(nxt)

    df = pd.concat(cleaned, ignore_index=True).drop(columns=["_min_date"])
    # tipi misti nell'xlsx: li rendo uniformi prima di passarli a DuckDB
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path, nargs="?", default=DEFAULT_XLSX)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "retail.duckdb")
    args = parser.parse_args()

    print(f"Leggo {args.xlsx} ...")
    df = read_sheets(args.xlsx)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.db))
    con.register("df_raw", df)
    con.execute("CREATE OR REPLACE TABLE transactions_raw AS SELECT * FROM df_raw")
    print(f"transactions_raw: {len(df):,} righe")

    print("Eseguo la pulizia SQL ...")
    con.execute(SQL_CLEAN.read_text(encoding="utf-8"))

    print("\nReport di pulizia:")
    print(con.execute("SELECT step, n_rows FROM cleaning_report ORDER BY ord").df().to_string(index=False))

    print("\nControlli rapidi:")
    print(con.execute("""
        SELECT COUNT(DISTINCT customer_id) AS clienti,
               MIN(invoice_date) AS data_min,
               MAX(invoice_date) AS data_max
        FROM purchases
    """).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()

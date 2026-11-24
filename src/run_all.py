"""Esegue l'intera pipeline, nell'ordine giusto.

Uso: tasto Play di VS Code su questo file, oppure da terminale:
    py src/run_all.py                # tutto, dall'Excel ai risultati
    py src/run_all.py --skip-load    # salta il caricamento dell'Excel (lento) se data/retail.duckdb esiste già

Prerequisiti: il file data/online_retail_II.xlsx e le librerie di requirements.txt.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["load_data.py", "build_dataset.py", "train_models.py", "analyze_results.py", "bootstrap_ci.py"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-load", action="store_true", help="salta load_data.py")
    args = parser.parse_args()

    steps = STEPS[1:] if args.skip_load else STEPS
    for i, step in enumerate(steps, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(steps)}] {step}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, str(SRC / step)], check=True)
    print("\nFatto. I risultati sono nella cartella reports/.")


if __name__ == "__main__":
    main()

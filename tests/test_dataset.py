from pathlib import Path
import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "retail.duckdb"


@pytest.fixture(scope="module")
def dataset():
    """Carica la tabella dataset da DuckDB una sola volta per tutti i test."""
    if not DB_PATH.exists():
        pytest.skip("Database data/retail.duckdb non trovato: esegui prima build_dataset.py")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM dataset").df()
    con.close()
    return df


def test_no_duplicates(dataset):
    """Ogni coppia (customer_id, snapshot_date) deve comparire una sola volta."""
    dups = dataset.duplicated(subset=["customer_id", "snapshot_date"]).sum()
    assert dups == 0, f"Trovate {dups} righe duplicate su (customer_id, snapshot_date)"


def test_target_and_key_columns_valid(dataset):
    """churn deve essere solo 0 o 1, e le colonne chiave non devono avere NULL."""
    assert set(dataset["churn"].unique()).issubset({0, 1})
    for col in ["customer_id", "snapshot_date", "recency_days", "net_spend", "split"]:
        assert dataset[col].isna().sum() == 0, f"Valori NULL trovati in {col}"


def test_active_population_and_no_future_leakage(dataset):
    """Recency deve essere tra 1 e 180 giorni (niente acquisti futuri, niente zombie)."""
    assert (dataset["recency_days"] > 0).all(), "Trovata recency <= 0 (possibile leakage)"
    assert (dataset["recency_days"] <= 180).all(), "Trovati clienti inattivi da > 180 giorni"
    assert (dataset["net_spend"] >= 0).all(), "Trovata net_spend negativa"


def test_temporal_splits_ordered(dataset):
    """Le date di train devono precedere val, e val deve precedere test (di almeno 90 giorni)."""
    max_train = dataset.loc[dataset["split"] == "train", "snapshot_date"].max()
    min_val = dataset.loc[dataset["split"] == "val", "snapshot_date"].min()
    max_val = dataset.loc[dataset["split"] == "val", "snapshot_date"].max()
    min_test = dataset.loc[dataset["split"] == "test", "snapshot_date"].min()

    assert (min_val - max_train).days >= 90, "Sovrapposizione tra train e val"
    assert (min_test - max_val).days >= 90, "Sovrapposizione tra val e test"
from __future__ import annotations
from pathlib import Path
import json, warnings
import pandas as pd

STRING_HINTS = {
    "ZipCode", "zipcode", "zipcode_raw", "zipcode_normalized", "Customer Id", "customerid",
    "Customer Key", "customerkey", "InvoiceAccountCustomerKey", "invoiceaccountcustomerkey",
    "CustomerShipTokey", "customershiptokey", "Product Key", "productkey",
    "Warehouse Location Key", "Warehouse_Location_Key", "warehouselocationkey",
    "ShiptoSeq", "shiptoseq", "InvoiceId", "invoiceid", "SalesId", "salesid",
}


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def choose_format(preferred: str = "parquet") -> str:
    if preferred == "parquet" and not _parquet_available():
        warnings.warn("pyarrow is not installed; using CSV for this run. Install requirements.txt locally for Parquet.")
        return "csv"
    return preferred


def write_df(df: pd.DataFrame, base: Path, preferred: str = "parquet") -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    fmt = choose_format(preferred)
    for ext in (".parquet", ".csv"):
        p = base.with_suffix(ext)
        if p.exists():
            p.unlink()
    if fmt == "parquet":
        p = base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    else:
        p = base.with_suffix(".csv")
        df.to_csv(p, index=False)
    return p


def read_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
        for c in df.columns:
            if c in STRING_HINTS:
                df[c] = df[c].astype("string")
        return df
    raise ValueError(f"Unsupported file: {path}")


def resolve_table(base: Path) -> Path:
    for ext in (".parquet", ".csv"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"No table found for {base} (.parquet/.csv)")


def read_table(base: Path) -> pd.DataFrame:
    return read_df(resolve_table(base))


def write_partition(df: pd.DataFrame, table_dir: Path, snapshot_week: str, preferred: str = "parquet") -> Path:
    part = table_dir / f"snapshot_week={snapshot_week}"
    part.mkdir(parents=True, exist_ok=True)
    return write_df(df, part / "data", preferred)


def read_partition(table_dir: Path, snapshot_week: str) -> pd.DataFrame:
    return read_table(table_dir / f"snapshot_week={snapshot_week}" / "data")


def update_manifest(root: Path, section: str, payload: dict) -> None:
    path = root / "data" / "local_pipeline_manifest.json"
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current[section] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import re
from typing import BinaryIO

import pandas as pd

from src.synthetic.contracts import SOURCE_CONTRACTS
from src.local_pipeline.io import STRING_HINTS


SOURCE_TABLE_ORDER = [
    "dimcustomer",
    "factsalesinvoice",
    "dimproduct",
    "dimwarehouselocation",
    "dimcustomershipto",
]

KEY_COLUMNS = {
    "dimcustomer": ["Customer Key"],
    "factsalesinvoice": [
        "InvoiceAccountCustomerKey",
        "Product Key",
        "Warehouse Location Key",
        "CustomerShipTokey",
    ],
    "dimproduct": ["Product Key"],
    "dimwarehouselocation": ["Warehouse Location Key"],
    "dimcustomershipto": ["CustomerShipTokey"],
}

DATE_COLUMNS = {
    "factsalesinvoice": [
        "InvoiceDate",
        "SalesOrderCreatedDate",
        "SalesOrderDate",
    ],
}

NUMERIC_COLUMNS = {
    "factsalesinvoice": ["NetAmountExtended", "Quantity"],
    "dimcustomer": ["Credit Max"],
}


class UploadValidationError(ValueError):
    """Raised when an uploaded source file cannot be read or validated."""


def template_csv(table_name: str) -> bytes:
    _validate_table_name(table_name)
    return (",".join(SOURCE_CONTRACTS[table_name]) + "\n").encode("utf-8")


def _validate_table_name(table_name: str) -> None:
    if table_name not in SOURCE_CONTRACTS:
        raise KeyError(f"Unknown source table: {table_name}")


def _safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise UploadValidationError(
            f"Unsupported file type for {filename}. Use CSV or Parquet."
        )
    return suffix


def read_uploaded_file(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = _safe_extension(filename)
    bio = BytesIO(payload)
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(bio)
        else:
            df = pd.read_csv(bio, low_memory=False)
    except Exception as exc:
        raise UploadValidationError(
            f"Could not read {filename}: {exc}"
        ) from exc

    for column in df.columns:
        if column in STRING_HINTS:
            df[column] = df[column].astype("string")
    return df


def _nonblank(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.notna() & text.str.strip().ne("")


def validate_source_table(table_name: str, df: pd.DataFrame) -> dict:
    _validate_table_name(table_name)
    required = SOURCE_CONTRACTS[table_name]
    required_set = set(required)
    present = set(df.columns)

    missing_columns = [c for c in required if c not in present]
    extra_columns = [c for c in df.columns if c not in required_set]

    checks: list[dict] = []

    def add_check(name: str, status: str, detail: str, count: int | None = None) -> None:
        checks.append(
            {
                "check": name,
                "status": status,
                "detail": detail,
                "count": count,
            }
        )

    if len(df) == 0:
        add_check("Rows present", "FAIL", "Uploaded table has no rows.", 0)
    else:
        add_check("Rows present", "PASS", f"{len(df):,} rows loaded.", len(df))

    if missing_columns:
        add_check(
            "Required columns",
            "FAIL",
            "Missing: " + ", ".join(missing_columns),
            len(missing_columns),
        )
    else:
        add_check(
            "Required columns",
            "PASS",
            f"All {len(required)} required columns are present.",
            0,
        )

    if extra_columns:
        add_check(
            "Extra columns",
            "INFO",
            f"{len(extra_columns)} extra columns will be ignored by the source contract.",
            len(extra_columns),
        )
    else:
        add_check("Extra columns", "PASS", "No extra columns.", 0)

    duplicate_rows = int(df.duplicated().sum())
    add_check(
        "Exact duplicate rows",
        "WARN" if duplicate_rows else "PASS",
        (
            f"{duplicate_rows:,} exact duplicate rows detected."
            if duplicate_rows
            else "No exact duplicate rows."
        ),
        duplicate_rows,
    )

    key_quality = {}
    for key in KEY_COLUMNS.get(table_name, []):
        if key not in df.columns:
            continue
        null_or_blank = int((~_nonblank(df[key])).sum())
        duplicate_key = None
        if table_name != "factsalesinvoice":
            usable = df.loc[_nonblank(df[key]), key]
            duplicate_key = int(usable.duplicated().sum())
        key_quality[key] = {
            "null_or_blank": null_or_blank,
            "duplicate": duplicate_key,
        }
        if null_or_blank:
            add_check(
                f"Key completeness: {key}",
                "WARN",
                f"{null_or_blank:,} rows have a null/blank key.",
                null_or_blank,
            )
        else:
            add_check(
                f"Key completeness: {key}",
                "PASS",
                "No null/blank values.",
                0,
            )
        if duplicate_key is not None:
            add_check(
                f"Key uniqueness: {key}",
                "WARN" if duplicate_key else "PASS",
                (
                    f"{duplicate_key:,} duplicate key values detected; the snapshot layer keeps the first row."
                    if duplicate_key
                    else "Key values are unique."
                ),
                duplicate_key,
            )

    date_quality = {}
    for column in DATE_COLUMNS.get(table_name, []):
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        nonblank = _nonblank(df[column])
        invalid = int((nonblank & parsed.isna()).sum())
        valid = parsed.dropna()
        date_quality[column] = {
            "invalid": invalid,
            "min": str(valid.min().date()) if len(valid) else None,
            "max": str(valid.max().date()) if len(valid) else None,
        }
        add_check(
            f"Date parse: {column}",
            "WARN" if invalid else "PASS",
            (
                f"{invalid:,} nonblank values could not be parsed as dates."
                if invalid
                else (
                    f"Date range {valid.min().date()} to {valid.max().date()}."
                    if len(valid)
                    else "No valid date values found."
                )
            ),
            invalid,
        )

    numeric_quality = {}
    for column in NUMERIC_COLUMNS.get(table_name, []):
        if column not in df.columns:
            continue
        parsed = pd.to_numeric(df[column], errors="coerce")
        nonblank = _nonblank(df[column])
        invalid = int((nonblank & parsed.isna()).sum())
        numeric_quality[column] = {"invalid": invalid}
        add_check(
            f"Numeric parse: {column}",
            "WARN" if invalid else "PASS",
            (
                f"{invalid:,} nonblank values could not be parsed as numeric."
                if invalid
                else "All nonblank values are numeric-compatible."
            ),
            invalid,
        )

    statuses = [c["status"] for c in checks]
    fatal = "FAIL" in statuses
    warnings = sum(s == "WARN" for s in statuses)

    if fatal:
        status = "FAIL"
    elif warnings:
        status = "READY_WITH_WARNINGS"
    else:
        status = "READY"

    return {
        "table": table_name,
        "status": status,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "required_columns": len(required),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "duplicate_rows": duplicate_rows,
        "key_quality": key_quality,
        "date_quality": date_quality,
        "numeric_quality": numeric_quality,
        "checks": checks,
    }


def validate_upload_bundle(
    uploads: dict[str, tuple[str, bytes]],
) -> tuple[dict, dict[str, pd.DataFrame]]:
    validations = {}
    frames: dict[str, pd.DataFrame] = {}

    for table_name in SOURCE_TABLE_ORDER:
        if table_name not in uploads:
            validations[table_name] = {
                "table": table_name,
                "status": "MISSING",
                "rows": 0,
                "columns": 0,
                "required_columns": len(SOURCE_CONTRACTS[table_name]),
                "missing_columns": SOURCE_CONTRACTS[table_name],
                "extra_columns": [],
                "duplicate_rows": 0,
                "key_quality": {},
                "date_quality": {},
                "numeric_quality": {},
                "checks": [
                    {
                        "check": "File uploaded",
                        "status": "FAIL",
                        "detail": "No file uploaded for this required source table.",
                        "count": 1,
                    }
                ],
            }
            continue

        filename, payload = uploads[table_name]
        try:
            frame = read_uploaded_file(filename, payload)
            frames[table_name] = frame
            validations[table_name] = validate_source_table(table_name, frame)
        except UploadValidationError as exc:
            validations[table_name] = {
                "table": table_name,
                "status": "FAIL",
                "rows": 0,
                "columns": 0,
                "required_columns": len(SOURCE_CONTRACTS[table_name]),
                "missing_columns": SOURCE_CONTRACTS[table_name],
                "extra_columns": [],
                "duplicate_rows": 0,
                "key_quality": {},
                "date_quality": {},
                "numeric_quality": {},
                "checks": [
                    {
                        "check": "File readable",
                        "status": "FAIL",
                        "detail": str(exc),
                        "count": 1,
                    }
                ],
            }

    statuses = [validations[t]["status"] for t in SOURCE_TABLE_ORDER]
    ready = all(s in {"READY", "READY_WITH_WARNINGS"} for s in statuses)
    warning_tables = sum(s == "READY_WITH_WARNINGS" for s in statuses)

    summary = {
        "status": (
            "READY_WITH_WARNINGS"
            if ready and warning_tables
            else "READY"
            if ready
            else "FAIL"
        ),
        "ready_to_stage": ready,
        "tables_received": int(sum(t in uploads for t in SOURCE_TABLE_ORDER)),
        "tables_required": len(SOURCE_TABLE_ORDER),
        "warning_tables": warning_tables,
        "total_rows": int(sum(v.get("rows", 0) for v in validations.values())),
        "tables": validations,
    }
    return summary, frames


def stage_upload_bundle(
    root: str | Path,
    uploads: dict[str, tuple[str, bytes]],
    validation: dict,
) -> dict:
    if not validation.get("ready_to_stage"):
        raise UploadValidationError(
            "Upload bundle is not ready to stage. Resolve validation failures first."
        )

    root = Path(root)
    run_id = "UPLOAD_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "data" / "uploads" / run_id
    source_dir = run_dir / "edw"
    source_dir.mkdir(parents=True, exist_ok=False)

    files = {}
    for table_name in SOURCE_TABLE_ORDER:
        filename, payload = uploads[table_name]
        suffix = _safe_extension(filename)
        path = source_dir / f"{table_name}{suffix}"
        path.write_bytes(payload)
        files[table_name] = str(path)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": validation["status"],
        "ready_to_promote": True,
        "source_dir": str(source_dir),
        "files": files,
        "validation": validation,
    }
    manifest_path = run_dir / "upload_validation.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def list_staged_uploads(root: str | Path, limit: int = 10) -> list[dict]:
    base = Path(root) / "data" / "uploads"
    if not base.exists():
        return []

    rows = []
    for run_dir in sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    ):
        manifest_path = run_dir / "upload_validation.json"
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "run_id": payload.get("run_id", run_dir.name),
                "created_at": payload.get("created_at"),
                "status": payload.get("status"),
                "ready_to_promote": payload.get("ready_to_promote", False),
                "total_rows": payload.get("validation", {}).get("total_rows", 0),
                "source_dir": payload.get("source_dir"),
            }
        )
        if len(rows) >= limit:
            break
    return rows

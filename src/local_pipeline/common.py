from __future__ import annotations
import re
import pandas as pd
from src.synthetic.contracts import RENAME_MAP

_BAD_COL_CHARS = [" ", ";", ",", "{", "}", "(", ")", "\n", "\t", "="]


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols=[]
    for c in out.columns:
        safe=str(c)
        for ch in _BAD_COL_CHARS:
            safe=safe.replace(ch,"_")
        cols.append(safe)
    out.columns=cols
    return out


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    mapping={}
    for raw,friendly in RENAME_MAP.items():
        mapping[raw]=friendly
        mapping[re.sub(r"[ ,;{}()\n\t=]+","_",raw).strip("_")]=friendly
    mapping["WarehouseLocationKey"]="warehouselocationkey"
    mapping["Warehouse_Location_Key"]="warehouselocationkey"
    for c in out.columns:
        if c in mapping:
            out=out.rename(columns={c:mapping[c]})
    return out


def is_bad_name(s: pd.Series) -> pd.Series:
    txt=s.astype("string")
    trimmed=txt.str.strip()
    return txt.isna() | trimmed.eq("") | trimmed.eq("0") | trimmed.str.upper().eq("UNASSIGNED")

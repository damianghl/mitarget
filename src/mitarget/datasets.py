"""
Loaders for the real-data benchmarks (OpenML ARFF exports).

Each loader returns a :class:`CategoricalTask`: an integer-coded
high-cardinality feature ``x``, an integer-coded target ``y`` and their
cardinalities. The ARFF files are not shipped with the package; point the
loaders at local copies (see ``benchmarks/README.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CategoricalTask:
    x: np.ndarray
    y: np.ndarray
    kx: int
    ky: int
    label: str

    @property
    def n(self) -> int:
        return len(self.x)


_ATTR_RE = re.compile(r"@attribute\s+('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)\s+(.*)$",
                      re.IGNORECASE)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1].replace("\\" + token[0], token[0])
    return token


def _split_data_line(line: str) -> list[str | None]:
    """Split one dense ARFF data row; handles '...' and "..." quoting and ``?`` missing."""
    out, i, n = [], 0, len(line)
    while i <= n:
        while i < n and line[i] in " \t":
            i += 1
        if i < n and line[i] in "'\"":
            q = line[i]
            j = i + 1
            buf = []
            while j < n and line[j] != q:
                if line[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(line[j])
                j += 1
            value: str | None = "".join(buf)
            j += 1  # closing quote
            while j < n and line[j] != ",":
                j += 1
        else:
            j = line.find(",", i)
            j = n if j < 0 else j
            raw = line[i:j].strip()
            value = None if raw == "?" else raw
        out.append(value)
        i = j + 1
    return out


def read_arff(path: str | Path) -> pd.DataFrame:
    """Read a dense ARFF file into a DataFrame of strings (``None`` for ``?``)."""
    columns: list[str] = []
    rows: list[list[str | None]] = []
    in_data = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                low = line.lower()
                if low.startswith("@attribute"):
                    m = _ATTR_RE.match(line)
                    if m is None:
                        raise ValueError(f"cannot parse attribute line: {line[:80]}")
                    columns.append(_unquote(m.group(1)))
                elif low.startswith("@data"):
                    in_data = True
                continue
            if line.startswith("{"):
                raise ValueError("sparse ARFF rows are not supported")
            row = _split_data_line(line)
            if len(row) != len(columns):
                raise ValueError(f"row has {len(row)} fields, expected {len(columns)}: {line[:80]}")
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _task(x_values: pd.Series, y_codes: np.ndarray, label: str) -> CategoricalTask:
    x_codes, x_unique = pd.factorize(x_values)
    return CategoricalTask(x=x_codes.astype(int), y=np.asarray(y_codes, dtype=int),
                           kx=len(x_unique), ky=int(np.max(y_codes)) + 1, label=label)


def load_employee_salaries(path, cat_col: str = "employee_position_title",
                           n_bins: int = 4) -> CategoricalTask:
    """``employee_salaries``: position title vs. salary quantile bin (multiclass)."""
    df = read_arff(path).dropna(subset=[cat_col, "current_annual_salary"])
    salary = df["current_annual_salary"].astype(float)
    y = pd.qcut(salary, q=n_bins, labels=False, duplicates="drop").to_numpy()
    return _task(df[cat_col], y, f"employee_salaries / {cat_col} (salary {n_bins}-quantile)")


def load_churn(path, cat_col: str = "state") -> CategoricalTask:
    df = read_arff(path)
    return _task(df[cat_col], df["class"].astype(int).to_numpy(), f"churn / {cat_col}")


def load_churn_composite_id(path) -> CategoricalTask:
    """``churn`` with a near-unique pseudo-ID: state x area_code x account_length."""
    df = read_arff(path)
    composite = df["state"] + "_" + df["area_code"] + "_" + df["account_length"]
    return _task(composite, df["class"].astype(int).to_numpy(),
                 "churn / state x area_code x account_length")


def load_airlines(path, cat_col: str = "Flight") -> CategoricalTask:
    df = read_arff(path)
    return _task(df[cat_col], df["Delay"].astype(int).to_numpy(), f"airlines / {cat_col}")


def load_click_prediction(path, cat_col: str = "user_id") -> CategoricalTask:
    df = read_arff(path)
    return _task(df[cat_col], df["click"].astype(int).to_numpy(),
                 f"click_prediction_small / {cat_col}")


#: Default file names of the OpenML ARFF exports (IDs 42125, 40701, 1169, 1220).
DATA_FILES = {
    "employee_salaries": "employee_salaries.arff",
    "churn": "churn.arff",
    "airlines": "airlines.arff",
    "click": "click_prediction_small.arff",
}

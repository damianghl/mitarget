"""
Real-data tasks for the paired benchmark: one high-cardinality feature and a
target per task, integer-coded.

Each group loader yields :class:`mitarget.datasets.CategoricalTask` objects.
Raw files are not shipped; pass the folder that holds them (see
``experiments/README.md`` for the layout and sources).
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from mitarget import datasets
from mitarget.datasets import CategoricalTask


def _task(label: str, xs, y) -> CategoricalTask:
    x, uniques = pd.factorize(pd.Series(xs).astype(str))
    y = np.asarray(y, dtype=int)
    return CategoricalTask(x=x.astype(np.int64), y=y, kx=len(uniques), ky=int(y.max()) + 1, label=label)


def openml(root: Path) -> Iterator[CategoricalTask]:
    """The four OpenML ARFF exports of the bakeoff (IDs 42125, 40701, 1169 as a 5% sample, 1220)."""
    f = datasets.DATA_FILES
    yield datasets.load_employee_salaries(root / f["employee_salaries"])
    yield datasets.load_churn(root / f["churn"])
    yield datasets.load_churn_composite_id(root / f["churn"])
    yield datasets.load_airlines(root / f["airlines"], cat_col="AirportFrom")
    yield datasets.load_airlines(root / f["airlines"], cat_col="Flight")
    yield datasets.load_click_prediction(root / f["click"], cat_col="advertiser_id")
    yield datasets.load_click_prediction(root / f["click"], cat_col="user_id")


def movielens(path: Path) -> Iterator[CategoricalTask]:
    """MovieLens 1M, ``y = rating >= 4`` (tab-separated ratings file)."""
    d = pd.read_csv(path, sep="\t", usecols=["user_id", "movie_id", "rating"])
    y = (d.rating >= 4).astype(int).values
    yield _task("movielens/movie_id", d.movie_id, y)
    yield _task("movielens/user_id", d.user_id, y)


def amazon(root: Path) -> Iterator[CategoricalTask]:
    """Kaggle Amazon Employee Access, ``y = ACTION``."""
    d = pd.read_csv(root / "train.csv")
    for c in ("RESOURCE", "MGR_ID", "ROLE_FAMILY_DESC"):
        yield _task(f"amazon/{c}", d[c], d.ACTION.values)


def book_crossing(root: Path) -> Iterator[CategoricalTask]:
    """
    Book-Crossing. Main target ``rating >= 8`` among explicit ratings;
    alternative target explicit vs implicit (``rating > 0``) on all rows.
    """
    r = pd.read_csv(root / "BX-Book-Ratings.csv", sep=";", encoding="latin-1")
    r.columns = ["user", "isbn", "rating"]
    r["user"] = r.user.astype(str)
    b = pd.read_csv(root / "BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip", dtype=str)
    b.columns = ["isbn", "title", "author", "year", "publisher", "a", "b", "c"]
    u = pd.read_csv(root / "BX-Users.csv", sep=";", encoding="latin-1", dtype=str)
    u.columns = ["user", "loc", "age"]
    e = r[r.rating > 0].merge(b[["isbn", "author", "publisher"]], on="isbn", how="left")
    e = e.merge(u, on="user", how="left")
    y = (e.rating >= 8).astype(int).values
    for c in ("isbn", "user", "author", "publisher", "loc"):
        m = e[c].notna().values
        yield _task(f"bx/{c}", e[c][m], y[m])
    yi = (r.rating > 0).astype(int).values
    yield _task("bx_explicit/isbn", r.isbn, yi)
    yield _task("bx_explicit/user", r.user, yi)


def criteo(root: Path, features=(1, 20, 22)) -> Iterator[CategoricalTask]:
    """Raw Criteo 1TB click-log parts (parquet), ``y = label``; missing values form one level."""
    import pyarrow.parquet as pq

    cols = ["label"] + [f"categorical_feature_{i}" for i in features]
    files = sorted(glob.glob(str(root / "*.parquet")))
    d = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    y = d.label.values
    for i in features:
        yield _task(f"criteo/C{i}", d[f"categorical_feature_{i}"].fillna("NA"), y)


def avazu(path: Path) -> Iterator[CategoricalTask]:
    """Raw Avazu train rows, ``y = click``; "user" is device_ip + device_model."""
    d = pd.read_csv(path, dtype=str, usecols=["click", "device_ip", "device_model"])
    y = d.click.astype(int).values
    yield _task("avazu/device_ip", d.device_ip, y)
    yield _task("avazu/user", d.device_ip + "|" + d.device_model, y)


#: group -> (loader, default path relative to the data root)
GROUPS = {
    "openml": (openml, "arff"),
    "movielens": (movielens, "movielens/ml1m_ratings.tsv"),
    "amazon": (amazon, "amazon_employee_access"),
    "bx": (book_crossing, "book_crossing"),
    "criteo": (criteo, "criteo_1tb_day0"),
    "avazu": (avazu, "avazu/avazu_train_4M.csv.gz"),
}

import numpy as np
import pandas as pd

from mitarget.benchmark import evaluate_cv, evaluate_synthetic, standard_methods, summarize
from mitarget.datasets import _split_data_line, read_arff
from mitarget.synthetic import oracle_three_point_encoding, scenario_parity

ARFF = """% comment
@RELATION test
@ATTRIBUTE name STRING
@attribute 'kind' {'a','b'}
@attribute value numeric

@DATA
'Smith, J.',a,1.5
"x",'b',?
'it\\'s',a,3
"""


def test_split_data_line():
    assert _split_data_line("'a, b',c,?") == ["a, b", "c", None]
    assert _split_data_line('"1",2,"x"') == ["1", "2", "x"]


def test_read_arff(tmp_path):
    p = tmp_path / "t.arff"
    p.write_text(ARFF)
    df = read_arff(p)
    assert list(df.columns) == ["name", "kind", "value"]
    assert df.iloc[0].tolist() == ["Smith, J.", "a", "1.5"]
    assert pd.isna(df.iloc[1, 2])
    assert df.iloc[2, 0] == "it's"


def test_evaluate_synthetic_pairs_methods():
    gt = scenario_parity(kx=150, seed=0)
    methods = standard_methods(mixture_ks=(2,), crp=False, sklearn=False)
    df = evaluate_synthetic(gt, 300, 2, methods,
                            oracles={"oracle": lambda n: oracle_three_point_encoding(n, 0.1)})
    assert set(df.method) == set(methods) | {"oracle"}
    assert (df.groupby("method").size() == 2).all()
    s = summarize(df)
    assert s.loc["oracle", "excess_CE_mean"] <= s.loc["Plug-in", "excess_CE_mean"]


def test_evaluate_cv_binary_and_multiclass():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 40, size=400)
    for ky in (2, 3):
        y = rng.integers(0, ky, size=400)
        df = evaluate_cv(x, y, standard_methods(mixture_ks=(), crp=False), n_splits=2)
        assert np.isfinite(df.logloss).all()
        assert ("auc" in df.columns) == (ky == 2)

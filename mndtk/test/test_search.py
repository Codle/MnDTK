from mndtk.measures import search as search_utils
import numpy as np

rels = np.array([3, 2, 3, 0, 1, 2])
ranks = np.array([1, 2, 3, 4, 5, 6])


def test_cg():
    assert search_utils.cg(rels) == 11


def test_dcg():
    assert f"{search_utils.dcg(rels, ranks):.3f}" == "6.861"

def test_ndcg():
    assert f"{search_utils.ndcg(rels, ranks):.3f}" == "0.961"

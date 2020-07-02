from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


def mrr(ranks):
    return np.mean(1.0 / ranks)


def cg(rels: np.array) -> np.float:
    return sum(rels)


def dcg(rels: np.array, ranks: np.array) -> np.float:
    return np.sum(rels / np.log2(ranks+1))


def pairwise_vec_mrr(src_vec: np.ndarray,
                     tgt_vec: np.ndarray,
                     distance_metric: str) -> Tuple[np.array, np.array]:
    """ 两个向量矩阵首先计算距离，根据距离排序并计算MRR得分，常用于语义匹配模型中
    """
    distances = cdist(src_vec, tgt_vec, metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    ranks = np.sum(distances <= correct_elements, axis=-1)
    return mrr(ranks)

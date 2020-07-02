from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


def mrr(ranks):
    return np.mean(1.0 / ranks)


def pairwise_vec_mrr(src_vec: np.ndarray,
                     tgt_vec: np.ndarray,
                     distance_metric: str) -> Tuple[np.array, np.array]:

    distances = cdist(src_vec, tgt_vec, metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    ranks = np.sum(distances <= correct_elements, axis=-1)
    return mrr(ranks)

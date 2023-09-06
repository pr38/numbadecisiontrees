import numpy as np
from numba import njit
import numba as nb


@njit(
    nb.types.Array(nb.float64, 1, "C", False, aligned=True)(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
    ),
)
def get_class_sum(y, n_classes, sample_weights):
    counts = np.zeros(n_classes)
    n_rows = y.shape[0]

    for i in range(n_rows):
        y_i = y[i]
        y_i_int = int(y_i)

        counts[y_i_int] += sample_weights[i]

    return counts


@njit(
    nb.types.Tuple(
        (
            nb.types.Array(nb.float64, 2, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 2, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        )
    )(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.float64,
        nb.int64,
    ),
    # parallel=True,
)
def split_data(X, y, weights, threshold, feature):

    left_mask_split = X[:, feature] <= threshold
    right_mask_split = ~left_mask_split

    right_X = X[right_mask_split]
    right_y = y[right_mask_split]
    right_weights = weights[right_mask_split]

    left_X = X[left_mask_split]
    left_y = y[left_mask_split]
    left_weights = weights[left_mask_split]

    return left_X, left_y, left_weights, right_X, right_y, right_weights

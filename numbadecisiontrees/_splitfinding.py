import numpy as np
from numba import njit, prange
import numba as nb

from ._utility import get_class_sum
from ._criterion import (
    get_MAE_proxy_impurity,
    get_Entropy_proxy_impurity,
    get_Gini_proxy_impurity,
    get_MSE_proxy_impurity,
)


@njit(
    nb.types.Tuple((nb.float64, nb.float64))(
        nb.types.Array(nb.float64, 1, "A", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.float64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
    ),
)
def get_best_threshold_on_col(
    col,
    y,
    weights,
    y_totals,
    weights_total,
    n_rows,
    min_samples_leaf,
    crit_code,
    n_classes,
):

    best_proxy_impurity = -np.inf
    best_threshold = -1.0

    arg_sort = np.argsort(col)
    col = col[arg_sort]
    y_sort = y[arg_sort]
    weights_sort = weights[arg_sort]

    last_value = col[-1]

    weighted_left_size = 0.0

    if crit_code in [2, 3]:  # if crit is Gini or Entropy
        left_class_counts = np.zeros(n_classes)
    else:  # if crit is MSE or MAE
        left_sum = 0.0

    for row_index in range(n_rows - 1):
        value_y = y_sort[row_index]
        value_col = col[row_index]
        weight_y = weights_sort[row_index]

        weighted_left_size += weight_y

        if crit_code in [2, 3]:
            left_class_counts[int(value_y)] += weight_y
        else:
            left_sum += value_y

        if (value_col != col[row_index + 1]) and (value_col != last_value):
            left_size = row_index + 1
            right_size = n_rows - left_size

            if (left_size >= min_samples_leaf) and (right_size >= min_samples_leaf):
                if crit_code in [2, 3]:
                    right_class_counts = y_totals - left_class_counts
                else:
                    right_sum = y_totals[0] - left_sum

                weighted_right_size = weights_total - weighted_left_size

                if crit_code == 1:
                    proxy_impurity = get_MAE_proxy_impurity(
                        left_sum, right_sum, weighted_left_size, weighted_right_size
                    )
                elif crit_code == 2:
                    proxy_impurity = get_Entropy_proxy_impurity(
                        left_class_counts,
                        right_class_counts,
                        n_classes,
                        weighted_left_size,
                        weighted_right_size,
                    )
                elif crit_code == 3:
                    proxy_impurity = get_Gini_proxy_impurity(
                        left_class_counts,
                        right_class_counts,
                        n_classes,
                        weighted_left_size,
                        weighted_right_size,
                    )
                else:
                    proxy_impurity = get_MSE_proxy_impurity(
                        left_sum, right_sum, weighted_left_size, weighted_right_size
                    )

                if proxy_impurity > best_proxy_impurity:
                    best_proxy_impurity = proxy_impurity
                    best_threshold = (value_col + col[row_index + 1]) / 2

    return best_threshold, best_proxy_impurity


@njit(
    nb.types.Tuple((nb.int64, nb.float64))(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
    ),
    parallel=True,
)
def get_best_threshold_on_data(
    X, y, weights, min_samples_leaf, crit_code, n_classes, max_features, random_state
):
    if random_state >= 0:
        np.random.seed(random_state)

    n_rows, n_cols = X.shape

    if max_features < 0:  # if max_features is not being used, examine all cols
        cols_to_examine = np.arange(n_cols)
    else:
        cols_to_examine = np.random.choice(n_cols, max_features, replace=False)

    if crit_code in [2, 3]:
        y_sums = get_class_sum(y, n_classes, weights)
    else:
        y_sums = np.array(
            [(y * weights).sum()]
        )  # rolling in weights for regression based criterion/split-finding

    weights_total = weights.sum()

    best_col_proxy_impurites = np.full(
        shape=n_cols, fill_value=-np.inf, dtype=np.float64
    )
    best_col_thresholds = np.zeros(n_cols)

    for col_index in prange(n_cols):
        col = X[:, col_index]
        if col_index in cols_to_examine:
            best_threshold, best_proxy_impurity = get_best_threshold_on_col(
                col,
                y,
                weights,
                y_sums,
                weights_total,
                n_rows,
                min_samples_leaf,
                crit_code,
                n_classes,
            )

            best_col_proxy_impurites[col_index] = best_proxy_impurity
            best_col_thresholds[col_index] = best_threshold

    best_cols_bool = best_col_proxy_impurites == max(best_col_proxy_impurites)
    best_col = int(
        np.random.choice(np.flatnonzero(best_cols_bool))
    )  # if 2 or more col share the best approximation, pick one at random. This is done to immitate sklean behaviour(see the splitter classes in the tree modual).

    best_threshold = best_col_thresholds[best_col]

    return best_col, best_threshold

import numpy as np
from numba import njit
import numba as nb

from ._utility import get_class_sum

### For getting true scores, ran after best split is found, lower the better. Primarily used for early stopping and other auxiliary tasks.


@njit(
    nb.float64(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
    )
)
def get_Entropy_true_score(y, n_classes, sample_weights):
    weight_n_samples = sample_weights.sum()

    class_sums = get_class_sum(y, n_classes, sample_weights)

    score = 0.0

    for class_sum in class_sums:
        class_percentage = class_sum / weight_n_samples
        score -= 0 if class_sum == 0 else (class_percentage * np.log2(class_percentage))

    return score


@njit(
    nb.float64(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
    )
)
def get_Gini_true_score(y, n_classes, sample_weights):
    weight_n_samples = sample_weights.sum()

    class_sums = get_class_sum(y, n_classes, sample_weights)
    prob_per_class_sqr = np.square(class_sums / weight_n_samples)

    return 1 - prob_per_class_sqr.sum()


@njit(
    nb.float64(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
    )
)
def get_MSE_true_score(y):
    mean = y.mean()

    rss = 0.0

    for y_i in y:
        diff = y_i - mean
        rss += diff * diff

    return rss / len(y)


@njit(
    nb.float64(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
    )
)
def get_MAE_true_score(y):
    mean = y.mean()

    rss = 0.0

    for y_i in y:
        diff = y_i - mean
        rss += np.absolute(rss + diff)

    return rss / len(y)


###'inverse approximations'; ran durring split search, higher the value the better


@njit(
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
    inline="always",
)
def get_MSE_proxy_impurity(left_sum, right_sum, weighted_n_left, weighted_n_right):
    return ((left_sum * left_sum) / weighted_n_left) + (
        (right_sum * right_sum) / weighted_n_right
    )


@njit(
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
    inline="always",
)
def get_MAE_proxy_impurity(left_sum, right_sum, weighted_n_left, weighted_n_right):
    return (left_sum * weighted_n_left) + (right_sum * weighted_n_right)


@njit(
    nb.float64(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.float64,
        nb.float64,
    ),
    inline="always",
)
def get_Entropy_proxy_impurity(
    left_class_counts,
    right_class_counts,
    n_classes,
    weighted_n_left,
    weighted_n_right,
):
    entropy_left = 0.0
    entropy_right = 0.0

    for c in range(n_classes):
        count_k = left_class_counts[c]
        if count_k > 0.0:
            count_k /= weighted_n_left
            entropy_left -= count_k * np.log(count_k)

        count_k = right_class_counts[c]
        if count_k > 0.0:
            count_k /= weighted_n_right
            entropy_right -= count_k * np.log(count_k)

    return -weighted_n_right * entropy_right - weighted_n_left * entropy_left


@njit(
    nb.float64(
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.float64,
        nb.float64,
    ),
    inline="always",
)
def get_Gini_proxy_impurity(
    left_class_counts,
    right_class_counts,
    n_classes,
    weighted_n_left,
    weighted_n_right,
):
    sq_count_left = 0.0
    sq_count_right = 0.0

    for c in range(n_classes):
        count_k = left_class_counts[c]
        sq_count_left += count_k * count_k

        count_k = right_class_counts[c]
        sq_count_right += count_k * count_k

    gini_left = 1.0 - sq_count_left / (weighted_n_left * weighted_n_left)

    gini_right = 1.0 - sq_count_right / (weighted_n_right * weighted_n_right)

    return -weighted_n_left * gini_left - weighted_n_right * gini_right

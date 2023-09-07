import numpy as np
from numba import njit
import numba as nb

from ._utility import get_class_sum, split_data
from ._criterion import (
    get_MSE_true_score,
    get_Entropy_true_score,
    get_Gini_true_score,
)

from ._splitfinding import get_best_threshold_on_data


@njit(
    nb.types.Tuple(
        (
            nb.boolean,
            nb.boolean,
            nb.types.Array(nb.float64, 1, "A", False, aligned=True),
            nb.int64,
            nb.float64,
            nb.float64,
            nb.int64,
            nb.int64,
            nb.int64,
            nb.float64,
            nb.types.optional(nb.types.Array(nb.float64, 2, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.float64,
            nb.types.optional(nb.types.Array(nb.float64, 2, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.float64,
        )
    )(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.boolean,
        nb.int64,
        nb.float64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int64,
        nb.int64,
    ),
)
def process_node(
    X_examined,
    y_examined,
    weights_examined,
    parent_index,
    is_left,
    depth,
    impurity,
    max_depth_limit,
    min_samples_leaf,
    min_samples_split,
    crit_code,
    n_classes,
    min_impurity_decrease,
    weighted_n_samples,
    min_weight_leaf,
    max_features,
    random_state,
):

    n_node_sample = len(y_examined)
    weighted_n_node_sample = weights_examined.sum()

    left_child_score = 0.0
    right_child_score = 0.0

    left_X = None
    left_y = None
    right_X = None
    right_y = None

    exceeds_depth = False if max_depth_limit <= 0 else depth >= max_depth_limit

    is_leaf = (
        exceeds_depth
        or n_node_sample < min_samples_split
        or n_node_sample < 2 * min_samples_leaf
        or weighted_n_node_sample < 2 * min_weight_leaf
    )

    if crit_code in [1, 2]:  # classification
        value = get_class_sum(y_examined, n_classes, weights_examined)
    else:  # regression
        value = np.array([np.average(y_examined,weights=weights_examined)])

    is_leaf = is_leaf or impurity <= np.finfo(np.float64).eps

    if not is_leaf:
        feature, threshold = get_best_threshold_on_data(
            X_examined,
            y_examined,
            weights_examined,
            min_samples_leaf,
            crit_code,
            n_classes,
            max_features,
            random_state,
        )

        if threshold == -1.0:
            is_leaf = True

    if not is_leaf:
        left_X, left_y, left_weights, right_X, right_y, right_weights = split_data(
            X_examined, y_examined, weights_examined, threshold, feature
        )

        if crit_code == 1:
            left_child_score = get_Entropy_true_score(left_y, n_classes, left_weights)
            right_child_score = get_Entropy_true_score(
                right_y, n_classes, right_weights
            )
        elif crit_code == 2:
            left_child_score = get_Gini_true_score(left_y, n_classes, left_weights)
            right_child_score = get_Gini_true_score(right_y, n_classes, right_weights)
        else:  # crit_code == 0
            left_child_score = get_MSE_true_score(left_y,weights_examined)
            right_child_score = get_MSE_true_score(right_y,weights_examined)

        weighted_n_node_right_y = right_weights.sum()
        weighted_n_node_left_y = left_weights.sum()

        improvement = (
            weighted_n_node_sample
            / weighted_n_samples
            * (
                impurity
                - (
                    (weighted_n_node_right_y / weighted_n_node_sample)
                    * right_child_score
                    + (weighted_n_node_left_y / weighted_n_node_sample)
                    * left_child_score
                )
            )
        )

        if (improvement + np.finfo(np.float64).eps) < min_impurity_decrease:
            is_leaf = True

    if is_leaf:
        threshold = -2
        feature = -2

        left_X = None
        left_y = None
        left_weights = None
        right_X = None
        right_y = None
        right_weights = None

    return (
        is_leaf,
        is_left,
        value,
        feature,
        threshold,
        impurity,
        depth,
        parent_index,
        n_node_sample,
        weighted_n_node_sample,
        left_X,
        left_y,
        left_weights,
        left_child_score,
        right_X,
        right_y,
        right_weights,
        right_child_score,
    )


@njit(
    nb.int64(
        nb.types.List(
            nb.types.Tuple(
                (
                    nb.int64,
                    nb.boolean,
                    nb.types.Array(nb.float64, 2, "C", False, aligned=True),
                    nb.types.Array(nb.float64, 1, "C", False, aligned=True),
                    nb.types.Array(nb.float64, 1, "C", False, aligned=True),
                    nb.int64,
                    nb.float64,
                )
            )
        )
    ),
    inline="always",
)
def get_index_with_highest_score(to_build_stack):
    best_score = np.inf
    best_index = -1

    for index, to_build_node in enumerate(to_build_stack):
        score = to_build_node[6]
        if score < best_score:
            best_score = score
            best_index = index

    return best_index


@njit(
    nb.types.Tuple(
        (
            nb.types.List(nb.int64),
            nb.types.List(nb.int64),
            nb.types.List(nb.int64),
            nb.types.List(nb.float64),
            nb.types.List(nb.float64),
            nb.types.List(nb.int64),
            nb.types.List(nb.float64),
            nb.types.List(nb.types.Array(nb.float64, 1, "A", False, aligned=True)),
            nb.int64,
        )
    )(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64,
        nb.float64,
        nb.int64,
        nb.int64,
        nb.int64,
    ),
)
def build_tree(
    X,
    y,
    weights,
    min_samples_leaf,
    max_depth_limit,
    min_samples_split,
    crit_code,
    n_classes,
    min_impurity_decrease,
    min_weight_leaf,
    max_features,
    max_leaf_nodes,
    random_state,
):
    left_childs = []
    right_childs = []
    features = []
    thresholds = []
    impurities = []
    n_node_samples = []
    weighted_n_node_samples = []
    values = []

    if crit_code == 1:
        root_impurity = get_Entropy_true_score(y, n_classes, weights)
    elif crit_code == 2:
        root_impurity = get_Gini_true_score(y, n_classes, weights)
    else:  # crit_code == 0
        root_impurity = get_MSE_true_score(y)

    to_build_stack = nb.typed.List()

    to_build_stack.append(
        (-1, True, X, y, weights, 0, root_impurity)
    )  # parentid,is_left,X,y,weights,depth,score

    current_index = 0
    depth = 0
    max_depth_build = 0
    weighted_n_samples = weights.sum()
    leaf_count = 0

    while len(to_build_stack) > 0:

        # if max_leaf_nodes is being used, get index of best nodes first
        if max_leaf_nodes < 0:
            index_to_pop = -1
        else:
            index_to_pop = get_index_with_highest_score(to_build_stack)

        # get from stack
        (
            parent_index,
            is_left,
            X_examined,
            y_examined,
            weights_examined,
            depth,
            impurity,
        ) = to_build_stack.pop(index_to_pop)

        # process node
        (
            is_leaf,
            is_left,
            value,
            feature,
            threshold,
            impurity,
            depth,
            parent_index,
            n_node_sample,
            weighted_n_node_sample,
            left_X,
            left_y,
            left_weights,
            left_child_impurity,
            right_X,
            right_y,
            right_weights,
            right_child_impurity,
        ) = process_node(
            X_examined,
            y_examined,
            weights_examined,
            parent_index,
            is_left,
            depth,
            impurity,
            max_depth_limit,
            min_samples_leaf,
            min_samples_split,
            crit_code,
            n_classes,
            min_impurity_decrease,
            weighted_n_samples,
            min_weight_leaf,
            max_features,
            random_state,
        )

        # add to tree
        left_childs.append(-1)
        right_childs.append(-1)
        features.append(feature)
        values.append(value)
        thresholds.append(threshold)
        impurities.append(impurity)
        n_node_samples.append(n_node_sample)
        weighted_n_node_samples.append(weighted_n_node_sample)

        # add child's coordinates's parent's position; so we know where the children of the parent is located.
        if parent_index != -1:  # skip the root/first node
            if is_left:
                left_childs[parent_index] = current_index
            else:
                right_childs[parent_index] = current_index

        # add to building stack
        if not is_leaf:
            to_build_stack.append(
                (
                    current_index,
                    False,
                    right_X,
                    right_y,
                    right_weights,
                    depth + 1,
                    right_child_impurity,
                )
            )
            to_build_stack.append(
                (
                    current_index,
                    True,
                    left_X,
                    left_y,
                    left_weights,
                    depth + 1,
                    left_child_impurity,
                )
            )
        else:
            leaf_count += 1

        # book keeping, for future nodes
        current_index = current_index + 1

        if depth > max_depth_build and not is_leaf:
            max_depth_build = depth

        # if max_leaf_nodes is being used, stop tree building if max_leaf_nodes limit is hit
        if max_leaf_nodes > 0:
            if leaf_count >= max_leaf_nodes:
                to_build_stack.clear()

    return (
        left_childs,
        right_childs,
        features,
        thresholds,
        impurities,
        n_node_samples,
        weighted_n_node_samples,
        values,
        max_depth_build,
    )

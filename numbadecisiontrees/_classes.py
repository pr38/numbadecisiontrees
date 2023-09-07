from ._treebuilding import build_tree
import numpy as np

from sklearn.tree._tree import Tree, _build_pruned_tree_ccp

from sklearn.utils import (
    check_random_state,
    compute_sample_weight,
)

from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from abc import ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier

from numbers import Integral
from math import ceil

from typing import Literal, Union, Optional


def build_sk_Tree_class(
    features,
    thresholds,
    childrenleft,
    childrenright,
    values,
    n_node_samples,
    weighted_n_node_samples,
    impurity,
    max_depth_build,
    n_classes,
    n_features,
):
    dt = {
        "names": [
            "left_child",
            "right_child",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
        ],
        "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8"],
    }
    nodes = np.zeros(len(childrenleft), dtype=dt)
    nodes["left_child"] = childrenleft
    nodes["right_child"] = childrenright
    nodes["feature"] = features
    nodes["threshold"] = thresholds
    nodes["impurity"] = impurity
    nodes["n_node_samples"] = n_node_samples
    nodes["weighted_n_node_samples"] = weighted_n_node_samples

    state = {
        "max_depth": max_depth_build,
        "node_count": len(features),
        "nodes": nodes,
        "values": np.array(values)[:, np.newaxis, :],
    }

    tree = Tree(n_features, np.array([n_classes]), 1)

    tree.__setstate__(state)

    return tree


def prune_sk_Tree_class(tree, n_features, n_classes, ccp_alpha):
    pruned_tree = Tree(n_features, np.array([n_classes]), 1)
    _build_pruned_tree_ccp(pruned_tree, tree, ccp_alpha)
    return pruned_tree


CRIT_MAP = {
    "squared_error": 0,
    "absolute_error": 1,
    "entropy": 2,
    "gini": 3,
    "log_loss": 2,
}


class BaseNBDecisionTree(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight,
        ccp_alpha,
    ):

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def check_fit_data_(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if not (isinstance(X, np.ndarray) and isinstance(X, np.ndarray)):
            raise ValueError("X and y be dense arrays")

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Estimator has no missing data support")

        if not (np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.number)):
            raise ValueError("Estimator only works with numeric data")

    def check_predict_data_(self, X):
        if not (isinstance(X, np.ndarray)):
            raise ValueError("X must be a dense array")

        if np.isnan(X).any():
            raise ValueError("Estimator has no missing data support")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Estimator only works with numeric data")

        if self.n_features != X.shape[1]:
            raise ValueError("X must have the same number of features as fitting data")

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
    ):
        if check_input:
            self.check_fit_data_(X, y)

        is_classification = is_classifier(self)

        # weights
        if self.class_weight is not None:
            expanded_class_weight = compute_sample_weight(self.class_weight, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        else:
            sample_weight = np.ones(len(y)).astype(np.float64)

        if self.class_weight is not None:
            sample_weight = sample_weight * expanded_class_weight

        if is_classification:
            self.classes_, y = np.unique(y, return_inverse=True)

        # fix types/order
        X = X.copy(order="C").astype(np.float64)
        y = y.copy(order="C").astype(np.float64)

        # random_state
        check_random_state(self.random_state)
        if self.random_state == None:
            random_state = -1
        else:
            random_state = self.random_state

        crit_code = CRIT_MAP[self.criterion]

        n_samples, n_features = X.shape

        self.n_features = n_features

        # deducting parameters
        # primary taken from sklearn's BaseDecisionTree
        max_depth = -1 if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if self.max_features is None:
            max_features = -1
        elif self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:  # float
            max_features = max(1, int(self.max_features * n_features))

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        min_weight_leaf = (
            0.0
            if self.min_weight_fraction_leaf is None
            else float(self.min_weight_fraction_leaf)
        )

        min_impurity_decrease = (
            0.0
            if self.min_impurity_decrease is None
            else float(self.min_impurity_decrease)
        )

        ccp_alpha = 0.0 if self.ccp_alpha is None else float(self.ccp_alpha)

        n_classes = len(np.unique(y)) if is_classification else 1
        self.n_classes = n_classes

        # building tree, entry point to actually building the tree
        (
            left_childs,
            right_childs,
            features,
            thresholds,
            impurities,
            n_node_samples,
            weighted_n_node_samples,
            values,
            max_depth_build,
        ) = build_tree(
            X,
            y,
            weights=sample_weight,
            min_samples_leaf=min_samples_leaf,
            max_depth_limit=max_depth,
            min_samples_split=min_samples_split,
            crit_code=crit_code,
            n_classes=n_classes,
            min_impurity_decrease=min_impurity_decrease,
            min_weight_leaf=min_weight_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )

        tree = build_sk_Tree_class(
            features,
            thresholds,
            left_childs,
            right_childs,
            values,
            n_node_samples,
            weighted_n_node_samples,
            impurities,
            max_depth_build,
            n_classes,
            n_features,
        )

        tree = prune_sk_Tree_class(
            tree=tree, n_features=n_features, n_classes=n_classes, ccp_alpha=ccp_alpha
        )

        self.tree_ = tree

        self._is_fitted = True

        return self

    # primary taken from sklearn's BaseDecisionTree
    def predict(self, X, check_input=True):
        check_is_fitted(self)

        if check_input:
            self.check_predict_data_(X)

        X = X.astype(np.float32)

        proba = self.tree_.predict(X)

        if is_classifier(self):
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)
        else:
            return proba[:, 0]

    def apply(self, X, check_input=True):
        check_is_fitted(self)

        if check_input:
            self.check_predict_data_(X)

        X = X.astype(np.float32)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        check_is_fitted(self)

        if check_input:
            self.check_predict_data_(X)

        X = X.astype(np.float32)
        return self.tree_.decision_path(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        return self.tree_.compute_feature_importances()


class NBDecisionTreeClassifier(ClassifierMixin, BaseNBDecisionTree):
    def __init__(
        self,
        *,
        criterion: Union[
            Literal["gini"], Literal["entropy"], Literal["log_loss"]
        ] = "gini",
        splitter: Literal["best"] = "best",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight=None,
        ccp_alpha: float = 0.0,
    ):

        if isinstance(criterion, str):
            if criterion not in ["gini", "entropy", "log_loss"]:
                raise ValueError(
                    """Only "gini", "entropy", "log_loss" are supported """
                )
        else:
            raise ValueError("criterion not properly set")

        if splitter != "best":
            raise ValueError("Only best splitter is implemented")

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def predict_proba(self, X, check_input=True):
        if check_input:
            self.check_predict_data_(X)

        X = X.astype(np.float32)
        proba = self.tree_.predict(X)

        # taken from sklearn
        proba = proba[:, :]
        proba = self.tree_.predict(X)
        proba = proba[:, : self.n_classes_]
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def predict_log_proba(self, X, check_input=True):
        proba = self.predict_proba(X, check_input=check_input)
        return np.log(proba)


class NBDecisionTreeRegressor(RegressorMixin, BaseNBDecisionTree):
    def __init__(
        self,
        *,
        criterion: Union[
            Literal["squared_error"], Literal["absolute_error"]
        ] = "squared_error",
        splitter: Literal["best"] = "best",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
    ):

        if isinstance(criterion, str):
            if criterion not in ["squared_error", "absolute_error"]:
                raise ValueError(
                    """Only "squared_error" and "absolute_error" are supported """
                )
        else:
            raise ValueError("criterion not properly set")

        if splitter != "best":
            raise ValueError("Only best splitter is implemented")

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            class_weight=None,
        )

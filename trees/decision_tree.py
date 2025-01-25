from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union
from statistics import fmean


def arg_sort(seq):
    """Returns the indices of values in sorted order"""
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

@dataclass(slots=True)
class Data:
    """
    A simple data container -- analogous to a very light dataframe.
    """
    column_names: List[str]
    X: List[List[float]] | List[float] # Rows of data, or Gains for XG Boost
    y: List[float] = None # can be null for predictions, Hessians for XG Boost
    gains: List[float] = None # Used by XG Boost, but not normal decision tree
    hessians: List[float] = None # Used by XG Boost, but not normal decision tree

@dataclass(slots=True)
class TreeConfig:
    """
    Hyperparameters for the Decision Tree, as well as any configuration between nodes in a tree.

    Supports logic for decision tree and XGBoost.
    """
    min_samples_leaf: int = 5
    max_depth: int = 6
    idxs: List[int] = None
    parent: Union["DecisionTree", None] = None
    initial_best_score: float = float('inf')  # For XG Boost, this will be 0

    min_child_weight: float = 1.0   # XG Boost config
    reg_lambda: float = 0.0         # XG Boost config -> set to 1.0 for XG Boost
    gamma: float = 0.0              # XG Boost config
    colsample_bynode: float = 1.0   # XG Boost config

    @property
    def is_decision_tree(self) -> bool:
        """
        Differentiates a decision tree from a XGBoost tree.  They have different initial best scores.
        """
        return self.initial_best_score == float('inf')

    def is_skippable(self, n_left: float, x_i: float, x_i_next: float) -> bool:
        """
        Determines if the current state is skippable.  Scenarios for skipping:
            1. This value and the next are the same
            2. Number of values in left node is below the minimum leaf samples for a Decision Tree
            3. Number of values in left node is less than minimum child weight for XGBoost.
        """
        if x_i == x_i_next:
            return True

        if self.is_decision_tree:
            return n_left < self.min_samples_leaf

        return n_left < self.min_child_weight

    def value_fxn(self, data:Data, idxs: List[int]) -> float:
        """
        A proxy that routes to the right value function (Decision Tree vs XGBoost)
        """
        if self.is_decision_tree:
            return self.decision_tree_value_fxn(data, idxs)

        return self.xgboost_value_fxn(data, idxs)

    def decision_tree_value_fxn(self, data:Data, idxs: List[int]) -> float:
        """Decision Tree Value Function is simply the mean of the Y values."""
        ys = [data.y[idx] for idx in idxs]
        return fmean(ys)  # node's prediction value

    def xgboost_value_fxn(self, data:Data, idxs: List[int]) -> float:
        """XGBoost Tree Value Function is the first deriv / second deriv: approximating how curved the function is"""
        gains = [data.gains[idx] for idx in idxs]
        hessians = [data.hessians[idx] for idx in idxs]

        return -sum(gains) / (sum(hessians) + self.reg_lambda)


class DecisionTree:
    """
    Decision Tree largely from: https://randomrealizations.com/posts/decision-tree-from-scratch/

    Changes:
    * Removed Numpy/Pandas
    * Added abstraction layer to support XGBoost as well
    * Improved `repr` support

    """

    __slots__ = ["config", "split_feature_idx", "is_root", "data", "row_count",
                 "column_count", "value", "best_score_so_far", "left", "right", "threshold"]


    @classmethod
    def from_values(cls, data: Data, min_samples_leaf = 5, max_depth = 6, idxs:List[int] = None, parent = None):
        """Makes a DecisionTree without the TreeConfig object"""
        return DecisionTree(data, TreeConfig(min_samples_leaf, max_depth, idxs, parent, float('inf'), 1.0, 0.0, 0.0, 1.0))

    @classmethod
    def from_xgb(cls, data: Data, params, max_depth = 6, idxs:List[int] = None, parent = None):
        config = TreeConfig(min_samples_leaf=1, max_depth=max_depth, idxs=idxs, parent=parent, initial_best_score=0.0,
                            min_child_weight=params.get('min_child_weight', 1.0), reg_lambda=params.get('reg_lambda', 1.0),
                            gamma=params.get('gamma', 0.0), colsample_bynode=params.get('colsample_bynode', 1.0))
        return DecisionTree(data, config)


    def __init__(self,data: Data, config: TreeConfig):
        """Initializes the Decision Tree"""
        assert config.max_depth >= 0, 'max_depth must be non-negative'
        assert config.min_samples_leaf > 0, 'min_samples_leaf must be positive'
        self.config = config

        self.split_feature_idx: int

        if config.idxs is None:
            if config.is_decision_tree:
                config.idxs = list(range(len(data.y)))
            else:
                config.idxs = list(range(len(data.gains)))
        #print(f"** IDXS:  {len(config.idxs)}")
        self.is_root = config.parent is None

        self.data = data
        self.row_count, self.column_count = len(config.idxs), len(data.X[0])

        # Only needed if linear regression for leaf node is added:
        # xs = [data.X[idx] for idx in config.idxs]
        self.value = config.value_fxn(data, config.idxs)

        self.best_score_so_far = config.initial_best_score  # initial loss before split finding
        if self.config.max_depth > 0:
            self._maybe_insert_child_nodes()

    def _maybe_insert_child_nodes(self):
        """Given the best split, divides the nodes according to the best split threshold"""
        for j in range(self.column_count):
            self._find_better_split(j)

        if self.is_leaf:  # do not insert children to a leaf node
            return

        less_than_threshold:List[int] = []
        greater_than_threshold:List[int] = []

        X = self.data.X
        config = self.config

        for idx in config.idxs:
            val = X[idx][self.split_feature_idx]

            if val <= self.threshold:
                if val != 0:
                    less_than_threshold.append(idx)
            elif val != 0:
                greater_than_threshold.append(idx)

        #print(f"** LEFT:{self.split_feature_idx} <= {self.threshold}")
        self.left = DecisionTree(self.data, TreeConfig(config.min_samples_leaf, config.max_depth - 1,
                                                       less_than_threshold, self, config.initial_best_score,
                                                       config.min_child_weight, config.reg_lambda,
                                                       config.gamma, config.colsample_bynode))
        #print(f"** RIGHT:{self.split_feature_idx} > {self.threshold}")
        self.right = DecisionTree(self.data, TreeConfig(config.min_samples_leaf, config.max_depth -1,
                                                       greater_than_threshold, self, config.initial_best_score,
                                                       config.min_child_weight, config.reg_lambda,
                                                       config.gamma, config.colsample_bynode))

    @property
    def is_leaf(self):
        """If the best score is the same as the initial score, this node is a leaf"""
        return self.best_score_so_far == self.config.initial_best_score

    def _find_better_split(self, feature_idx):
        """
        For a given feature, scans through the values to find the best split for that feature.

        If this split is better than the best found so far, this split replaces the best found.
        """
        config = self.config
        data = self.data
        x = [data.X[idx][feature_idx] for idx in config.idxs]
        y = [data.y[idx] for idx in config.idxs] if config.is_decision_tree \
            else  [data.gains[idx] for idx in config.idxs]
        hessians = [1 for idx in config.idxs] if config.is_decision_tree \
            else [data.hessians[idx] for idx in config.idxs]

        # for XGBoost, Y is gains, n is hessians
        sort_idx = arg_sort(x)
        sort_y = [y[idx] for idx in sort_idx]
        sort_x = [x[idx] for idx in sort_idx]
        sort_hess = [hessians[idx] for idx in sort_idx]

        sum_y, n = sum(y), sum(hessians)
        sum_y_right, n_right = sum_y, n
        sum_y_left, n_left = 0., 0

        for i in range(0, self.row_count - config.min_samples_leaf):
            y_i = sort_y[i]
            n_i = sort_hess[i]
            x_i, x_i_next = sort_x[i], sort_x[i + 1]
            sum_y_left += y_i
            sum_y_right -= y_i
            n_left += n_i   #For XGBoost, assumes Mean Squared Objective Hessian: 1
            n_right -= n_i

            if config.is_skippable(n_left, x_i, x_i_next):
                continue

            if not config.is_decision_tree and n_right < config.min_child_weight:
                break

            score = - sum_y_left ** 2 / (n_left + config.reg_lambda) \
                    - sum_y_right ** 2 / (n_right + config.reg_lambda)\
                    + sum_y ** 2 / (n + config.reg_lambda)

            if not config.is_decision_tree:
                score *= -0.5 # flip the signs
                score -= config.gamma / 2

            #print(f"[{self.data.column_names[feature_idx]}]{score} > {self.best_score_so_far}")
            if (config.is_decision_tree and score < self.best_score_so_far) or \
                    (not config.is_decision_tree and score > self.best_score_so_far):
                #print(f"^^^^SELECTED")
                self.best_score_so_far = score
                self.split_feature_idx = feature_idx
                self.threshold = (x_i + x_i_next) / 2


    def __repr__(self):
        """
        This implementation displays the tree in a manner very similar (with some extras) to the sklearn `extract_text`
        function.
        """
        orig_depth = self._get_orig_depth()
        indent_level = orig_depth - self.config.max_depth
        indent_node = "|---"
        indent_spacing = "|   "
        indenting = f"{indent_spacing*indent_level}"

        if self.is_leaf:
            s = f"{indenting}{indent_node} value: [{self.value:0.2f}]\t\t<{self.row_count}>"
        else:
            split_feature_name = self.data.column_names[self.split_feature_idx]
            s = f"{indenting}{indent_node} {split_feature_name} <= {self.threshold:0.2f} \t\t\t<{self.row_count}> <<{self.best_score_so_far}>>\n"
            s += f"{repr(self.left)}\n"
            s += f"{indenting}{indent_node} {split_feature_name} > {self.threshold:0.2f}\n"
            s += f"{repr(self.right)}"

        return s

    def _get_orig_depth(self) -> int:
        """
        Gets the allowed Tree Depth
        """
        if self.is_root:
            return self.config.max_depth
        else:
            return self.config.parent._get_orig_depth()

    def predict(self, data:Data) -> List[float]:
        """
        Returns a list of predictions for the input list of data
        """
        return [self._predict_row(row) for row in data.X]

    def _predict_row(self, row:List[float]) -> float:
        """
        Returns an individual prediction for a row of data
        """
        if self.is_leaf:
            return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold else self.right
        return child._predict_row(row)

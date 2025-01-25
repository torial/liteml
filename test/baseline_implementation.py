import math

import numpy as np


class SquaredErrorObjectiveOrig():
    def loss(self, y, pred): return np.mean((y - pred)**2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))


class TreeBooster:
    """ Original Decision Tree for XGBoost -- needs to be ported away."""
    # essentially: min_samples_leaf = 1
    def __init__(self, X, gains, hessians, params, max_depth, idxs=None, parent=None):
        self.params = params
        self.max_depth = max_depth
        assert self.max_depth >= 0, 'max_depth must be nonnegative'
        self.min_child_weight = params.get('min_child_weight', 1.0)
        self.reg_lambda = params.get('reg_lambda', 1.0)
        self.gamma = params.get('gamma', 0.0)
        self.colsample_bynode = params.get('colsample_bynode', 1.0)
        if idxs is None:
            idxs = np.arange(len(gains))
        #print(f"** IDXS:  {len(idxs)}")
        self.is_root = parent is None
        self.parent = parent

        self.X, self.gains, self.hessians, self.idxs = X, gains, hessians, idxs
        self.row_count, self.column_count = len(idxs), X.shape[1]

        self.value = -gains[idxs].sum() / (hessians[idxs].sum() + self.reg_lambda)  # Eq (5)
        self.best_score_so_far = 0.
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()

    def _maybe_insert_child_nodes(self):
        for i in range(self.column_count):
            self._find_better_split(i)
        if self.is_leaf:
            return
        x = self.X.values[self.idxs, self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        self.left = TreeBooster(self.X, self.gains, self.hessians, self.params,
                                self.max_depth - 1, self.idxs[left_idx], self)
        self.right = TreeBooster(self.X, self.gains, self.hessians, self.params,
                                 self.max_depth - 1, self.idxs[right_idx], self)

    @property
    def is_leaf(self):
        return self.best_score_so_far == 0.

    def _find_better_split(self, feature_idx):
        x = self.X.values[self.idxs, feature_idx]
        gains, hessians = self.gains[self.idxs], self.hessians[self.idxs]
        sort_idx = np.argsort(x)
        sort_gains, sort_hessians, sort_x = gains[sort_idx], hessians[sort_idx], x[sort_idx]
        sum_gains, sum_hessians = gains.sum(), hessians.sum()
        sum_gains_right, sum_hessians_right = sum_gains, sum_hessians
        sum_gains_left, sum_hessians_left = 0., 0.

        for i in range(0, self.row_count - 1):
            gain_i, hessian_i= sort_gains[i], sort_hessians[i]
            x_i, x_i_next = sort_x[i], sort_x[i + 1]
            sum_gains_left += gain_i
            sum_gains_right -= gain_i
            sum_hessians_left += hessian_i
            sum_hessians_right -= hessian_i

            if sum_hessians_left < self.min_child_weight or x_i == x_i_next:
                continue
            if sum_hessians_right < self.min_child_weight:
                break

            gain = 0.5 * ((sum_gains_left ** 2 / (sum_hessians_left + self.reg_lambda))
                          + (sum_gains_right ** 2 / (sum_hessians_right + self.reg_lambda))
                          - (sum_gains ** 2 / (sum_hessians + self.reg_lambda))
                          ) - self.gamma / 2  # Eq(7) in the xgboost paper
            #print(f"[{self.X.columns[feature_idx]}]{gain} > {self.best_score_so_far}")
            if gain > self.best_score_so_far:
                #print(f"^^^^SELECTED")
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2

    def predict(self, X):
        return np.array([self._predict_row(row) for i, row in X.iterrows()])

    def _predict_row(self, row):
        if self.is_leaf:
            return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold \
            else self.right
        return child._predict_row(row)

    def __repr__(self):
        """
        This implementation displays the tree in a manner very similar (with some extras) to the sklearn `extract_text`
        function.
        """
        orig_depth = self._get_orig_depth()
        indent_level = orig_depth - self.max_depth
        indent_node = "|---"
        indent_spacing = "|   "
        indenting = f"{indent_spacing*indent_level}"

        if self.is_leaf:
            s = f"{indenting}{indent_node} value: [{self.value:0.2f}]\t\t<{self.row_count}>"
        else:
            split_feature_name = list(self.X.columns)[self.split_feature_idx]
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
            return self.max_depth
        else:
            return self.parent._get_orig_depth()


class XGBoostModelOrig:
    '''XGBoost from Scratch
    '''

    def __init__(self, params, random_seed=None):
        self.params = params
        self.subsample = self.params.get('subsample', 1.0)
        self.learning_rate = self.params.get('learning_rate', 0.3)
        self.base_prediction = self.params.get('base_score', 0.5)
        self.max_depth = self.params.get('max_depth', 5)
        self.rng = np.random.default_rng(seed=random_seed)
        self.boosters: list

    def fit(self, X, y, objective, num_boost_round, verbose=False):
        current_predictions = self.base_prediction * np.ones(shape=y.shape)
        self.boosters = []
        for i in range(num_boost_round):
            gradients = np.array(objective.gradient(y, current_predictions))
            hessians = np.array(objective.hessian(y, current_predictions))
            sample_idxs = None if self.subsample == 1.0 \
                else self.rng.choice(len(y),
                                     size=math.floor(self.subsample * len(y)),
                                     replace=False)
            booster = TreeBooster(X, gradients, hessians,
                                  self.params, self.max_depth, sample_idxs)
            prediction = booster.predict(X)
            #print("*********")
            #print(booster)
            current_predictions += self.learning_rate * prediction
            #print(current_predictions.tolist())
            self.boosters.append(booster)
            if verbose:
                print(f'[{i}] train loss = {objective.loss(y, current_predictions)}')

    def predict(self, X):
        return (self.base_prediction + self.learning_rate
                * np.sum([booster.predict(X) for booster in self.boosters], axis=0))

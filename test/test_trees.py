from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error
import numpy as np
import pytest


from trees.decision_tree import DecisionTree, Data, TreeConfig
from trees.xgboost import SquaredErrorObjective, XGBoostModel
from test.baseline_implementation import SquaredErrorObjectiveOrig, TreeBooster, XGBoostModelOrig


@pytest.fixture(scope="session")
def data():
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    return ((X_train, y_train), (X_test, y_test))


def test_decision_tree_as_tree_booster(data):
    """
    This test confirms that the Decision Tree implementation can replace the Tree Booster
    """

    # ARRANGE
    (X_train, y_train),(X_test, y_test) = data

    initial_predictions = [0.5] * y_train.shape[0]

    gains = SquaredErrorObjective.gradient(y_train, initial_predictions)
    hessians = SquaredErrorObjective.hessian(y_train, initial_predictions)
    d = Data(list(X_train.columns), X_train.values.tolist(), y_train.values.tolist(), gains, hessians)

    # These config options are necessary to be compatible with
    config = TreeConfig(initial_best_score=0, colsample_bynode= 1.0, reg_lambda=1.0, min_samples_leaf=1)

    tree = DecisionTree(d, config)
    tree_boost = TreeBooster(X_train, np.array(gains), np.array(hessians),{},config.max_depth)
    #print(repr(tree))
    #print("*"*50)
    #print(repr(tree_boost))

    # ACT
    tree_predicted = tree.predict(Data(list(X_test.columns),X_test.values.tolist()))

    tree_boost_predicted = tree_boost.predict(X_test)

    # ASSERT
    # MSE less than or equal to the baseline is acceptable!
    assert mean_squared_error(y_test, tree_predicted) <= mean_squared_error(y_test, tree_boost_predicted)

def test_xgboost_implementation_against_baseline(data):

    # ARRANGE
    (X_train, y_train),(X_test, y_test) = data

    depth = 5

    params = {
        'learning_rate': 0.2,
        'max_depth': depth,
        'subsample': 1,
        'reg_lambda': 1.5,
        'gamma': 1.5,
        'min_child_weight': 42,
        'base_score': 0.0,
        'tree_method': 'exact',
    }
    num_boost_round = 50

    # train the from-scratch XGBoost model
    model_baseline = XGBoostModelOrig(params, random_seed=42)
    model_baseline.fit(X_train, y_train, SquaredErrorObjectiveOrig(), num_boost_round)

    model_ported = XGBoostModel(params, random_seed=42)
    model_ported.fit(Data(list(X_train.columns), X_train.values.tolist(), y_train.values.tolist()), SquaredErrorObjective(), num_boost_round)

    # ACT
    predicted = model_baseline.predict(X_test)
    baseline_mse = round(mean_squared_error(y_test, predicted), 6)

    predicted = model_ported.predict(Data(list(X_test.columns),X_test.values.tolist()))
    ported_mse = round(mean_squared_error(y_test, predicted), 6)

    # ASSERT
    assert ported_mse <= baseline_mse


from typing import List
import math
from trees.decision_tree import DecisionTree, Data
import random
from operator import add

class SquaredErrorObjective:
    @staticmethod
    def loss(y: List[float], pred: List[float]) -> float:
        if len(y) == 0:
            return 0
        square_error = [(y_val - pred_val)**2 for (y_val, pred_val) in zip(y, pred)]
        return sum(square_error) / len(square_error)

    @staticmethod
    def gradient(y: List[float], pred: List[float]) -> List[float]:
        error = [(pred_val - y_val) for (y_val, pred_val) in zip(y, pred)]
        return error

    @staticmethod
    def hessian(y: List[float], pred: List[float]) -> List[float]:
        return [1] * len(y)



class XGBoostModel:
    '''XGBoost from Scratch --
    '''

    def __init__(self, params, random_seed=None):
        self.params = params
        self.subsample = self.params.get('subsample', 1.0)
        self.learning_rate = self.params.get('learning_rate', 0.3)
        self.base_prediction = self.params.get('base_score', 0.5)
        self.max_depth = self.params.get('max_depth', 5)
        random.seed(random_seed)
        self.boosters: list

    def fit(self, base_data: Data, objective, num_boost_round, verbose=False):
        current_predictions = [self.base_prediction] * len(base_data.y)
        self.boosters = []
        for i in range(num_boost_round):
            data = Data(column_names=base_data.column_names, X=base_data.X, y=base_data.y)
            data.gains = objective.gradient(data.y, current_predictions)
            data.hessians = objective.hessian(data.y, current_predictions)
            sample_idxs = None if self.subsample == 1.0 \
                else random.sample(list(range(len(data.y))), math.floor(self.subsample * len(data.y)))
            booster = DecisionTree.from_xgb(data, self.params, self.max_depth, sample_idxs)
            prediction = booster.predict(data)
            #print("*********")
            #print(booster)
            #print(prediction)
            current_predictions = list(map(add, current_predictions, map(lambda x: x * self.learning_rate, prediction)))
            #print(current_predictions)
            self.boosters.append(booster)
            if verbose:
                print(f'[{i}] train loss = {objective.loss(data.y, current_predictions)}')

    def predict(self, data: Data):
        result = [self.base_prediction] * len(data.X)
        for booster in self.boosters:
            lst = booster.predict(data)
            result = list(map(add, result, map(lambda x: x * self.learning_rate, lst)))

        return result






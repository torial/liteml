import random
import numpy as np
import matplotlib.pyplot as plt

from micrograd import Value
from nn import Neuron, Layer, MLP
from sklearn.datasets import make_moons, make_blobs
from charts import display_data, display_decision_boundary, draw_dot

class ExperimentRunner:
    __slots__ = ['model']

    def __init__(self, model: MLP):
        self.model = model

    def l2_regularization_loss(self, alpha=1e-4):
        params = (p * p for p in self.model.parameters())
        reg_loss = alpha * sum(params)
        return reg_loss

    @staticmethod
    def svm_max_margin_loss(batchY, scores):
        losses = ((1 + -y_i * score_i).relu() for y_i, score_i in zip(batchY, scores))
        # len(batchY) was len(losses), but changing to generator required the same length
        data_loss = sum(losses) * (1.0 / len(batchY))
        return data_loss

    def loss(self, X, y, batch_size=None):
        # inline data loader
        if batch_size is None:
            batch_src, batch_target = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            batch_src, batch_target = X[ri], y[ri]

        inputs = [list(map(Value, row_x)) for row_x in batch_src]

        # forward the model to get scores
        scores = list(map(self.model, inputs))

        total_loss = self.svm_max_margin_loss(batch_target, scores) + self.l2_regularization_loss()

        # also get accuracy
        accuracy = [(y_i > 0) == (score_i.data > 0) for y_i, score_i in zip(batch_target, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    def fit(self, X, y, number_epochs):
        for k in range(number_epochs):

            # forward
            total_loss, acc = self.loss(X, y)

            # backward
            self.model.zero_grad()
            total_loss.backward()

            # update (sgd)
            learning_rate = 1.0 - 0.9 * k / 100
            for p in self.model.parameters():
                p.data -= learning_rate * p.grad

            if k % 1 == 0:
                print(f" step {k} loss {total_loss.data}, accuracy {acc * 100}%")

        total_loss, acc = self.loss(X, y)
        print(total_loss, acc)


def main():
    np.random.seed(1337)
    random.seed(1337)

    # make up a dataset
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1  # make y = -1 or 1

    display_data(X, y)

    model = MLP(2, [16, 16, 1]) # 2-layer neural network
    print(model)
    print(f"number of parameters: {len(model.parameters())}")

    experimenter = ExperimentRunner(model)
    experimenter.fit(X, y, number_epochs=100)

    display_decision_boundary(model, X, y)

    plt.show()


if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), 'project_profile.stat')
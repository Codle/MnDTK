import numpy as np


class LinearRegression:

    def __init__(self, lr=1e-3, step=1000) -> None:
        self.lr = lr
        self.W = None
        self.step = step

    def fit(self, X: np.array, y: np.array) -> None:
        """train the model

        Parameters
        ----------
        X : np.array
            feature matrix, shape is m * n, which m is the number of train
            examples and n is features' count.
        y : np.array
            label matrix, shape is 1D arry or 1 * m.
        """ 
        y = y.reshape(-1, 1)
        # add bias
        bias = np.ones(X.shape[0], 1)
        X = np.concatenate([X, bias], axis=1)
        m, n = X.shape

        # init weight
        self.W = np.random.randn((1, n))

        # train loop
        for i in range(self.step):
            z = np.exp(np.dot(self.W, X))
            y_hat = z / (1 + z)

            # compute crossentry loss
            L = np.dot(y, np.log(y_hat).T) + np.dot(1-y, np.log(1 - y_hat).T)
            L = -1.0 / m * L

            # gd
            grident = -1.0 / m * X.dot((y - y_hat).T)
            self.weight -= self.lr * grident
        return

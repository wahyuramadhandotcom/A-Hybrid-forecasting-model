import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GrnnArticel:

    def __init__(self, X_train, y_train, X_test, y_test, sigma, actifation='euclidean', p_minkowski=2):
        self.sigma = sigma
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.actifation = actifation
        self.p_minkowski = p_minkowski
        self.std = np.ones((1, self.y_train.size))

    def activation_func(self, distances):
        # Ensure correct operation order in the exponent
        return np.exp(- (distances ** 2) / (2 * (self.std ** 2)))

    def output(self, i):
        if self.actifation == "euclidean":
            distances= np.sqrt(np.sum((self.X_test[i]-self.X_train)**2,axis=1))
        elif self.actifation == "manhattan":
            distances = np.sum(np.abs(self.X_test[i] - self.X_train), axis=1)
        elif self.actifation == "chebyshev":
            distances = np.max(np.abs(self.X_test[i] - self.X_train), axis=1)
        elif self.actifation == "minkowski":
            distances = np.sum(np.abs(self.X_test[i] - self.X_train) ** 2, axis=1) ** (1 / 2)
        elif self.actifation == "cosine":
            X_test = self.X_test[i]
            dot_product = np.sum(self.X_train * X_test, axis=1)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            norm_test = np.linalg.norm(X_test)
            distances = dot_product / (norm_train * norm_test)
        elif self.actifation == "hamming":
            distances = np.sum(self.X_test[i] != self.X_train, axis=1)
        elif self.actifation == "mahalanobis":
            X_test = self.X_test[i]
            VI = np.linalg.inv(np.cov(self.X_train.T)).T
            diff = self.X_train - X_test
            distances = np.sqrt(np.sum(np.dot(diff, VI) * diff, axis=1))

        return distances

    def gaussian_kernel(self, distance):
        return np.exp(- (distance ** 2) / (2 * (self.sigma ** 2)))

    def denominator(self, i):
        return np.sum(self.gaussian_kernel(self.output(i)))

    def numerator(self, i):
        return np.sum(self.gaussian_kernel(self.output(i)) * self.y_train)

    def predict(self):
        predict_array = np.array([])
        for i in range(len(self.y_test)):
            numerator = self.numerator(i)
            denominator = self.denominator(i)
            predict = np.array([numerator / denominator if denominator != 0 else 0])
            predict_array = np.append(predict_array, predict)
        return predict_array

    def mean_squared_error(self):
        predictions = self.predict()
        return np.mean((predictions - self.y_test) ** 2)

    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())
    
    def plot_results(self):
        predictions = self.predict()
        mse = self.mean_squared_error()
        rmse = self.root_mean_squared_error()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test, label='Actual', marker='o')
        plt.plot(predictions, label='Predicted', marker='x')
        plt.title('Actual vs Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()
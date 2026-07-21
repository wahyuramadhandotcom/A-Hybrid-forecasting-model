import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import random

class GEO:
    def __init__(self, obj_func, dim, bounds, n_agents=10, max_iter=50):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.n_agents = n_agents
        self.max_iter = max_iter

    def optimize(self):
        # Inisialisasi posisi eagle (solusi)
        X = np.random.rand(self.n_agents, self.dim)
        for i in range(self.dim):
            lb, ub = self.bounds[i]
            X[:, i] = lb + (ub - lb) * X[:, i]

        fitness = np.array([self.obj_func(x) for x in X])
        gbest = X[np.argmin(fitness)].copy()
        gbest_fit = np.min(fitness)

        for t in range(self.max_iter):
            # Parameter kontrol (meniru berburu)
            alpha = 2 * (1 - t / self.max_iter)

            for i in range(self.n_agents):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * alpha * r1 - alpha
                C = 2 * r2

                D = abs(C * gbest - X[i])
                newX = gbest - A * D

                # Boundary handling
                for d in range(self.dim):
                    lb, ub = self.bounds[d]
                    newX[d] = np.clip(newX[d], lb, ub)

                # Evaluasi fitness
                new_fit = self.obj_func(newX)
                if new_fit < fitness[i]:
                    X[i] = newX
                    fitness[i] = new_fit

            # Update global best
            if np.min(fitness) < gbest_fit:
                gbest = X[np.argmin(fitness)].copy()
                gbest_fit = np.min(fitness)

            print(f"Iter {t+1}/{self.max_iter} | Best RMSE: {gbest_fit:.4f}")

        return gbest, gbest_fit
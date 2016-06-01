import numpy as np
from scipy.optimize import minimize

DIM = 5
N = 400

def cube_euclidean_projection(z):
    if np.all(z >= -1) and np.all(z <= 1):
        return z
    d = z.shape[0]
    func = lambda x: np.linalg.norm(z - x)
    bounds = tuple([(-1, 1) for _ in range(d)])
    x_0 = np.zeros(d)
    result = minimize(func, x_0, bounds = bounds)
    return result.x

def ball_euclidean_projection(z):
    if np.linalg.norm(z) <= 1:
        return z
    return z / np.linalg.norm(z)

def generate_data(scenario, n, sigma):
    X = np.empty((n, DIM - 1))
    y = np.random.binomial(1, 0.5, n)
    y[y == 0] = -1
    mean_0 = np.array([-0.25, -0.25, -0.25, -0.25])
    mean_1 = np.array([0.25, 0.25, 0.25, 0.25])
    cov = np.square(sigma) * np.eye(DIM - 1)
    for i in range(n):
        if y[i] == -1:
            X[i] = np.random.multivariate_normal(mean_0, cov)
        else:
            X[i] = np.random.multivariate_normal(mean_1, cov)
        if scenario == 1:
            X[i] = cube_euclidean_projection(X[i])
        else:
            X[i] = ball_euclidean_projection(X[i])
    return X, y

def loss_gradient(w, x, y):
    x_tilde = np.hstack((x, [1]))
    return -y * np.exp(-y * w.dot(x_tilde)) * x_tilde / (1 + np.exp(-y * w.dot(x_tilde)))

def step_size(t, scenario, n):
    if scenario == 1:
        return 1 / np.sqrt(n)
    else:
        return 1 / np.sqrt(2 * n)

def stochastic_gradient_descent(X, y, scenario):
    n = X.shape[0]
    w = np.zeros((n + 1, DIM))
    for t in range(n):
        alpha = step_size(t + 1, scenario, n)
        G = loss_gradient(w[t], X[t], y[t])
        if scenario == 1:
            w[t + 1] = cube_euclidean_projection(w[t] - alpha * G)
        else:
            w[t + 1] = ball_euclidean_projection(w[t] - alpha * G)
    return np.mean(w[1:], axis = 0)

def average_logistic_loss(w, X, y):
    n = X.shape[0]
    X_tilde = np.hstack((X, np.ones(n).reshape(n, 1)))
    return np.mean(np.log(1 + np.exp(-X_tilde.dot(w) * y)))

def average_binary_error(w, X, y):
    n = X.shape[0]
    X_tilde = np.hstack((X, np.ones(n).reshape(n, 1)))
    return np.mean(np.sign(X_tilde.dot(w)) != y)

if __name__ == '__main__':
    for scenario in [1, 2]:
        for sigma in [0.05, 0.25]:
            # Xte, yte = generate_data(scenario, N, sigma)
            # np.save('test_data/Xte_{0}_{1}.npy'.format(scenario, sigma), Xte)
            # np.save('test_data/yte_{0}_{1}.npy'.format(scenario, sigma), yte)
            Xte = np.load('test_data/Xte_{0}_{1}.npy'.format(scenario, sigma))
            yte = np.load('test_data/yte_{0}_{1}.npy'.format(scenario, sigma))
            logistic_loss_min = 100000
            for n in [50, 100, 500, 1000]:
                trials = 50
                logistic_loss = np.empty(trials)
                binary_error = np.empty(trials)
                for i in range(trials):
                    Xtr, ytr = generate_data(scenario, n, sigma)
                    w = stochastic_gradient_descent(Xtr, ytr, scenario)
                    logistic_loss[i] = average_logistic_loss(w, Xte, yte)
                    binary_error[i] = average_binary_error(w, Xte, yte)
                if np.min(logistic_loss) < logistic_loss_min:
                    logistic_loss_min = np.min(logistic_loss)
                print('Scenario {0}, n = {1}, sigma = {2}'.format(scenario, n, sigma))
                print('Average logistic loss = {0}, deviation = {1}'.format(np.mean(logistic_loss), np.std(logistic_loss)))
                print('Average binary error = {0}, deviation = {1}\n'.format(np.mean(binary_error), np.std(binary_error)))
            print('Min logistic loss = {0}\n'.format(logistic_loss_min))

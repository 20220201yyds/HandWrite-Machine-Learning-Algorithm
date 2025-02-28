import numpy as np


def polynomial_kernal(X1,X2,degree=2,coef0=1):
    return (np.dot(X1,X2.T)+coef0)**degree


def rbf_kernal(X1,X2,gamma=1):
    X1_sq=np.sum(X1**2)
    X2_sq=np.sum(X2**2)
    dist_sq=X1_sq+X2_sq-2*np.dot(X1,X2.T)
    return np.exp(-gamma*dist_sq)


def kernal_ridge_predict(K_train,y_train,K_test,lambd):
    n_train=K_train.shape[0]
    # alpha = (K_train + λ I)^(-1) y_train
    # np.linalg.solve(a,b) solve ax=b
    alpha = np.linalg.solve(K_train + lambd * np.eye(n_train), y_train)
    return np.dot(K_test,alpha)


def kfold_kernal_ridge_gridsearch(X,y,kernal_type='rbf',kernal_params_grid=[{'gamma':1.0}],lambd_grid=[0.1],k=5):
    """
    K-fold cross-validation for selecting the kernel hyperparameters and regularization parameter
    in kernel ridge regression.

    Parameters:
      - X: Feature matrix with shape (n_samples, n_features)
      - y: Label vector with shape (n_samples,)
      - kernel_type: 'rbf' or 'poly'
      - kernel_params_grid: A list of dictionaries, each containing the hyperparameters for the corresponding kernel.
            For example: [{'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 10.0}] or
                         [{'degree': 2, 'coef0': 1}, {'degree': 3, 'coef0': 1}]
      - lambda_grid: A list of regularization parameters, e.g. [0.01, 0.1, 1.0]
      - k: Number of folds
      - random_seed: Random seed (used for shuffling the data)

    Returns:
      - best_params: Dictionary containing the optimal kernel parameters and regularization parameter
      - best_score: The corresponding average mean squared error (MSE)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0

    best_score = float('inf')
    best_params = {}

    for kp in kernal_params_grid:
        for lambd in lambd_grid:
            mse_scores = []
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                val_idx = indices[start:stop]
                train_idx = np.concatenate((indices[:start], indices[stop:]))

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                if kernal_type == 'rbf':
                    gamma = kp.get('gamma', 1.0)
                    K_train = rbf_kernal(X_train, X_train, gamma=gamma)
                    K_val = rbf_kernal(X_val, X_train, gamma=gamma)
                elif kernal_type == 'poly':
                    degree = kp.get('degree', 2)
                    coef0 = kp.get('coef0', 1)
                    K_train = polynomial_kernal(X_train, X_train, degree=degree, coef0=coef0)
                    K_val = polynomial_kernal(X_val, X_train, degree=degree, coef0=coef0)
                else:
                    raise ValueError("Unsupported kernel type. Use 'rbf' or 'poly'.")

                y_pred = kernal_ridge_predict(K_train, y_train, K_val, lambd)
                mse = np.mean((y_val - y_pred) ** 2)
                mse_scores.append(mse)

                current += fold_size
            avg_mse = np.mean(mse_scores)
            print(f"Kernel Params: {kp}, Lambda: {lambd}, CV MSE: {avg_mse:.4f}")
            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {'kernel_params': kp, 'lambda': lambd}

    return best_params, best_score


if __name__ == "__main__":

    X = np.random.randn(100, 5)

    true_coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X.dot(true_coef) + np.random.randn(100) * 0.5

    rbf_params = [{'gamma': 0.1}, {'gamma': 1.0}, {'gamma': 10.0}]
    lambda_grid = [0.01, 0.1, 1.0]

    best_params, best_score = kfold_kernal_ridge_gridsearch(X, y, kernal_type='rbf',
                                                    kernal_params_grid=rbf_params,
                                                    lambd_grid=lambda_grid,
                                                    k=5)
    print("Best Parameters:", best_params)
    print("Best CV MSE:", best_score)
import torch
from sklearn.kernel_approximation import Nystroem


class PLS:
    """Pytorch implementation of the basic PLS algorithm.

    Implemented with reference to the 'scikit-learn PLSRegression' model.
    'https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression'

    Parameters
    ----------
    n_components : int. Dimension of latent variables in the PLS algorithm.

    solver : {'iter', 'svd'}. Solver type of the PLS algorithm.

    max_iter : int. The maximum number of iterations of the power method.
        Only effective when solver == 'iter'.

    tol : float. The tolerance used as convergence criteria in the power method.
        Only effective when solver == 'iter'.
    """

    def __init__(self, n_components, solver, max_iter=500, tol=1e-06):
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        X = X.clone()
        Y = Y.clone()
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        n_components = self.n_components

        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        Xk, Yk, self._x_mean, self._y_mean = X, Y, x_mean, y_mean

        self.x_weights_ = torch.zeros((p, n_components))
        self.y_weights_ = torch.zeros((q, n_components))
        self._x_scores = torch.zeros((n, n_components))
        self._y_scores = torch.zeros((n, n_components))
        self.x_loadings_ = torch.zeros((p, n_components))
        self.y_loadings_ = torch.zeros((q, n_components))
        self.n_iter_ = []

        Y_eps = torch.finfo(torch.float64).eps
        for k in range(n_components):
            Yk_mask = torch.all(torch.abs(Yk) < 10 * Y_eps, dim=0)
            Yk[:, Yk_mask] = 0.0
            if self.solver == 'iter':
                x_weights, y_weights, n_iter_ = _get_first_singular_vectors_power_method(
                    Xk, Yk, max_iter=self.max_iter, tol=self.tol
                )
                self.n_iter_.append(n_iter_)
            elif self.solver == 'svd':
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)
            else:
                raise NameError('PLS solver not supported')

            _svd_flip_1d(x_weights, y_weights)
            x_scores = torch.matmul(Xk, x_weights)
            y_ss = torch.matmul(y_weights, y_weights)
            y_scores = torch.matmul(Yk, y_weights) / y_ss
            x_loadings = torch.matmul(x_scores, Xk) / torch.matmul(x_scores, x_scores)
            Xk -= torch.einsum('i,j->ij', x_scores, x_loadings)
            y_loadings = torch.matmul(x_scores, Yk) / torch.matmul(x_scores, x_scores)
            Yk -= torch.einsum('i,j->ij', x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings
            self.x_scores_ = self._x_scores
            self.y_scores_ = self._y_scores

        self.x_rotations_ = torch.matmul(
            self.x_weights_, torch.pinverse(torch.matmul(self.x_loadings_.T, self.x_weights_))
        )
        self.y_rotations_ = torch.matmul(
            self.y_weights_, torch.pinverse(torch.matmul(self.y_loadings_.T, self.y_weights_))
        )

        self.coef_ = torch.matmul(self.x_rotations_, self.y_loadings_.T)
        return self

    def predict(self, X):
        X = X.clone()
        X -= self._x_mean
        Y_pred = torch.matmul(X, self.coef_)
        return Y_pred + self._y_mean

    def transform(self, X):
        X = X.clone()
        X -= self._x_mean
        x_scores = torch.matmul(X, self.x_rotations_)
        return x_scores


class Model:
    """Deep PLS and generalized deep PLS models.

    Please see the article 'Deep PLS: A Lightweight Deep Learning Model for Interpretable and Efficient Data Analytics,
    https://dx.doi.org/10.1109/TNNLS.2022.3154090' for more details.

    Parameters
    ----------
    lv_dimensions : list of ints. Dimension of latent variables in each PLS layer.

    pls_solver : {'iter', 'svd'}. Solver type of the PLS algorithm.

    use_nonlinear_mapping : bool. Whether to use nonlinear mapping or not.

    mapping_dimensions : list of ints. Dimension of nonlinear features in each nonlinear mapping layer.
        Only effective when use_nonlinear_mapping == True.

    nys_gamma_values : list of floats. Gamma values of Nystroem function in each nonlinear mapping layer.
        Only effective when use_nonlinear_mapping == True.

    stack_previous_lv1 : bool. Whether to stack the first latent variable of the previous PLS layer
        into the current nonlinear features. See the right column of Page 8 in the original article.
        Only effective when use_nonlinear_mapping == True.
    """
    supported_tasks = ['ml_soft_sensor', 'ml_rul_estimation', 'ml_process_monitoring']

    def __init__(self, configs):
        self.lv_dimensions = configs.lv_dimensions
        self.device = configs.device
        self.n_layers = len(self.lv_dimensions)
        self.pls_solver = configs.pls_solver
        self.latent_variables = []

        self.pls_funcs = [
            PLS(n_components=self.lv_dimensions[i], solver=self.pls_solver) 
            for i in range(self.n_layers)
        ]

        self.use_nonlinear_mapping = configs.use_nonlinear_mapping
        self.mapping_dimensions = configs.mapping_dimensions
        self.nys_gamma_values = configs.nys_gamma_values

        if self.use_nonlinear_mapping:
            assert len(self.lv_dimensions) == len(self.mapping_dimensions)
            assert len(self.mapping_dimensions) == len(self.nys_gamma_values)
            self.mapping_funcs = [
                Nystroem(
                    kernel='rbf', gamma=self.nys_gamma_values[i], 
                    n_components=self.mapping_dimensions[i], n_jobs=-1
                ) for i in range(self.n_layers)
            ]

        self.stack_previous_lv1 = configs.stack_previous_lv1

    def fit(self, X, Y):
        X = torch.tensor(X)  # [N, Dx]
        Y = torch.tensor(Y)  # [N, Dy]

        for layer_index in range(self.n_layers):
            if self.use_nonlinear_mapping:
                nys_func = self.mapping_funcs[layer_index]

                X_backup = X
                X = nys_func.fit_transform(X)  # [N, Dm]
                X = torch.tensor(X)
                if self.stack_previous_lv1 and layer_index > 0:
                    lv1_previous_layer = X_backup[:, [0]]
                    X = torch.hstack((lv1_previous_layer, X))

            pls = self.pls_funcs[layer_index]
            pls.fit(X, Y)

            latent_variables = pls.x_scores_
            self.latent_variables.append(latent_variables)
            X = latent_variables

    def predict(self, test_X):
        test_X = torch.tensor(test_X)

        Y_pred = None
        for layer_index in range(self.n_layers):
            if self.use_nonlinear_mapping:
                test_X_backup = test_X.clone()
                test_X = self.mapping_funcs[layer_index].transform(test_X)
                test_X = torch.tensor(test_X)
                if self.stack_previous_lv1 and layer_index > 0:
                    lv1_previous_layer = test_X_backup[:, [0]]
                    test_X = torch.hstack((lv1_previous_layer, test_X))

            if layer_index + 1 == self.n_layers:
                Y_pred = self.pls_funcs[layer_index].predict(test_X)
            test_X = self.pls_funcs[layer_index].transform(test_X)

        return Y_pred


def _get_first_singular_vectors_power_method(X, Y, max_iter=500, tol=1e-06):
    eps = torch.finfo(torch.float64).eps
    try:
        y_score = next(col for col in Y.T if torch.any(torch.abs(col) > eps))
    except StopIteration as e:
        raise StopIteration("Y residual is constant") from e

    x_weights_old = 100
    for i in range(max_iter):
        x_weights = torch.matmul(X.T, y_score) / torch.matmul(y_score, y_score)
        x_weights /= torch.sqrt(torch.matmul(x_weights, x_weights)) + eps
        x_score = torch.matmul(X, x_weights)
        y_weights = torch.matmul(Y.T, x_score) / torch.matmul(x_score.T, x_score)
        y_score = torch.matmul(Y, y_weights) / (torch.matmul(y_weights, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if torch.matmul(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights
    n_iter = i + 1
    return x_weights, y_weights, n_iter


def _get_first_singular_vectors_svd(X, Y):
    C = torch.matmul(X.T, Y)
    U, _, Vt = torch.linalg.svd(C)
    return U[:, 0], Vt[0, :]


def _svd_flip_1d(u, v):
    biggest_abs_val_idx = torch.argmax(torch.abs(u))
    sign = torch.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign

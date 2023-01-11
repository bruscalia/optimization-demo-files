import numpy as np
from optimizers import QuasiNewton, ConjugateGradient, Newton, SteepestDescent
from sklearn.preprocessing import OneHotEncoder


# ----------------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------------

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the softmax function
def softmax(z):
    return np.exp(z) / (np.sum(np.exp(z), axis=-1, keepdims=True))

# Hessian in case of sigmoid
def hessian_sigmoid(betas, X):
    return X.T.dot(np.diag(betas)).dot(X)

# ----------------------------------------------------------------------------------
# MAIN CLASS
# ----------------------------------------------------------------------------------

class LogisticRegression:
    
    def __init__(
        self,
        fit_intercept=True,
        l2=0.0,
        tol=1e-8,
        optimizer=None,
        preprocess=True,
        seed=None,
        random_scale=1e-2,
        **kwargs
    ):
        """Logistic Regression class

        Parameters
        ----------
        fit_intercept : bool, optional
            Whether or not to define a Beta 0 term, by default True
        l2 : float, optional
            L2 regularization parameter, by default 0.0
        tol : float, optional
            Numerical tolerance on logistic functions, by default 1e-8
        optimizer : Optimizer, optional
            Optimizer, by default None
        preprocess : bool, optional
            Whether or not to use linear regression as a warm start for betas, by default True
        seed : int, optional
            Random seed passed to numpy, by default None
        random_scale : float, optional
            Random scale for random initialization of betas, by default 1e-2
        """
        
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.encoder = OneHotEncoder(sparse=False)
        self.tol = tol
        self.preprocess = preprocess
        self.seed = seed
        self.random_scale = random_scale
        
        if optimizer is None:
            self.optimizer = QuasiNewton
        else:
            self.optimizer = optimizer
        
        
        if not "x_tol" in kwargs:
            kwargs["x_tol"] = 1e-2
        
        if not "f_tol" in kwargs:
            kwargs["f_tol"] = 1e-6
        
        self.opt_kwargs = kwargs
    
    def fit(self, X, y=None):
        """Fit model to data

        Parameters
        ----------
        X : numpy.array (n_samples, n_var)
            Independent variables
        
        y : numpy.array (n_samples,) or (n_samples, n_classes), optional
            Response observations, by default None
        """
        
        X = self._prepare_X(X)
        n, p = X.shape
        
        y = self._prepare_y(y)
        k = y.shape[1]
        
        self.n = n
        self.p = p
        self.k = k
        
        if self.preprocess:
            C = np.linalg.inv(X.T.dot(X))
            b = X.T.dot(y)
            betas_start = C.dot(b).flatten()
            
        else:
            np.random.seed(self.seed)
            betas_start = np.zeros((p, k)).flatten()
            betas_start = np.random.rand(p * k) * self.random_scale
        
        self.opt = self.optimizer(self.loss, gradient=self.gradient, **self.opt_kwargs)
        self.opt.optimize(betas_start, X, y, p, k, n)
        
        self.betas = self.opt.x.reshape((p, k))
        self.coef_ = self.betas[1:, :].T
        self.intercept_ = self.betas[0, :].T
    
    def predict_proba(self, X, y=None):
        """Predict the probability of each item belonging to each class

        Parameters
        ----------
        X : numpy.array (n_samples, n_var)
            Independent variables

        Returns
        -------
        numpy.array
            Predictions of shape (n_samples,) in case of binary classification 
            or (n_samples, n_classes) otherwise
        """
        
        X = self._prepare_X(X)
        Z = X.dot(self.betas).clip(-1e2, 1e2)
        A = self.activation(Z)
        
        return A
    
    def predict(self, X, y=None):
        """Predict to which class each item of a set belongs to

        Parameters
        ----------
        X : numpy.array (n_samples, n_var)
            Independent variables

        Returns
        -------
        numpy.array
            Predictions of shape (n_samples,) in case of binary classification 
            or (n_samples, n_classes) otherwise
        """
        
        A = self.predict_proba(X)
        
        if self.return_flatten:
            
            if self.multi_class:
                choices = np.argmax(A, axis=1)
                y_pred = self.encoder.categories_[0][choices]
                
            else:
                # choices = 1 - np.round(A, 0).flatten().astype(int)
                # y_pred = self.encoder.categories_[0][choices]
                y_pred = np.round(A, 0).flatten().astype(int)
        
        else:
            choices = np.argmax(A, axis=1)
            y_pred = self.encoder.fit_transform(choices)
            
        return y_pred
    
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        y_pred = self.predict(X)
        return y_pred
    
    def softmax_loss(self, betas, X, y, p, k, n):
            
        betas_mat = betas.copy().reshape((p, k))
        
        Z = X.dot(betas_mat).clip(-1e2, 1e2)
        A = softmax(Z).clip(self.tol, 1.0 - self.tol)
        J = -np.sum(y * np.log(A)) / n
        
        if self.fit_intercept:
            betas_params = betas_mat[1:, :].flatten()
        else:
            betas_params = betas.copy()
        
        l2_penalty = self.l2 * betas_params.dot(betas_params)

        return J + l2_penalty
    
    def gradient_softmax(self, betas, X, y, p, k, n):
        
        betas_mat = betas.copy().reshape((p, k))
        
        Z = X.dot(betas_mat).clip(-1e2, 1e2)
        A = softmax(Z)
        grad = X.T.dot(A - y)
        
        if self.fit_intercept:
            betas_params = betas_mat.copy()
            betas_params[1, :] = 0.0
            betas_params = betas_params.flatten()
        else:
            betas_params = betas.copy()
        
        l2_gradient = 2 * self.l2 * betas_params
        
        return grad.flatten() / n + l2_gradient
    
    def gradient_sigmoid(self, betas, X, y, p, k, n):
        
        betas_mat = betas.copy().reshape((p, k))
        
        Z = X.dot(betas_mat).clip(-1e2, 1e2)
        A = sigmoid(Z)
        grad = X.T.dot(A - y)
        
        if self.fit_intercept:
            betas_params = betas_mat.copy()
            betas_params[1, :] = 0.0
            betas_params = betas_params.flatten()
        else:
            betas_params = betas.copy()
        
        l2_gradient = 2 * self.l2 * betas_params
        
        return grad.flatten() / n + l2_gradient
    
    def sigmoid_loss(self, betas, X, y, p, k, n):
            
        betas_mat = betas.copy().reshape((p, k))
        
        Z = X.dot(betas_mat).clip(-1e2, 1e2)
        A = sigmoid(Z).clip(self.tol, 1.0 - self.tol)
        J = -np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) / n
        
        if self.fit_intercept:
            betas_params = betas_mat[1:, :].flatten()
        else:
            betas_params = betas.copy()
        
        l2_penalty = self.l2 * betas_params.dot(betas_params)

        return J + l2_penalty
    
    def _prepare_X(self, X):
        
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.atleast_2d(X).reshape((-1, 1))
        
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.column_stack((ones, X))
        
        else:
            pass

        return X
    
    def _prepare_y(self, y):
        
        y = np.array(y)
        
        n_dimensions = len(y.shape)
        self.return_flatten = True
        
        if n_dimensions == 2:
            self.return_flatten = False
            y = self._prepare_two_dimensional_y(y)
            
        elif n_dimensions == 1:
            y = self._prepare_one_dimensional_y(y)
            
        else:
            raise ValueError("y must be either one-dimensional or two-dimensional")
        
        return y
    
    def _prepare_one_dimensional_y(self, y):
        
        self.n_classes = len(np.unique(y))
        
        self.activation = softmax
        self.loss = self.softmax_loss
        self.gradient = self.gradient_softmax
        self.multi_class = True
        
        if self.n_classes == 2:
            self.multi_class = False
            assert np.all(np.isin(y, np.array([0, 1]))), "y must contain only 0 and 1"
            y = y.reshape((-1, 1))
            
        elif self.n_classes < 2:
            raise ValueError("Need at least two classes")
        
        else:
            y = self.encoder.fit_transform(y.reshape((-1, 1)))
        
        if self.n_classes == 2:
            y = y[:, :1]
            self.activation = sigmoid
            self.loss = self.sigmoid_loss
            self.gradient = self.gradient_sigmoid
        
        return y
    
    def _prepare_two_dimensional_y(self, y):
        
        assert np.sum(y, axis=1) == 1, "y must contain only one True value per row"
        assert np.all(np.isin(y, np.array([0, 1]))), "y must contain only 0 and 1"
        
        return y
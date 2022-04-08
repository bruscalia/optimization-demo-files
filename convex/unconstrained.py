import numpy as np
from scipy.optimize import minimize, line_search, BFGS, SR1
from numdifftools import Gradient

_small_number = np.sqrt(np.finfo(float).eps)

class DescentAlgorithm:
    
    def __init__(self, fun, gradient=None, hess=None, nd={}, wolfe_c1=1e-4, wolfe_c2=0.1,
                 x_tol=1e-6, f_tol=1e-6, max_iter=50, save_history=False):
        """White label solver for Descent Algorithms.

        Args:
            fun (callable): Objective function f(x, *args) of minimization problem.
            gradient (callable or Nons, optional): Gradient of objective function gradient(x, *args)
                of minimization problem. If None, numdifftools Gradient is created. Defaults to None.
            hess (callable or HessianUpdateStrategy, optional): Hessian of the objective function.
                In Newton method, should be provided as hess(x, *args), whereas in QuasiNewton methods
                it should be a scipy.optimize.HessianUpdateStrategy instance, Defaults to None.
            nd (dict, optional): Keyword arguments passed to numdifftools is gradient is not provided.
                Defaults to {}.
            wolfe_c1 (float, optional): Wolfe line search c1. Defaults to 1e-4.
            wolfe_c2 (float, optional): Wolfe line search c2. Defaults to 0.1.
            x_tol (float, optional): Tolerance for advance in x. Defaults to 1e-6.
            f_tol (float, optional): Tolerance for advance in f(x). Defaults to 1e-6.
            max_iter (int, optional): Max iterations. Defaults to 50.
            save_history (bool, optional): Either of not to store history of iterations. Defaults to False.
        """
        
        self.fun = fun
        
        if gradient is None:
            self.gradient = Gradient(fun, **nd)
        else:
            self.gradient = gradient
        
        self.hess = hess
        self.wolfe_coefs = wolfe_c1, wolfe_c2
        self.x_tol = x_tol
        self.f_tol = f_tol
        self.max_iter = max_iter
        self.save_history = save_history
        self.history = []
    
    def optimize(self, x0, *args, **kwargs):
        
        x0 = np.atleast_1d(x0).astype(float)
        self.history = []

        xk = x0.copy()
        fk = self.fun(x0, *args, **kwargs)
        gradk = self.gradient(x0, *args, **kwargs)
        
        fc, gc = 1, 1
        
        pk = self.prepare_initial_step(xk, fk, gradk, *args, **kwargs)
        
        advance_x, advance_f, advance_max = True, True, True
        k = 0
        
        if self.save_history:
            self.history.append({"x":xk, "f":fk, "grad":gradk})
        
        while (advance_x or advance_f) and (k <= self.max_iter):
            
            alpha, fc_, gc_, fnew, fk, gradnew = line_search(self.fun, self.gradient,
                                                             xk, pk, gradk, fk, args=args,
                                                             c1=self.wolfe_coefs[0],
                                                             c2=self.wolfe_coefs[1],
                                                             maxiter=15)
            
            if alpha is None:
                alpha = 1
                fnew = self.fun(xk + alpha * pk, *args, **kwargs)
                gradnew = self.gradient(xk + alpha * pk, *args, **kwargs)
            
            xnew = xk + alpha * pk
            fc = fc + fc_
            gc = gc + gc_
            
            if gradnew is None:
                gradnew = self.gradient(xnew)
            
            advance_f = abs(fnew - fk) > self.f_tol
            advance_x = np.linalg.norm(xnew - xk) > self.x_tol
            
            xk, fk, gradk, pk = self.prepare_next_step(xk, fk, gradk, pk, xnew, fnew, gradnew, *args, **kwargs)

            k = k + 1
            
            if self.save_history:
                self.history.append({"x":xk, "f":fk, "grad":gradk})
            
            if np.linalg.norm(pk) < np.sqrt(np.finfo(float).eps):
                self.message = 'Negligible step'
                self.success = True
                break
        
        if not (advance_x or advance_f):
            self.success = True
            self.message = 'Tolerance reached'
            
        elif k > self.max_iter:
            self.success = False
            self.message = 'Max iterations reached'
        
        self.x = xk
        self.f = fk
        self.grad = gradk
        self.fc = fc
        self.gc = gc
        self.result = {"x":xk, "f":fk, "grad":gradk, "iter":k, "message":self.message, "success":self.success}
    
    def prepare_next_step(self, xk, fk, gradk, pk, xnew, fnew, gradnew, *args, **kwargs):
        pass
    
    def prepare_initial_step(self, xk, fk, gradk, *args, **kwargs):
        pass


class SteepestDescent(DescentAlgorithm):

    def prepare_next_step(self, xk, fk, gradk, pk, xnew, fnew, gradnew, *args, **kwargs):
        return xnew, fnew, gradnew, -gradnew
    
    def prepare_initial_step(self, xk, fk, gradk, *args, **kwargs):
        return -gradk


class ConjugateGradient(SteepestDescent):

    def prepare_next_step(self, xk, fk, gradk, pk, xnew, fnew, gradnew, *args, **kwargs):
        return xnew, fnew, gradnew, -gradnew + pk * gradnew.dot(gradnew) / gradk.dot(gradk)


class Newton(DescentAlgorithm):
    
    def __init__(self, fun, gradient=None, hess=None, nd={}, wolfe_c1=1e-4, wolfe_c2=0.9,
                 x_tol=1e-6, f_tol=1e-6, max_iter=50, save_history=False):
        
        if hess is None:
            raise TypeError("Must provide hessian")
        
        super().__init__(fun, gradient=gradient, hess=hess, nd=nd, wolfe_c1=wolfe_c1, wolfe_c2=wolfe_c2,
                         x_tol=x_tol, f_tol=f_tol, max_iter=max_iter, save_history=save_history)
    
    def prepare_next_step(self, xk, fk, gradk, pk, xnew, fnew, gradnew, *args, **kwargs):
        H = self.hess(xnew, *args, **kwargs)
        return xnew, fnew, gradnew, np.linalg.solve(H, -gradnew)
    
    def prepare_initial_step(self, xk, fk, gradk, *args, **kwargs):
        H = self.hess(xk, *args, **kwargs)
        return np.linalg.solve(H, -gradk)

class QuasiNewton(Newton):

    def __init__(self, fun, gradient=None, hess=None, nd={}, wolfe_c1=1e-4, wolfe_c2=0.9,
                 x_tol=1e-6, f_tol=1e-6, max_iter=50, save_history=False):
        
        if hess is None:
            hess = BFGS(exception_strategy="damp_update", min_curvature=0.2)
        
        super().__init__(fun, gradient=gradient, hess=hess, nd=nd, wolfe_c1=wolfe_c1, wolfe_c2=wolfe_c2,
                         x_tol=x_tol, f_tol=f_tol, max_iter=max_iter, save_history=save_history)
    
    def prepare_next_step(self, xk, fk, gradk, pk, xnew, fnew, gradnew, *args, **kwargs):
        self.hess.update(xnew - xk, gradnew - gradk)
        H = self.hess.get_matrix()
        return xnew, fnew, gradnew, np.linalg.solve(H, -gradnew)
    
    def prepare_initial_step(self, xk, fk, gradk, *args, **kwargs):
        self.hess.initialize(xk.shape[0], "hess")
        H = self.hess.get_matrix()
        return np.linalg.solve(H, -gradk)

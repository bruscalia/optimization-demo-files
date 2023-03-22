# optimization-demo-files
Here you might find some examples and tutorials on numerical optimization.

Check my [Medium profile](https://medium.com/@bruscalia12) for some interesting articles about (most of) these examples.

## Linear Programming (continuous space)

- [Product-mix problem](./convex/linear/product_mix.ipynb)
- [Transportation problem](./convex/linear/transportation.ipynb)
- [Turbogenerators industrial example](./convex/linear/turbogenerators.ipynb)

## Nonlinear (convex)

- Implementation from scratch of usual [gradient-based](./convex/nonlinear/unconstrained.py) line search algorithms in Python.
- Solution of a [constrained problem](./convex/nonlinear/convex_problems.ipynb) using the scipy.optimize minimize function.
- Optimization of a [chemical reactor problem](./convex/nonlinear/example_xylene.ipynb).
- Applications of nonlinear optimization to [classification problems](./classification/logistic_regression.ipynb).

## MIP

- Variants of the knapsack problem: [simple](./mip/knapsack/notebooks/simple_knapsack.ipynb), [multi-dimensional](./mip/knapsack/notebooks/simple_knapsack.ipynb) and [multiple knapsacks](./mip/knapsack/notebooks/multiple_knapsacks.ipynb).
- [Dynamic lot-size model](./mip/dynamic_lot_size/notebooks/dynamic_lot_size.ipynb)
- [Job-shop scheduling](https://github.com/bruscalia/jobshop/blob/master/notebooks/mip_models.ipynb) (external repository)
- [Branch & Bound](./mip/branch_and_bound/graphical_example.ipynb) graphical example

## Nonconvex

- Implementation [Differential Evolution](./nonconvex/de_scipy.ipynb) using scipy.optimize.
- Solutions of convex, nondifferentiable, and nonconvex problems using DE and classic algorithms.

## Regression

- Implementation of [Linear Regression from scratch](./regression/notebooks/linear_regression.ipynb) in Python.
- Examples of Linear Regression applications, residual analysis, and feature selection.

## Portfolio MOO
- Solution of the Efficient Frontier problem from the Modern Portfolio Theory using a [multi-objective approach](./portfolio-moo/portfolio.ipynb) with pymoo and pymoode.

## Contact
bruscalia12@gmail.com

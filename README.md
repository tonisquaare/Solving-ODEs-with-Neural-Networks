# Solving ODEs with Neural Networks.

This repository contains scripts and functions to solve ordinary differential equations (ODEs) using feedforward neural networks. The optimizer used is ADAM combined with LRExponential as scheduler.

The ODEs must be defined in the form $F(x,f,f',...,f^{(n)}) = 0$

Initial conditions for the initial value problems are enforced through trial functions, whose structure will always be the same:

$$T_f(x) = A(x)\mathcal{N}(x) + B(x)$$

where $A(x)$ and its derivatives vashish in the points where initial conditions apply; on the other hand, $B(x)$ must satisfy these initial conditions.

In addition to the equation and the trial function, there are some parameters that must be specified:
- The domain of the solution, as an interval: (a,b)
- The order of the equation (by default is 1)
- The learning rate of the optimizer (by default is 1e-3)
- The parameter gamma of the scheduler (by default is 0.99).
- The maximum number of epochs: max_iters (by default will be 1000).
- The acceptable tolerance of the loss function (by default is 1e-7). In case we want to ignore it, and run for a certain number of iterations, we can set it to 0.

The useful files of the project are ODEs_solver_1D.py, which includes a function that create the nn and solves the equation; and main.py, which will allow us to run such function. An order 4 ODE is provided (together with its analytic solution) as an example

 There are also two files tagged as 'order2' that are a primitive version of the code, and can only solve up to order 2.

 All of this work follows the method described in the following article by Lagaris, Likas and Fotiadis (1998):

 Lagaris, I.E., Likas, A., & Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks, 9(5), 987–1000. DOI: [10.1109/72.712178](https://ieeexplore.ieee.org/document/712178).




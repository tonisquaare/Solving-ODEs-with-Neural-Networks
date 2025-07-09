# Solving ODEs with Neural Networks.

This repository contains scripts and functions to solve systems of ordinary differential equations (ODEs) using feedforward neural networks. The optimizer used is ADAM combined with LRExponential as scheduler. Curriculum learning is also used.

The ODEs must be defined in the form $F(x,f_1,f_1',...,f^{(n)},f_2,...) = 0$

Initial conditions for the initial value problems are enforced through trial functions, whose structure will always be the same:

$$T_f(x) = A(x)\mathcal{N}(x) + B(x)$$

where $A(x)$ and its derivatives vashish in the points where initial conditions apply; on the other hand, $B(x)$ must satisfy these initial conditions.

In addition to the equation and the trial function, there are some parameters that must be specified:
- The domain of the solution, as an interval: (a,b)
- The 'multiorder' of the equation: the order of each equation of the system, as a list. (by default is 1)
- The number of curriculum 'steps' (by default will be one)
- The learning rate of the optimizer (by default is 1e-3)
- The parameter gamma of the scheduler (by default is 0.99).
- The maximum number of epochs for each curriculum step: max_iters (by default will be 1000).

The useful files of the project are ODEs_solver.py, which includes a function that create the nn and solves the system of equations; and main.py, which will allow us to run such function. An system of two equations is provided (together with its analytic solution) as an example (Problem 4, Lagaris).

 There are also two files tagged as '1D' that are a primitive version of the code, and can only solve a single scalar equation (of an arbitrary order).

 All of this work follows the method described in the following article by Lagaris, Likas and Fotiadis (1998):

 Lagaris, I.E., Likas, A., & Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks, 9(5), 987–1000. DOI: [10.1109/72.712178](https://ieeexplore.ieee.org/document/712178).

 # Improvements to be implemented
 - Batch training: currently I'm using a single batch.
 - Hyperparameters optimization: it's done by hand right now.
 - Normalization of the inputs and changing the equation to work in the normalized space (the derivatives basically).
 - Better initialization of parameters: Xavier maybe...
 - ...
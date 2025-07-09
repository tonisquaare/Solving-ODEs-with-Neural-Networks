import numpy as np
import matplotlib.pyplot as plt
from ODEs_solver_1D import solve_ODE
import torch

tol = 1e-7
max_iters = 5000

a = 0
b = 1

order = 4

# args is a list that contains the function f and its derivatives, in ascending order : args[:,0:1]=f, args[:,1:2]=f_x, args[:,2:3]=f_xx...
def equation(x,args):
    f = args[0]
    f_x = args[1]
    f_xx = args[2]
    f_3x = args[3]
    f_4x = args[4]

    ODE = f_4x - 2*f_xx + f
    return ODE
            
# Trial functions are of the form of A(x)N(x) + B(x), A vanishes in the fixed points and B satisfies explicitly the IC in those points
def trial_function(x,N): #f(0) = 1; f'(0) = f''(0) = f'''(0) = 0
    return x**4*N + 1

f,v_loss = solve_ODE(a,b,equation,trial_function,order=order,max_iters=max_iters,tol=tol,lr=5e-3,gamma=0.93)

# The analytic solution, just to compare.
def sol_analyt(x):
    return (1/2 - 1/4*x)*np.exp(x) + (1/2 + 1/4*x)*np.exp(-x)  # Symbolab...

# PLOT THE RESULTS
x = np.linspace(a,b,100)
fig,(axis_sol,axis_loss) = plt.subplots(ncols=2,figsize=(16,6))
axis_sol.grid(True)
axis_sol.set_xlabel("x")
axis_sol.set_ylabel("f(x)")
axis_sol.set_title("Solution")
axis_sol.scatter(x,f,s=10)  # NN solution
axis_sol.plot(x,sol_analyt(x),color="r")  # Analytic solution

axis_loss.grid(True)
axis_loss.set_yscale('log')
axis_loss.set_xlabel("Epoch")
axis_loss.set_ylabel("Loss")
axis_loss.set_title("Loss function")
axis_loss.plot(v_loss)
fig.savefig("resultados.png",dpi=400)
plt.show()
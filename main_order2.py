import numpy as np
import matplotlib.pyplot as plt
from ODEs_solver_1D_order_2 import solve_ODE
import torch

tol = 1e-7
max_iters = 1000

a = 0
b = 2

def equation(x,f,df,d2f=0):
    return d2f + 1/5 * df + f + 1/5 * torch.exp(-x/5) * torch.cos(x)

def trial_function(x,N):
    return x + x**2*N

f,v_loss = solve_ODE(a,b,equation,trial_function,max_iters=max_iters,tol=tol,lr=1e-2,gamma=0.99)

# The analytic solution, just to compare.
def sol_analyt(x):
    return np.exp(-x/5)*np.sin(x)

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
fig.savefig("resultados.png")
plt.show()
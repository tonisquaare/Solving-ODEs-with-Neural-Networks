import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from ODEs_solver import solve_ODE
import torch

tol = 0
max_iters = 5000

a = 0
b = 3

multiorder = (1,1) # List with the order of each equation

# args is a list of nested lists that contain the functions f_k and theor respective derivatives, 
# in ascending order : args[0][0]=f, args[0][1]]=f_x, args[1][0]=g...The example provided is the 4th problem of Lagaris article.
def equation(x,args):
    f = args[0][0]
    f_x = args[0][1]
    g = args[1][0]
    g_x = args[1][1]

    ODE_1 = f_x - torch.cos(x) - f**2 - g + (1+x**2 + torch.sin(x)**2)
    ODE_2 = g_x - 2*x + (1+x**2)*torch.sin(x) - f*g
    return (ODE_1,ODE_2)
            
# Trial functions are of the form of A(x)N(x) + B(x), A vanishes in the fixed points and B satisfies explicitly the IC in those points

def trial_function(x,N): #f(0) = 1; f'(0) = f''(0) = f'''(0) = 0
    trial1 = x*N[0]
    trial2 = 1 + x*N[1]
    return (trial1,trial2)


f,v_loss = solve_ODE(a,b,equation,trial_function,n_curr=3,multiorder=multiorder,max_iters=max_iters,lr=5e-3,gamma=0.94)

# The analytic solution, just to compare.
def sol_analyt(x):
    return (np.sin(x),1+x**2)  # Symbolab...

# PLOT THE RESULTS
x = np.linspace(a,b,100)
fig,(axis_sol,axis_loss) = plt.subplots(ncols=2,figsize=(16,6))
axis_sol.grid(True)
axis_sol.set_xlabel("x")
axis_sol.set_ylabel("f(x)")
axis_sol.set_title("Solution")

n = len(multiorder)
cmap = plt.get_cmap('gist_rainbow')
if n == 1:
    colors = 'r'
else:
    colors = [cmap(u) for u in np.linspace(0,1,n)]

analytic = sol_analyt(x)
for i in range(n):
    axis_sol.scatter(x,f[i],s=10,color=colors[i], label=f"f_{i+1}")  # NN solution
    axis_sol.plot(x,analytic[i],color='b',linestyle='dashed',label=f"analytic")  # Analytic solution
axis_sol.legend()

axis_loss.grid(True)
axis_loss.set_yscale('log')
axis_loss.set_xlabel("Epoch")
axis_loss.set_ylabel("Loss")
axis_loss.set_title("Loss function")
axis_loss.plot(v_loss)

fig.savefig("resultados.png",dpi=400)
plt.show()
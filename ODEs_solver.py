import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim


NEURONS = 30
HIDDEN_LAYERS = 2
POINTS = 100


def solve_ODE(a,b,equation,trial_function,multiorder,n_curr=1,max_iters=1000,lr=1e-3,gamma=0.98):
    """This function solves an arbitrary ODE using a neural network. It uses the 'trial_function' and the 'equation' to calculate the loss function,
    which is minimized using an Adam optimizer. We use curriculum learning, the number of steps can be controlled through n_curr. 
    The parameters required are:
    a: float, the start of the interval.
    b: float, the end of the interval.
    equation: tuple/list of the ODEs of the system to be solved.
    trial_function: tuple/list, the trial functions to be used.
    multiorder: tuple/list, the orders of each ODE.
    n_curr=1
    max_iters: int, the maximum number of iterations for the optimizer for each curriculum step. (default is 1000).
    lr: float, the learning rate for the optimizer.(default is 1e-3).
    gamma: float, the decay rate for the learning rate scheduler. (default is 0.98).
    """

    n = len(multiorder)  # Number of equations 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_np = np.linspace(a,b,POINTS)
    x = torch.Tensor(x_np,device=device).unsqueeze(1)
    x.requires_grad_(True)

    # NEURAL NETWORK
    class NetODE(nn.Module):
        def __init__(self):
            super(NetODE,self).__init__()
            self.input = nn.Linear(1,NEURONS)
            self.hidden = nn.ModuleList([nn.Linear(NEURONS, NEURONS) for _ in range(HIDDEN_LAYERS - 1)])
            self.output = nn.Linear(NEURONS,n)

            self.act = nn.Tanh()
        
        def forward(self, x):
            x = self.input(x)
            x = self.act(x)
            for linear_N_N in self.hidden:
                x = linear_N_N(x)
                x = self.act(x)
            x = self.output(x)
            return x

    # LOSS FUNCTION
    mse = nn.MSELoss()
    def f_loss(NN,x):
        N = []
        for i in range(n):
            N.append(NN[:,i:i+1])

        Tf = trial_function(x,N)  # Stores the trial functions
        args = []

        for i in range(n):
            args.append([Tf[i]])  # functions
            for j in range(multiorder[i]):
                args[i].append(torch.autograd.grad(args[i][j],x,create_graph=True, retain_graph=True,grad_outputs=torch.ones_like(args[i][j]))[0])  # their derivatives

        ODEs = equation(x,args)
        total = 0
        for i in range(n):
            total += mse(ODEs[i],torch.zeros_like(ODEs[i],device=device))

        return total

    net = NetODE().to(device)

    v_loss = []

    # TRAINING

    n_curr = round(b-a)

    for k in range(n_curr): # Curriculum training
        x_np_curr = np.linspace(a,a+(b-a)*(k+1)/n_curr,POINTS)
        x = torch.Tensor(x_np_curr,device=device).unsqueeze(1)
        x.requires_grad_(True)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        for epoch in range(max_iters):
            NN = net(x)

            loss = f_loss(NN,x)
            v_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #if loss < tol: # In case the value of our loss function is lower than the tolerance, we stop the training.
            #    print(f"{epoch} epochs. Loss: {loss}")
            #    break

            if epoch % 200 == 0:
                print(f"{epoch} epochs. Loss: {loss}")
                scheduler.step()

    N = [NN[:,i:i+1] for i in range(n)]
    Tfs = trial_function(x,N)

    res = [Tf.detach().cpu().numpy() for Tf in Tfs]

    np.savetxt(f'results.txt', np.hstack((x_np.reshape(-1,1),*res))) 

    print(f"Loss: {v_loss[-1]} after {epoch} epochs.")
    return res,v_loss
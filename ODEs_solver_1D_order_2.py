import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim


NEURONS = 10
HIDDEN_LAYERS = 2
POINTS = 200


def solve_ODE(a,b,equation,trial_function,max_iters,tol,lr,gamma):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_np = np.linspace(a,b,100)
    x = torch.Tensor(x_np,device=device).unsqueeze(1)
    x.requires_grad_(True)
    
    class NetODE1(nn.Module):
        def __init__(self):
            super(NetODE1,self).__init__()
            self.input = nn.Linear(1,NEURONS)
            self.hidden = nn.ModuleList([nn.Linear(NEURONS, NEURONS) for _ in range(HIDDEN_LAYERS - 1)])
            self.output = nn.Linear(NEURONS,1)

            self.act = nn.Tanh()
        
        def forward(self, x):
            x = self.input(x)
            x = self.act(x)
            for linear_N_N in self.hidden:
                x = linear_N_N(x)
                x = self.act(x)
            x = self.output(x)
            return x

    mse = nn.MSELoss()
    def f_loss(NN,x):
        N1 = NN[:,0:1]

        Tf = trial_function(x,N1) #TRIAL FUNCTION
        Tf_x = torch.autograd.grad(Tf,x,create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(Tf))[0]
        #The calculation of the second derivative could be avoided if the ecuation doesn't use it. It would improve performance.
        Tf_xx = torch.autograd.grad(Tf_x,x,create_graph=True,retain_graph=True,grad_outputs=torch.ones_like(Tf_x))[0] 
        ODE = equation(x,Tf,Tf_x,Tf_xx)
        return mse(ODE,torch.zeros_like(ODE,device=device))

    net = NetODE1().to(device)

    v_loss = []


    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
    for epoch in range(max_iters):
        NN = net(x)

        loss = f_loss(NN,x)
        v_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss < tol: # In case the value of our loss function is lower than the tolerance, we stop the training.
            print(f"{epoch} epochs. Loss: {loss}")
            break

        if epoch % 200 == 0:
            print(f"{epoch} epochs. Loss: {loss}")
            scheduler.step()

    Tf = trial_function(x,NN)
    res = Tf.detach().cpu().numpy()
    np.savetxt(f'results.txt', np.hstack((x_np.reshape(-1,1),res)),header="x\tPsi(x)") 
    print(f"Loss: {v_loss[-1]} after {epoch} epochs.")
    return res,v_loss
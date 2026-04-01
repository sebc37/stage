import numpy as np
import matplotlib.pyplot as plt
import torch
import  architecture

torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






#############################
# modèle lorenz 63 avec RK4 #
#############################
nb    = 1000 # number of times
n=3
time  = np.array(range(nb))

### define the nonlinear dynamic system (Lorenz-63) using the Runge-Kutta integration method
def m(x_past):
    
    # physical parameters
    dT=0.01
    sigma=10
    rho=28
    beta=8/3
    
    # Runge-Kutta (4,5) integration method
    X1 = np.copy(x_past)
    k1 = np.zeros(X1.shape)
    k1[0] = sigma*(X1[1] - X1[0])
    k1[1] = X1[0]*(rho-X1[2]) - X1[1]
    k1[2] = X1[0]*X1[1] - beta*X1[2]
    X2 = np.copy(x_past+k1/2*dT)
    k2 = np.zeros(x_past.shape)
    k2[0] = sigma*(X2[1] - X2[0])
    k2[1] = X2[0]*(rho-X2[2]) - X2[1]
    k2[2] = X2[0]*X2[1] - beta*X2[2]   
    X3 = np.copy(x_past+k2/2*dT)
    k3 = np.zeros(x_past.shape)
    k3[0] = sigma*(X3[1] - X3[0])
    k3[1] = X3[0]*(rho-X3[2]) - X3[1]
    k3[2] = X3[0]*X3[1] - beta*X3[2]
    X4 = np.copy(x_past+k3*dT)
    k4 = np.zeros(x_past.shape)
    k4[0] = sigma*(X4[1] - X4[0])
    k4[1] = X4[0]*(rho-X4[2]) - X4[1]
    k4[2] = X4[0]*X4[1] - beta*X4[2]

    # return the state in the near future
    x_future = x_past + dT/6.*(k1+2*k2+2*k3+k4)
    
    return x_future


x = np.zeros((n,nb))
x[:,0] = np.array([8,0,30])

for t in range(1,nb):
    x[:,t] = m(x[:,t-1])



#################################################
# Dataset LR63 avec BC, IC, Collocation points  #
#################################################

x_bc = x[0,:]
x_ic = x[:,0]

i_cl = np.random.choice(range(nb), 500)
x_cl = x[1:3,i_cl]

### dataset mis sous forme de tensor
X_BC = torch.tensor(x_bc) # x
X_CL = torch.tensor(x_cl) # y,z point aléatoire
X_IC = torch.tensor(x_ic) # x(0),y(0),z(0)

T = torch.tensor(time,dtype=torch.float32)
T_CL = torch.stack((torch.tensor(i_cl,dtype=torch.float32),torch.tensor(i_cl,dtype=torch.float32)),dim=0)
T_IC = torch.zeros(3)

dataset_bc = torch.utils.data.TensorDataset(T,X_BC)
dataset_cl = torch.utils.data.TensorDataset(T_CL,X_CL)
dataset_ic = torch.utils.data.TensorDataset(T_IC,X_IC)

plt.figure()
plt.plot(time,x_bc.T)
plt.plot(i_cl,x_cl.T,'*')
plt.show()

plt.figure()
plt.plot(T,X_BC)
plt.plot(T_CL.mT,X_CL.mT,'*')
plt.show()

##############
# Dataloader #
##############

trainloader_bc = torch.utils.data.DataLoader(dataset_bc, batch_size=128, shuffle=True, drop_last=False)
trainloader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=128, shuffle=True, drop_last=False)

###############################
# loss BC, IC ,CL et physique #
###############################

#############
#   modèle  #
# ###########            
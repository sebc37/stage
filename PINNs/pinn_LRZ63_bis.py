import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import  architecture
import tqdm

torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = ""

list_of_datasets = glob.glob(PATH+"dataset_*.npy")

# faire différent dataset : parti facile de l'attracteur et point d'equilibre 
# regarder avec des (ti,xi,yi,zi) qui predise des(xj,yj,zj) avec j>i
# faire une fonction closure pour LBFGS



#############################
# modèle lorenz 63 avec RK4 #
#############################
nb    = 1000 # number of times
n=3
time  = np.array(range(nb))

sigma=10
rho=28
beta=8/3

### define the nonlinear dynamic system (Lorenz-63) using the Runge-Kutta integration method
def m(x_past,dT,sigma,rho,beta):
    
    # physical parameters
    # dT=0.01
    # sigma=10
    # rho=28
    # beta=8/3
    
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


def make_dataset(nb,n,CI,dT,sigma,rho,beta,name):

    x = np.zeros((n,nb))
    x[:,0] = CI
    for t in range(1,nb):
        x[:,t] = m(x[:,t-1],dT,sigma,rho,beta)
    
    if PATH+f"dataset_{name}_LR63.npy" not in list_of_datasets:
        np.save(PATH+f"dataset_{name}_LR63.npy",x)
    return 0

make_dataset(nb=27000,n=n,CI=np.array([8,0,30]),dT=1e-3,sigma=10,rho=28,beta=8/3,name="long_attractor")
for i in range(3):
    make_dataset(nb=3000,n=n,CI=np.array([0,0,0])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"zero_attractor_{i}")
    make_dataset(nb=3000,n=n,CI=np.array([-np.sqrt(beta*(rho-1)),-np.sqrt(beta*(rho-1)),rho-1])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"equil_{i}")
    make_dataset(nb=3000,n=n,CI=np.array([np.sqrt(beta*(rho-1)),np.sqrt(beta*(rho-1)),rho-1])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"equilibrium_{i}")
for i in range(9):
    make_dataset(nb=9000,n=n,CI=np.array([8,0,30])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"short_attractor_{i}")

# x = np.zeros((n,nb))
# x[:,0] = np.array([8,0,30])

# for t in range(1,nb):
#     x[:,t] = m(x[:,t-1])
easy_dataset = [p for p in list_of_datasets if(('zero' in p) or ('equil' in p)) ]
medium_dataset = [p for p in list_of_datasets if('short' in p)]
hard_dataset = [p for p in list_of_datasets if('long' in p)]

def plot_attractor(p):
    x = np.load(PATH+p)
    xyzs = x.T
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor {p.split('.')[0]}")

    plt.show()

list_of_datasets = glob.glob(PATH+"dataset_*.npy")

for p in list_of_datasets:
    plot_attractor(p)
#################################################
# Dataset LR63 avec BC, IC, Collocation points  #
#################################################




x_bc = x[0,:]
x_ic = x[:,0]



iy_cl = np.random.choice(range(nb), 500)
iz_cl = np.random.choice(range(nb), 500)

# y_cl = np.c_[iy_cl,np.array([1 for _ in range(500)],dtype=np.float32),x[1,iy_cl]]
# z_cl = np.c_[iz_cl,np.array([2 for _ in range(500)],dtype=np.float32),x[2,iz_cl]]
i_cl =np.c_[iy_cl,iz_cl]
x_cl = np.c_[x[1,iy_cl],x[2,iz_cl]]


### dataset mis sous forme de tensor
X_BC = torch.tensor(x_bc,dtype=torch.float32) # x
X_CL = torch.tensor(x_cl,dtype=torch.float32) # y,z point aléatoire
X_IC = torch.tensor(x_ic,dtype=torch.float32) # x(0),y(0),z(0)
X_ALL = torch.tensor(x,dtype=torch.float32) # x,y,z 
### grille 
T = torch.tensor(time,dtype=torch.float32)

# X = torch.stack((T,torch.zeros_like(T,dtype=torch.float32),torch.tensor(x[0,:],dtype=torch.float32)))
# Y = torch.stack((T,torch.ones_like(T,dtype=torch.float32),torch.tensor(x[1,:],dtype=torch.float32)))
# Z = torch.stack((T,torch.ones_like(T,dtype=torch.float32)*2,torch.tensor(x[2,:],dtype=torch.float32)))

# FULL_GRID = torch.cat((X,Y,Z),dim=1).mT


#T_CL = torch.tensor(i_cl,dtype=torch.float32)#torch.stack((torch.tensor(i_cl,dtype=torch.float32),torch.tensor(i_cl,dtype=torch.float32)),dim=0)
# T_IC = torch.stack((torch.zeros(n,dtype=torch.float32),torch.tensor([i for i in range(n)],dtype=torch.float32)))
# POS = torch.zeros_like(T)



#dataset_bc = torch.utils.data.TensorDataset(torch.stack((T,POS,X_BC)).mT)
#dataset_cl = torch.utils.data.TensorDataset(X_CL) # indice dans T_CL ou X_CL donne y=0 ou z=1 


# plt.figure()
# plt.plot(time,x_bc.T)
# plt.plot(i_cl,x_cl.T,'*')
# plt.show()

# plt.figure()
# plt.plot(T,X_BC)
# plt.plot(T_CL.mT,X_CL.mT,'*')
# plt.show()

##############
# Dataloader #
##############

# trainloader_bc = torch.utils.data.DataLoader(dataset_bc, batch_size=128, shuffle=True, drop_last=False)
# trainloader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=128, shuffle=True, drop_last=False)

# for idx,d in enumerate(trainloader_bc):
#     print(type(d))
#     print(d)


###############################
# loss BC, IC ,CL et physique #
###############################

sigma=10
rho=28
beta=8/3

def loss_bc(x,y):
    return torch.mean((x-y)**2)

def loss_ic(x,y):
    return torch.mean((x-y)**2)

def loss_phy(x,y,z,T):
    
    #dxyz = torch.autograd.grad(XYZ,T,torch.ones_like(XYZ),create_graph=True)
    # dx = dxyz[:,0:1]
    # dy = dxyz[:,1:2]
    # dz = dxyz[:,2:3]
    dx = grad(x,T)
    dy = grad(y,T)
    dz = grad(z,T)

    res_x = dx - sigma*(y-x)
    res_y = dy - (x*(rho-z) - y)
    res_z = dz - (x*y - beta*z)
    loss_phy = torch.mean(res_x**2) + torch.mean(res_y**2) + torch.mean(res_z**2)
    #dydt = torch.autograd.grad(x[:,1],grid,torch.ones_like(x),create_graph=True)[0][:,0]
    # LOSS_PHY = torch.zeros_like(x,dtype=torch.float32).to(device)
    # LOSS_PHY[0:nb] = dx[0:nb] - sigma*(x[nb:2*nb] - x[0:nb])
    # LOSS_PHY[nb:2*nb] = dx[nb:2*nb] - (rho*x[0:nb] - x[nb:2*nb] - x[0:nb]*x[2*nb:3*nb])
    # LOSS_PHY[2*nb:3*nb] = dx[2*nb:3*nb] - (x[0:nb]*x[nb:2*nb]- beta*x[2*nb:3*nb])
    # loss = torch.mean(LOSS_PHY**2)
    return loss_phy

# a = loss_bc(X_BC,X_BC)
# print("a")

def grad(y, t):
    return torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),   # sum over N points
        create_graph=True,                 # ← keep graph for higher-order or loss backprop
        retain_graph=True
    )[0]   



#############
#   Train   #
#############

class Train_PINN():

    def __init__(self,learning_rate,nbr_iteration,w_1,w_2,w_3,optim):
        self.learning_rate = learning_rate
        self.nbr_iteration = nbr_iteration
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        
        if type(optim) != None:
            self.optimizer = optim
        else:
            self.optimizer = torch.optim.Adam(model.parameters(),lr = self.learning_rate)
        
    def train(self,*dataset):
        
        loss = np.zeros((self.nbr_iteration,1))
        loss_bc_tracker = np.zeros((self.nbr_iteration,1))
        loss_ic_tracker = np.zeros((self.nbr_iteration,1))
        loss_cl_tracker = np.zeros((self.nbr_iteration,1))
        loss_phy_tracker = np.zeros((self.nbr_iteration,1))

        for iteration in tqdm.tqdm(range(self.nbr_iteration)):
            self.optimizer.zero_grad()

            # #initial conditions Loss
            # initial_train_data = torch.cat((T_IC,X_IC.unsqueeze(0))).mT.to(device)#torch.tensor(initial_train_dataset)
            # u_pd_ini = model(initial_train_data[:, 0:2])
            # u_exa_ini = initial_train_data[:,2:3]
            # loss_initital_conditions = self.w_1*torch.mean((u_pd_ini-u_exa_ini)**2)
           
            #Boundary conditions Loss
            # boundary_train_data = dataset_bc
            # u_pd_bou = model(boundary_train_data[:,0:2][0].to(device))
            # u_exa_bou = boundary_train_data[:,2:3][0].to(device)
            # loss_boundary_conditions = self.w_2*torch.mean((u_pd_bou-u_exa_bou)**2)
            
            t = T.unsqueeze(-1).requires_grad_().to(device)
            u_pd_all = model(t)
            u_exa_all = X_ALL.mT.to(device)

            loss_boundary_conditions = self.w_2*torch.mean((u_pd_all-u_exa_all)**2)


            # Collocation conditions Loss
            # collocation_train_data = dataset_cl
            # u_pd_cl = model(collocation_train_data[:,0:2][0].to(device))
            # u_exa_cl = collocation_train_data[:,2:3][0].to(device)
            # loss_collocation_conditions = self.w_3*torch.mean((u_pd_cl-u_exa_cl)**2)

            #Physical Loss
            # train_data = FULL_GRID[:,0:2].requires_grad_(True).to(device)
            # u_pd = model(train_data)
            #a = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)
            #u_t = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]

            # u_x = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]
            # u_xx = torch.autograd.grad(u_x, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]
            #physics = u_t + u_pd*u_x - nu*u_xx
            u_pd_all.requires_grad_()
            loss_physics = self.w_3*loss_phy(x=u_pd_all[:,0:1],y=u_pd_all[:,1:2],z=u_pd_all[:,2:3],T=t)

            #Total Loss
            total_loss =  loss_boundary_conditions + loss_physics#+ loss_physics #loss_initital_conditions +  loss_collocation_conditions
            total_loss.backward()
            self.optimizer.step()
            
            # Save losses
            loss[iteration]=total_loss.cpu().detach().numpy()
            loss_bc_tracker[iteration] = loss_boundary_conditions.cpu().detach().numpy()
            #loss_cl_tracker[iteration] = loss_collocation_conditions.cpu().detach().numpy()
            loss_phy_tracker[iteration] = loss_physics.cpu().detach().numpy()
            #loss_ic_tracker[iteration] = loss_initital_conditions.cpu().detach().numpy()
        return loss,loss_ic_tracker,loss_bc_tracker,loss_cl_tracker,loss_phy_tracker,self.optimizer.state_dict()


#u_t = torch.autograd.grad(u_pd, grid_train_data, torch.ones_like(u_pd), create_graph=True)[0][:,1:2].to(device)
#############
#   modèle  #
# ###########            

model = architecture.GOY_PINN(1,3,64,4)
model.to(device)

lr = 1.0e-4
nb_iter = 1000
t =Train_PINN(learning_rate=lr,nbr_iteration=nb_iter,w_1=1,w_2=1,w_3=1)
l,li,lb,lc,lp,dict_optim =  t.train()

U_pred = model(T.unsqueeze(-1).to(device))
U = U_pred.view((n,nb)).mT.cpu().detach().numpy()

#################
# loop training #
#################

# séparer les dataset 

easy_dataset = [p for p in list_of_datasets if(('zero' in p) or ('equil' in p)) ]
medium_dataset = [p for p in list_of_datasets if('short' in p)]
hard_dataset = [p for p in list_of_datasets if('long' in p)]

for p in list_of_datasets:
    
    # training on each type of dataset
    #t = Train_PINN(nbr_iteration=nb_iter,optim=)


    torch.save({
            'epoch': nb_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': dict_optim,
            'loss': l,
            
            }, PATH + "model_save")


################
# saving model #
################

torch.save({
            'epoch': nb_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': dict_optim,
            'loss': l,
            
            }, PATH + "model_save")



#############
#   plots   #
#############

# prediction vs realité
plt.figure()
plt.plot(time,x_bc.T)
plt.plot(i_cl.T,x_cl.T,'*')
plt.plot(time,U[:,0],label="x pred")
plt.plot(time,U[:,1],label="y pred")
plt.plot(time,U[:,2],label="z pred")
plt.legend()
plt.savefig(PATH+"prediction")
#plt.show()

# losses
plt.figure()
plt.plot(l,label="Total loss")
plt.plot(li,label="IC loss")
plt.plot(lb,label="BC loss")
plt.plot(lc,label="CL loss")
plt.plot(lp,label="PHY loss")
plt.legend()
plt.savefig(PATH+"losses")
#plt.show()


import os
import numpy as np
from custom_sampler import *
from architecture import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim
import tqdm
import matplotlib.pyplot as plt



class Train_PINN():

    def __init__(self,learning_rate,nbr_iteration,w_1,w_2,w_3,w_4,iteration=False,epoch=1000):
        self.learning_rate = learning_rate
        self.nbr_iteration = nbr_iteration
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.w_4 = w_4
        self.optimizer = torch.optim.Adam(model.parameters(),lr = self.learning_rate)
        self.iteration = iteration
        self.epoch = epoch

    def train(self):

        loss = np.zeros((self.nbr_iteration,1))
        loss_physics_tracker = np.zeros((self.nbr_iteration,1))
        loss_colocation_tracker = np.zeros((self.nbr_iteration,1))
        loss_boundary_conditions_tracker = np.zeros((self.nbr_iteration,1))
        loss_initial_conditions_tracker = np.zeros((self.nbr_iteration,1))
        if self.iteration:
            for iteration in tqdm.tqdm(range(self.nbr_iteration)):
                self.optimizer.zero_grad()

                #initial conditions Loss
                
                initial_train_data = initial_train_dataset.tensor_data.to(device) #torch.tensor(initial_rows, dtype=torch.float32)
                u_pd_ini = model(initial_train_data[:, 0:2]).to(device)
                u_exa_ini = initial_train_data[:,2:3]
                loss_initital_conditions = self.w_1*torch.mean((u_pd_ini-u_exa_ini)**2).to(device)

                #Boundary conditions Loss
                boundary_train_data = boundary_train_dataset.tensor_data_bc.to(device)#torch.tensor(boundary_rows, dtype=torch.float32)
                u_pd_bou = model(boundary_train_data[:, 0:2]).to(device)
                u_exa_bou = boundary_train_data[:,2:3]
                loss_boundary_conditions = self.w_2*torch.mean((u_pd_bou-u_exa_bou)**2).to(device)


                
                colocation_train_data = colocation_dataset.tensor_data_colocation.to(device)
                u_pd_colocation = model(colocation_train_data[:,0:2]).to(device)
                loss_colocation = self.w_3*torch.mean((u_pd_colocation-colocation_train_data[:,2])**2).to(device)

                grid_train_data = grid_dataset.grille.requires_grad_(True).to(device)
                u_pd = model(grid_train_data).to(device)
                #a = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)
                u_t = torch.autograd.grad(u_pd, grid_train_data, torch.ones_like(u_pd), create_graph=True)[0][:,1:2].to(device)
        
                # u_x = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]
                # u_xx = torch.autograd.grad(u_x, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]
                
                u_t_split = torch.split(u_t,int(grid_train_data.shape[0]/(k_max-k_min)))
                tuple_u_t = tuple(k for k in u_t_split)
                u_t = torch.cat(tuple_u_t,1).to(device)
                
                #u_pd = u_pd.view(int(grid_train_data.shape[0]/(k_max-k_min)),int(k_max-k_min))
                u_pd_split = torch.split(u_t,int(grid_train_data.shape[0]/(k_max-k_min)))
                tuple_u_pd = tuple(k for k in u_pd_split)
                u_pd = torch.cat(tuple_u_pd,1).to(device)         

                # on veut calculer la loss physique sur le shells où il y a des collocations points
                GOY_physics = torch.zeros(Npts,(k_max-2 - k_min_collocation)).to(device)
                #print(GOY_physics.shape)
                for i in range (k_min_collocation,k_max-2):
                    GOY_physics[:,i-k_min_collocation] = u_t[:,i] -K[i]*u_pd[:,i+1]*u_pd[:,i+2] +K[i]*eps/lmb*u_pd[:,i-1]*u_pd[:,i+1] + K[i]*((eps-1)/lmb**2)*u_pd[:,i-2]*u_pd[:,i-1] + nu*K[i]*K[i]*u_pd[:,i] # à confirmer
                loss_physics = self.w_4*torch.mean(GOY_physics**2).to(device)

                #Total Loss
                total_loss = loss_initital_conditions + loss_boundary_conditions + loss_physics + loss_colocation
                total_loss.backward()
                self.optimizer.step()
                loss[iteration]=total_loss.cpu().detach().numpy()
                loss_physics_tracker[iteration] = loss_physics.cpu().detach().numpy()
                loss_colocation_tracker[iteration] = loss_colocation.cpu().detach().numpy()
                loss_boundary_conditions_tracker[iteration] = loss_boundary_conditions.cpu().detach().numpy()
                loss_initial_conditions_tracker[iteration] = loss_initital_conditions.cpu().detach().numpy()
        else:
                # regarder comment calculer les loss 
                loss_func = torch.nn.MSELoss()
                for ep in range(0,self.epoch):
                    loss_boundary_conditions = 0
                    loss_initital_conditions = 0
                    loss_colocation = 0
                    loss_physics = 0
                    for idx,ic in enumerate(Dataloader_ic):
                        # ic[0].to(device)
                        # ic[1].to(device)
                        ic_pred = torch.stack((ic[0],ic[1]))
                        ic_pred.requires_grad_()
                        u_pd_ini = model(ic_pred.T.to(device))
                        u_exa_ini = ic[2].to(device)
                        loss_initital_conditions = loss_initital_conditions + loss_func(u_pd_ini,u_exa_ini)
                    
                        
                    for idx,bc in enumerate(Dataloader_bc):
                        # bc[0].to(device)
                        # bc[1].to(device)
                        bc_pred = torch.stack((bc[0],bc[1]))
                        u_pd_bc = model(bc_pred.T.to(device))
                        u_exa_bc = bc[2].to(device)
                        loss_boundary_conditions = loss_boundary_conditions +  loss_func(u_pd_bc,u_exa_bc)

                    for idx,cl in enumerate(Dataloader_cl):
                        cl_pred = torch.stack((cl[0],cl[1]))
                        cl_pred.requires_grad_()
                        u_pd_cl = model(cl_pred.T.to(device))
                        u_exa_cl = cl[2].to(device)
                        loss_colocation = loss_colocation +  loss_func(u_pd_cl,u_exa_cl)

                    for grid in enumerate(Dataloader_grid):
                        loss_physics
        print(iteration)
        print(total_loss)
            
            
        return loss,loss_physics_tracker,loss_colocation_tracker,loss_boundary_conditions_tracker,loss_initial_conditions_tracker




def filter_mode(X,mode_min:int,mode_max:int,t_min:int,ratio:float,seed):
    
    np.random.seed(seed=seed)
    X_subset = np.copy(X)
    X_subset[:,0:mode_min] = None #enlève les modes inférieurs à mode_min
    X_subset[0:t_min,:] = None #enlève la phase de stabilisation
    nb_column = np.shape(X_subset)[1]
    nb_line = np.shape(X_subset)[0]
    X_filtered = np.copy(X_subset)
    X_posx = []
    X_posy = []
    X_value = []
    X_dataset = []

    var_mode = [np.var(X[:,j]) for j in range(nb_column)]
    std_mode = [np.std(X[:,j]) for j in range(nb_column)]
    mean_mode = [np.mean(X[:,j]) for j in range(nb_column)]

    for j in range(mode_min,mode_max):
        for i in range(t_min,nb_line):
            if (np.random.random()<=ratio):
                X_filtered[i,j] = X_subset[i,j]
                X_posx.append(j)
                X_posy.append(i)
                X_value.append((X_subset[i,j]-mean_mode[j])/std_mode[j]) # centré réduit

            else:
                X_filtered[i,j] = None
    
    pourcentage_filtered = nb_line*ratio/nb_line*100
    X_dataset.append(X_posx)
    X_dataset.append(X_posy)
    X_dataset.append(X_value)
    return X_filtered,X_dataset,mean_mode,var_mode,std_mode,pourcentage_filtered


def reduced_center(X,mean,std):
    for k in range(X.shape[1]):
        X[:,k]= (X[:,k]-mean[k])/std[k]
    return X

def test_loss(Data_train,grille_datatset):

    # grid_train_data = grille_datatset.grille # donne la grille des (k,t)
    # grid_train_data.requires_grad = True # permet d'appliqué grad
    data = Data_train[:,k_min_collocation:k_max]
    grille = [t for t in range(Npts)]
    u = torch.tensor(Data_train[:,k_min:k_max],requires_grad=True) #tensor des U(k,t)  #view(Npts,k_max-k_min)
    #tensor_split = torch.split(tensor_data,k_max-k_min,1)
    u_t = np.copy(Data_train[:,k_min_collocation:k_max])
    for k in range(k_min_collocation,k_max):
        for t in range(0,Npts-1):
            u_t[t,k-k_min_collocation] = (data[t+1,k-k_min_collocation]*np.exp(nu*K[k]*dt)-data[t,k-k_min_collocation])/dt


    u_t = torch.tensor(u_t)
    GOY_physics = torch.zeros(Npts,(k_max - k_min_collocation))
 
    
    for i in range (k_min_collocation,k_max-2):
            GOY_physics[:,i-k_min_collocation] = u_t[:,i-k_min_collocation] -K[i]*u[:,i+1]*u[:,i+2] +K[i]*eps/lmb*u[:,i-1]*u[:,i+1] - K[i]*((eps-1)/lmb**2)*u[:,i-2]*u[:,i-1] + nu*K[i]*K[i]*u[:,i] #deltat =1
            plt.figure()
            plt.plot(grille,GOY_physics[:,i-k_min_collocation].detach().numpy(),label=f'Loss shell{i-k_min_collocation}')
            plt.legend()
            plt.savefig(PATH + f"loss_shell_{i-k_min_collocation}")
    print(torch.max(GOY_physics),torch.min(GOY_physics),torch.std_mean(GOY_physics))
    loss_physics = torch.mean(GOY_physics**2)
    return loss_physics,GOY_physics

#########################################################################
#               Paramètres Pinn et entrainement                         #
#########################################################################

##### changer l'ordre  des k ==> k1,t0,k2,t0...kn,t0;k1,t1,k2,t1...kn,t1 etc#########


# check if cuda available ==> log
command = torch.cuda.is_available()
print(f'cuda is available : {command}')

######### The data generated by the shell model #######
PATH =r".\\GOY-main\\"# r"/home/s26calme/Documents/code_stage/Donnees/GOY_modele/Parametre_Ewen/"
path_data = PATH + "data.dat"

data =  np.loadtxt(path_data) # charge le jeu de données
Nmax = np.shape(data)[0] # nombres de pas de temps
debut = int(0.1*Nmax) # skip la phase de stabilisation

Data_shell = data[debut:Nmax,::2] # on garde que la partie réelle de chaque shell
Npts = np.shape(Data_shell)[0] # nombre de pas dans le temps


# retourne un dataset pour plot , var,std,et mean pour chaque mode et les colocation point centré réduit
Data_filtered, Data_train, mean, Var_mode, Std_mode, perc, = filter_mode(Data_shell,4,10,0,0.001,123456)


Data_shell = reduced_center(Data_shell,mean,Std_mode) # centré réduit tous les modes 
Data_ic = Data_shell[0,:] # prends tous le spoints en t=0
Data_bc = Data_shell[:,0:4] # prends tous les points de bords (shell allant de 0->3 avec 3 shell de forçage)

print("Moyenne de l'ensemble des mode",np.mean(Data_shell))
print("std de tous les modes ", np.std(Data_shell))


#########  carateristics of the shell model ############

k0 = 0.125    # largest scale
lmb = 2.0     #ratio between consecutive scales
eps =  0.5     # for the NL coefficients
nu = 1.e-7     # vicosité
nb_shell = 22 
dt = 1
############ parameters for the PINN ############

# nb of shells selected for training the PINN on collocatin point
k_min_collocation = 4 
k_max_collocation = 10 

#nb of shells for training on boundary conditions
k_bc_min = 0
k_bc_max = 4 

# k_min et k_max sur l'ensemble des shells à reconstituter 
k_min = min(k_min_collocation,k_bc_min)
k_max = max(k_bc_max,k_max_collocation)

# list of coefficient kn for all shells
K = [k0*lmb**i for i in range(k_min,k_max)]



############ initialization of model ###########
torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GOY_PINN(n_input=2,n_output=1,n_hidden=20,n_layers=8)
model.to(device)


nbr_initial_t = 1     # Only t=0 for defining the initial condition
t_min = 0    # t initial pour la grille          
t_max = Npts # tmax pour la grille 


############# initialization of dataset for training #################

# sample m points consécutifs à un tps au hasard sur les shells pour calculer loss physic 


initial_train_dataset = initials_variables_data(Data_ic,nbr_initial_t,k_min,k_max)
boundary_train_dataset = boundary_variables_data(X_boundary=Data_bc,Npts=Npts)
colocation_dataset = colocations_variables_data(Data_train)
grid_dataset = grid_data(k_min,k_max,t_min,t_max)


batch_size_bc = int(0.01*Npts)
batch_size_cl = int(0.1*colocation_dataset.nb_colocation_pnt)
batch_size_grid = int(0.01*(k_max*Npts))

sampler_grid = SamplerOverGrid(10,k_min_collocation-2,k_max,Npts=Npts) 
#batch_sampler_grid = BatchSampler(sampler=sampler_grid,batch_size=batch_size_grid,drop_last=False)

Dataloader_ic = DataLoader(initial_train_dataset,batch_size=10,drop_last=False)
Dataloader_bc = DataLoader(boundary_train_dataset,batch_size=batch_size_bc,shuffle=True,drop_last=False)
Dataloader_cl = DataLoader(colocation_dataset,batch_size=batch_size_cl,shuffle=True,drop_last=False)
Dataloader_grid = DataLoader(grid_dataset,batch_sampler=sampler_grid)


# for idx,data in enumerate(Dataloader_grid):
#     print(data)
#     print(idx)



# boundary_train_dataset.tensor_data_bc.to(device)
# colocation_dataset.tensor_data_colocation.to(device)
# grid_dataset.grille.to(device)



#test_loss(Data_train=Data_shell,grille_datatset=grid_dataset)



learning_rate,nbr_iteration,w_1,w_2,w3,w_4 = 0.001,1000,1,1,1,1
t = Train_PINN(learning_rate,nbr_iteration,w_1,w_2,w3,w_4)
Total_loss = t.train()
model.eval().to(device)

U = model(grid_dataset.grille.to(device))
U_split = torch.split(U,Npts)
U = torch.cat(tuple(k for k in U_split),1).to(device)
U = U.cpu().detach().numpy()#.detach().numpy().reshape(t_max-t_min,k_max-k_min)
U_exa = Data_shell[t_min:t_max,k_min:k_max]

square_error = (U-U_exa)**2
rmse = np.sqrt(np.mean(square_error))
print("RMSE:", rmse)



for i in range(U.shape[1]):
    plt.figure()
    plt.plot(U[:,i],label=f'Predicted u{i}')
    plt.plot(U_exa[:,i],label=f'Exact u{i}')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.savefig(PATH + f"prediction_u{i}.png")



#torch.save(model.state_dict(),'/Odyssey/private/s26calme/code_stage/')


plt.figure()
plt.plot(Total_loss[0],label='Total Loss')
plt.plot(Total_loss[1],label='Physics Loss')
plt.plot(Total_loss[2],label='Colocation Loss')
plt.plot(Total_loss[3],label='Boundary Conditions Loss')
plt.plot(Total_loss[4],label='initial Conditions Loss' )  
plt.xlabel('Iterations')
plt.ylabel('Losses')
plt.legend()
plt.savefig(PATH + "losses.png")

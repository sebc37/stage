import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim
import numpy as np




class GOY_PINN(nn.Module):

    def __init__(self,n_input,n_output,n_hidden,n_layers):
        super().__init__()

        activation = nn.Tanh
        self.input_layer = nn.Sequential(*[
                                    nn.Linear(n_input, n_hidden),
                                    activation()])#.to(device)

        self.hidden_layers = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(n_hidden, n_hidden),
                            activation()]) for _ in range(n_layers-1)])#.to(device)

        self.output_layer = nn.Linear(n_hidden, n_output)#.to(device)

    def forward(self,x):

        x=self.input_layer(x)
        x=self.hidden_layers(x)
        x=self.output_layer(x)

        return x
    
################################################# DATASET ###############################################



# classe pour transformer les données en jeu de donnée des condittions initiales 
class initials_variables_data(Dataset):
    def __init__(self,X_ic,nbr_initial_t,k_min,k_max): # conditions initiales à t=0 et k sur l'ensemble des k
        
        #initialise the variables
        self.k_min = k_min
        self.k_max = k_max
        self.X_ic = X_ic[self.k_min:self.k_max*2]  # select only the shells between k_min and k_max
        self.nbr_initial_t = nbr_initial_t
        
        self.x_initial = np.array([k for k in range(k_min,k_max*2)],dtype="float32") # shells selected for initial conditions
        self.t_initial = np.array([0 for _ in range(self.nbr_initial_t)],dtype="float32")  # 0 since the initial condition is defined for t=0
        
        # transform data and grid to the shape (k,t,u)
        self.tensor_data_ic = torch.tensor(self.X_ic, dtype=torch.float32)
        self.grille = np.meshgrid(self.x_initial,self.t_initial)
        self.grille = torch.tensor(self.grille, dtype=torch.float32).T.view(np.shape(self.x_initial)[0]*np.shape(self.t_initial)[0],2)
        
        self.tensor_data = torch.ones((np.shape(self.x_initial)[0]*np.shape(self.t_initial)[0],3), dtype=torch.float32)
        self.tensor_data[:,0:2] = self.grille
        self.tensor_data[:,2] = self.tensor_data_ic
        print(f'shape tensor for initial conditions : {self.tensor_data.shape}')
    
    def __len__(self):
        #return the lenght of the dataset
        return self.x_initial.shape[0]*self.t_initial.shape[0]

    def __getitem__(self,idx):

        return self.tensor_data[idx,0], self.tensor_data[idx,1], self.tensor_data[idx,2]  #return the x and t values of the grid and value of X for the initial condition
    

# classe pour transformer les données en jeu de donnée des conditions de bord 
class boundary_variables_data(Dataset):
    def __init__(self,X_boundary,Npts,time,f,dt):
        
        
        self.X_boundary = X_boundary[0:Npts,:]
        self.nb_k = np.shape(X_boundary)[1]
        self.nb_t = np.shape(X_boundary)[0]
        self.tensor_data_bc = torch.ones((self.nb_k*self.nb_t,3), dtype=torch.float32) #columns: k, t, u
        #X_test = torch.tensor(X_boundary,dtype=torch.float32).T.contiguous().view(self.nb_k*self.nb_t,1)
        # trouver solution tq pour tout t, t!=0
        time = torch.arange(0.1*time,time,10*f*dt,dtype=torch.float32)
        shell = torch.arange(0,self.X_boundary.shape[1],1,dtype=torch.float32)
        grid_shell,grid_time = torch.meshgrid(shell,time,indexing="xy")
        grid_shell = grid_shell.T.contiguous().view(Npts*self.X_boundary.shape[1],1)
        grid_time = grid_time.T.contiguous().view(Npts*self.X_boundary.shape[1],1)
        u_bc = torch.tensor(X_boundary).T.contiguous().view(Npts*self.X_boundary.shape[1],1)
        self.tensor_data_bc_bis = torch.stack((grid_shell,grid_time,u_bc),1).view(Npts*self.X_boundary.shape[1],3)
        for k in range(self.nb_k*self.nb_t): # data ordered as (k,t,u) in the grid
            self.tensor_data_bc[k,0],self.tensor_data_bc[k,1],self.tensor_data_bc[k,2] = k//(self.nb_t),k%(self.nb_t),self.X_boundary[k%self.nb_t,k//self.nb_t]
        print(f'shape tensor for boundary conditions: {self.tensor_data_bc.shape}')
        t = torch.sub(self.tensor_data_bc_bis[:,2],self.tensor_data_bc[:,2])
        print(t)
        # for k in range(self.nb_k):
        #     print(self.tensor_data_bc[k*(self.nb_t-3):k*(self.nb_t+3)]) 
        #print(X_test)
        
    def __len__(self):
        #return the lenght of the dataset
        return self.nb_k*self.nb_t
    def __getitem__(self,idx):

        # return the element in that index (k,t,u)[idx] 
        return self.tensor_data_bc[idx,0],self.tensor_data_bc[idx,1],self.tensor_data_bc[idx,2]
    

# classe pour transformer les données en jeu de donnée des collocations points
class colocations_variables_data(Dataset):
    def __init__(self,X_Data_train):
        #initialise the variables

        self.X_Data_train = X_Data_train
        self.nb_colocation_pnt = len(self.X_Data_train[1])
        #colocation points aranged as (k,t,u) in the grid
        self.tensor_data_colocation = torch.tensor(self.X_Data_train, dtype=torch.float32).T.view(self.nb_colocation_pnt,3)
        print(f'shape of tensor for collocation points :  {self.tensor_data_colocation.shape}')


    def __len__(self):
        #return the lenght of the dataset
        return self.nb_colocation_pnt
   

    def __getitem__(self,idx):

        # return the element in that index (k,t,u)[idx]
        return self.tensor_data_colocation[idx,0],self.tensor_data_colocation[idx,1],self.tensor_data_colocation[idx,2] #float(self.X_Data_train[idx][0]),float(self.X_Data_train[idx][1]) ,torch.tensor(self.X_Data_train[idx][2])   #return the x and t values of the grid and value of X for the physics loss
        #return self.x[j],self.t[i]  # This class only returns the x and t values of the grid not the velocity


class grid_data(Dataset): # créer la grille sur laquelle on veut inferer U(k,t) sous la forme (k,t)
    def __init__(self,k_min,k_max,t_min,t_max): # 
        #initialise the variables
        self.k_min = k_min
        self.k_max = k_max
        self.t_min = t_min
        self.t_max = t_max
        self.x = np.array([k for k in range(k_min,2*k_max)],dtype="float32")
        self.t = np.array([t for t in range(t_min,t_max)],dtype="float32")
        self.x, self.t = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.t, dtype=torch.float32)
        self.grille = torch.ones((2*k_max-k_min)*(t_max-t_min),2, dtype=torch.float32)
        for k in range((2*k_max-k_min)*(t_max-t_min)):
            i = k%(t_max-t_min)
            j = k//(t_max-t_min)
            self.grille[k,0],self.grille[k,1] = self.x[j],self.t[i]

        self.N_k = np.shape(self.x)[0] 
        self.N_t = np.shape(self.t)[0]
        print(f'shape tensor of the grid : {self.grille.shape}')

    def __len__(self):
        #return the lenght of the dataset
        return self.N_t*self.N_k

    def __getitem__(self,idx):
        print(f'idx {idx}')
        return self.grille[idx,0],self.grille[idx,1],idx  # This class only returns the x and t values of the grid not the velocity

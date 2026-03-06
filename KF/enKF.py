import numpy as np


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
                X_value.append((X_subset[i,j]-mean_mode[j])/std_mode[j] + np.random.random()) # centré réduit

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


PATH = "/home/s26calme/Documents/code_stage/GOY-main/"
path_data = PATH + "data.dat"

data =  np.loadtxt(path_data,dtype=np.float32) # charge le jeu de données
Nmax = np.shape(data)[0] # nombres de pas de temps
debut = int(0.1*Nmax) # skip la phase de stabilisation

Data_shell = data[debut:Nmax,:] # on garde  partie réelle de chaque shell
Npts = np.shape(Data_shell)[0] # nombre de pas dans le temps

# nb of shells selected for training the PINN on collocatin point
k_min_collocation = 4 
k_max_collocation = 10 

#nb of shells for training on boundary conditions
k_bc_min = 0
k_bc_max = 4 



# retourne un dataset pour plot , var,std,et mean pour chaque mode et les colocation point centré réduit
Data_filtered, Data_train, mean, Var_mode, Std_mode, perc, = filter_mode(Data_shell,2*k_min_collocation,2*k_max_collocation,0,0.001,123456)


Data_shell = reduced_center(Data_shell,mean,Std_mode)


### parameters
n     = 20 # state size  on veut estimer les Un de 1 à 10 avec Re et Im donc 20 variables d'état
p     = 12 # On observe Un n=5,6,7,8,9,10 avec Re et Im donc 12 variables d'observations 
nb    = Npts # number of times
time  = np.array(range(nb)) # time vector
var_Q = 0.1 # error variance of the model (in Kalman)
var_R = 1 # error variance of the observations (in Kalman)
x_0   = np.zeros((n)) # initial coundition (mean)
P_0   = np.eye(n,n) # initial coundition (covariance)

### variables
Q      = var_Q*np.eye(n,n)
R      = var_R*np.eye(p,p)
# x1     = sqrt(time)*cos(2*pi*time/(nb/2)) # true x1 position
# x2     = sqrt(time)*sin(2*pi*time/(nb/2)) # true x2 position
# x1_dot = x1[time[1:-1]]-x1[time[0:-2]] # true x1 speed
# x2_dot = x2[time[1:-1]]-x2[time[0:-2]] # true x2 speed
# x1_dot = r_[x1_dot[0], x1_dot, x1_dot[-1]]
# x2_dot = r_[x2_dot[0], x2_dot, x2_dot[-1]]

# ### true state and noisy observations
# x = c_[x1, x2, x1_dot, x2_dot].T # true state
# y = c_[x1, x2].T + randn(p,nb) # noisy observations


### nonlinear and linear operators of the state-space model
def m(x_past):
    x_future = x_past
    x_future[0] = x_past[0]+x_past[2] ### A CACHER
    x_future[1] = x_past[1]+x_past[3] ### A CACHER
    x_future[2] = x_past[2] ### A CACHER
    x_future[3] = x_past[3] ### A CACHER
    return x_future
H = array([[1,0,0,0], [0,1,0,0]])

### Ensemble Kalman initialization
Ne = 100                   # number of ensembles
x_f_enkf = zeros((n,nb))   # forecast state
P_f_enkf = zeros((n,n,nb)) # forecast error covariance matrix
x_a_enkf = zeros((n,nb))   # analysed state
P_a_enkf = zeros((n,n,nb)) # analysed error covariance matrix

### Ensemble Kalman filter
x_a_enkf_tmp = zeros((n,Ne)) 
x_f_enkf_tmp = zeros((n,Ne))
y_f_enkf_tmp = zeros((p,Ne))
# initial step
for i in range(Ne):
    x_a_enkf_tmp[:,i] = random.multivariate_normal(x_0, P_0)
x_a_enkf[:,0]   = mean(x_a_enkf_tmp,1) # initial state
P_a_enkf[:,:,0] = cov(x_a_enkf_tmp)    # initial state covariance
for k in range(1,nb): # forward in time
    # prediction step
    for i in range(Ne):
        x_f_enkf_tmp[:,i] = m(x_a_enkf_tmp[:,i]) + random.multivariate_normal(zeros(n), Q) ### A CACHER
        y_f_enkf_tmp[:,i] = H @ x_f_enkf_tmp[:,i] + random.multivariate_normal(zeros(p), R) ### A CACHER
    P_f_enkf_tmp = cov(x_f_enkf_tmp) ### A CACHER
    # Kalman gain
    K = P_f_enkf_tmp @ H.T @ inv(H @ P_f_enkf_tmp @ H.T + R) ### A CACHER
    # update step
    for i in range(Ne):
        x_a_enkf_tmp[:,i] = x_f_enkf_tmp[:,i] + K @ (y[:,k] - y_f_enkf_tmp[:,i]) ### A CACHER
    P_a_enkf_tmp = cov(x_a_enkf_tmp) ### A CACHER
    # store results
    x_f_enkf[:,k]   = mean(x_f_enkf_tmp,1)
    P_f_enkf[:,:,k] = P_f_enkf_tmp
    x_a_enkf[:,k]   = mean(x_a_enkf_tmp,1)
    P_a_enkf[:,:,k] = P_a_enkf_tmp
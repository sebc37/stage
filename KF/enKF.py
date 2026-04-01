import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tqdm
from goy import GoyModel



# plt.plot(time,X2[:,0], 'gray',label='Mode 1',alpha=0.5)
# plt.plot(time,X2[:,1], 'gray',label='Mode 2',alpha=0.5)
# plt.plot(time,X2[:,2], 'gray',label='Mode 3',alpha=0.5)
# plt.plot(time,X2[:,3], 'black',label='Mode 4 Forcing',alpha=1)
# plt.plot(time,X2[:,5], 'gray',label='Mode 6',alpha=0.5)
# plt.plot(time,X2[:,7], 'gray',label='Mode 8',alpha=0.5)
# plt.plot(time,X2[:,9], 'gray',label='Mode 10',alpha=0.5)
# plt.savefig(r"/home/s26calme/Documents/code_stage/KF/c_encapsulate")


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
                X_filtered[i,j] =  X_subset[i,j] #+ np.random.normal(0,1) 
                X_posx.append(j)
                X_posy.append(i)
                X_value.append(X_subset[i,j])#-mean_mode[j])/std_mode[j]) # centré réduit

            else:
                X_filtered[i,j] = None
    
    X_filtered[:,2*k_max_collocation:] = None
    # Y = np.zeros((nb_line,nb_column))
    # Y[:,:] = None
    # Y[X_posy,X_posx] =  X_value #X_filtered[X_posy,X_posx] # on ajoute du bruit gaussien aux observations

    pourcentage_filtered = nb_line*ratio/nb_line*100
    X_dataset.append(X_posx)
    X_dataset.append(X_posy)
    X_dataset.append(X_value)


    return X_filtered,X_dataset,mean_mode,var_mode,std_mode,pourcentage_filtered#,Y


def reduced_center(X,mean,std):
    for k in range(X.shape[1]):
        X[:,k]= (X[:,k]-mean[k])/std[k]
    return X

# etape 1
# filter doit sortir une matrice des Uk,t donc transposé de ce qu'il il ya maintement et de meme taille que data_shell avec des nan pour les valeurs non sélectionnées

# etape 2 
# écrire la matrice H pour les observations on prend les modes 5,6,7,8,9,10

#etape 3 
# écrire la fonction m qui met a jour le modèle dynamique des shells avec equatioons différentes pour les modes 1,2
# et les modes 9 et 10. le schéma d'intégration doit être Adam bashforth 2

# étape 4
# implémenter l'enKF avec les fonctions m et H écrites précédement et tester l'enKF
# utiliser l'algo d'optimisation de la variance pour trouver les meilleurs var_Q et var_R
# comparer réultats des enKF 



PATH = "/home/s26calme/Documents/code_stage/GOY-main/"
path_data = PATH + "data_test.dat"
SAVE = "/home/s26calme/Documents/code_stage/KF/"

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

k0 = 0.125
lmb = 2.0
# retourne un dataset pour plot , var,std,et mean pour chaque mode et les colocation point centré réduit
Data_filtered, Data_train, mean, Var_mode, Std_mode, perc= filter_mode(Data_shell,2*k_min_collocation,2*k_max_collocation,0,0.001,123456)
#Data_shell = reduced_center(Data_shell,mean=mean,std=Std_mode)
#Data_filtered = reduced_center(Data_filtered,mean=mean,std=Std_mode) # données réelles

K = np.array([k0*lmb**i for i in range(22)],dtype=np.float32)



# for i in range(np.shape(y_obs)[0]):
#     plt.figure()
#     plt.plot(y_obs[i,:],'*')
#     plt.plot(Data_shell[:,i],'gray',alpha=0.5)
#     plt.savefig(PATH + "ploty",dpi=300)
shell_array = np.array(Data_shell)
MS = np.array([(0.05**2)*np.mean(shell_array[:,k]**2) for k in range(44)])


### parameters
n     = 44 # state size  on veut estimer les Un de 1 à 10 avec Re et Im donc 20 variables d'état
p     = 12 # On observe Un n=5,6,7,8,9,10 avec Re et Im donc 12 variables d'observations 
nb    = Npts # number of times
time  = np.array(range(nb)) # time vector
var_Q = 1.0e-10 # error variance of the model (in Kalman)
var_R = 0.1 # error variance of the observations (in Kalman)
x_0   = np.zeros((n)) # initial coundition (mean)
P_0   = np.eye(n,n)*1.e-12 # initial coundition (covariance)


### variables

m = MS[2*4:2*10]
R = np.eye(p,p)
Q      = var_Q*np.eye(n,n)
for i in range(p):
    for j in range(p):
        if i==j:
            R[i,j] = m[i]
#R      = np.fill_diagonal(R,list(m[:]))#var_R*np.eye(p,p)

##############  noisy observations ##################
y_obs = Data_shell.T  
y_obs = y_obs[2*k_min_collocation:2*k_max_collocation,:]
for t in range(Npts):
    y_obs[:,t]  = y_obs[:,t] + np.random.multivariate_normal(np.zeros(p),R)


# ### true state and noisy observations
# x = c_[x1, x2, x1_dot, x2_dot].T # true state
# y = c_[x1, x2].T + randn(p,nb) # noisy observations


### nonlinear and linear operators of the state-space model
def NL(x_past_real,x_past_imag):
    
    eps = 0.5
    lmb = 2.0
    n = np.shape(x_past_real)[0]
    NL_re = np.zeros(n)
    NL_im = np.zeros(n)    
    
    NL_re[0] = x_past_real[2]*x_past_imag[1] + x_past_imag[2]*x_past_real[1]
    NL_re[1] = x_past_real[3]*x_past_imag[2] + x_past_imag[3]*x_past_real[2] - (eps/lmb)*K[1]*(x_past_real[0]*x_past_imag[2] + x_past_imag[0]*x_past_real[2])

    NL_im[0] = x_past_real[2]*x_past_real[1] - x_past_imag[2]*x_past_imag[1]
    NL_im[1] = x_past_real[3]*x_past_real[2] - x_past_imag[3]*x_past_imag[2] - (eps/lmb)*K[1]*(x_past_real[0]*x_past_real[2] - x_past_imag[0]*x_past_imag[2])

    for i in range(2,n-2):
        NL_im[i]  =  K[i]*(x_past_real[i+1]*x_past_imag[i+2] + x_past_imag[i+1]*x_past_real[i+2]) 

        -K[i]*(eps/lmb)*(x_past_real[i-1]*x_past_imag[i+1] + x_past_imag[i-1]*x_past_real[i+1]) 
                
        + K[i]*((eps-1)/lmb**2)*(x_past_real[i-2]*x_past_imag[i-1] + x_past_imag[i-2]*x_past_real[i-1])

        

        NL_re[i] = K[i]*(x_past_real[i+1]*x_past_real[i+2] - x_past_imag[i+1]*x_past_imag[i+2]) 
        
        -K[i]*(eps/lmb)*(x_past_real[i-1]*x_past_real[i+1] - x_past_imag[i-1]*x_past_imag[i+1]) 
            
        + K[i]*((eps-1)/lmb**2)*(x_past_real[i-2]*x_past_real[i-1] - x_past_imag[i-2]*x_past_imag[i-1]) 

    NL_re[n-2] = -(eps/lmb)*K[n-2]*(x_past_real[n-3]*x_past_imag[n-1] + x_past_imag[n-3]*x_past_real[n-1])  
    + K[n-2]*((eps-1)/lmb**2)*(x_past_real[n-4]*x_past_imag[n-3] + x_past_imag[n-4]*x_past_real[n-3])
    
    NL_re[n-1] = K[n-1]*((eps-1)/lmb**2)*(x_past_real[n-3]*x_past_imag[n-2] + x_past_imag[n-3]*x_past_real[n-2])
    
    NL_im[n-2] = -(eps/lmb)*K[n-2]*(x_past_real[n-3]*x_past_real[n-1] + x_past_imag[n-3]*x_past_imag[n-1])
    + K[n-2]*((eps-1)/lmb**2)*(x_past_real[n-4]*x_past_real[n-3] + x_past_imag[n-4]*x_past_imag[n-3])

    NL_im[n-1] = K[n-1]*((eps-1)/lmb**2)*(x_past_real[n-3]*x_past_real[n-2] - x_past_imag[n-3]*x_past_imag[n-2])


    return NL_re, NL_im


TIME      = 1000.
DT        = 1e-5
FS        = 100.
FORCE     = 0.005
N_FORCE   = 4
FORCE_RND = 0
N_fs      = int(1.0 / DT / FS)  
model = GoyModel(dt=DT, force=FORCE, N_force=N_FORCE, force_rnd=FORCE_RND)
N = model.N  # 22
nu =1.0e-7



def m_b(x_past,N_fs,n_steps_first,start=False,second = False,custom=False):
    # ── choix du point de départ ──────────────────────────────────────────────────
    #i      = 10000   # ligne du fichier depuis laquelle on repart
     # nombre de lignes suivantes à reproduire

     # pas entre deux lignes du fichier
     # pas spéciaux pour la ligne 0 (voir run_goy.py)

    # ── reconstruction de Xpp (état à t_i - dt) ──────────────────────────────────
    # On part de la ligne i-1 et on intègre 998 pas → on arrive à t_i - dt
    if start:
        # cas particulier : ligne 0, on repart des CI
        Xpp0, Ypp0, Xp0, Yp0 = model.init_fields()
        (Xpp, Ypp), (Xp, Yp) = model.integrate(Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first - 1)
    else:
        # ligne i-1 → intègre N_fs-1 pas → arrive à t_i - dt
        if custom:
            Xpp0,Ypp0,Xp0,Yp0 = model.init_fields(Xpp=x_past[0::2],Ypp=x_past[1::2])
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first)
            return cur_Xpp,cur_Ypp,cur_Xp,cur_Yp
        if second:
            Xpp0, Ypp0, Xp0, Yp0 = model.init_fields()
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first)
        else:
            Xp_prev2 = x_past[0, 0::2];  Yp_prev2 = x_past[0, 1::2]
            Xp_prev1 = x_past[1, 0::2];  Yp_prev1 = x_past[1, 1::2]
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xp_prev2, Yp_prev2, Xp_prev1, Yp_prev1, n_steps=N_fs - 1)

        (Xpp, Ypp), (Xp, Yp) = model.integrate(
            cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=1)
        # maintenant Xp/Yp = ref[i] à la précision machine, Xpp/Ypp = état à t_i - dt
    return Xpp,Ypp,Xp,Yp

def m_step(Xpp, Ypp, Xp, Yp):
    """
    Intègre le modèle GOY de N_fs pas.
    Entrée  : Xpp,Ypp (t-dt)  Xp,Yp (t)
    Sortie  : Xpp_new,Ypp_new (t+N_fs*dt - dt)  Xp_new,Yp_new (t+N_fs*dt)
    """
    (Xpp_new, Ypp_new), (Xp_new, Yp_new) = model.integrate(
        Xpp, Ypp, Xp, Yp, n_steps=N_fs)
    return Xpp_new, Ypp_new, Xp_new, Yp_new

j_ = 50
nb_iter = 3
count_init   = int(TIME / DT)  
n_steps_first = count_init % N_fs  
series = np.zeros((n,nb_iter)).T
series[0,:] = Data_shell[j_,:]
series[1,:] = Data_shell[j_+1,:]#x_past[0,:]



# for k in range(2,nb_iter):
#     x_past = series[k-1:k,:].copy()
    
#     _,_,cur_Xp,cur_Yp = m_b(x_past,N_fs=999,n_steps_first=99,custom=True)
#     series[k, 0::2] = cur_Xp.copy()
#     series[k, 1::2] = cur_Yp.copy()
#     # print("serie:",series[k,0:2])

# Rmse = np.sqrt(np.mean(Data_shell[j_:j_+nb_iter,:]-series)**2)

# plt.figure()

# for i in range(5):
#      #plt.plot(np.abs(series[:,2*i]-Data_shell[:,2*i]))
#      plt.plot(series[:,2*i],label=f"shell_encaps_{2*i}")
#      plt.plot(Data_shell[j_:j_+nb_iter,2*i],label=f"shell_reel_{2*i}")
#      plt.legend()
#      plt.show()
# plt.savefig("test")


# plt.figure()
# for i in range(10):
#      plt.plot(np.abs(base[2*i,:]-data[2*i,0:9999]))
# plt.savefig("test_2")

### Generate observations and covariance
def generate_observations(p, H):
    y = np.zeros((p,nb))
    R = var_R*np.eye(p,p) # observation covariance
    for t in range(1,nb):
        y[:,t] = H @ y_obs[:,t] + np.random.multivariate_normal(np.zeros(p), R) # noisy observations
    #y[:,i_nan] = y[:,i_nan]*np.nan # remove observations 
    return y, R

H = np.eye(44,44) #array([[1,0,0,0], [0,1,0,0]])
H = H[2*k_min_collocation:2*k_max_collocation,:] # on observe que les modes de 5 à 10 avec Re et Im donc 12 variables d'observations



### Ensemble Kalman initialization
Ne = 50                   # number of ensembles
x_f_enkf = np.zeros((n,nb))   # forecast state
P_f_enkf = np.zeros((n,n,nb)) # forecast error covariance matrix
x_a_enkf = np.zeros((n,nb))   # analysed state
P_a_enkf = np.zeros((n,n,nb)) # analysed error covariance matrix

### Ensemble Kalman filter
x_a_enkf_tmp = np.zeros((n,Ne)) # shell,t-2 t-1, Ne
x_f_enkf_tmp = np.zeros((n,Ne))
y_f_enkf_tmp = np.zeros((p,Ne))
# initial step

ens_Xpp = np.zeros((Ne, n))
ens_Ypp = np.zeros((Ne, n))
ens_Xp  = np.zeros((Ne, n))
ens_Yp  = np.zeros((Ne, n))


fens_Xpp = np.zeros((Ne, n))
fens_Ypp = np.zeros((Ne, n))
fens_Xp  = np.zeros((Ne, n))
fens_Yp  = np.zeros((Ne, n))


# condition initiales
j_start = 2

(cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
    Data_shell[j_start-2, 0::2], Data_shell[j_start-2, 1::2],
    Data_shell[j_start-1, 0::2], Data_shell[j_start-1, 1::2],
    n_steps=N_fs - 1)
(ref_Xpp, ref_Ypp), (ref_Xp, ref_Yp) = model.integrate(
    cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=1)


amp = np.std(Data_shell, axis=0)

nb = 44
for i in range(Ne):
    x_a_enkf_tmp[:,i] = np.random.multivariate_normal(x_0, P_0)
    
    noise = np.random.randn(n) * amp * 0.01   # perturbation 1%
    ens_Xpp[i] = ref_Xpp + noise[0::2] * 0.1  # Xpp varie peu
    ens_Ypp[i] = ref_Ypp + noise[1::2] * 0.1
    ens_Xp[i]  = ref_Xp  + noise[0::2]
    ens_Yp[i]  = ref_Yp  + noise[1::2]

x_a_enkf[:,0]   = np.mean(x_a_enkf_tmp,1) # initial state
P_a_enkf[:,:,0] = np.cov(x_a_enkf_tmp)    # initial state covariance

for k in tqdm.tqdm(range(nb)): # forward in time #nb
    # prediction step
    # il faut un initialisation custom pour chaque Ne
    for i in range(Ne):
        Xpp_n, Ypp_n, Xp_n, Yp_n = m_step(
                ens_Xpp[i], ens_Ypp[i], ens_Xp[i], ens_Yp[i])

        fens_Xpp[i] = Xpp_n
        fens_Ypp[i] = Ypp_n
        fens_Xp[i]  = Xp_n
        fens_Yp[i]  = Yp_n

        x_f_enkf_tmp[0::2,i] = Xp_n.T
        x_f_enkf_tmp[1::2,i] = Yp_n.T
        _,_,forward_x,forward_y  = m_b(x_a_enkf_tmp[:,i].T,N_fs=999,n_steps_first=998,custom=True)
        x_f_enkf_tmp[0::2,i],x_f_enkf_tmp[1::2,i] = forward_x.T,forward_y.T ### A CACHER
        x_f_enkf_tmp[:,i] += np.random.multivariate_normal(np.zeros(n), Q)
        y_f_enkf_tmp[:,i] = H @ x_f_enkf_tmp[:,i] + np.random.multivariate_normal(np.zeros(p), R) ### A CACHER
    
    P_f_enkf_tmp = np.cov(x_f_enkf_tmp) ### A CACHER
    # Kalman gain
    
    K_g = P_f_enkf_tmp @ H.T @ np.linalg.inv(H @ P_f_enkf_tmp @ H.T + R) ### A CACHER
    # update step
    if(sum(np.isfinite(y_obs[:,k]))>0):
        for i in range(Ne):
            x_a_enkf_tmp[:,i] = x_f_enkf_tmp[:,i] + K_g @ (y_obs[:,k] - y_f_enkf_tmp[:,i]) ### A CACHER
        P_a_enkf_tmp = np.cov(x_a_enkf_tmp) ### A CACHER
    else:
            #x_a_enkf_tmp[:,:,0] = x_a_enkf_tmp[:,:,1]
            x_a_enkf_tmp = x_f_enkf_tmp
            P_a_enkf_tmp = P_f_enkf_tmp 
    # store results
    x_f_enkf[:,k]   = np.mean(x_f_enkf_tmp,1)
    P_f_enkf[:,:,k] = P_f_enkf_tmp
    x_a_enkf[:,k]   = np.mean(x_a_enkf_tmp,1)
    P_a_enkf[:,:,k] = P_a_enkf_tmp


### plot trajectories (true, observed, KF, EnKF)
plt.figure()
plt.plot(Data_shell.T[8,0:nb], 'b', label='True state ($x$)')
plt.plot(y_obs[8,0:nb], '.k', label='Observations ($y$)')
plt.plot(x_a_enkf[8,0:nb], 'r', label='EnKF ($x^a$)')
plt.xlabel('$time$', fontsize=20)
plt.ylabel('$U_8', fontsize=20)
plt.legend(fontsize=20)
plt.savefig(SAVE + "fig1enKF")
### plot state variables
plt.figure()
y_label=('$U_4$', '$U_5$', '$U_6$', '$U_7$')
for i in range(4,8):
    plt.subplot(2,2,i-4+1)
    plt.plot(time[0:nb], Data_shell.T[2*i,0:nb], 'b')
    if ((i==1) or (i==2)):
        plt.plot(time[0:nb], y_obs[2*i,0:nb], '.k') 
    plt.plot(time[0:nb], x_a_enkf[2*i,0:nb], 'r')
    plt.fill_between(time[0:nb], x_a_enkf[i,0:nb] - 1.96*np.sqrt(P_a_enkf[i,i,0:nb]), x_a_enkf[i,0:nb] + 1.96*np.sqrt(P_a_enkf[i,i,0:nb]), facecolor='red', alpha=0.5)
    plt.xlabel('Time', size=20)
    plt.ylabel(y_label[i-4], size=20)
plt.savefig(SAVE + "fig2enKF")
### compute Root Mean Squared Errors (RMSE) of the positions
print('RMSE(obs):', np.sqrt(np.mean((y_obs[range(4,9),0:nb] - Data_shell.T[range(4,9),0:nb])**2))) ### A CACHER
print('RMSE(EnKF):', np.sqrt(np.mean((x_a_enkf[range(4,9),0:nb] - Data_shell.T[range(4,9),0:nb])**2))) ### A CACHER

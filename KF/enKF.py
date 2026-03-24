import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tqdm
from goy import GoyModel




# # --- Run 2 : init_custom avec les mêmes CI ---
# X0 = sh**(-1/3)
# Y0 = np.ones(sim.N) * 1e-4
# sim.init_custom(X0, Y0, dt=1e-5, force=0.005, force_rnd=True)
# X2, Y2, t = sim.run(T=1.0, save_every=100)

# Comparaison
# print("Max diff X :", np.max(np.abs(X1 - X2)))
# print("Max diff Y :", np.max(np.abs(Y1 - Y2)))

# def plot_im_xy(X,Y,t):
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#     # --- X ---
#     im1 = axes[0].imshow(np.abs(X).T , norm=LogNorm(), cmap='inferno', aspect='auto', origin='lower',
#                         extent=[t[0], t[-1], 0, sim.N-1],
#                         )
#     axes[0].set_xlabel("Temps")
#     axes[0].set_ylabel("Shell n°")
#     axes[0].set_title("X (partie réelle)")
#     plt.colorbar(im1, ax=axes[0], label="Amplitude")

#     # --- Y ---
#     im2 = axes[1].imshow(np.abs(Y).T , norm=LogNorm(), cmap='inferno', aspect='auto', origin='lower',
#                         extent=[t[0], t[-1], 0, sim.N-1],
#                         )
#     axes[1].set_xlabel("Temps")
#     axes[1].set_ylabel("Shell n°")
#     axes[1].set_title("Y (partie imaginaire)")
#     plt.colorbar(im2, ax=axes[1], label="Amplitude")

#     plt.tight_layout()
#     plt.savefig("XY_intensite.png", dpi=150)
#     plt.show()
#     E = X**2 + Y**2   # shape (n_snapshots, N)

#     fig, ax = plt.subplots(figsize=(10, 5))
#     im = ax.imshow(E.T, aspect='auto', origin='lower',
#                 extent=[t[0], t[-1], 0, sim.N-1],
#                 norm=LogNorm(), cmap='inferno')
#     ax.set_xlabel("Temps")
#     ax.set_ylabel("Shell n°")
#     ax.set_title("Énergie par shell E(n,t) = X² + Y²")
#     plt.colorbar(im, ax=ax, label="Énergie")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig("Energie.png", dpi=150)

# plot_im_xy(X2,Y2)

# sim.init_default()
# X2, Y2, t2 = sim.run(T=100.0, save_every=100)
# print(f"\nRésultat exemple 2 : shape X = {X2.shape}")
# time = np.arange(0,X2.shape[0],1)

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
                X_value.append( (X_subset[i,j]-mean_mode[j])/std_mode[j]) # centré réduit

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



PATH = "/home/seb/Documents/stage/GOY-main/"
path_data = PATH + "data.dat"
SAVE = "/home/seb/Documents/stage/KF"

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

y_obs = Data_filtered.T + np.random.normal(0,1.0e-4,size=(44,90090)) #noisy observations

y_obs = y_obs[2*k_min_collocation:2*k_max_collocation,:]

# for i in range(np.shape(y_obs)[0]):
#     plt.figure()
#     plt.plot(y_obs[i,:],'*')
#     plt.plot(Data_shell[:,i],'gray',alpha=0.5)
#     plt.savefig(PATH + "ploty",dpi=300)




### parameters
n     = 44 # state size  on veut estimer les Un de 1 à 10 avec Re et Im donc 20 variables d'état
p     = 12 # On observe Un n=5,6,7,8,9,10 avec Re et Im donc 12 variables d'observations 
nb    = Npts # number of times
time  = np.array(range(nb)) # time vector
var_Q = 0.01 # error variance of the model (in Kalman)
var_R = 0.1 # error variance of the observations (in Kalman)
x_0   = np.zeros((n)) # initial coundition (mean)
P_0   = np.eye(n,n)*1.e-4 # initial coundition (covariance)

### variables
Q      = var_Q*np.eye(n,n)
R      = var_R*np.eye(p,p)


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

# def m(x_past):

#     dT = 1.0e-5
#     eps = 0.5
#     lmb = 2.0
#     force = 0.005
#     force_rnd = True
#     k0 = 0.125
#     N = 22
#     nu = 1.0e-7
#     fs = 100
#     time_end = 1.0e-2

#     X0 = x_past[0::2]
#     Y0 = x_past[1::2]

#     sim.init_custom(X0, Y0, dt=1e-5, force=0.005, force_rnd=True)
   
#     Xc, Yc = sim.step() #sim.run(T=1.0, save_every=100)#
    
    

#     x_future = np.zeros_like(x_past)
#     x_future[0::2] = Xc
#     x_future[1::2] = Yc
#     return x_future

#     '''
#     param = integration.GOYParams(force=force,N_force=4,
#                           force_rnd=force_rnd,k0=k0,lmb=lmb,
#                           eps=eps,nu=nu,N=22,dt=dT,fs=100,time=time_end)

#     x_past_real = np.copy(x_past[0::2,0]) # Reels
#     x_past_imag = np.copy(x_past[1::2,0]) # Imaginaires

#     x_future_real = np.copy(x_past[0::2,1]) # Un Reels
#     x_future_imag = np.copy(x_past[1::2,1]) # Un Imagin
#     integrate = integration.GOYShellModel(params=param,Xpp=x_past_real,Ypp=x_past_imag,Xp=x_future_real,Yp=x_future_imag)
#     X,Y = integrate.run()
#     # n = np.shape(x_past_real)[0]

#     # NL_re_pp, NL_im_pp = NL(x_past_real,x_past_imag)
#     # NL_re_p, NL_im_p = NL(x_future_real,x_future_imag)
    
#     x_future = np.zeros((n,2))

#     x_future[0::2,0] = x_past[0::2,1]
#     x_future[1::2,0] = x_past[1::2,1]

#     x_future[0::2,1] = X
#     x_future[1::2,1] = Y
#     '''
#     # x_future[:,0] = x_past[:,1] # x(t-1) => x(t)
#     # x_future_r = np.zeros(n)
#     # x_future_i = np.zeros(n)
#     # for i in range(n):
#     #     if i!=3:
#     #         x_future_i[i] = np.exp(-nu*(K[i]**2)*dT)*(x_future_imag[i] + dT*((3/2)*NL_im_p[i] - (1/2)*NL_im_pp[i]))
#     #         x_future_r[i] = np.exp(-nu*(K[i]**2)*dT)*(x_future_real[i] + dT*((3/2)*NL_re_p[i] - (1/2)*NL_re_pp[i]))

#     #     else:
#     #         x_future_i[i] = np.exp(-nu*(K[i]**2)*dT)*(x_future_imag[i] + dT*((3/2)*NL_im_p[i] - (1/2)*NL_im_pp[i])) + 0.005*dT*np.random.normal(0,1) # on ajoute du bruit pour le mode 4
#     #         x_future_r[i] = np.exp(-nu*(K[i]**2)*dT)*(x_future_real[i] + dT*((3/2)*NL_re_p[i] - (1/2)*NL_re_pp[i])) + 0.005*dT*np.random.normal(0,1) # x1(t+1) = x1(t) + x1_dot(t)
        
#     #     x_future[2*i,1] = x_future_r[i]
#     #     x_future[2*i+1,1] = x_future_i[i]
    
#     #return x_future

# def m_bis(x_past, n_steps=1):
#     X_in  = np.ascontiguousarray(x_past[0::2], dtype=np.float64)
#     Y_in  = np.ascontiguousarray(x_past[1::2], dtype=np.float64)
#     X_out = np.zeros(N, dtype=np.float64)
#     Y_out = np.zeros(N, dtype=np.float64)

#     lib.step_n(X_in.ctypes.data_as(_ptr),
#                Y_in.ctypes.data_as(_ptr),
#                X_out.ctypes.data_as(_ptr),
#                Y_out.ctypes.data_as(_ptr),
#                ctypes.c_int(n_steps))

#     x_future = np.zeros_like(x_past)
#     x_future[0::2] = X_out
#     x_future[1::2] = Y_out
#     return x_future
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
            return cur_Xpp,cur_Xp,cur_Ypp,cur_Yp
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

j_ = 50
nb_iter = 500
count_init   = int(TIME / DT)  
n_steps_first = count_init % N_fs  
series = np.zeros((n,nb_iter)).T
series[0,:] = Data_shell[j_,:]
series[1,:] = Data_shell[j_+1,:]#x_past[0,:]



for k in range(2,nb_iter):
    x_past = series[k-2:k,:].copy()
    
    _,_,cur_Xp,cur_Yp = m_b(x_past,N_fs=999,n_steps_first=100)
    series[k, 0::2] = cur_Xp.copy()
    series[k, 1::2] = cur_Yp.copy()
    # print("serie:",series[k,0:2])

Rmse = np.sqrt(np.mean(Data_shell[j_:j_+nb_iter,:]-series)**2)

plt.figure()

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
Ne = 100                   # number of ensembles
x_f_enkf = np.zeros((n,nb))   # forecast state
P_f_enkf = np.zeros((n,n,nb)) # forecast error covariance matrix
x_a_enkf = np.zeros((n,nb))   # analysed state
P_a_enkf = np.zeros((n,n,nb)) # analysed error covariance matrix

### Ensemble Kalman filter
x_a_enkf_tmp = np.zeros((n,Ne)) # shell,t-2 t-1, Ne
x_f_enkf_tmp = np.zeros((n,Ne))
y_f_enkf_tmp = np.zeros((p,Ne))
# initial step

for i in range(Ne):
    x_a_enkf_tmp[:,i] = np.random.multivariate_normal(x_0, P_0)

x_a_enkf[:,0]   = np.mean(x_a_enkf_tmp,1) # initial state
P_a_enkf[:,:,0] = np.cov(x_a_enkf_tmp)    # initial state covariance

for k in tqdm.tqdm(range(nb)): # forward in time
    # prediction step
    # il faut un initialisation custom pour chaque Ne
    for i in range(Ne):
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
plt.plot(Data_shell.T[8,:], 'b', label='True state ($x$)')
plt.plot(y_obs[8,:], '.k', label='Observations ($y$)')
plt.plot(x_a_enkf[8,:], 'r', label='EnKF ($x^a$)')
plt.xlabel('$time$', fontsize=20)
plt.ylabel('$U_8', fontsize=20)
plt.legend(fontsize=20)
plt.savefig(SAVE + "fig1enKF")
### plot state variables
plt.figure()
y_label=('$U_4$', '$U_5$', '$U_6$', '$U_7$')
for i in range(4,8):
    plt.subplot(2,2,i+1)
    plt.plot(time, Data_shell.T[2*i,:], 'b')
    if ((i==1) or (i==2)):
        plt.plot(time, y_obs[2*i,:], '.k') 
    plt.plot(time, x_a_enkf[2*i,:], 'r')
    plt.fill_between(time, x_a_enkf[i,:] - 1.96*np.sqrt(P_a_enkf[i,i,:]), x_a_enkf[i,:] + 1.96*np.sqrt(P_a_enkf[i,i,:]), facecolor='red', alpha=0.5)
    plt.xlabel('Time', size=20)
    plt.ylabel(y_label[i], size=20)
plt.savefig(SAVE + "fig2enKF")
### compute Root Mean Squared Errors (RMSE) of the positions
print('RMSE(obs):', np.sqrt(np.mean((y_obs[range(4,8),:] - Data_shell.T[range(4,9),:])**2))) ### A CACHER
print('RMSE(EnKF):', np.sqrt(np.mean((x_a_enkf[range(4,8),:] - Data_shell.T[range(4,9),:])**2))) ### A CACHER

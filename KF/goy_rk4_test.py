import numpy as np
import tqdm 
from goy_rk4_wrapper import GoyRK4, GoyParams

# Paramètres identiques à parameters.h
p = GoyParams(N=22, k0=0.125, lmb=2.0, eps=0.5, nu=1e-7, dt=1e-7, force=0.005, N_force=4, force_rnd=0)

# Conditions initiales identiques au code original (F[i] et 1e-4)
sh = p.k0 * (p.lmb ** np.arange(p.N))
X = sh ** (-1.0/3.0)
Y = np.full(p.N, 1e-4)
N_FS = int(1/(1e-5)/100)
# Lancement de la simulation RK4
goy = GoyRK4("/home/s26calme/Documents/code_stage/KF/libgoy.so")
goy.init(p)

XYhistory =np.zeros((100100,44)) 
for i in tqdm.tqdm(range(100100)):

    goy.integrate(X, Y, n_steps=99999)
    XYhistory[i,0::2] = X
    XYhistory[i,1::2] = Y
PATH = '/home/s26calme/Documents/code_stage/KF/'
np.save(PATH + "goy_rk4.npy",XYhistory)
print(f"État du mode forced (4) après 1000 pas : {X[4]} + i{Y[4]}")
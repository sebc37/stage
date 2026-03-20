import ctypes
import numpy as np
import matplotlib.pyplot as plt

lib = ctypes.CDLL("/home/s26calme/Documents/code_stage/KF/enkf_lib.so")
_ptr = ctypes.POINTER(ctypes.c_double)

lib.step_n.restype  = None
lib.step_n.argtypes = [_ptr, _ptr,   # Xpp, Ypp (entrée)
                        _ptr, _ptr,   # Xout, Yout (sortie)
                        ctypes.c_int] # n_steps
lib.get_N.restype = ctypes.c_int
N = lib.get_N()

def make_array(a):
    return np.ascontiguousarray(a, dtype=np.float64)

# Puis dans l'appel :
Xpp_arr = make_array(X0)
Ypp_arr = make_array(Y0)
Xout    = np.zeros(N, dtype=np.float64)
Yout    = np.zeros(N, dtype=np.float64)

lib.step_n(Xpp_arr.ctypes.data_as(_ptr),
           Ypp_arr.ctypes.data_as(_ptr),
           Xout.ctypes.data_as(_ptr),
           Yout.ctypes.data_as(_ptr),
           ctypes.c_int(n_steps))

def m(Xpp, Ypp, n_steps=1):
    # Tableaux explicites — pas de temporaires
    Xpp_c = np.ascontiguousarray(Xpp, dtype=np.float64)
    Ypp_c = np.ascontiguousarray(Ypp, dtype=np.float64)
    Xout  = np.zeros(N, dtype=np.float64)
    Yout  = np.zeros(N, dtype=np.float64)

    lib.step_n(Xpp_c.ctypes.data_as(_ptr),
               Ypp_c.ctypes.data_as(_ptr),
               Xout.ctypes.data_as(_ptr),
               Yout.ctypes.data_as(_ptr),
               ctypes.c_int(n_steps))
    return Xout, Yout


# Lecture données référence
data  = np.loadtxt("/home/s26calme/Documents/code_stage/GOY-main/data_test.dat")
X_ref = data[:, 0::2]   # (n_snap, N)
Y_ref = data[:, 1::2]
n_snap = len(X_ref)
print(f"Référence : {n_snap} snapshots")

# Simulation — on fournit les deux états consécutifs à chaque pas
X_sim = np.zeros_like(X_ref)
Y_sim = np.zeros_like(Y_ref)
X_sim[0,:] = X_ref[0,:]
Y_sim[0,:] = Y_ref[0,:]

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

n = 44
k0 = 0.125
lmb = 2.0
K = np.array([k0*lmb**i for i in range(22)],dtype=np.float64)
nu = 1.0e-7
x_past = np.zeros((2,n))
x_past[0,0::2] = K**(-1./3) #[0:int(n/2)]
x_past[0,1::2] = 1e-4 
x_p_r,x_p_i = NL(x_past[0,0::2],x_past[0,1::2])
x_past[1,0::2] = np.exp(-nu*K**2*1.0e-5)*(x_past[0,0::2] + 1.0e-5*x_p_r ) #np.random.normal(0,1.e-4,size=((n,2))) # state at time t-1 and t-2 for the model m
x_past[1,1::2] = np.exp(-nu*K**2*1.0e-5)*(x_past[0,1::2] + 1.0e-5*x_p_i ) #np.random.normal(0,1.e-4,size=((n,2))) # state at time t-1 and t-2 for the model m

# Pour le premier pas on n'a qu'un état → on calcule t=1 depuis (t=0, t=0)
# puis on a les deux états pour tous les suivants
X_sim[1,:], Y_sim[1,:] = x_past[1,0::2], x_past[1,1::2]

#m(X_ref[0], Y_ref[0], X_ref[0], Y_ref[0], N_fs)

for i in range(1, n_snap):
    X_sim[i], Y_sim[i] = m(
                            X_sim[i-1], Y_sim[i-1], n_steps=1001)

# Comparaison
rmse = np.sqrt(np.mean((X_sim - X_ref)**2 + (Y_sim - Y_ref)**2, axis=1))
print(f"RMSE max : {rmse.max():.4e}")

t = np.arange(n_snap) * 1e-5

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].semilogy(t, rmse)
axes[0].set_title("RMSE") ; axes[0].grid(True)

axes[1].plot(t, X_ref[:,0], label="Référence")
axes[1].plot(t, X_sim[:,0], '--', label="Pas à pas")
axes[1].set_title("Shell 0 — X(t)") ; axes[1].legend() ; axes[1].grid(True)

diff = np.abs(np.log10(X_ref**2+Y_ref**2+1e-20) - np.log10(X_sim**2+Y_sim**2+1e-20))
im = axes[2].imshow(diff.T, aspect='auto', origin='lower',
                    extent=[t[0],t[-1],0,N-1], cmap='hot')
axes[2].set_title("|Diff| log10 énergie")
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig("comparaison.png", dpi=150)
plt.show()
'''

## Pourquoi ça marchait pas avant

Adams-Bashforth ordre 2 calcule :
```
X_{t+1} = A·X_t + 1.5·dt·A·NL(t) - 0.5·dt·A²·NL(t-1)
'''
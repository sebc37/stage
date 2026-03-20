import ctypes
import numpy as np
import matplotlib.pyplot as plt

# ================================================================== #
#  Chargement lib                                                     #
# ================================================================== #
lib = ctypes.CDLL("/home/s26calme/Documents/code_stage/KF/enkf_lib.so")
_ptr = ctypes.POINTER(ctypes.c_double)

lib.step_n.restype  = None
lib.step_n.argtypes = [_ptr, _ptr,    # Xpp, Ypp (entrée)
                        _ptr, _ptr,    # Xout, Yout (sortie)
                        ctypes.c_int]  # n_steps
lib.get_N.restype = ctypes.c_int
N = lib.get_N()   # 22

def m(Xpp, Ypp, n_steps=1):
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

# ================================================================== #
#  Paramètres                                                         #
# ================================================================== #
k0  = 0.125
lmb = 2.0
nu  = 1.0e-7
dt  = 1.0e-5
fs  = 100
N_fs = int(1.0 / (dt * fs))-1 # 1000 pas entre chaque snapshot
K = np.array([k0 * lmb**i for i in range(N)], dtype=np.float64)

# ================================================================== #
#  Lecture données référence                                          #
# ================================================================== #
data  = np.loadtxt("/home/s26calme/Documents/code_stage/GOY-main/data_100.dat")
X_ref = data[:, 0::2]   # (n_snap, N)
Y_ref = data[:, 1::2]
n_snap = len(X_ref)
print(f"Référence : {n_snap} snapshots, N_fs={N_fs}")

# ================================================================== #
#  Simulation RK4 pas à pas                                          #
# ================================================================== #
X_sim    = np.zeros_like(X_ref)
Y_sim    = np.zeros_like(Y_ref)

# CI = premier snapshot de la référence
X_sim[0,:] = X_ref[0,:]
Y_sim[0,:] = Y_ref[0,:]

print("Intégration RK4...")
for i in range(1, n_snap):
    X_sim[i,:], Y_sim[i,:] = m(X_sim[i-1,:], Y_sim[i-1,:], n_steps=N_fs)
    if i % 500 == 0:
        print(f"  snapshot {i}/{n_snap}")

print("Terminé.")

# ================================================================== #
#  Comparaison                                                        #
# ================================================================== #
rmse = np.sqrt(np.mean((X_sim - X_ref)**2 + (Y_sim - Y_ref)**2, axis=1))
print(f"RMSE max : {rmse.max():.4e} | RMSE min : {rmse[1:].min():.4e}")

t = np.arange(n_snap) * N_fs * dt

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].semilogy(t, rmse)
axes[0].set_title("RMSE RK4 vs référence")
axes[0].set_xlabel("Temps")
axes[0].grid(True)

axes[1].plot(t, X_ref[:, 0], label="Référence", lw=1)
axes[1].plot(t, X_sim[:, 0], '--', label="RK4", lw=1)
axes[1].set_title("Shell 0 — X(t)")
axes[1].set_xlabel("Temps")
axes[1].legend()
axes[1].grid(True)

diff = np.abs(np.log10(X_ref**2 + Y_ref**2 + 1e-20)
            - np.log10(X_sim**2 + Y_sim**2 + 1e-20))
im = axes[2].imshow(diff.T, aspect='auto', origin='lower',
                    extent=[t[0], t[-1], 0, N-1], cmap='hot')
axes[2].set_title("|Diff| log10 énergie")
axes[2].set_xlabel("Temps")
axes[2].set_ylabel("Shell")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("comparaison.png", dpi=150)
plt.show()

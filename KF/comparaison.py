import ctypes
import numpy as np
import matplotlib.pyplot as plt

lib = ctypes.CDLL("/home/s26calme/Documents/code_stage/KF/enkf_lib.so")
_ptr = ctypes.POINTER(ctypes.c_double)

lib.init.restype    = None
lib.init.argtypes   = [_ptr, _ptr]
lib.step_n.restype  = None
lib.step_n.argtypes = [_ptr, _ptr, ctypes.c_int]
lib.get_N.restype   = ctypes.c_int

N    = lib.get_N()
N_fs = 1000-1   # 1/(dt*fs) = 1/(1e-5 * 100)

def to_ptr(arr):
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(_ptr)

# Lecture données référence
data  = np.loadtxt("/home/s26calme/Documents/code_stage/GOY-main/data_test.dat")
X_ref = data[:, 0::2]   # (n_snap, N)
Y_ref = data[:, 1::2]
n_snap = len(X_ref)
print(f"Référence : {n_snap} snapshots")

# Simulation pas à pas — init UNE SEULE FOIS
X_sim = np.zeros_like(X_ref)
Y_sim = np.zeros_like(Y_ref)
X_sim[0] = X_ref[0]
Y_sim[0] = Y_ref[0]

lib.init(to_ptr(X_ref[0]), to_ptr(Y_ref[0]))   # ← une seule fois

for i in range(1, n_snap):
    X_out = np.zeros(N, dtype=np.float64)
    Y_out = np.zeros(N, dtype=np.float64)
    lib.step_n(to_ptr(X_out), to_ptr(Y_out), ctypes.c_int(N_fs))
    X_sim[i] = X_out
    Y_sim[i] = Y_out

# Comparaison
rmse = np.sqrt(np.mean((X_sim - X_ref)**2 + (Y_sim - Y_ref)**2, axis=1))
print(f"RMSE max : {rmse.max():.4e} | RMSE min : {rmse.min():.4e}")

t = np.arange(n_snap) * N_fs * 1e-5

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].semilogy(t, rmse)
axes[0].set_title("RMSE") ; axes[0].set_xlabel("Temps") ; axes[0].grid(True)

axes[1].plot(t, X_ref[:,0], label="Référence")
axes[1].plot(t, X_sim[:,0], '--', label="Pas à pas")
axes[1].set_title("Shell 0 — X(t)") ; axes[1].legend() ; axes[1].grid(True)

E_ref = np.log10(X_ref**2 + Y_ref**2 + 1e-20)
E_sim = np.log10(X_sim**2 + Y_sim**2 + 1e-20)
diff  = np.abs(E_ref - E_sim)
im = axes[2].imshow(diff.T, aspect='auto', origin='lower',
                    extent=[t[0],t[-1],0,N-1], cmap='hot')
axes[2].set_title("|Diff| log10 énergie") ; axes[2].set_xlabel("Temps")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("comparaison.png", dpi=150)
plt.show()
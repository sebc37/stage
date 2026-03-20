"""
simulation.py
Interface Python pour integrate_lib.so (modèle shell GOY/Sabra).

Compilation préalable :
    gcc -shared -fPIC -O2 -o integrate_lib.so integrate_lib.c integrate.c -lm

Utilisation :
    python simulation.py
"""

import ctypes
import numpy as np

# ------------------------------------------------------------------ #
# Chargement de la bibliothèque                                       #
# ------------------------------------------------------------------ #
lib = ctypes.CDLL(r"/home/s26calme/Documents/code_stage/KF/integrate_lib.so")

# Récupérer N depuis la lib (défini par #define N 22 dans parameters.h)
lib.lib_get_N.restype  = ctypes.c_int
lib.lib_get_N.argtypes = []
N = lib.lib_get_N()
print(f"N = {N} shells")

# ------------------------------------------------------------------ #
# Déclaration des signatures                                          #
# ------------------------------------------------------------------ #
_ptr = ctypes.POINTER(ctypes.c_double)

lib.lib_init.restype  = None
lib.lib_init.argtypes = []

lib.lib_step.restype  = None
lib.lib_step.argtypes = []

lib.lib_get_X.restype  = None
lib.lib_get_X.argtypes = [_ptr]

lib.lib_get_Y.restype  = None
lib.lib_get_Y.argtypes = [_ptr]

lib.lib_get_sh.restype  = None
lib.lib_get_sh.argtypes = [_ptr]

# ------------------------------------------------------------------ #
# Buffers numpy (passage direct des pointeurs, sans copie inutile)    #
# ------------------------------------------------------------------ #
X_buf  = np.zeros(N, dtype=np.float64)
Y_buf  = np.zeros(N, dtype=np.float64)
sh_buf = np.zeros(N, dtype=np.float64)

x_ptr  = X_buf.ctypes.data_as(_ptr)
y_ptr  = Y_buf.ctypes.data_as(_ptr)
sh_ptr = sh_buf.ctypes.data_as(_ptr)

# ------------------------------------------------------------------ #
# Initialisation                                                      #
# ------------------------------------------------------------------ #
lib.lib_init()
lib.lib_get_sh(sh_ptr)
print("Nombres d'onde (sh) :", sh_buf)

# ------------------------------------------------------------------ #
# Boucle de simulation                                                #
# ------------------------------------------------------------------ #
# Paramètres (doivent correspondre à parameters.h)
dt       = 1e-5
fs       = 100          # fréquence de sauvegarde (Hz)
N_fs     = int(1.0 / dt / fs)   # nb de pas entre 2 sauvegardes
n_total  = int(1e6)     # nb total de pas (= time/dt avec time=10)

snapshots_X = []
snapshots_Y = []

print(f"Démarrage : {n_total} pas, sauvegarde tous les {N_fs} pas...")

for step in range(n_total):
    lib.lib_step()

    if step % N_fs == 0:
        lib.lib_get_X(x_ptr)
        lib.lib_get_Y(y_ptr)
        snapshots_X.append(X_buf.copy())
        snapshots_Y.append(Y_buf.copy())

        if step % (N_fs * 100) == 0:
            print(f"  step {step:8d} | X[0] = {X_buf[0]:.6e} | Y[0] = {Y_buf[0]:.6e}")

# ------------------------------------------------------------------ #
# Résultats                                                           #
# ------------------------------------------------------------------ #
snapshots_X = np.array(snapshots_X)   # shape : (n_snapshots, N)
snapshots_Y = np.array(snapshots_Y)

print(f"\nSimulation terminée : {len(snapshots_X)} snapshots, shape = {snapshots_X.shape}")

# Sauvegarde numpy (optionnel)
np.save("snapshots_X.npy", snapshots_X)
np.save("snapshots_Y.npy", snapshots_Y)
print("Données sauvegardées dans snapshots_X.npy et snapshots_Y.npy")

# ------------------------------------------------------------------ #
# Exemple d'analyse : spectre d'énergie moyen                        #
# ------------------------------------------------------------------ #
energy = np.mean(snapshots_X**2 + snapshots_Y**2, axis=0)
print("\nSpectre d'énergie moyen par shell :")
for i in range(N):
    print(f"  shell {i:2d} | k = {sh_buf[i]:.4f} | E = {energy[i]:.4e}")

import ctypes
import numpy as np

# Charger la lib
lib = ctypes.CDLL(r"/home/s26calme/Documents/code_stage/KF/integrate_lib.so")

# Déclarer les signatures
lib.lib_init.argtypes = [
    ctypes.c_int,                          # n_shells
    ctypes.c_double, ctypes.c_double,      # lmb, k0
    ctypes.c_double, ctypes.c_double,      # nu, dt
    ctypes.c_double, ctypes.c_double,      # eps, force
    ctypes.c_int                           # force_rnd
]
lib.lib_init.restype = None
lib.lib_step.restype = None
lib.lib_init_fields.restype = None

lib.lib_get_X.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.lib_get_Y.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

# Paramètres de la simulation
N      = 22
lmb    = 2.0
k0     = 0.0625
nu     = 1e-7
dt     = 1e-4
eps    = 0.5
force  = 1.0
force_rnd = 0

# Initialisation
lib.lib_init(N, lmb, k0, nu, dt, eps, force, force_rnd)
lib.lib_init_fields()

# Buffers numpy pour récupérer les résultats
X_out = np.zeros(N, dtype=np.float64)
Y_out = np.zeros(N, dtype=np.float64)
x_ptr = X_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
y_ptr = Y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Boucle de simulation
n_steps = 10000
results = []

for step in range(n_steps):
    lib.lib_step()

    if step % 100 == 0:
        lib.lib_get_X(x_ptr, N)
        lib.lib_get_Y(y_ptr, N)
        results.append((X_out.copy(), Y_out.copy()))

print(f"Simulation terminée : {len(results)} snapshots")
print("X[0] =", results[-1][0][0])
import ctypes
import numpy as np

class GoyParams(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_int), ("k0", ctypes.c_double), ("lmb", ctypes.c_double),
        ("eps", ctypes.c_double), ("nu", ctypes.c_double), ("dt", ctypes.c_double),
        ("force", ctypes.c_double), ("N_force", ctypes.c_int), ("force_rnd", ctypes.c_int)
    ]

class GoyRK4:
    def __init__(self, lib_path="./libgoy.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.goy_init.argtypes = [ctypes.POINTER(GoyParams)]
        self.lib.goy_step_rk4.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64)
        ]
        self.lib.goy_integrate_rk4.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_int
        ]

    def init(self, params):
        self.lib.goy_init(ctypes.byref(params))

    def integrate(self, X, Y, n_steps):
        return self.lib.goy_integrate_rk4(X, Y, n_steps)
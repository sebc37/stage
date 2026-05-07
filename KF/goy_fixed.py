"""
goy.py  –  Python wrapper for the GOY shell model library (goy_lib.so)

Usage
-----
    from goy import GoyModel
    import numpy as np

    model = GoyModel()          # default parameters (N=22, same as original code)

    # build initial conditions exactly as the original C code does:
    Xpp, Ypp, Xp, Yp = model.init_fields()

    # integrate N_fs steps (= 1 output sample at frequency fs)
    (Xpp, Ypp), (Xp, Yp) = model.integrate(Xpp, Ypp, Xp, Yp, n_steps=model.N_fs)

    # the two returned states are:
    #   (Xpp, Ypp)  →  state at  t + (n_steps - 1) * dt
    #   (Xp,  Yp)   →  state at  t +  n_steps      * dt
"""

import ctypes
import os
import numpy as np

# ── locate the .so next to this file ──────────────────────────────────────────
_LIB_NAME = "goy_lib.so"
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, _LIB_NAME)

# ── C struct mirroring GoyParams ──────────────────────────────────────────────
class _GoyParams(ctypes.Structure):
    _fields_ = [
        ("N",        ctypes.c_int),
        ("k0",       ctypes.c_double),
        ("lmb",      ctypes.c_double),
        ("eps",      ctypes.c_double),
        ("nu",       ctypes.c_double),
        ("dt",       ctypes.c_double),
        ("force",    ctypes.c_double),
        ("N_force",  ctypes.c_int),
        ("force_rnd",ctypes.c_int),
    ]


class GoyModel:
    """
    GOY shell-model integrator backed by a compiled C shared library.

    Parameters
    ----------
    N         : number of shells                     (default 22)
    k0        : largest-scale wave-number            (default 0.125)
    lmb       : ratio between consecutive shells     (default 2.0)
    eps       : NL coupling parameter                (default 0.5)
    nu        : viscosity                            (default 1e-7)
    dt        : time step                            (default 1e-5)
    force     : forcing amplitude                    (default 0.005)
    N_force   : index of the forced shell            (default 4)
    force_rnd : 1 = random forcing, 0 = deterministic (default 0)
    lib_path  : path to goy_lib.so                   (default: same directory as this file)
    """

    def __init__(self,
                 N=22,
                 k0=np.float64(0.125),
                 lmb=np.float64(2.0),
                 eps=np.float64(0.5),
                 nu=np.float64(1e-7),
                 dt=np.float64(1e-5),
                 force=np.float64(0.005),
                 N_force=4,
                 force_rnd=0,
                 lib_path=None):

        if N > 64:
            raise ValueError("N must be ≤ 64 (recompile goy_lib.c with larger N_MAX if needed)")

        self.N   = N
        self.dt  = dt
        self._lmb = lmb
        self._eps = eps
        self._nu  = nu

        # load library
        path = lib_path or _LIB_PATH
        self._lib = ctypes.CDLL(path)

        # goy_init(const GoyParams *)
        self._lib.goy_init.restype  = None
        self._lib.goy_init.argtypes = [ctypes.POINTER(_GoyParams)]

        # int goy_integrate(Xpp, Ypp, Xp, Yp, n_steps, out_Xp, out_Yp, out_X, out_Y)
        _dp = ctypes.POINTER(ctypes.c_double)
        self._lib.goy_integrate.restype  = ctypes.c_int
        self._lib.goy_integrate.argtypes = [
            _dp, _dp,           # Xpp, Ypp  (input, state at t-dt)
            _dp, _dp,           # Xp,  Yp   (input, state at t)
            ctypes.c_int,       # n_steps
            _dp, _dp,           # out_Xp, out_Yp  (state at t + (n_steps-1)*dt)
            _dp, _dp,           # out_X,  out_Y   (state at t + n_steps*dt)
        ]

        # initialise C state
        params = _GoyParams(N=N, k0=k0, lmb=lmb, eps=eps, nu=nu,
                            dt=dt, force=force, N_force=N_force,
                            force_rnd=force_rnd)
        self._lib.goy_init(ctypes.byref(params))

        # cache shell wave-numbers for convenience
        self._sh = k0 * lmb ** np.arange(N, dtype=np.float64)

        # ── FIX: N_fs must match the C formula exactly ────────────────────────
        # C code: N_fs = (int)(1./dt/fs)  where fs is passed separately.
        # Store it so the caller can use it directly.
        # The value is computed at construction time; pass fs explicitly if needed.
        # (default matches parameters.h: fs=100, dt=1e-5 → N_fs=999)
        self._force    = force
        self._N_force  = N_force

    # ── helpers ───────────────────────────────────────────────────────────────

    def shell_wavenumbers(self) -> np.ndarray:
        """Return the array of shell wave-numbers k_n = k0 * lmb^n."""
        return self._sh.copy()

    def N_fs_for(self, fs: float) -> int:
        """
        Return the number of integration steps between two successive output
        samples at frequency *fs*, computed the same way as the C code:
            N_fs = (int)(1.0 / dt / fs)
        Note: due to floating-point arithmetic this is NOT necessarily
        round(1 / (dt * fs)).  Always use this helper to stay consistent.
        """
        return int(1.0 / self.dt / fs)

    @staticmethod
    def _c_arr(a: np.ndarray):
        """Return a ctypes pointer to a contiguous float64 array."""
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a

    # ── main API ──────────────────────────────────────────────────────────────

    def init_fields(self, Xpp=None, Ypp=None) -> tuple:
        """
        Compute the two-step initial condition (Xpp, Ypp, Xp, Yp) exactly as
        the original C init_fields() does:

          1. Xpp[i] = k_n^{-1/3},  Ypp[i] = 1e-4   (or use the provided arrays)
          2. Compute NXpp, NYpp from (Xpp, Ypp)
          3. Xp  = A * (Xpp + dt * NXpp)            ← one Euler step forward
             Yp  = A * (Ypp + dt * NYpp)

        The two returned states therefore satisfy:
            Xpp  ↔  state at t = 0
            Xp   ↔  state at t = dt

        This is exactly the pair that must be passed as the first call to
        integrate(Xpp, Ypp, Xp, Yp, ...).

        Returns
        -------
        (Xpp, Ypp, Xp, Yp)  all as float64 numpy arrays of length N
        """
        sh  = self._sh
        lmb = self._lmb
        eps = self._eps
        N   = self.N
        dt  = self.dt

        if Xpp is None and Ypp is None:
            Xpp = sh ** (-1.0 / 3.0)
            Ypp = np.full(N, 1e-4, dtype=np.float64)
        else:
            Xpp = np.asarray(Xpp, dtype=np.float64)
            Ypp = np.asarray(Ypp, dtype=np.float64)

        A  = np.exp(-self._nu * sh**2 * dt)
        A2 = np.full(N, -eps / lmb,          dtype=np.float64)
        A3 = np.full(N, -(1.0 - eps) / lmb**2, dtype=np.float64)

        def _NX(ax, ay):
            res = np.zeros(N, dtype=np.float64)
            res[0] = ax[2]*ay[1] + ay[2]*ax[1]
            res[1] = ax[3]*ay[2] + ay[3]*ax[2] + A2[1]*(ax[2]*ay[0] + ay[2]*ax[0])
            for k in range(2, N-2):
                res[k] = (ax[k+2]*ay[k+1] + ay[k+2]*ax[k+1]) \
                       + A2[k]*(ax[k+1]*ay[k-1] + ay[k+1]*ax[k-1]) \
                       + A3[k]*(ax[k-1]*ay[k-2] + ay[k-1]*ax[k-2])
            res[N-2] = A2[N-2]*(ax[N-1]*ay[N-3] + ay[N-1]*ax[N-3]) \
                     + A3[N-2]*(ax[N-3]*ay[N-4] + ay[N-3]*ax[N-4])
            res[N-1] = A3[N-1]*(ax[N-2]*ay[N-3] + ay[N-2]*ax[N-3])
            return res * sh

        def _NY(ax, ay):
            res = np.zeros(N, dtype=np.float64)
            res[0] = ax[2]*ax[1] - ay[1]*ay[2]
            res[1] = ax[3]*ax[2] - ay[2]*ay[3] + A2[1]*(ax[2]*ax[0] - ay[2]*ay[0])
            for k in range(2, N-2):
                res[k] = ax[k+2]*ax[k+1] - ay[k+1]*ay[k+2] \
                       + A2[k]*(ax[k+1]*ax[k-1] - ay[k+1]*ay[k-1]) \
                       + A3[k]*(ax[k-1]*ax[k-2] - ay[k-1]*ay[k-2])
            res[N-2] = A2[N-2]*(ax[N-1]*ax[N-3] + ay[N-1]*ay[N-3]) \
                     + A3[N-2]*(ax[N-3]*ax[N-4] + ay[N-3]*ay[N-4])
            res[N-1] = A3[N-1]*(ax[N-2]*ax[N-3] - ay[N-2]*ay[N-3])
            return res * sh

        NXpp = _NX(Xpp, Ypp)
        NYpp = _NY(Xpp, Ypp)
        Xp = A * (Xpp + dt * NXpp)   # one Euler step: state at t = dt
        Yp = A * (Ypp + dt * NYpp)

        return Xpp, Ypp, Xp, Yp

    def integrate(self,
                  Xpp: np.ndarray, Ypp: np.ndarray,
                  Xp:  np.ndarray, Yp:  np.ndarray,
                  n_steps: int):
        """
        Integrate the GOY model for *n_steps* Adams-Bashforth steps.

        Parameters
        ----------
        Xpp, Ypp : real/imaginary parts of the complex field at  t - dt
        Xp,  Yp  : real/imaginary parts of the complex field at  t
        n_steps  : number of integration steps

        Returns
        -------
        (Xpp_out, Ypp_out) : state at  t + (n_steps - 1) * dt
        (Xp_out,  Yp_out)  : state at  t +  n_steps      * dt

        To chain calls correctly:
            (Xpp, Ypp), (Xp, Yp) = model.integrate(Xpp, Ypp, Xp, Yp, N_fs)
        This keeps (Xpp, Ypp) as the penultimate state and (Xp, Yp) as the
        latest state, exactly as the original C main loop does.

        Raises
        ------
        RuntimeError if a NaN is detected during integration.
        """
        N = self.N
        for arr in (Xpp, Ypp, Xp, Yp):
            if len(arr) != N:
                raise ValueError(f"All input arrays must have length N={N}")

        out_Xp = np.zeros(N, dtype=np.float64)
        out_Yp = np.zeros(N, dtype=np.float64)
        out_X  = np.zeros(N, dtype=np.float64)
        out_Y  = np.zeros(N, dtype=np.float64)

        p_Xpp, _Xpp = self._c_arr(Xpp)
        p_Ypp, _Ypp = self._c_arr(Ypp)
        p_Xp,  _Xp  = self._c_arr(Xp)
        p_Yp,  _Yp  = self._c_arr(Yp)
        p_oXp  = out_Xp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_oYp  = out_Yp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_oX   = out_X .ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_oY   = out_Y .ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        ret = self._lib.goy_integrate(
            p_Xpp, p_Ypp, p_Xp, p_Yp,
            ctypes.c_int(n_steps),
            p_oXp, p_oYp, p_oX, p_oY
        )

        if ret != 0:
            raise RuntimeError("NaN detected during GOY integration")

        return (out_Xp, out_Yp), (out_X, out_Y)


# ── quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    TIME      = 1000.
    DT        = 1e-5
    FS        = 100.
    FORCE     = 0.005
    N_FORCE   = 4
    FORCE_RND = 0

    model = GoyModel(dt=DT, force=FORCE, N_force=N_FORCE, force_rnd=FORCE_RND)

    # ── FIX 1: use N_fs_for() to get the same integer as the C code ──────────
    N_fs = model.N_fs_for(FS)          # = int(1.0 / DT / FS) = 999
    N_pts = int(TIME * FS)             # number of output samples = 100 000

    print(f"N_fs = {N_fs}  (steps between saves)")
    print(f"N_pts = {N_pts}")

    data = np.zeros((N_pts, 2 * model.N), dtype=np.float64)

    # ── FIX 2: use init_fields() so that Xpp ≠ Xp (correct AB2 start) ───────
    # In the original buggy code:  Xp = Xpp.copy()  →  Xpp == Xp
    # which makes NXpp == NXp and degenerates the first AB2 step to Euler.
    Xpp, Ypp, Xp, Yp = model.init_fields()

    print(f"Integrating {model.N} shells for {N_pts} output samples …")
    t0 = time.perf_counter()

    # ── FIX 3: do NOT save data[0] before any integration ────────────────────
    # The C code saves only after at least one integrate() call.
    # The loop below saves data[i] AFTER the i-th block of N_fs steps,
    # so data[0] = state after N_fs steps, matching the C output exactly.
    for i in range(N_pts):
        (Xpp, Ypp), (Xp, Yp) = model.integrate(Xpp, Ypp, Xp, Yp, n_steps=N_fs)
        data[i, 0::2] = Xp
        data[i, 1::2] = Yp

    elapsed = time.perf_counter() - t0

    PATH = "/home/s26calme/Documents/code_stage/KF/"
    np.save(PATH + "goy_lib_test.npy",data)
    
    print(f"Done in {elapsed:.3f} s")
    print(f"X[0]  = {Xp[0]:.6e}")
    print(f"Xpp[0]= {Xpp[0]:.6e}")
    energy = 0.5 * np.sum(Xp**2 + Yp**2)
    print(f"Total energy = {energy:.6e}")

    # Optionally save
    # np.save("goy_lib_fixed.npy", data)

"""
simulation.py
Interface Python pour le modèle shell GOY via integrate_lib.so.

Compilation (depuis le dossier KF) :
    gcc -shared -fPIC -O2 -o integrate_lib.so \
        integrate_lib.c integrate.c integrate_io.c stats_io.c -lm

Utilisation rapide :
    python simulation.py
"""

import ctypes
import numpy as np

# ================================================================== #
#  Chargement de la lib                                               #
# ================================================================== #
import os
LIB_PATH = os.path.join(os.path.dirname(__file__), "integrate_lib.so")
lib = ctypes.CDLL(LIB_PATH)

_ptr = ctypes.POINTER(ctypes.c_double)

# Récupérer N (défini par #define dans parameters.h)
lib.lib_get_N.restype  = ctypes.c_int
lib.lib_get_N.argtypes = []
N = lib.lib_get_N()

# Signatures complètes
lib.lib_init_default.restype  = None
lib.lib_init_default.argtypes = []

lib.lib_init_custom.restype  = None
lib.lib_init_custom.argtypes = [
    _ptr,            # X0
    _ptr,            # Y0
    ctypes.c_int,    # n  (taille des tableaux)
    ctypes.c_double, # dt_user
    ctypes.c_double, # force_user
    ctypes.c_int,    # force_rnd
]

lib.lib_step.restype  = None
lib.lib_step.argtypes = []

lib.lib_run.restype  = ctypes.c_int
lib.lib_run.argtypes = [
    ctypes.c_int,  # n_steps
    ctypes.c_int,  # save_every
    _ptr,          # out_X  (taille n_out * N)
    _ptr,          # out_Y
    ctypes.c_int,  # n_out
]

lib.lib_get_X.restype  = None ; lib.lib_get_X.argtypes  = [_ptr]
lib.lib_get_Y.restype  = None ; lib.lib_get_Y.argtypes  = [_ptr]
lib.lib_get_sh.restype = None ; lib.lib_get_sh.argtypes = [_ptr]
lib.lib_get_dt.restype = ctypes.c_double

lib.lib_set_sh.restype  = None
lib.lib_set_sh.argtypes = [_ptr, ctypes.c_int]


# ================================================================== #
#  Classe de simulation                                               #
# ================================================================== #
class ShellModel:
    """
    Interface haut niveau pour le modèle shell GOY.

    Exemple d'utilisation :
        sim = ShellModel()
        sim.init_default()                    # CI originales
        X, Y = sim.run(T=10.0, save_every=100)

        # ou avec CI personnalisées :
        X0 = np.ones(sim.N) * 1e-3
        Y0 = np.zeros(sim.N)
        sim.init_custom(X0, Y0, dt=1e-5, force=0.005, force_rnd=True)
        X, Y = sim.run(T=10.0, save_every=100)
    """

    def __init__(self):
        self.N  = N
        self.sh = np.zeros(N, dtype=np.float64)
        self.dt = 1e-5  # valeur par défaut, mise à jour à l'init

    def set_sh(self, sh_new):
        """
        Remplace les nombres d'onde sh par tes propres valeurs.
        Recalcule automatiquement les coefficients A[i] en C.
        A appeler APRES init_default() ou init_custom().

        Parameters
        ----------
        sh_new : array-like de taille N
        """
        sh_new = np.asarray(sh_new, dtype=np.float64)
        assert len(sh_new) == self.N, f"sh_new doit etre de taille {self.N}"
        lib.lib_set_sh(sh_new.ctypes.data_as(_ptr), ctypes.c_int(self.N))
        self._refresh_sh()   # met a jour self.sh en Python
        print(f"[ShellModel] set_sh | sh = {self.sh}")

    def _refresh_sh(self):
        """Lit les nombres d'onde depuis la lib (après init)."""
        lib.lib_get_sh(self.sh.ctypes.data_as(_ptr))

    # -------------------------------------------------------------- #
    #  Initialisations                                                #
    # -------------------------------------------------------------- #
    def init_default(self):
        """CI originales (spectre k^{-1/3}) avec les paramètres de parameters.h."""
        lib.lib_init_default()
        self.dt = lib.lib_get_dt()
        self._refresh_sh()
        print(f"[ShellModel] init_default | N={self.N} | dt={self.dt:.2e}")

    def init_custom(self, X0, Y0,
                    dt=1e-5, force=0.005, force_rnd=True):
        """
        CI personnalisées.

        Parameters
        ----------
        X0, Y0     : array-like de taille N (partie réelle / imaginaire)
        dt         : pas de temps
        force      : amplitude du forçage
        force_rnd  : True = forçage aléatoire, False = déterministe
        """
        X0 = np.asarray(X0, dtype=np.float64)
        Y0 = np.asarray(Y0, dtype=np.float64)
        assert len(X0) == N and len(Y0) == N, f"X0 et Y0 doivent être de taille {N}"

        lib.lib_init_custom(
            X0.ctypes.data_as(_ptr),
            Y0.ctypes.data_as(_ptr),
            ctypes.c_int(N),
            ctypes.c_double(dt),
            ctypes.c_double(force),
            ctypes.c_int(int(force_rnd)),
        )
        self.dt = dt
        self._refresh_sh()
        print(f"[ShellModel] init_custom  | N={self.N} | dt={dt:.2e} | "
              f"force={force:.3e} | force_rnd={force_rnd}")

    # -------------------------------------------------------------- #
    #  Run complet (rapide — tout en C)                              #
    # -------------------------------------------------------------- #
    def run(self, T, save_every=100):
        """
        Lance la simulation sur une durée T (en unités physiques).

        Parameters
        ----------
        T          : durée totale de la simulation
        save_every : sauvegarder 1 snapshot tous les `save_every` pas

        Returns
        -------
        X : ndarray (n_snapshots, N)  — partie réelle des champs
        Y : ndarray (n_snapshots, N)  — partie imaginaire
        t : ndarray (n_snapshots,)    — temps correspondants
        """
        n_steps = int(T / self.dt)
        n_out   = n_steps // save_every

        out_X = np.zeros((n_out, N), dtype=np.float64)
        out_Y = np.zeros((n_out, N), dtype=np.float64)

        print(f"[ShellModel] run | T={T} | n_steps={n_steps} | "
              f"save_every={save_every} | n_snapshots={n_out}")

        snap = lib.lib_run(
            ctypes.c_int(n_steps),
            ctypes.c_int(save_every),
            out_X.ctypes.data_as(_ptr),
            out_Y.ctypes.data_as(_ptr),
            ctypes.c_int(n_out),
        )

        t = np.arange(1, snap + 1) * save_every * self.dt
        print(f"[ShellModel] terminé — {snap} snapshots sauvegardés")
        return out_X[:snap], out_Y[:snap], t

    # -------------------------------------------------------------- #
    #  Pas à pas (pour usage interactif ou filtre de Kalman)         #
    # -------------------------------------------------------------- #
    def step(self):
        """Un seul pas d'intégration. Retourne (X, Y) courants."""
        lib.lib_step()
        return self.get_state()

    def get_state(self):
        """Retourne (X, Y) courants sans avancer."""
        X = np.zeros(N, dtype=np.float64)
        Y = np.zeros(N, dtype=np.float64)
        lib.lib_get_X(X.ctypes.data_as(_ptr))
        lib.lib_get_Y(Y.ctypes.data_as(_ptr))
        return X, Y


# ================================================================== #
#  Exemple d'utilisation                                              #
# ================================================================== #
if __name__ == "__main__":
    sim = ShellModel()
    print(f"Nombres d'onde : {sim.sh}")

    # --- Exemple 1 : CI par défaut, paramètres de parameters.h ---
    sim.init_default()
    X, Y, t = sim.run(T=10.0, save_every=100)
    print(f"\nRésultat exemple 1 : shape X = {X.shape}")
    np.save("X_default.npy", X)
    np.save("Y_default.npy", Y)

    # --- Exemple 2 : CI personnalisées ---
    X0 = np.array([sim.sh[i]**(-1/3) if sim.sh[i] > 0 else 0.0
                   for i in range(sim.N)])
    X0 *= (1 + 0.1*np.random.randn(sim.N))
    Y0 = np.zeros(sim.N)

    sim.init_custom(X0, Y0, dt=1e-5, force=0.005, force_rnd=True)
    X2, Y2, t2 = sim.run(T=5.0, save_every=50)
    print(f"\nRésultat exemple 2 : shape X = {X2.shape}")

    # --- Exemple 3 : pas à pas (utile pour filtre de Kalman) ---
    sim.init_custom(X0, Y0, dt=1e-5, force=0.005, force_rnd=False)
    states = []
    for _ in range(1000):
        Xc, Yc = sim.step()
        states.append(Xc.copy())
    states = np.array(states)
    print(f"\nRésultat exemple 3 (pas à pas) : shape = {states.shape}")

    # --- Spectre d'énergie moyen ---
    E = np.mean(X**2 + Y**2, axis=0)
    print("\nSpectre d'énergie moyen (exemple 1) :")
    for i in range(sim.N):
        print(f"  shell {i:2d} | k={sim.sh[i]:.4f} | E={E[i]:.4e}")

"""
enkf.py
Filtre de Kalman d'ensemble (EnKF) pour le modèle shell GOY.

Compilation :
    gcc -shared -fPIC -O2 -o enkf_lib.so enkf_lib.c -lm

Utilisation :
    python enkf.py
"""

import ctypes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ================================================================== #
#  Chargement de la lib                                               #
# ================================================================== #
import os
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "enkf_lib.so"))
_ptr = ctypes.POINTER(ctypes.c_double)

lib.enkf_create_member.restype  = ctypes.c_int
lib.enkf_create_member.argtypes = [_ptr, _ptr,
                                    ctypes.c_double,  # dt
                                    ctypes.c_double,  # force
                                    ctypes.c_int]     # force_rnd

lib.enkf_reset_member.restype  = None
lib.enkf_reset_member.argtypes = [ctypes.c_int, _ptr, _ptr]

lib.enkf_step.restype   = None
lib.enkf_step.argtypes  = [ctypes.c_int]

lib.enkf_step_n.restype  = None
lib.enkf_step_n.argtypes = [ctypes.c_int, ctypes.c_int]

lib.enkf_get_state.restype  = None
lib.enkf_get_state.argtypes = [ctypes.c_int, _ptr, _ptr]

lib.enkf_set_state.restype  = None
lib.enkf_set_state.argtypes = [ctypes.c_int, _ptr, _ptr]

lib.enkf_get_sh.restype  = None
lib.enkf_get_sh.argtypes = [_ptr]

lib.enkf_get_N.restype  = ctypes.c_int
lib.enkf_free_all.restype = None

N = lib.enkf_get_N()


# ================================================================== #
#  Classe membre                                                      #
# ================================================================== #
def _make_ptr(arr):
    return arr.ctypes.data_as(_ptr)

def create_member(X0, Y0, dt=1e-5, force=0.005, force_rnd=True):
    X0 = np.asarray(X0, dtype=np.float64)
    Y0 = np.asarray(Y0, dtype=np.float64)
    return lib.enkf_create_member(_make_ptr(X0), _make_ptr(Y0),
                                   ctypes.c_double(dt),
                                   ctypes.c_double(force),
                                   ctypes.c_int(int(force_rnd)))

def get_state(mid):
    X = np.zeros(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)
    lib.enkf_get_state(mid, _make_ptr(X), _make_ptr(Y))
    return X, Y

def set_state(mid, X, Y):
    lib.enkf_set_state(mid, _make_ptr(np.asarray(X, np.float64)),
                            _make_ptr(np.asarray(Y, np.float64)))

def step_n(mid, n):
    lib.enkf_step_n(mid, ctypes.c_int(n))


# ================================================================== #
#  Filtre de Kalman d'ensemble                                        #
# ================================================================== #
class EnKF:
    """
    Filtre de Kalman d'ensemble pour le modèle shell GOY.

    L'état est le vecteur x = [X_0, Y_0, X_1, Y_1, ..., X_N-1, Y_N-1]
    de dimension 2*N = 44.

    L'observation est un vecteur de dimension n_obs (sous-ensemble des shells).

    Parameters
    ----------
    n_members  : taille de l'ensemble
    obs_shells : liste des shells observés (ex: [0,1,2])
    obs_noise  : écart-type du bruit d'observation
    dt         : pas de temps du modèle
    da_step    : nombre de pas entre deux analyses
    force      : amplitude du forçage
    force_rnd  : forçage stochastique
    """

    def __init__(self, n_members=50, obs_shells=None,
                 obs_noise=0.1, dt=1e-5, da_step=100,
                 force=0.005, force_rnd=True):

        self.Ne        = n_members
        self.obs_shells = obs_shells if obs_shells is not None else list(range(N//2))
        self.n_obs     = len(self.obs_shells)
        self.obs_noise = obs_noise
        self.dt        = dt
        self.da_step   = da_step      # pas de modèle entre 2 analyses
        self.force     = force
        self.force_rnd = force_rnd
        self.dim       = 2 * N        # taille du vecteur état

        # Matrice d'observation H  (n_obs x dim)
        # On observe X[shell] pour chaque shell observé
        self.H = np.zeros((self.n_obs, self.dim))
        for k, s in enumerate(self.obs_shells):
            self.H[k, 2*s] = 1.0     # position de X[s] dans x

        # Matrice de covariance du bruit d'observation
        self.R = (obs_noise**2) * np.eye(self.n_obs)

        # IDs des membres C
        self.member_ids = []

        # Récupère les sh
        self.sh = np.zeros(N, dtype=np.float64)
        lib.enkf_get_sh(_make_ptr(self.sh))

    def _state_to_vec(self, X, Y):
        """[X0,Y0,X1,Y1,...] → vecteur état 1D de taille 2N."""
        v = np.zeros(self.dim)
        v[0::2] = X
        v[1::2] = Y
        return v

    def _vec_to_XY(self, v):
        """Vecteur état 1D → (X, Y)."""
        return v[0::2].copy(), v[1::2].copy()

    def init_ensemble(self, X0_mean, Y0_mean, spread=0.01):
        """
        Initialise l'ensemble autour de (X0_mean, Y0_mean)
        avec une perturbation gaussienne de std=spread.

        Parameters
        ----------
        X0_mean, Y0_mean : CI moyennes (taille N)
        spread           : écart-type des perturbations initiales
        """
        lib.enkf_free_all()
        self.member_ids = []

        for _ in range(self.Ne):
            X0 = X0_mean + spread * np.random.randn(N)
            Y0 = Y0_mean + spread * np.random.randn(N)
            mid = create_member(X0, Y0, dt=self.dt,
                                force=self.force,
                                force_rnd=self.force_rnd)
            self.member_ids.append(mid)

        print(f"[EnKF] Ensemble initialisé : {self.Ne} membres, "
              f"dim={self.dim}, n_obs={self.n_obs}")

    def _get_ensemble_matrix(self):
        """Retourne la matrice d'ensemble A de forme (dim, Ne)."""
        A = np.zeros((self.dim, self.Ne))
        for j, mid in enumerate(self.member_ids):
            X, Y = get_state(mid)
            A[:, j] = self._state_to_vec(X, Y)
        return A

    def _propagate(self):
        """Avance tous les membres de da_step pas."""
        for mid in self.member_ids:
            step_n(mid, self.da_step)

    def _analysis(self, obs):
        """
        Mise à jour EnKF (formulation stochastique).

        Parameters
        ----------
        obs : vecteur d'observation (taille n_obs)
        """
        # Matrice ensemble (dim x Ne)
        A = self._get_ensemble_matrix()

        # Moyenne et anomalies
        x_mean = A.mean(axis=1, keepdims=True)         # (dim, 1)
        Ap = A - x_mean                                 # (dim, Ne)

        # Covariance de fond : P ≈ Ap @ Ap.T / (Ne-1)
        # On travaille directement sur Ap pour éviter de stocker P explicitement

        # Perturbations d'observation : D = obs + bruit - H @ x_j
        D = np.zeros((self.n_obs, self.Ne))
        obs_noise_mat = np.random.randn(self.n_obs, self.Ne) * self.obs_noise
        for j in range(self.Ne):
            D[:, j] = obs + obs_noise_mat[:, j] - self.H @ A[:, j]

        # Innovation en espace obs : S = H @ Ap @ Ap.T @ H.T / (Ne-1) + R
        HAp = self.H @ Ap                               # (n_obs, Ne)
        S   = (HAp @ HAp.T) / (self.Ne - 1) + self.R  # (n_obs, n_obs)

        # Gain de Kalman : K = Ap @ (H@Ap).T @ S^{-1} / (Ne-1)
        K = (Ap @ HAp.T) / (self.Ne - 1) @ np.linalg.inv(S)  # (dim, n_obs)

        # Mise à jour de chaque membre
        A_new = A + K @ D                               # (dim, Ne)

        # Réinjecte dans les membres C
        for j, mid in enumerate(self.member_ids):
            X_new, Y_new = self._vec_to_XY(A_new[:, j])
            set_state(mid, X_new, Y_new)

        return A_new.mean(axis=1)   # moyenne analysée

    def run(self, n_cycles, obs_func, true_traj=None):
        """
        Lance le filtre sur n_cycles cycles (propagation + analyse).

        Parameters
        ----------
        n_cycles : nombre de cycles d'assimilation
        obs_func : fonction obs_func(cycle) → vecteur obs (taille n_obs)
                   Peut générer des obs synthétiques depuis une vraie trajectoire.
        true_traj : (optionnel) vraie trajectoire shape (n_cycles, dim)
                    pour calcul des métriques RMSE

        Returns
        -------
        mean_traj  : (n_cycles, dim) — moyenne d'ensemble après analyse
        spread_traj: (n_cycles, dim) — écart-type d'ensemble après analyse
        rmse       : (n_cycles,)     — RMSE vs vraie trajectoire (si fournie)
        """
        mean_traj   = np.zeros((n_cycles, self.dim))
        spread_traj = np.zeros((n_cycles, self.dim))
        rmse        = np.zeros(n_cycles)

        for cyc in range(n_cycles):
            # 1. Propagation
            self._propagate()

            # 2. Observation
            obs = obs_func(cyc)

            # 3. Analyse
            x_mean = self._analysis(obs)

            # 4. Statistiques
            A = self._get_ensemble_matrix()
            mean_traj[cyc]   = x_mean
            spread_traj[cyc] = A.std(axis=1)

            if true_traj is not None:
                rmse[cyc] = np.sqrt(np.mean((x_mean - true_traj[cyc])**2))

            if cyc % 50 == 0:
                print(f"  cycle {cyc:4d}/{n_cycles} | "
                      f"RMSE={rmse[cyc]:.4e} | "
                      f"spread={spread_traj[cyc].mean():.4e}")

        lib.enkf_free_all()
        return mean_traj, spread_traj, rmse


# ================================================================== #
#  Exemple complet avec vérité terrain synthétique                    #
# ================================================================== #
if __name__ == "__main__":

    np.random.seed(12345)

    dt       = 1e-5
    da_step  = 100        # pas de modèle entre 2 analyses
    n_cycles = 200        # nombre de cycles d'assimilation
    Ne       = 50         # taille de l'ensemble

    # --- Nombres d'onde ---
    sh = np.array([0.125 * 2.0**i for i in range(N)])

    # --- CI "vraie" trajectoire ---
    X0_true = sh**(-1/3)
    Y0_true = np.ones(N) * 1e-4

    # 1. Génération de la vraie trajectoire (membre unique, forçage déterministe)
    print("Génération de la vraie trajectoire...")
    true_id = create_member(X0_true, Y0_true, dt=dt,
                            force=0.005, force_rnd=True)
    true_traj = np.zeros((n_cycles, 2*N))
    for cyc in range(n_cycles):
        step_n(true_id, da_step)
        Xt, Yt = get_state(true_id)
        true_traj[cyc, 0::2] = Xt
        true_traj[cyc, 1::2] = Yt
    lib.enkf_free_all()
    print(f"Vraie trajectoire : shape={true_traj.shape}")

    # 2. Fonction d'observation : X des shells 0..10 + bruit
    obs_noise = 0.05
    obs_shells = list(range(8,19))   # on observe les 11 premiers shells

    def obs_func(cyc):
        obs_true = true_traj[cyc, 0::2][obs_shells]   # X[shell] vrais
        return obs_true + obs_noise * np.random.randn(len(obs_shells))

    # 3. CI initiales de l'ensemble : vraie CI + perturbation
    X0_ens = X0_true.copy()
    Y0_ens = Y0_true.copy()

    # 4. Lancement de l'EnKF
    enkf = EnKF(n_members=Ne,
                obs_shells=obs_shells,
                obs_noise=obs_noise,
                dt=dt,
                da_step=da_step,
                force=0.005,
                force_rnd=True)

    enkf.init_ensemble(X0_ens, Y0_ens, spread=0.05)

    print(f"\nDémarrage EnKF : {n_cycles} cycles, {Ne} membres...")
    mean_traj, spread_traj, rmse = enkf.run(n_cycles, obs_func,
                                            true_traj=true_traj)

    # ================================================================ #
    #  Visualisation                                                    #
    # ================================================================ #
    t_axis = np.arange(n_cycles) * da_step * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- RMSE ---
    axes[0,0].semilogy(t_axis, rmse)
    axes[0,0].set_xlabel("Temps")
    axes[0,0].set_ylabel("RMSE")
    axes[0,0].set_title("RMSE EnKF vs vraie trajectoire")
    axes[0,0].grid(True)

    # --- Spread moyen ---
    axes[0,1].semilogy(t_axis, spread_traj.mean(axis=1))
    axes[0,1].set_xlabel("Temps")
    axes[0,1].set_ylabel("Spread moyen")
    axes[0,1].set_title("Spread de l'ensemble")
    axes[0,1].grid(True)

    # --- X moyen analysé vs vrai (shell 0) ---
    axes[1,0].plot(t_axis, true_traj[:, 0],  label="Vrai",    lw=1.5)
    axes[1,0].plot(t_axis, mean_traj[:, 0],  label="EnKF",    lw=1.5, ls='--')
    axes[1,0].fill_between(t_axis,
                            mean_traj[:,0] - spread_traj[:,0],
                            mean_traj[:,0] + spread_traj[:,0],
                            alpha=0.3, label="±1σ")
    axes[1,0].set_xlabel("Temps")
    axes[1,0].set_title("Shell 0 — X(t)")
    axes[1,0].legend()
    axes[1,0].grid(True)

    # --- Énergie par shell (vraie vs analysée, dernier cycle) ---
    E_true = true_traj[-1, 0::2]**2 + true_traj[-1, 1::2]**2
    E_enkf = mean_traj[-1, 0::2]**2 + mean_traj[-1, 1::2]**2
    sh_vals = np.array([0.125 * 2.0**i for i in range(N)])
    axes[1,1].loglog(sh_vals, E_true, 'o-', label="Vrai")
    axes[1,1].loglog(sh_vals, E_enkf, 's--', label="EnKF")
    axes[1,1].set_xlabel("k (nombre d'onde)")
    axes[1,1].set_ylabel("Énergie")
    axes[1,1].set_title("Spectre d'énergie — dernier cycle")
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig("enkf_results.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : enkf_results.png")

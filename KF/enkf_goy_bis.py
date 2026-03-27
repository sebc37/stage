import numpy as np
import matplotlib.pyplot as plt
import tqdm
from goy import GoyModel

# ── chemins ───────────────────────────────────────────────────────────────────
PATH      = "/home/s26calme/Documents/code_stage/GOY-main/"
SAVE      = "/home/s26calme/Documents/code_stage/KF/"
path_data = PATH + "data_test_precis.dat"

# ── données de référence ──────────────────────────────────────────────────────
data       = np.loadtxt(path_data)
Nmax       = data.shape[0]
debut      = int(0.1 * Nmax)
Data_shell = data[debut:Nmax, :]   # shape (Npts, 44)
Npts       = Data_shell.shape[0]

# ── modèle GOY ────────────────────────────────────────────────────────────────
DT        = 1e-5
FS        = 100.0
FORCE     = 0.005
N_FORCE   = 4
FORCE_RND = 0
count_init    = int(1000.0 / DT)        # 99999999
N_fs          = int(1.0 / DT / FS)      # 999
n_steps_first = count_init % N_fs       # 99  (premier save)

model = GoyModel(dt=DT, force=FORCE, N_force=N_FORCE, force_rnd=FORCE_RND)
N_shells = model.N   # 22

# ── paramètres EnKF ───────────────────────────────────────────────────────────
n  = 44    # taille état [X0,Y0,X1,Y1,...,X21,Y21]
p  = 12    # observations shells 4-9 Re+Im (indices 8..19)
Ne = 50    # nombre de membres

k_min_obs = 4
k_max_obs = 10

# bruit observation : 5% de la variance de chaque shell observé
shell_var = np.array([np.mean(Data_shell[:, k]**2) for k in range(n)])
R_diag    = 0.05**2 * shell_var[2*k_min_obs : 2*k_max_obs]
R         = np.diag(R_diag)

# bruit modèle : très petit (GOY est déterministe)
var_Q = 1e-10
Q     = var_Q * np.eye(n)

# ── matrice d'observation ─────────────────────────────────────────────────────
H         = np.eye(n)[2*k_min_obs : 2*k_max_obs, :]   # shape (12, 44)

# ── observations bruitées ─────────────────────────────────────────────────────
noise_obs = np.random.normal(0, R_diag[:, None],      #
                             size=(p, Npts))
y_obs     = Data_shell[:, 2*k_min_obs:2*k_max_obs].T + noise_obs  # (p, Npts)

# ── fonction m : intègre 1 pas de fichier (N_fs itérations AB2) ───────────────
def m_step(Xpp, Ypp, Xp, Yp):
    """
    Intègre le modèle GOY de N_fs pas.
    Entrée  : Xpp,Ypp (t-dt)  Xp,Yp (t)
    Sortie  : Xpp_new,Ypp_new (t+N_fs*dt - dt)  Xp_new,Yp_new (t+N_fs*dt)
    """
    (Xpp_new, Ypp_new), (Xp_new, Yp_new) = model.integrate(
        Xpp, Ypp, Xp, Yp, n_steps=N_fs)
    return Xpp_new, Ypp_new, Xp_new, Yp_new

# ── initialisation de l'ensemble à partir des données réelles ─────────────────
# On initialise chaque membre en prenant la ligne j du fichier
# + une petite perturbation physiquement cohérente (std = 1% de l'amplitude)
nb_enkf = min(500, Npts - 2)   # nombre de pas EnKF

j_start  = 1000    # on commence à la ligne 2 (besoin de j_start-1 et j_start-2)

# amplitude physique par composante pour calibrer la perturbation initiale
amp = np.std(Data_shell, axis=0)   # std temporelle de chaque composante

# Pour chaque membre: on stocke (Xpp, Ypp, Xp, Yp) séparément
# shape: (Ne, 22) pour chacun des 4 tableaux
ens_Xpp = np.zeros((Ne, N_shells))
ens_Ypp = np.zeros((Ne, N_shells))
ens_Xp  = np.zeros((Ne, N_shells))
ens_Yp  = np.zeros((Ne, N_shells))

# Etat de référence à j_start :
# ligne j_start-1 = Xpp (état à t - N_fs*dt), mais on a besoin de t - dt
# => on intègre N_fs-1 pas depuis (j_start-2, j_start-1) pour avoir Xpp exact
(cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
    Data_shell[j_start-2, 0::2], Data_shell[j_start-2, 1::2],
    Data_shell[j_start-1, 0::2], Data_shell[j_start-1, 1::2],
    n_steps=N_fs - 1)
(ref_Xpp, ref_Ypp), (ref_Xp, ref_Yp) = model.integrate(
    cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=1)
# ref_Xp/Yp = Data_shell[j_start] à la précision machine
# ref_Xpp/Ypp = état à t - dt

for i in range(Ne):
    noise = np.random.randn(n) * amp * 0.01   # perturbation 1%
    ens_Xpp[i] = ref_Xpp + noise[0::2] * 0.1  # Xpp varie peu
    ens_Ypp[i] = ref_Ypp + noise[1::2] * 0.1
    ens_Xp[i]  = ref_Xp  + noise[0::2]
    ens_Yp[i]  = ref_Yp  + noise[1::2]

# ── tableaux de résultats ─────────────────────────────────────────────────────
x_f_enkf = np.zeros((n, nb_enkf))
x_a_enkf = np.zeros((n, nb_enkf))
P_a_diag = np.zeros((n, nb_enkf))

# ── boucle EnKF ───────────────────────────────────────────────────────────────
x_f_tmp = np.zeros((n, Ne))
y_f_tmp = np.zeros((p, Ne))

# tableaux pour stocker (Xpp,Ypp,Xp,Yp) après prévision
fens_Xpp = np.zeros((Ne, N_shells))
fens_Ypp = np.zeros((Ne, N_shells))
fens_Xp  = np.zeros((Ne, N_shells))
fens_Yp  = np.zeros((Ne, N_shells))

for k in tqdm.tqdm(range(nb_enkf)):  #nb_enkf

    # ── étape de prévision ────────────────────────────────────────────────────
    nan_members = 0
    for i in range(Ne):
        try:
            Xpp_n, Ypp_n, Xp_n, Yp_n = m_step(
                ens_Xpp[i], ens_Ypp[i], ens_Xp[i], ens_Yp[i])

            # bruit modèle sur Xp uniquement (Xpp reste cohérent)
            noise_q = np.random.randn(n) * np.sqrt(var_Q)
            Xp_n += noise_q[0::2]
            Yp_n += noise_q[1::2]

        except RuntimeError:
            # membre divergé : le réinitialiser autour de la moyenne courante
            nan_members += 1
            mean_Xp = np.mean(fens_Xp if k > 0 else ens_Xp, axis=0)
            mean_Yp = np.mean(fens_Yp if k > 0 else ens_Yp, axis=0)
            noise = np.random.randn(n) * amp * 0.05
            Xpp_n = mean_Xp + noise[0::2]*0.1
            Ypp_n = mean_Yp + noise[1::2]*0.1
            Xp_n  = mean_Xp + noise[0::2]
            Yp_n  = mean_Yp + noise[1::2]

        fens_Xpp[i] = Xpp_n
        fens_Ypp[i] = Ypp_n
        fens_Xp[i]  = Xp_n
        fens_Yp[i]  = Yp_n

        # vecteur état aplati pour Kalman
        x_f_tmp[0::2, i] = Xp_n
        x_f_tmp[1::2, i] = Yp_n
        y_f_tmp[:, i]    = H @ x_f_tmp[:, i] + \
                            np.random.multivariate_normal(np.zeros(p), R)

    if nan_members > 0:
        print(f"\n  t={k+j_start}: {nan_members} membres réinitialisés")

    # ── gain de Kalman ────────────────────────────────────────────────────────
    P_f = np.cov(x_f_tmp)
    K_g = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)

    # ── étape d'analyse ───────────────────────────────────────────────────────
    obs_k = y_obs[:, k + j_start]

    if np.any(np.isfinite(obs_k)):
        for i in range(Ne):
            innov = obs_k - y_f_tmp[:, i]
            dx    = K_g @ innov          # correction (n,)

            # on corrige Xp/Yp seulement — Xpp reste inchangé (cohérence AB2)
            fens_Xp[i] += dx[0::2]
            fens_Yp[i] += dx[1::2]
    # sinon on garde la prévision telle quelle

    # l'ensemble analysé devient l'entrée du prochain pas
    ens_Xpp[:] = fens_Xpp
    ens_Ypp[:] = fens_Ypp
    ens_Xp[:]  = fens_Xp
    ens_Yp[:]  = fens_Yp

    # ── stockage ──────────────────────────────────────────────────────────────
    x_f_enkf[:, k] = np.mean(x_f_tmp, axis=1)
    x_a_enkf[0::2, k] = np.mean(fens_Xp, axis=0)
    x_a_enkf[1::2, k] = np.mean(fens_Yp, axis=0)
    P_a_diag[:, k] = np.var(
        np.column_stack([
            np.column_stack((fens_Xp[:, j],fens_Yp[:,j])) for j in range(N_shells)
        ]), axis=0)[:n]   # approximation diagonale

# ── plots ─────────────────────────────────────────────────────────────────────
time = np.arange(nb_enkf)
ref  = Data_shell[j_start : j_start + nb_enkf, :]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
shells_plot = [4, 5, 6, 7]
for idx, s in enumerate(shells_plot):
    ax = axes[idx//2, idx%2]
    ax.plot(time, ref[:, 2*s],        'b',  label='Vérité',  lw=1)
    ax.plot(time, y_obs[s-k_min_obs, j_start:j_start+nb_enkf],
                                       '.k', label='Obs',     ms=2, alpha=0.5)
    ax.plot(time, x_a_enkf[2*s, :],   'r',  label='EnKF',    lw=1)
    #plt.fill_between(time, x_a_enkf - 1.96*np.sqrt(P_a_diag[idx,idx]), x_a_enkf[idx] + 1.96*np.sqrt(P_a_diag[idx,idx]), facecolor='red', alpha=0.5)
    plt.xlabel('Time', size=20)
    ax.set_title(f'Shell {s} (Re)')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(SAVE + "enkf_shells.png", dpi=150)
plt.close()

# RMSE
rmse_obs  = np.sqrt(np.mean((y_obs[0:p, j_start:j_start+nb_enkf]
                              - ref[:, 2*k_min_obs:2*k_max_obs].T)**2))
rmse_enkf = np.sqrt(np.mean((x_a_enkf[2*k_min_obs:2*k_max_obs, :]
                              - ref[:, 2*k_min_obs:2*k_max_obs].T)**2))
print(f"RMSE obs  : {rmse_obs:.4e}")
print(f"RMSE EnKF : {rmse_enkf:.4e}")

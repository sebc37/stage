from goy import GoyModel
import numpy as np
import matplotlib.pyplot as plt

model = GoyModel(force_rnd=0)

ref = np.loadtxt("/home/s26calme/Documents/code_stage/GOY-main/data_test_precis.dat")

def init(x_past,N_fs,n_steps_first):
    # ── choix du point de départ ──────────────────────────────────────────────────
    #i      = 10000   # ligne du fichier depuis laquelle on repart
     # nombre de lignes suivantes à reproduire

    N_fs          = 999   # pas entre deux lignes du fichier
    n_steps_first = 100   # pas spéciaux pour la ligne 0 (voir run_goy.py)

    # ── reconstruction de Xpp (état à t_i - dt) ──────────────────────────────────
    # On part de la ligne i-1 et on intègre 998 pas → on arrive à t_i - dt
    if i == 0:
        # cas particulier : ligne 0, on repart des CI
        Xpp0, Ypp0, Xp0, Yp0 = model.init_fields()
        (Xpp, Ypp), (Xp, Yp) = model.integrate(Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first - 1)
    else:
        # ligne i-1 → intègre N_fs-1 pas → arrive à t_i - dt
        if i == 1:
            Xpp0, Ypp0, Xp0, Yp0 = model.init_fields()
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first)
        else:
            Xp_prev2 = x_past[0, 0::2];  Yp_prev2 = x_past[0, 1::2]
            Xp_prev1 = x_past[1, 0::2];  Yp_prev1 = x_past[1, 1::2]
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xp_prev2, Yp_prev2, Xp_prev1, Yp_prev1, n_steps=N_fs - 1)

        (Xpp, Ypp), (Xp, Yp) = model.integrate(
            cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=1)
        # maintenant Xp/Yp = ref[i] à la précision machine, Xpp/Ypp = état à t_i - dt
    return Xpp,Ypp,Xp,Yp

i=5000
N_fs          = 999
Xpp,Ypp,Xp,Yp = init(ref[i-2:i],N_fs=N_fs,n_steps_first=100)
n_pred = 10  
# ── vérification que Xp coïncide bien avec ref[i] ────────────────────────────
print(f"Vérification ligne {i}:")
print(f"  |Xp - ref[i]|_max = {np.max(np.abs(Xp - ref[i, 0::2])):.3e}")
print(f"  |Yp - ref[i]|_max = {np.max(np.abs(Yp - ref[i, 1::2])):.3e}")

# ── intégration des n_pred lignes suivantes ───────────────────────────────────
pred = np.zeros((n_pred, 44))
cur_Xpp, cur_Ypp = Xpp.copy(), Ypp.copy()
cur_Xp,  cur_Yp  = Xp.copy(),  Yp.copy()

for k in range(n_pred):
    (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
        cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=N_fs)
    pred[k, 0::2] = cur_Xp
    pred[k, 1::2] = cur_Yp

# ── comparaison avec les lignes i+1 .. i+n_pred du fichier ───────────────────
ref_window = ref[i+1 : i+1+n_pred]
diff_X = pred[:, 0::2] - ref_window[:, 0::2]
diff_Y = pred[:, 1::2] - ref_window[:, 1::2]
plt.figure()
plt.plot(ref[i:i+n_pred,9],label="truth")
plt.plot(pred[:,9],label="pred")
plt.legend()
plt.show()
plt.figure()
plt.plot(ref[i:i+n_pred,16],label="truth")
plt.plot(pred[:,16],label="pred")
plt.legend()
plt.show()

print(f"\nComparaison lib vs ref sur {n_pred} lignes à partir de i={i} :")
print(f"{'ligne':>6}  {'|ΔX|_max':>12}  {'|ΔY|_max':>12}")
for k in range(n_pred):
    print(f"  {i+1+k:4d}   {np.max(np.abs(diff_X[k])):.3e}    {np.max(np.abs(diff_Y[k])):.3e}")
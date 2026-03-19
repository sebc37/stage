import numpy as np
def make_collocation_dataset(Data_shell, k_min_collocation, k_max_collocation,
                              mean, std, ratio, t_min, t_max, seed=123456):
    """
    Génère un dataset de collocation points de la forme (k, t, u).

    Paramètres
    ----------
    Data_shell        : np.ndarray, shape (Npts, nb_shells) — données déjà slicées [debut:Nmax]
                        NON encore centrées-réduites
    k_min_collocation : int — indice de shell minimum (ex: 4)
    k_max_collocation : int — indice de shell maximum (ex: 10)
    mean              : list — moyenne par colonne (retournée par filter_mode ou calculée sur Data_shell)
    std               : list — écart-type par colonne
    ratio             : float — fraction de points conservés aléatoirement
    t_min             : float — temps physique correspondant à debut (ex: 0.1)
    t_max             : float — temps physique correspondant à Nmax (ex: 1.0)
    seed              : int

    Retourne
    --------
    X_dataset : list[list] — [X_posx, X_posy, X_value] avec
                X_posx  : indice de shell k (entier, dans [k_min_collocation, k_max_collocation))
                X_posy  : temps normalisé dans [t_min, t_max]
                X_value : valeur u centrée-réduite
    perc      : float — pourcentage de points conservés
    """
    np.random.seed(seed)

    Npts = Data_shell.shape[0]   # = Nmax - debut

    # colonnes réelles et imaginaires à parcourir
    col_min = 2 * k_min_collocation
    col_max = 2 * k_max_collocation

    X_posx  = []   # indice de shell k (réel ou imaginaire → même k)
    X_posy  = []   # temps dans [t_min, t_max]
    X_value = []   # u centré-réduit
    X_is_im = []   # 0 = partie réelle, 1 = partie imaginaire (utile si n_output=2)

    for j in range(col_min, col_max):
        k_shell  = j // 2          # ✅ indice de shell réel dans [k_min, k_max)
        is_imag  = j % 2           # 0 → Re, 1 → Im

        for i in range(Npts):
            if np.random.random() <= ratio:
                u_raw = Data_shell[i, j]

                # ✅ temps normalisé dans [t_min, t_max] cohérent avec grid_data
                t_norm = t_min + (i / Npts) * (t_max - t_min)

                # ✅ normalisation u avec les stats de la colonne j
                u_norm = (u_raw - mean[j]) / std[j]

                X_posx.append(k_shell)
                X_posy.append(t_norm)
                X_value.append(u_norm)
                X_is_im.append(is_imag)

    perc = ratio * 100
    X_dataset = [X_posx, X_posy, X_value, X_is_im]
    return X_dataset, perc
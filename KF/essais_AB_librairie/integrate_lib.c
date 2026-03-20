/*
 * integrate_lib.c
 * Wrapper pour appel depuis Python via ctypes.
 * Permet de fixer les conditions initiales, le dt et la durée depuis Python.
 *
 * Compilation :
 *   gcc -shared -fPIC -O2 -o integrate_lib.so \
 *       integrate_lib.c integrate.c integrate_io.c stats_io.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "parameters.h"

/* ------------------------------------------------------------------ */
/* Variables globales définies dans integrate.c                        */
/* ------------------------------------------------------------------ */
extern double sh[N];
extern double X[N], Xp[N], Xpp[N];
extern double Y[N], Yp[N], Ypp[N];
extern double NXp[N], NXpp[N];
extern double NYp[N], NYpp[N];
extern double A[N], A2[N], A3[N], D[N];
extern int N_steps, N_fs;

/* ------------------------------------------------------------------ */
/* Macros compute_NX / compute_NY (GOY par défaut)                    */
/* ------------------------------------------------------------------ */
#ifdef SABRA_MODEL
  void compute_NX_Sabra(double*, double*, double*);
  void compute_NY_Sabra(double*, double*, double*);
  #define compute_NX compute_NX_Sabra
  #define compute_NY compute_NY_Sabra
#else
  void compute_NX_GOY(double*, double*, double*);
  void compute_NY_GOY(double*, double*, double*);
  #define compute_NX compute_NX_GOY
  #define compute_NY compute_NY_GOY
#endif

/* ------------------------------------------------------------------ */
/* Paramètres dynamiques (modifiables depuis Python)                   */
/* ------------------------------------------------------------------ */
static double _dt       = dt;
static double _force    = force;
static int    _force_rnd = force_rnd;

/* ------------------------------------------------------------------ */
/* Initialisation des shells et coefficients (utilise _dt)            */
/* ------------------------------------------------------------------ */
static void _init_coeffs()
{   int i;
    for (i=0; i<N; i++) sh[i]  = k0 * pow(lmb, (double)i);
    for (i=0; i<N; i++) A[i]   = exp(-nu * sh[i]*sh[i] * _dt);
    for (i=0; i<N; i++) A2[i]  = -eps / lmb;
    for (i=0; i<N; i++) A3[i]  = -(1.0 - eps) / (lmb * lmb);
    for (i=0; i<N; i++) D[i]   = 0.0;
    D[N_force] = 1.0;
}

/* ------------------------------------------------------------------ */
/* API Python — initialisation                                         */
/* ------------------------------------------------------------------ */

/*
 * lib_init_default()
 * CI originales du code : spectre en k^{-1/3} pour X, 1e-4 pour Y.
 */
void lib_init_default()
{
    int i;
    _init_coeffs();

    for (i=0; i<N; i++)
    {   Xpp[i] = pow(sh[i], -1./3.);
        Ypp[i] = 1.e-4;
    }
    compute_NX(Xpp, Ypp, NXpp);
    compute_NY(Xpp, Ypp, NYpp);

    for (i=0; i<N; i++)
    {   Xp[i] = A[i] * (Xpp[i] + _dt * NXpp[i]);
        Yp[i] = A[i] * (Ypp[i] + _dt * NYpp[i]);
    }
    compute_NX(Xp, Yp, NXp);
    compute_NY(Xp, Yp, NYp);
    N_steps = 0;
}

/*
 * lib_init_custom(X0, Y0, n, dt_user, force_user, force_rnd_user)
 * X0, Y0     : conditions initiales (tableaux numpy de taille N)
 * n          : taille (doit valoir N=22)
 * dt_user    : pas de temps
 * force_user : amplitude du forçage
 * frnd       : forçage aléatoire (1) ou déterministe (0)
 */
void lib_init_custom(double *X0, double *Y0, int n,
                     double dt_user, double force_user, int frnd)
{
    int i;
    _dt        = dt_user;
    _force     = force_user;
    _force_rnd = frnd;

    _init_coeffs();

    for (i=0; i<N; i++)
    {   Xpp[i] = (i < n) ? X0[i] : 0.0;
        Ypp[i] = (i < n) ? Y0[i] : 0.0;
    }
    compute_NX(Xpp, Ypp, NXpp);
    compute_NY(Xpp, Ypp, NYpp);

    for (i=0; i<N; i++)
    {   Xp[i] = A[i] * (Xpp[i] + _dt * NXpp[i]);
        Yp[i] = A[i] * (Ypp[i] + _dt * NYpp[i]);
    }
    compute_NX(Xp, Yp, NXp);
    compute_NY(Xp, Yp, NYp);
    N_steps = 0;
}

/* ------------------------------------------------------------------ */
/* API Python — un seul pas d'intégration                             */
/* ------------------------------------------------------------------ */
void lib_step()
{
    int i;
    double force1, force2;

    if (_force_rnd)
    {   force1 = _force * drand48();
        force2 = _force * drand48();
    }
    else
    {   force1 = _force;
        force2 = _force;
    }

    for (i=0; i<N; i++)
    {   X[i] = A[i]*Xp[i]
              + 1.5*_dt*A[i]*NXp[i]
              - 0.5*_dt*A[i]*A[i]*NXpp[i]
              + _dt*force1*D[i];
        Y[i] = A[i]*Yp[i]
              + 1.5*_dt*A[i]*NYp[i]
              - 0.5*_dt*A[i]*A[i]*NYpp[i]
              + _dt*force2*D[i];
    }

    for (i=0; i<N; i++)
    {   Xpp[i]  = Xp[i];    Ypp[i]  = Yp[i];
        NXpp[i] = NXp[i];   NYpp[i] = NYp[i];
        Xp[i]   = X[i];     Yp[i]   = Y[i];
    }
    compute_NX(Xp, Yp, NXp);
    compute_NY(Xp, Yp, NYp);
}

/* ------------------------------------------------------------------ */
/* API Python — run complet (le plus rapide, tout en C)               */
/* ------------------------------------------------------------------ */
/*
 * lib_run(n_steps, save_every, out_X, out_Y, n_out)
 * Exécute n_steps pas et sauvegarde tous les save_every pas.
 * out_X, out_Y : buffers numpy de forme (n_out, N) applatis en (n_out*N,)
 * Retourne le nombre de snapshots effectivement écrits.
 */
int lib_run(int n_steps, int save_every,
            double *out_X, double *out_Y, int n_out)
{
    int step, snap = 0, i;

    for (step = 0; step < n_steps && snap < n_out; step++)
    {
        lib_step();
        if ((step + 1) % save_every == 0)
        {   for (i=0; i<N; i++)
            {   out_X[snap*N + i] = X[i];
                out_Y[snap*N + i] = Y[i];
            }
            snap++;
        }
    }
    return snap;
}

/* ------------------------------------------------------------------ */
/* Lecture des champs courants                                         */
/* ------------------------------------------------------------------ */
void   lib_get_X (double *out) { int i; for(i=0;i<N;i++) out[i]=X[i];  }
void   lib_get_Y (double *out) { int i; for(i=0;i<N;i++) out[i]=Y[i];  }
void   lib_get_sh(double *out) { int i; for(i=0;i<N;i++) out[i]=sh[i]; }
int    lib_get_N ()            { return N;   }
double lib_get_dt()            { return _dt; }

/* ------------------------------------------------------------------ */
/* Modification des nombres d'onde sh[]                               */
/* ------------------------------------------------------------------ */

/*
 * lib_set_sh(sh_new, n)
 * Remplace sh[] par les valeurs fournies ET recalcule A[] en conséquence.
 * A appeler APRES lib_init_default() ou lib_init_custom(),
 * et AVANT lib_run() / lib_step().
 *
 * sh_new : tableau de taille n (doit valoir N=22)
 */
void lib_set_sh(double *sh_new, int n)
{   int i;
    for (i=0; i<N; i++)
        sh[i] = (i < n) ? sh_new[i] : 0.0;

    /* Recalcul de A[i] = exp(-nu * sh[i]^2 * dt) avec les nouveaux sh */
    for (i=0; i<N; i++)
        A[i] = exp(-nu * sh[i]*sh[i] * _dt);
}

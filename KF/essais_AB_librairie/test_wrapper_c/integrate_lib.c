/*
 * integrate_lib.c
 * Wrapper minimal autour de integrate.c pour l'appel depuis Python via ctypes.
 * Ne modifie pas integrate.c — utilise directement ses variables globales.
 *
 * Compilation :
 *   gcc -shared -fPIC -O2 -o integrate_lib.so integrate_lib.c integrate.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include "parameters.h"   /* #define N, dt, force, lmb, k0, nu, eps, ... */

/* ------------------------------------------------------------------ */
/* Déclaration des variables globales définies dans integrate.c        */
/* ------------------------------------------------------------------ */
extern double sh[N];
extern double X[N], Xp[N], Xpp[N];
extern double Y[N], Yp[N], Ypp[N];
extern double Xf[N], Yf[N];
extern double NXp[N], NXpp[N];
extern double NYp[N], NYpp[N];
extern double A[N], A1[N], A2[N], A3[N];
extern double D[N], F[N];
extern int N_steps, N_fs;

/* ------------------------------------------------------------------ */
/* Prototypes des fonctions de integrate.c                             */
/* ------------------------------------------------------------------ */
void init_NL();
void init_forcing();
void init_shell();
void init_fields();

/* compute_NX / compute_NY sont des macros dans integrate.h
   qui pointent vers GOY ou Sabra selon le #define dans parameters.h  */
#ifdef SABRA_MODEL
  void compute_NX_Sabra(double *ax, double *ay, double *res);
  void compute_NY_Sabra(double *ax, double *ay, double *res);
  #define compute_NX compute_NX_Sabra
  #define compute_NY compute_NY_Sabra
#else
  void compute_NX_GOY(double *ax, double *ay, double *res);
  void compute_NY_GOY(double *ax, double *ay, double *res);
  #define compute_NX compute_NX_GOY
  #define compute_NY compute_NY_GOY
#endif

/* ------------------------------------------------------------------ */
/* API exportée vers Python                                            */
/* ------------------------------------------------------------------ */

/* Initialisation complète (appelle les 3 inits de integrate.c) */
void lib_init()
{
    init_NL();
    init_forcing();
    init_shell();
    init_fields();
}

/* Un pas d'intégration Adams-Bashforth + mise à jour des champs */
void lib_step()
{
    int i;
    double force1, force2;

    /* Forçage : aléatoire ou déterministe selon parameters.h */
    if (force_rnd)
    {   force1 = force * drand48();
        force2 = force * drand48();
    }
    else
    {   force1 = force;
        force2 = force;
    }

    /* Schéma Adams-Bashforth ordre 2 (identique à integrate()) */
    for (i = 0; i < N; i++)
    {   X[i] = A[i]*Xp[i]
              + 1.5*dt*A[i]*NXp[i]
              - 0.5*dt*A[i]*A[i]*NXpp[i]
              + dt*force1*D[i];
        Y[i] = A[i]*Yp[i]
              + 1.5*dt*A[i]*NYp[i]
              - 0.5*dt*A[i]*A[i]*NYpp[i]
              + dt*force2*D[i];
    }

    /* Décalage temporel : pp <- p <- courant */
    for (i = 0; i < N; i++)
    {   Xpp[i]  = Xp[i];   Ypp[i]  = Yp[i];
        NXpp[i] = NXp[i];  NYpp[i] = NYp[i];
        Xp[i]   = X[i];    Yp[i]   = Y[i];
    }

    /* Recalcul du terme non-linéaire pour le prochain pas */
    compute_NX(Xp, Yp, NXp);
    compute_NY(Xp, Yp, NYp);
}

/* Copie X courant dans le buffer fourni par Python */
void lib_get_X(double *out)
{   int i;
    for (i = 0; i < N; i++) out[i] = X[i];
}

/* Copie Y courant dans le buffer fourni par Python */
void lib_get_Y(double *out)
{   int i;
    for (i = 0; i < N; i++) out[i] = Y[i];
}

/* Copie les nombres d'onde dans le buffer fourni par Python */
void lib_get_sh(double *out)
{   int i;
    for (i = 0; i < N; i++) out[i] = sh[i];
}

/* Retourne N (utile côté Python pour allouer les buffers) */
int lib_get_N()
{   return N; }

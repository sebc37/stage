/*
 * enkf_lib.c
 * Intégration du modèle shell GOY par RK4.
 * Entrée : Xpp, Ypp (état initial)
 * Sortie : X(t+n), Y(t+n) après n pas RK4
 *
 * Compilation :
 *   gcc -shared -fPIC -O2 -o enkf_lib.so enkf_lib.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "parameters.h"

#define DT        dt
#define FORCE_AMP force
#define FORCE_RND force_rnd

/* ================================================================== */
/* Coefficients (calculés une seule fois)                             */
/* ================================================================== */
static double sh[N], A_diss[N], A2[N], A3[N], D[N];
static int coeffs_ready = 0;

static void init_coeffs()
{   int i;
    for (i=0; i<N; i++) sh[i]     = k0 * pow(lmb, (double)i);
    for (i=0; i<N; i++) A_diss[i] = exp(-nu * sh[i]*sh[i] * DT);
    for (i=0; i<N; i++) A2[i]     = -eps / lmb;
    for (i=0; i<N; i++) A3[i]     = -(1.0-eps) / (lmb*lmb);
    for (i=0; i<N; i++) D[i]      = 0.0;
    D[N_force] = 1.0;
    coeffs_ready = 1;
}

/* ================================================================== */
/* Termes non-linéaires GOY : NX et NY                               */
/* ================================================================== */
static void compute_NX(double *ax, double *ay, double *res)
{   int k;
    res[0] = ax[2]*ay[1] + ay[2]*ax[1];
    res[1] = ax[3]*ay[2] + ay[3]*ax[2]
           + A2[1]*(ax[2]*ay[0] + ay[2]*ax[0]);
    for (k=2; k<N-2; k++)
        res[k] = (ax[k+2]*ay[k+1] + ay[k+2]*ax[k+1])
               + A2[k]*(ax[k+1]*ay[k-1] + ay[k+1]*ax[k-1])
               + A3[k]*(ax[k-1]*ay[k-2] + ay[k-1]*ax[k-2]);
    res[N-2] = A2[N-2]*(ax[N-1]*ay[N-3] + ay[N-1]*ax[N-3])
             + A3[N-2]*(ax[N-3]*ay[N-4] + ay[N-3]*ax[N-4]);
    res[N-1] = A3[N-1]*(ax[N-2]*ay[N-3] + ay[N-2]*ax[N-3]);
    for (k=0; k<N; k++) res[k] *= sh[k];
}

static void compute_NY(double *ax, double *ay, double *res)
{   int k;
    res[0] = ax[2]*ax[1] - ay[1]*ay[2];
    res[1] = ax[3]*ax[2] - ay[2]*ay[3]
           + A2[1]*(ax[2]*ax[0] - ay[2]*ay[0]);
    for (k=2; k<N-2; k++)
        res[k] = ax[k+2]*ax[k+1] - ay[k+1]*ay[k+2]
               + A2[k]*(ax[k+1]*ax[k-1] - ay[k+1]*ay[k-1])
               + A3[k]*(ax[k-1]*ax[k-2] - ay[k-1]*ay[k-2]);
    res[N-2] = A2[N-2]*(ax[N-1]*ax[N-3] + ay[N-1]*ay[N-3])
             + A3[N-2]*(ax[N-3]*ax[N-4] + ay[N-3]*ay[N-4]);
    res[N-1] = A3[N-1]*(ax[N-2]*ax[N-3] - ay[N-2]*ay[N-3]);
    for (k=0; k<N; k++) res[k] *= sh[k];
}

/* ================================================================== */
/* Dérivée complète du système : dX/dt et dY/dt                      */
/*                                                                    */
/* Le modèle GOY avec dissipation s'écrit :                          */
/*   dX/dt = NX(X,Y) - nu*sh^2 * X + force*D                        */
/*   dY/dt = NY(X,Y) - nu*sh^2 * Y + force*D                        */
/* ================================================================== */
static void deriv(double *X, double *Y,
                  double f1,  double f2,
                  double *dX, double *dY)
{   int i;
    double NX[N], NY[N];
    compute_NX(X, Y, NX);
    compute_NY(X, Y, NY);
    for (i=0; i<N; i++) {
        dX[i] = NX[i] - nu*sh[i]*sh[i]*X[i] + f1*D[i];
        dY[i] = NY[i] - nu*sh[i]*sh[i]*Y[i] + f2*D[i];
    }
}

/* ================================================================== */
/* Un pas RK4                                                         */
/* ================================================================== */
static void rk4_step(double *X, double *Y, double *Xout, double *Yout)
{   int i;
    double f1, f2;
    double k1X[N], k1Y[N];
    double k2X[N], k2Y[N];
    double k3X[N], k3Y[N];
    double k4X[N], k4Y[N];
    double Xtmp[N], Ytmp[N];

    /* Forçage constant sur le pas (tiré une fois par pas) */
    f1 = FORCE_RND ? FORCE_AMP * drand48() : FORCE_AMP;
    f2 = FORCE_RND ? FORCE_AMP * drand48() : FORCE_AMP;

    /* k1 = f(X, Y) */
    deriv(X, Y, f1, f2, k1X, k1Y);

    /* k2 = f(X + dt/2 * k1, Y + dt/2 * k1) */
    for (i=0; i<N; i++) {
        Xtmp[i] = X[i] + 0.5*DT*k1X[i];
        Ytmp[i] = Y[i] + 0.5*DT*k1Y[i];
    }
    deriv(Xtmp, Ytmp, f1, f2, k2X, k2Y);

    /* k3 = f(X + dt/2 * k2, ...) */
    for (i=0; i<N; i++) {
        Xtmp[i] = X[i] + 0.5*DT*k2X[i];
        Ytmp[i] = Y[i] + 0.5*DT*k2Y[i];
    }
    deriv(Xtmp, Ytmp, f1, f2, k3X, k3Y);

    /* k4 = f(X + dt * k3, ...) */
    for (i=0; i<N; i++) {
        Xtmp[i] = X[i] + DT*k3X[i];
        Ytmp[i] = Y[i] + DT*k3Y[i];
    }
    deriv(Xtmp, Ytmp, f1, f2, k4X, k4Y);

    /* Combinaison finale */
    for (i=0; i<N; i++) {
        Xout[i] = X[i] + (DT/6.0)*(k1X[i] + 2.0*k2X[i] + 2.0*k3X[i] + k4X[i]);
        Yout[i] = Y[i] + (DT/6.0)*(k1Y[i] + 2.0*k2Y[i] + 2.0*k3Y[i] + k4Y[i]);
    }
}

/* ================================================================== */
/* API                                                                 */
/* ================================================================== */

/*
 * step_n(Xpp, Ypp, Xout, Yout, n_steps)
 *
 * Xpp, Ypp   : état initial
 * Xout, Yout : état après n_steps pas RK4
 * n_steps    : nombre de pas
 */
void step_n(double *Xpp, double *Ypp,
            double *Xout, double *Yout,
            int n_steps)
{   int s, i;
    double cur_X[N], cur_Y[N];
    double nxt_X[N], nxt_Y[N];

    if (!coeffs_ready) init_coeffs();

    /* Copie de l'état initial */
    for (i=0; i<N; i++) { cur_X[i]=Xpp[i]; cur_Y[i]=Ypp[i]; }

    /* n_steps pas RK4 */
    for (s=0; s<n_steps; s++) {
        rk4_step(cur_X, cur_Y, nxt_X, nxt_Y);
        for (i=0; i<N; i++) { cur_X[i]=nxt_X[i]; cur_Y[i]=nxt_Y[i]; }
    }

    for (i=0; i<N; i++) { Xout[i]=cur_X[i]; Yout[i]=cur_Y[i]; }
}

int  get_N()             { return N; }
void get_sh(double *out) {
    int i;
    if (!coeffs_ready) init_coeffs();
    for (i=0; i<N; i++) out[i] = sh[i];
}

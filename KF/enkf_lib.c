/*
 * enkf_lib.c
 * Intègre X, Y sur n pas de temps en conservant l'historique entre appels.
 *
 * Compilation :
 *   gcc -shared -fPIC -O2 -o enkf_lib.so enkf_lib.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include "parameters.h"

#define DT        dt
#define FORCE_AMP force
#define FORCE_RND force_rnd

typedef struct {
    double X[N], Xp[N], Xpp[N];
    double Y[N], Yp[N], Ypp[N];
    double NXp[N], NXpp[N];
    double NYp[N], NYpp[N];
    double sh[N], A[N], A2[N], A3[N], D[N];
    int initialized;
} State;

static void compute_NX(State *s, double *ax, double *ay, double *res)
{   int k;
    res[0] = ax[2]*ay[1] + ay[2]*ax[1];
    res[1] = ax[3]*ay[2] + ay[3]*ax[2] + s->A2[1]*(ax[2]*ay[0] + ay[2]*ax[0]);
    for (k=2; k<N-2; k++)
        res[k] = (ax[k+2]*ay[k+1] + ay[k+2]*ax[k+1])
               + s->A2[k]*(ax[k+1]*ay[k-1] + ay[k+1]*ax[k-1])
               + s->A3[k]*(ax[k-1]*ay[k-2] + ay[k-1]*ax[k-2]);
    res[N-2] = s->A2[N-2]*(ax[N-1]*ay[N-3] + ay[N-1]*ax[N-3])
             + s->A3[N-2]*(ax[N-3]*ay[N-4] + ay[N-3]*ax[N-4]);
    res[N-1] = s->A3[N-1]*(ax[N-2]*ay[N-3] + ay[N-2]*ax[N-3]);
    for (k=0; k<N; k++) res[k] *= s->sh[k];
}

static void compute_NY(State *s, double *ax, double *ay, double *res)
{   int k;
    res[0] = ax[2]*ax[1] - ay[1]*ay[2];
    res[1] = ax[3]*ax[2] - ay[2]*ay[3] + s->A2[1]*(ax[2]*ax[0] - ay[2]*ay[0]);
    for (k=2; k<N-2; k++)
        res[k] = ax[k+2]*ax[k+1] - ay[k+1]*ay[k+2]
               + s->A2[k]*(ax[k+1]*ax[k-1] - ay[k+1]*ay[k-1])
               + s->A3[k]*(ax[k-1]*ax[k-2] - ay[k-1]*ay[k-2]);
    res[N-2] = s->A2[N-2]*(ax[N-1]*ax[N-3] + ay[N-1]*ay[N-3])
             + s->A3[N-2]*(ax[N-3]*ax[N-4] + ay[N-3]*ay[N-4]);
    res[N-1] = s->A3[N-1]*(ax[N-2]*ax[N-3] - ay[N-2]*ay[N-3]);
    for (k=0; k<N; k++) res[k] *= s->sh[k];
}

static void init_coeffs(State *s)
{   int i;
    for (i=0; i<N; i++) s->sh[i] = k0 * pow(lmb, (double)i);
    for (i=0; i<N; i++) s->A[i]  = exp(-nu * s->sh[i]*s->sh[i] * DT);
    for (i=0; i<N; i++) s->A2[i] = -eps / lmb;
    for (i=0; i<N; i++) s->A3[i] = -(1.0-eps) / (lmb*lmb);
    for (i=0; i<N; i++) s->D[i]  = 0.0;
    s->D[N_force] = 1.0;
}

static void do_step(State *s)
{   int i;
    double f1 = FORCE_RND ? FORCE_AMP * drand48() : FORCE_AMP;
    double f2 = FORCE_RND ? FORCE_AMP * drand48() : FORCE_AMP;
    double tmpX[N], tmpY[N];

    for (i=0; i<N; i++) {
        tmpX[i] = s->A[i]*s->Xp[i]
                + 1.5*DT*s->A[i]*s->NXp[i]
                - 0.5*DT*s->A[i]*s->A[i]*s->NXpp[i]
                + DT*f1*s->D[i];
        tmpY[i] = s->A[i]*s->Yp[i]
                + 1.5*DT*s->A[i]*s->NYp[i]
                - 0.5*DT*s->A[i]*s->A[i]*s->NYpp[i]
                + DT*f2*s->D[i];
    }
    for (i=0; i<N; i++) {
        s->Xpp[i]=s->Xp[i];   s->Ypp[i]=s->Yp[i];
        s->NXpp[i]=s->NXp[i]; s->NYpp[i]=s->NYp[i];
        s->Xp[i]=tmpX[i];     s->Yp[i]=tmpY[i];
        s->X[i]=tmpX[i];      s->Y[i]=tmpY[i];
    }
    compute_NX(s, s->Xp, s->Yp, s->NXp);
    compute_NY(s, s->Xp, s->Yp, s->NYp);
}

/* ================================================================== */
/* État global unique                                                  */
/* ================================================================== */
static State _s = {.initialized = 0};

/* ================================================================== */
/* API                                                                 */
/* ================================================================== */

/*
 * init(X_in, Y_in)
 * Initialise l'historique Adams-Bashforth depuis (X_in, Y_in).
 * À appeler UNE SEULE FOIS au début, ou après reset().
 */
void init(double *X_in, double *Y_in)
{   int i;
    if (!_s.initialized) init_coeffs(&_s);

    /* Xpp = CI */
    for (i=0; i<N; i++) { _s.Xpp[i]=X_in[i]; _s.Ypp[i]=Y_in[i]; }
    compute_NX(&_s, _s.Xpp, _s.Ypp, _s.NXpp);
    compute_NY(&_s, _s.Xpp, _s.Ypp, _s.NYpp);

    /* Xp = un pas Euler pour amorcer */
    for (i=0; i<N; i++) {
        _s.Xp[i] = _s.A[i]*(_s.Xpp[i] + DT*_s.NXpp[i]);
        _s.Yp[i] = _s.A[i]*(_s.Ypp[i] + DT*_s.NYpp[i]);
    }
    compute_NX(&_s, _s.Xp, _s.Yp, _s.NXp);
    compute_NY(&_s, _s.Xp, _s.Yp, _s.NYp);

    _s.initialized = 1;
}

/*
 * step_n(X_out, Y_out, n_steps)
 * Avance de n_steps pas et retourne l'état final.
 * L'historique est CONSERVÉ entre les appels successifs.
 * Appeler init() avant le premier step_n().
 */
void step_n(double *X_out, double *Y_out, int n_steps)
{   int s;
    for (s=0; s<n_steps; s++) do_step(&_s);
    int i;
    for (i=0; i<N; i++) { X_out[i]=_s.X[i]; Y_out[i]=_s.Y[i]; }
}

/*
 * reset()
 * Force la réinitialisation (à appeler si tu changes l'état extérieurement).
 */
void reset() { _s.initialized = 0; }

int  get_N()             { return N; }
void get_sh(double *out) {
    int i; State tmp; init_coeffs(&tmp);
    for (i=0; i<N; i++) out[i] = tmp.sh[i];
}

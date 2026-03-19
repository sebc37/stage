/*
 * enkf_lib.c
 * Intègre X, Y sur n pas de temps.
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

static void init_history(State *s, double *X_in, double *Y_in)
{   int i;
    for (i=0; i<N; i++) { s->Xpp[i]=X_in[i]; s->Ypp[i]=Y_in[i]; }
    compute_NX(s, s->Xpp, s->Ypp, s->NXpp);
    compute_NY(s, s->Xpp, s->Ypp, s->NYpp);
    for (i=0; i<N; i++) {
        s->Xp[i] = s->A[i]*(s->Xpp[i] + DT*s->NXpp[i]);
        s->Yp[i] = s->A[i]*(s->Ypp[i] + DT*s->NYpp[i]);
    }
    compute_NX(s, s->Xp, s->Yp, s->NXp);
    compute_NY(s, s->Xp, s->Yp, s->NYp);
    s->initialized = 1;
}

static void do_step(State *s, double *X_out, double *Y_out)
{   int i;
    double f1 = FORCE_RND ? FORCE_AMP * drand48() : FORCE_AMP;
    double f2 = FORCE_RND ? FORCE_AMP * drand48() : FORCE_AMP;

    for (i=0; i<N; i++) {
        X_out[i] = s->A[i]*s->Xp[i]
                 + 1.5*DT*s->A[i]*s->NXp[i]
                 - 0.5*DT*s->A[i]*s->A[i]*s->NXpp[i]
                 + DT*f1*s->D[i];
        Y_out[i] = s->A[i]*s->Yp[i]
                 + 1.5*DT*s->A[i]*s->NYp[i]
                 - 0.5*DT*s->A[i]*s->A[i]*s->NYpp[i]
                 + DT*f2*s->D[i];
    }
    for (i=0; i<N; i++) {
        s->Xpp[i]=s->Xp[i];   s->Ypp[i]=s->Yp[i];
        s->NXpp[i]=s->NXp[i]; s->NYpp[i]=s->NYp[i];
        s->Xp[i]=X_out[i];    s->Yp[i]=Y_out[i];
    }
    compute_NX(s, s->Xp, s->Yp, s->NXp);
    compute_NY(s, s->Xp, s->Yp, s->NYp);
}

/* ================================================================== */
/* API                                                                 */
/* ================================================================== */
static State _s = {.initialized = 0};

/*
 * step_n(X_in, Y_in, X_out, Y_out, n_steps)
 * Intègre n_steps pas à partir de (X_in, Y_in).
 * Retourne l'état final dans (X_out, Y_out).
 */
void step_n(double *X_in, double *Y_in,
            double *X_out, double *Y_out,
            int n_steps)
{   int i, s;
    double tmpX[N], tmpY[N];

    if (!_s.initialized) init_coeffs(&_s);
    init_history(&_s, X_in, Y_in);

    /* n_steps - 1 pas intermédiaires (résultat dans tmpX/tmpY) */
    for (s=0; s<n_steps-1; s++)
        do_step(&_s, tmpX, tmpY);

    /* dernier pas → directement dans X_out / Y_out */
    do_step(&_s, X_out, Y_out);
}

/*
 * reset()
 * Réinitialise l'historique (à appeler après une mise à jour externe).
 */
void reset() { _s.initialized = 0; }

int  get_N()            { return N; }
void get_sh(double *out) {
    int i; State tmp; init_coeffs(&tmp);
    for (i=0; i<N; i++) out[i] = tmp.sh[i];
}

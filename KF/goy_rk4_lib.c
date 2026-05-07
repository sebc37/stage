#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef N_MAX
#define N_MAX 64
#endif

typedef struct {
    int    N;
    double k0;
    double lmb;
    double eps;
    double nu;
    double dt;
    double force;
    int    N_force;
    int    force_rnd;
} GoyParams;

static GoyParams p;
static double sh[N_MAX];
static double A2[N_MAX];
static double A3[N_MAX];
static double D[N_MAX];

void goy_init(const GoyParams *params) {
    int i;
    p = *params;
    for (i = 0; i < p.N; i++) {
        sh[i] = p.k0 * pow(p.lmb, (double)i);
        A2[i] = -p.eps / p.lmb;
        A3[i] = -(1.0 - p.eps) / (p.lmb * p.lmb);
        D[i]  = (i == p.N_force) ? 1.0 : 0.0;
    }
}

// Calcul de la dérivée complète : dU/dt = NL(U) - nu*k^2*U + Force
static void goy_deriv(const double *X, const double *Y, double *dX, double *dY) {
    int k, n = p.N;
    double NX[N_MAX], NY[N_MAX];

    // Termes non-linéaires - Partie Réelle (NX)
    NX[0] = X[2]*Y[1] + Y[2]*X[1];
    NX[1] = X[3]*Y[2] + Y[3]*X[2] + A2[1] * (X[2]*Y[0] + Y[2]*X[0]);
    for (k = 2; k < n-2; k++)
        NX[k] = (X[k+2]*Y[k+1] + Y[k+2]*X[k+1]) + A2[k]*(X[k+1]*Y[k-1] + Y[k+1]*X[k-1]) + A3[k]*(X[k-1]*Y[k-2] + Y[k-1]*X[k-2]);
    NX[n-2] = A2[n-2] * (X[n-1]*Y[n-3] + Y[n-1]*X[n-3]) + A3[n-2] * (X[n-3]*Y[n-4] + Y[n-3]*X[n-4]);
    NX[n-1] = A3[n-1] * (X[n-2]*Y[n-3] + Y[n-2]*X[n-3]);

    // Termes non-linéaires - Partie Imaginaire (NY)
    NY[0] = X[2]*X[1] - Y[1]*Y[2];
    NY[1] = X[3]*X[2] - Y[2]*Y[3] + A2[1] * (X[2]*X[0] - Y[2]*Y[0]);
    for (k = 2; k < n-2; k++)
        NY[k] = X[k+2]*X[k+1] - Y[k+1]*Y[k+2] + A2[k] * (X[k+1]*X[k-1] - Y[k+1]*Y[k-1]) + A3[k] * (X[k-1]*X[k-2] - Y[k-1]*Y[k-2]);
    NY[n-2] = A2[n-2] * (X[n-1]*X[n-3] + Y[n-1]*Y[n-3]) + A3[n-2] * (X[n-3]*X[n-4] + Y[n-3]*Y[n-4]);
    NY[n-1] = A3[n-1] * (X[n-2]*X[n-3] - Y[n-2]*Y[n-3]);

    for (k = 0; k < n; k++) {
        dX[k] = NX[k] * sh[k] - p.nu * sh[k] * sh[k] * X[k] + p.force * D[k];
        dY[k] = NY[k] * sh[k] - p.nu * sh[k] * sh[k] * Y[k] + p.force * D[k];
    }
}

// Un pas d'intégration RK4
void goy_step_rk4(double *X, double *Y) {
    int i, n = p.N;
    double dt = p.dt;
    double k1x[N_MAX], k1y[N_MAX], k2x[N_MAX], k2y[N_MAX], k3x[N_MAX], k3y[N_MAX], k4x[N_MAX], k4y[N_MAX];
    double tx[N_MAX], ty[N_MAX];

    goy_deriv(X, Y, k1x, k1y);
    for(i=0; i<n; i++) { tx[i] = X[i] + 0.5*dt*k1x[i]; ty[i] = Y[i] + 0.5*dt*k1y[i]; }
    goy_deriv(tx, ty, k2x, k2y);
    for(i=0; i<n; i++) { tx[i] = X[i] + 0.5*dt*k2x[i]; ty[i] = Y[i] + 0.5*dt*k2y[i]; }
    goy_deriv(tx, ty, k3x, k3y);
    for(i=0; i<n; i++) { tx[i] = X[i] + dt*k3x[i]; ty[i] = Y[i] + dt*k3y[i]; }
    goy_deriv(tx, ty, k4x, k4y);

    for(i=0; i<n; i++) {
        X[i] += (dt/6.0)*(k1x[i] + 2.0*k2x[i] + 2.0*k3x[i] + k4x[i]);
        Y[i] += (dt/6.0)*(k1y[i] + 2.0*k2y[i] + 2.0*k3y[i] + k4y[i]);
    }
}

// Fonction utilitaire pour intégrer plusieurs pas d'un coup
int goy_integrate_rk4(double *X, double *Y, int n_steps) {
    for (int step = 0; step < n_steps; step++) {
        goy_step_rk4(X, Y);
        if (X[0] != X[0]) return -1; // Détection de NaN
    }
    return 0;
}
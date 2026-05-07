#ifndef GOY_LIB_H
#define GOY_LIB_H

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

void goy_init(const GoyParams *params);
void goy_step_rk4(double *X, double *Y);
int goy_integrate_rk4(double *X, double *Y, int n_steps);

#endif
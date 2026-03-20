// integrate_lib.h  (nouveau fichier)
#ifndef INTEGRATE_LIB_H
#define INTEGRATE_LIB_H

void lib_init(int n_shells, double lmb_, double k0_, double nu_, 
              double dt_, double eps_, double force_, int force_rnd_);
void lib_init_fields();
void lib_step();           // une seule itération
void lib_get_X(double *out, int n);
void lib_get_Y(double *out, int n);
int  lib_get_N();

#endif
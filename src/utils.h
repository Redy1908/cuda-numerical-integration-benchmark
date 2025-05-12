#ifndef UTILS_H
#define UTILS_H

#include "functions.h"

#ifdef __cplusplus
extern "C" {
#endif

void get_integration_parameters(double *a_ptr, double *b_ptr, int *n_ptr);
void select_integration_function();
double get_cur_time();
cudaDeviceProp check_GPU_and_architecture();

#ifdef __cplusplus
}
#endif

#endif
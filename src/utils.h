#ifndef UTILS_H
#define UTILS_H

#include "functions.h"

#ifdef __cplusplus
extern "C" {
#endif

int get_positive_integer_input(const char* prompt);
int parse_double_or_constant(const char* prompt, double* value);
void get_integration_parameters(double* a, double* b, int* n);
void get_function_choice(FunctionChoice* choice_ptr);
double get_cur_time();
cudaDeviceProp check_GPU_and_architecture();

#ifdef __cplusplus
}
#endif

#endif
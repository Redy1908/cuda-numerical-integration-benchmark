#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "utils.h"

int get_positive_integer_input(const char* display_prompt, int* value_to_update) {
    char line_buffer[100];
    int temp_value;

    printf("%s", display_prompt);
    if (fgets(line_buffer, sizeof(line_buffer), stdin) == NULL) {
        printf("Error reading input line.\n");
        return 0; 
    }

    line_buffer[strcspn(line_buffer, "\n")] = 0;

    if (line_buffer[0] == '\0') {
        return 1; 
    }

    char* endptr;
    temp_value = strtol(line_buffer, &endptr, 10);
    if (endptr != line_buffer && *endptr == '\0' && temp_value > 0) {
        *value_to_update = temp_value;
        return 1;
    }
    
    printf("Invalid input. Please enter a positive integer or press Enter for current value.\n");
    return 0;
}

int parse_double_or_constant(const char* display_prompt, double* value_to_update) {
  char line_buffer[100];
  char input_str[100];
  char numeric_part_str[100];
  char* endptr_full_string;
  char* endptr_numeric_part;
  double multiplier;
  double temp_value;

  printf("%s", display_prompt);
  if (fgets(line_buffer, sizeof(line_buffer), stdin) == NULL) {
      printf("Error reading input line.\n");
      return 0;
  }

  line_buffer[strcspn(line_buffer, "\n")] = 0;

  if (line_buffer[0] == '\0') {
      return 1; 
  }

  strncpy(input_str, line_buffer, sizeof(input_str) - 1);
  input_str[sizeof(input_str) - 1] = '\0';

  for (int i = 0; input_str[i]; i++) {
      input_str[i] = toupper(input_str[i]);
  }

  if (strcmp(input_str, "E") == 0) {
      *value_to_update = M_E;
      return 1;
  } else if (strcmp(input_str, "PI") == 0) {
      *value_to_update = M_PI;
      return 1;
  }

  size_t len = strlen(input_str);

  if (len > 2 && strcmp(input_str + len - 2, "PI") == 0) {
      strncpy(numeric_part_str, input_str, len - 2);
      numeric_part_str[len - 2] = '\0';
      multiplier = strtod(numeric_part_str, &endptr_numeric_part);
      if (endptr_numeric_part != numeric_part_str && *endptr_numeric_part == '\0') {
          *value_to_update = multiplier * M_PI;
          return 1;
      }
  }
  else if (len > 1 && input_str[len - 1] == 'E') {
      strncpy(numeric_part_str, input_str, len - 1);
      numeric_part_str[len - 1] = '\0';
      multiplier = strtod(numeric_part_str, &endptr_numeric_part);
      if (endptr_numeric_part != numeric_part_str && *endptr_numeric_part == '\0') {
          *value_to_update = multiplier * M_E;
          return 1;
      }
  }

  temp_value = strtod(input_str, &endptr_full_string);
  if (endptr_full_string != input_str && *endptr_full_string == '\0') {
      *value_to_update = temp_value;
      return 1;
  }

  printf("Invalid input. Please enter a number, 'E', 'PI', a combination, or press Enter for current value.\n");
  return 0;
}

void get_integration_parameters(double* a_ptr, double* b_ptr, int* n_ptr) {
  char prompt_str[256];
  char choice_buffer[10];

  printf("\n--- Integration Parameters ---\n");
  printf("Default parameters are: a = 0, b = 2PI, n = 1000000\n");
  printf("Do you want to use these default parameters? (Y/n, default is Y): ");

  if (fgets(choice_buffer, sizeof(choice_buffer), stdin) == NULL) {
      printf("Error reading choice. Using default parameters.\n");
  } else {
      choice_buffer[strcspn(choice_buffer, "\n")] = 0; 
  }

  if (choice_buffer[0] == '\0' || toupper(choice_buffer[0]) == 'Y') {
      printf("Using default integration parameters.\n");
  } else if (toupper(choice_buffer[0]) == 'N') {
      printf("Please enter your custom integration parameters (press Enter at any prompt to keep its current value):\n");

      while (1) {
        snprintf(prompt_str, sizeof(prompt_str), "Enter the lower integration limit (a) [current: %f]: ", *a_ptr);
        if (parse_double_or_constant(prompt_str, a_ptr) == 1) break;
      }

      while (1) {
        snprintf(prompt_str, sizeof(prompt_str), "Enter the upper integration limit (b) [current: %f]: ", *b_ptr);
        if (parse_double_or_constant(prompt_str, b_ptr) == 1) {
            if (*b_ptr > *a_ptr) {
                break;
            } else {
                printf("The upper limit (b=%f) must be greater than the lower limit (a=%f).\n", *b_ptr, *a_ptr);
            }
        }
      }

      while (1) {
        snprintf(prompt_str, sizeof(prompt_str), "Enter the number of trapezoids (n) [current: %d]: ", *n_ptr);
        if (get_positive_integer_input(prompt_str, n_ptr) == 1) break;
      }
  } else {
      printf("Invalid choice. Using default integration parameters.\n");
  }

  printf("------------------------------------\n\n");
  printf("--- Integration Parameters Set ---\n");
  printf("Lower limit (a): %f\n", *a_ptr);
  printf("Upper limit (b): %f\n", *b_ptr);
  printf("Number of trapezoids (n): %d\n", *n_ptr);
  printf("------------------------------------\n\n");
}

const char* get_function_name_as_string(FunctionChoice choice) {
    switch (choice) {
        case FUNC_DEFAULT: return "x * e^(-x) * cos(2x)";
        case FUNC_SIN_X:   return "sin(x)";
        case FUNC_X_SQUARED: return "x^2";
        default: return "Unknown Function";
    }
}

void print_function_options() {
    printf("Available functions f(x):\n");
    for (int i = 0; i < NUM_FUNCTION_CHOICES; ++i) {
        printf("  %d. %s\n", i + 1, get_function_name_as_string((FunctionChoice)i));
    }
}

void get_function_choice(FunctionChoice* current_choice_ptr) {
    char line_buffer[100];
    int temp_value_int;
    int choice_made = 0;

    printf("\n--- Function f(x) Selection ---\n");
    print_function_options();

    while (!choice_made) {
        printf("Select function f(x) by number [current: %s (%d)]: ",
               get_function_name_as_string(*current_choice_ptr), *current_choice_ptr + 1);

        if (fgets(line_buffer, sizeof(line_buffer), stdin) == NULL) {
            printf("Error reading input line. Keeping current selection.\n");
            choice_made = 1; 
            continue;
        }
        line_buffer[strcspn(line_buffer, "\n")] = 0; 

        if (line_buffer[0] == '\0') {
            printf("No input. Keeping current selection: %s\n", get_function_name_as_string(*current_choice_ptr));
            choice_made = 1;
            continue;
        }

        char* endptr;
        temp_value_int = strtol(line_buffer, &endptr, 10);

        if (endptr != line_buffer && *endptr == '\0') {
            FunctionChoice selected_enum_val = (FunctionChoice)(temp_value_int - 1); // Adjust to 0-based

            if (selected_enum_val >= 0 && selected_enum_val < NUM_FUNCTION_CHOICES) {
                *current_choice_ptr = selected_enum_val;
                printf("Selected function: %s\n", get_function_name_as_string(*current_choice_ptr));
                choice_made = 1;
            } else {
                printf("Invalid selection. Please choose from the available options.\n");
                print_function_options();
            }
        } else {
            printf("Invalid input. Please enter a number corresponding to the function.\n");
            print_function_options();
        }
    }
    printf("Function for integration set to: %s\n", get_function_name_as_string(*current_choice_ptr));
    printf("------------------------------------\n\n");
}

double get_cur_time() {
    struct timespec ts;
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

cudaDeviceProp check_GPU_and_architecture() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d -> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Can't retrive the number of GPUs. Exiting...\n");
        exit(0);
    }
  
    if (deviceCount == 0) {
        printf("No CUDA GPU found. Exiting...\n");
        exit(0);
    }
  
    int device = 0;
    cudaSetDevice(device);
  
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
  
    FILE* fp_gpu_property = fopen("../results/gpu_property.csv", "w");
    fprintf(fp_gpu_property, "GPUName,SMs,MaxThreadsPerBlock,MaxThreadsPerSM,WarpSize\n");
  
    fprintf(fp_gpu_property, "%s,%d,%d,%d,%d\n",
      deviceProp.name,
      deviceProp.multiProcessorCount,
      deviceProp.maxThreadsPerBlock,
      deviceProp.maxThreadsPerMultiProcessor,
      deviceProp.warpSize
    );
  
    fclose(fp_gpu_property);
  
    return deviceProp;
  }
  
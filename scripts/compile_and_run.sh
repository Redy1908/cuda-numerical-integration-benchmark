#!/bin/bash
cd "$(dirname "$0")"

echo "Compiling CUDA code..."

CUDA_VERSION_OUTPUT=$(nvcc --version 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: nvcc (CUDA Toolkit) not found or not executable. Please ensure it's installed and in your PATH."
  exit 1
fi

CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION_OUTPUT" | grep -oP 'release \K[0-9]+' | head -n 1)

if [ -z "$CUDA_MAJOR_VERSION" ]; then
  echo "Error: Could not determine CUDA Toolkit version from 'nvcc --version'."
  echo "Output was:"
  echo "$CUDA_VERSION_OUTPUT"
  exit 1
fi

echo "Detected CUDA Toolkit major version: $CUDA_MAJOR_VERSION"

if [ "$CUDA_MAJOR_VERSION" -lt 9 ]; then
  echo "Error: CUDA Toolkit version $CUDA_MAJOR_VERSION.x is not supported. Version 9.0 or higher is required."
  exit 1
else
  echo "CUDA Toolkit version $CUDA_MAJOR_VERSION.x meets the requirement (>= 9.0)."
fi

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 2>/dev/null)
SM_VERSION=$(echo $COMPUTE_CAP | sed 's/\.//' 2>/dev/null)

ARCH_FLAG=""

if [ -z "$SM_VERSION" ]; then
  echo "Warning: Could not automatically detect GPU Compute Capability via nvidia-smi."
  echo "         Falling back to a default architecture (sm_70)."
  echo "         For optimal performance, ensure nvidia-smi is working or specify architecture manually."
  ARCH_FLAG="-arch=sm_70"
else
  echo "Detected GPU Compute Capability: $COMPUTE_CAP (sm_$SM_VERSION)"
  ARCH_FLAG="-arch=sm_$SM_VERSION"
fi

echo "Using architecture flag: $ARCH_FLAG"

nvcc ../src/main.cu ../src/functions.cu ../src/utils.cu -o ../bin/main_app -std=c++11 -lm $ARCH_FLAG
if [ $? -ne 0 ]; then
  echo -e "Compilation failed.\n"
  exit 1
fi
echo -e "Compilation successful.\n"

echo "Executing application..."
../bin/main_app

echo -e "\nGenerating plots..."
python3 plot.py
echo "Plots generated and saved to /plots."
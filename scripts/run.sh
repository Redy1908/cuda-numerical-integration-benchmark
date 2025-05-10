#!/bin/bash
cd "$(dirname "$0")"

echo "Application ecxution..."
../bin/main_app

echo -e "\nGenerating plots..."

python3 plot.py
echo "Plot generated and saved to /plots."
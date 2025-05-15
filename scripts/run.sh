#!/bin/bash
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found."
    echo "Python 3 is required by this script to execute 'plot.py', which generates graphical plots of the benchmark results."
    echo "Please install Python 3 to generate plots."
    echo "You can typically install it using your system's package manager, e.g.:"
    echo "sudo apt update && sudo apt install python3 python3-pip"
    echo "Then, install the required packages using: pip install -r ../requirements.txt"
    exit 1
fi

if [ ! -f "../bin/main_app" ]; then
    echo "Error: Executable ../bin/main_app not found."
    echo "Please compile the project first using ./scripts/compile.sh or ./scripts/compile_and_run.sh"
    exit 1
fi

echo "Application ecxution..."
../bin/main_app

echo -e "\nGenerating plots..."

python3 plot.py
echo "Plot generated and saved to /plots."
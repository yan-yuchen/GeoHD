#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated test runner script for a set of Python test files.

This script will import the necessary warnings filter to ignore specific warnings,
and then sequentially run a series of test scripts while handling potential errors.

Usage:
    python test_automation.py

Make sure that the test scripts (visualize_shapefile_test.py, plot_g_function_test.py, etc.)
are located in the same directory as this script, or provide the correct path to them.
"""

import warnings
import os
import subprocess
import sys

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def run_test_script(script_name):
    """
    Run a given test script using subprocess while handling potential errors.

    Parameters:
    script_name (str): The name of the Python test script to run.
    """
    try:
        print(f"Running {script_name}...")
        # Run the script and capture the output
        result = subprocess.run(["python", script_name], capture_output=True, text=True, check=True)
        # Print the output if needed
        # print(result.stdout)
        # Print the error if needed
        # print(result.stderr)
        print(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        # If there is an error, print it but continue with the next scripts
        print(f"An error occurred while running {script_name}: {e.output}")
        # If you want to stop the execution on error, use 'exit(1)' instead of 'pass'
        pass

# Redirect standard error to discard error messages
original_stderr = sys.stderr
# sys.stderr = open('/dev/null', 'w')  # For Unix-like systems
sys.stderr = open(r'NUL', 'w')  # For Windows systems

# List of test scripts to run
test_scripts = [
    "visualize_shapefile_test.py",
    "plot_g_function_test.py",
    "grid_visualization_test.py",
    "plot_density_raster_test.py",
    "adaptive_kde_test.py",
    "extract_hotspots_test.py"
]

# Run the test scripts
for script in test_scripts:
    run_test_script(script)

# Restore standard error after all tests have run
sys.stderr = original_stderr
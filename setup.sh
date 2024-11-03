#!/bin/bash

# This script sets up the Python environment for Streamlit Cloud.

# Specify the Python version
PYTHON_VERSION=3.9

# Create a virtual environment with the specified Python version
python$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

#!/bin/bash
# Ensure dynamic linker finds the arm64 libomp.dylib
export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"
source venv/bin/activate
python3 src/evaluate.py

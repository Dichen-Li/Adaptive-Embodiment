#!/bin/bash

# Ensure two arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <URDF_PATH> <USD_PATH>"
  exit 1
fi

# Assign arguments to variables
URDF_PATH=$1
USD_PATH=$2

# Run the command with the provided paths
${ISAAC_LAB_PATH}/isaaclab.sh -p ${ISAAC_LAB_PATH}/source/standalone/tools/convert_urdf.py \
  "$URDF_PATH" \
  "$USD_PATH" \
  --make-instanceable
  --headless
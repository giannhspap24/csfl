#!/bin/bash

for file in *.py
do
  echo "Running $file..."
  python "$file"  
  if [ $? -ne 0 ]; then
    echo "Error running $file. Exiting."
    exit 1  # Exit if any script fails
  fi
done

echo "All scripts ran successfully."

#!/bin/bash

# Base paths
SRC_BASE=~/miso/TRACER/results
DEST_BASE=~/miso/user-simulator/examples

# Loop through each domain in the source directory
for domain in "$SRC_BASE"/*; do
  domain_name=$(basename "$domain")
  
  # Loop through each execution_X directory inside the domain
  for exec_dir in "$domain"/execution_*; do
    exec_name=$(basename "$exec_dir")
    src_profiles="$exec_dir/profiles"

    # If the profiles directory exists
    if [ -d "$src_profiles" ]; then
      dest_profiles="$DEST_BASE/$domain_name/profiles/$exec_name"
      
      # Create destination directory if it doesn't exist
      mkdir -p "$dest_profiles"
      
      # Copy all profile files
      cp "$src_profiles"/* "$dest_profiles"
      
      echo "Copied profiles from $src_profiles to $dest_profiles"
    fi
  done
done

echo "All profile directories copied successfully."


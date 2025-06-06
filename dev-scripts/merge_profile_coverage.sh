#!/bin/bash

# Base directory for results (relative to TRACER root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$TRACER_ROOT/results"

echo "Starting automatic profile coverage merger..."

# Find all execution directories and process them
find "$RESULTS_DIR" -name "execution_*" -type d | while read -r execution_dir; do
    # Extract domain from path (e.g., "bikeshop" from "results/bikeshop/execution_1")
    domain=$(basename $(dirname "$execution_dir"))
    execution=$(basename "$execution_dir")
    
    # Look for profile_logs directory
    logs_dir="$execution_dir/profile_logs"
    
    if [ ! -d "$logs_dir" ]; then
        echo "Skipping $execution_dir - no profile_logs directory found"
        continue
    fi
    
    # Find all unique basenames in the logs directory
    # Extract basename from files like "bikeshop_profile_1_uuid" -> "bikeshop_profile_1"
    basenames=$(find "$logs_dir" -maxdepth 1 -type f ! -name ".*" | \
                sed 's/.*\///' | \
                sed 's/_[0-9a-f-]\{36\}$//' | \
                sort -u)
    
    if [ -z "$basenames" ]; then
        echo "No profile coverage files found in $logs_dir"
        continue
    fi
    
    # Process each basename found
    for basename in $basenames; do
        output_dir="$execution_dir/profile_coverage"
        
        echo "Processing $basename in $domain/$execution..."
        
        python "$TRACER_ROOT/src/scripts/coverage_merger.py" "$basename" \
            --input-dir "$logs_dir/" \
            --output-dir "$output_dir/"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully merged $basename"
        else
            echo "✗ Failed to merge $basename"
        fi
    done
done

echo "Automatic profile coverage merging completed!"

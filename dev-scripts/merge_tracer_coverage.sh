#!/bin/bash

# Base directory for results
RESULTS_DIR="results"

echo "Starting automatic coverage merger..."

# Find all execution directories and process them
find "$RESULTS_DIR" -name "execution_*" -type d | while read -r execution_dir; do
    # Extract domain from path (e.g., "bikeshop" from "results/bikeshop/execution_1")
    domain=$(basename $(dirname "$execution_dir"))
    execution=$(basename "$execution_dir")
    
    # Look for tracer_logs directory
    logs_dir="$execution_dir/tracer_logs"
    
    if [ ! -d "$logs_dir" ]; then
        echo "Skipping $execution_dir - no tracer_logs directory found"
        continue
    fi
    
    # Find all unique basenames in the logs directory
    # Extract basename from files like "bikeshop1_uuid" -> "bikeshop1"
    basenames=$(find "$logs_dir" -maxdepth 1 -type f ! -name ".*" | \
                sed 's/.*\///' | \
                sed 's/_[0-9a-f-]\{36\}$//' | \
                sort -u)
    
    if [ -z "$basenames" ]; then
        echo "No coverage files found in $logs_dir"
        continue
    fi
    
    # Process each basename found
    for basename in $basenames; do
        output_dir="$execution_dir/tracer_coverage"
        
        echo "Processing $basename in $domain/$execution..."
        
        python src/scripts/coverage_merger.py "$basename" \
            --input-dir "$logs_dir/" \
            --output-dir "$output_dir/"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully merged $basename"
        else
            echo "✗ Failed to merge $basename"
        fi
    done
done

echo "Automatic coverage merging completed!"

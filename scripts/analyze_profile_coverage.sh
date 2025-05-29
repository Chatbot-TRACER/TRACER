#!/bin/bash

# Base directory for results
RESULTS_DIR="results"

echo "Starting automatic profile coverage analysis..."

# Find all profile coverage JSON files and analyze them
find "$RESULTS_DIR" -path "*/profiles_coverage/*_coverage.json" -type f | while read -r coverage_file; do
    echo "Analyzing: $coverage_file"
    
    python src/scripts/coverage_analyzer.py "$coverage_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully analyzed $(basename "$coverage_file")"
    else
        echo "✗ Failed to analyze $(basename "$coverage_file")"
    fi
    echo "---"
done

echo "Profile coverage analysis completed for all files!"

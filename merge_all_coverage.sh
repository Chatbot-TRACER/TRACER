#!/bin/bash
# filepath: /home/ivan/miso/TRACER/merge_all_coverage.sh

# Base directory for results
RESULTS_DIR="results"

# Function to merge coverage for a specific execution
merge_coverage() {
    local domain=$1
    local execution=$2
    local basename=$3
    
    local input_dir="${RESULTS_DIR}/${domain}/${execution}/exploration_logs/"
    local output_dir="${RESULTS_DIR}/${domain}/${execution}/exploration_coverage/"
    
    echo "Processing ${basename} in ${domain}/${execution}..."
    
    if [ -d "$input_dir" ]; then
        python src/scripts/coverage_merger.py "$basename" \
            --input-dir "$input_dir" \
            --output-dir "$output_dir"
    else
        echo "Warning: Input directory $input_dir does not exist"
    fi
}

# Main execution
echo "Starting coverage merger for all executions..."

# Bikeshop executions
merge_coverage "bikeshop" "execution_1" "bikeshop1"
merge_coverage "bikeshop" "execution_2" "bikeshop2"
merge_coverage "bikeshop" "execution_3" "bikeshop3"

# Photography executions
merge_coverage "photography" "execution_1" "photography1"
merge_coverage "photography" "execution_2" "photography2"
merge_coverage "photography" "execution_3" "photography3"

# Pizza order executions
merge_coverage "pizzaorder" "execution_1" "pizzaorder1"
merge_coverage "pizzaorder" "execution_2" "pizzaorder2"
merge_coverage "pizzaorder" "execution_3" "pizzaorder3"

# Veterinary executions
merge_coverage "veterinary" "execution_1" "veterinary1"
merge_coverage "veterinary" "execution_2" "veterinary2"
merge_coverage "veterinary" "execution_3" "veterinary3"

echo "Coverage merging completed for all executions!"

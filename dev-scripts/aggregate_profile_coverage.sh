#!/usr/bin/env bash
set -euo pipefail

# Base directory for results (relative to TRACER root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$TRACER_ROOT/results"

echo "▶︎ Aggregating ALL profile logs per domain across all executions and profiles…"

for domain_dir in "${RESULTS_DIR}"/*; do
  [ -d "$domain_dir" ] || continue
  domain=$(basename "$domain_dir")
  echo -e "\n Domain: $domain"

  # Find all execution directories for this domain
  mapfile -t execution_dirs < <(find "$domain_dir" -name "execution_*" -type d | sort)
  if [ "${#execution_dirs[@]}" -eq 0 ]; then
    echo "    no execution directories found, skipping"
    continue
  fi

  # Collect ALL profile log files from ALL executions (not grouped by profile)
  all_profile_files=()
  for execution_dir in "${execution_dirs[@]}"; do
    logs_dir="$execution_dir/profile_logs"

    if [ ! -d "$logs_dir" ]; then
      echo "    Skipping $(basename "$execution_dir") - no profile_logs directory found"
      continue
    fi

    # Find all profile log files in this execution
    for log_file in "$logs_dir"/*; do
      [ -f "$log_file" ] || continue
      all_profile_files+=("$log_file")
    done
  done

  if [ "${#all_profile_files[@]}" -eq 0 ]; then
    echo "    no profile log files found across executions, skipping"
    continue
  fi

  echo "   Found ${#all_profile_files[@]} profile log files across all executions and profiles"

  # Create temporary file list
  temp_list=$(mktemp)
  printf '%s\n' "${all_profile_files[@]}" > "$temp_list"

  agg_cov="$domain_dir/${domain}_aggregate_profile_coverage.json"
  echo "   Merging ${#all_profile_files[@]} profile log files → $(basename "$agg_cov")"

  # Use file-list approach to merge all profile logs into one aggregate
  python3 "$TRACER_ROOT/src/scripts/coverage_merger.py" \
    --file-list "$temp_list" \
    --output-file "$agg_cov"

  rm "$temp_list"

  if [ $? -eq 0 ]; then
    echo "  📊 Running CoverageAnalyzer on domain aggregate…"
    python3 "$TRACER_ROOT/src/scripts/coverage_analyzer.py" "$agg_cov" \
      -o "$domain_dir/${domain}_aggregate_profile_report"
  else
    echo "  ✗ Failed to merge domain profile coverage"
  fi
done

echo -e "\n All profile coverage aggregation done."

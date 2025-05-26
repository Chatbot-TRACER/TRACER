#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results"

echo "▶︎ Aggregating coverage per domain…"

for domain_dir in "${RESULTS_DIR}"/*; do
  [ -d "$domain_dir" ] || continue
  domain=$(basename "$domain_dir")
  echo -e "\n Domain: $domain"

  # find all per-execution coverage.json files
  mapfile -t cov_files < <(find "$domain_dir" -maxdepth 3 -type f -name "*_coverage.json")
  if [ "${#cov_files[@]}" -eq 0 ]; then
    echo "    no coverage files found, skipping"
    continue
  fi

  # create temporary file list
  temp_list=$(mktemp)
  printf '%s\n' "${cov_files[@]}" > "$temp_list"

  agg_cov="$domain_dir/${domain}_aggregate_coverage.json"
  echo "   Merging ${#cov_files[@]} coverage files → $(basename "$agg_cov")"

  python3 src/scripts/coverage_merger.py \
    --file-list "$temp_list" \
    --output-file "$agg_cov"

  rm "$temp_list"

  echo "  📊 Running CoverageAnalyzer on aggregate…"
  python3 src/scripts/coverage_analyzer.py "$agg_cov" \
    -o "$domain_dir/${domain}_aggregate_report"
done

echo -e "\n All done."

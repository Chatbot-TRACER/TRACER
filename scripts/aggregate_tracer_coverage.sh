#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results"

echo "▶︎ Aggregating tracer coverage per domain…"

for domain_dir in "${RESULTS_DIR}"/*; do
  [ -d "$domain_dir" ] || continue
  domain=$(basename "$domain_dir")
  echo -e "\n Domain: $domain"

  # find all tracer/explorer coverage files from tracer_coverage directories
  mapfile -t tracer_files < <(find "$domain_dir" -path "*/tracer_coverage/*_tracer_coverage.json" -type f)
  if [ "${#tracer_files[@]}" -eq 0 ]; then
    echo "    no tracer coverage files found, skipping"
    continue
  fi

  # create temporary file list
  temp_list=$(mktemp)
  printf '%s\n' "${tracer_files[@]}" > "$temp_list"

  agg_cov="$domain_dir/${domain}_aggregate_tracer_coverage.json"
  echo "   Merging ${#tracer_files[@]} tracer coverage files → $(basename "$agg_cov")"

  python3 src/scripts/coverage_merger.py \
    --file-list "$temp_list" \
    --output-file "$agg_cov"

  rm "$temp_list"

  echo "  📊 Running CoverageAnalyzer on aggregate…"
  python3 src/scripts/coverage_analyzer.py "$agg_cov" \
    -o "$domain_dir/${domain}_aggregate_tracer_report"
done

echo -e "\n All tracer coverage aggregation done."

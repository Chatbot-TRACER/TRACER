#!/usr/bin/env python3
"""Script to update goal_style configuration in YAML profile files.
Replaces:
  goal_style:
    all_answered:
      export: false
      limit: X

With:
  goal_style:
    steps: X
"""

import glob
import re
from pathlib import Path


def update_goal_style_in_file(file_path):
    """Update goal_style configuration in a single YAML file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Pattern to match the goal_style section we want to replace
        # This pattern looks for the goal_style section with all_answered configuration
        pattern = r"(goal_style:\s*\n\s+)all_answered:\s*\n\s+export:\s*false\s*\n\s+limit:\s*(\d+)"

        # Replace with steps configuration
        def replacement(match):
            indent = match.group(1)
            limit_value = match.group(2)
            return f"{indent}steps: {limit_value}"

        # Check if pattern exists before replacement
        if re.search(pattern, content):
            updated_content = re.sub(pattern, replacement, content)

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print(f"✓ Updated: {file_path}")
            return True
        print(f"- No matching pattern found in: {file_path}")
        return False

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all profile files."""
    results_dir = Path("/home/ivan/miso/TRACER/results")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Find all YAML files in profiles directories
    profile_pattern = str(results_dir / "**/profiles/*.yaml")
    profile_files = glob.glob(profile_pattern, recursive=True)

    if not profile_files:
        print("No profile YAML files found!")
        return

    print(f"Found {len(profile_files)} profile files to process...\n")

    updated_count = 0
    total_count = len(profile_files)

    # Process each file
    for file_path in sorted(profile_files):
        if update_goal_style_in_file(file_path):
            updated_count += 1

    print("\nProcessing complete!")
    print(f"Files updated: {updated_count}/{total_count}")

    if updated_count < total_count:
        print(f"Files skipped: {total_count - updated_count} (no matching pattern)")


if __name__ == "__main__":
    main()

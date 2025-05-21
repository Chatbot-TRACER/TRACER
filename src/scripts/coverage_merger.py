import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def find_coverage_files(basename: str, directory: str = ".") -> List[str]:
    """Find all coverage files matching the basename pattern"""
    path = Path(directory)
    return sorted(f for f in path.glob(f"{basename}_*") if not f.suffix)


def merge_footprints(files: List[str]) -> Dict[str, Any]:
    """Merge multiple coverage files into a combined footprint"""
    merged = {"specification": None, "footprint": {"modules": set(), "unknown": set()}}

    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)

            # Keep first specification as reference
            if merged["specification"] is None:
                merged["specification"] = data["specification"]

            # Merge modules list
            merged["footprint"]["modules"].update(data["footprint"].get("modules", []))

            # Merge unknown questions
            merged["footprint"]["unknown"].update(data["footprint"].get("unknown", []))

            # Merge module data
            for module, fields in data["footprint"].items():
                if module in ["modules", "unknown"]:
                    continue

                if module not in merged["footprint"]:
                    merged["footprint"][module] = defaultdict(set)

                for field, values in fields.items():
                    if isinstance(values, list):
                        merged["footprint"][module][field].update(values)
                    else:
                        merged["footprint"][module][field].add(values)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error processing {file} - {e}")

    # Convert sets to lists for JSON serialization
    merged["footprint"]["modules"] = sorted(merged["footprint"]["modules"])
    merged["footprint"]["unknown"] = sorted(merged["footprint"]["unknown"])

    for module in merged["footprint"]:
        if module in ["modules", "unknown"]:
            continue
        for field in merged["footprint"][module]:
            merged["footprint"][module][field] = sorted(
                merged["footprint"][module][field]
            )

    return merged


def save_merged_coverage(
    basename: str, output_dir: str = ".", merged: Dict[str, Any] = None
):
    """Save merged coverage to a file"""
    if merged is None:
        files = find_coverage_files(basename, output_dir)
        merged = merge_footprints(files)

    output_path = Path(output_dir) / f"{basename}_coverage.json"
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    return str(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge chatbot coverage files")
    parser.add_argument(
        "basename", help='Base name prefix of files to merge (e.g. "bikeshop")'
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to output merged file"
    )
    args = parser.parse_args()

    output_file = save_merged_coverage(args.basename, args.output_dir)
    print(f"Merged coverage saved to: {output_file}")


if __name__ == "__main__":
    main()


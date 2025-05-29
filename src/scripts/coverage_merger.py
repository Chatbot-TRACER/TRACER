import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_coverage_files(basename: str, input_dir: str = ".") -> list[str]:
    """Find all coverage files matching the basename pattern"""
    path = Path(input_dir)
    # Extract just the filename from basename if it contains a path
    basename_name = Path(basename).name
    return sorted(str(f) for f in path.glob(f"{basename_name}_*") if not f.suffix)


def merge_footprints(files: list[str]) -> dict[str, Any]:
    """Merge multiple coverage files into a combined footprint"""
    merged = {"specification": None, "footprint": {"modules": set(), "unknown": set()}}

    for file in files:
        try:
            with open(file) as f:
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
            merged["footprint"][module][field] = sorted(merged["footprint"][module][field])

    return merged


def save_merged_coverage(basename: str, input_dir: str = ".", output_dir: str = ".", merged: dict[str, Any] = None):
    """Save merged coverage to a file"""
    if merged is None:
        files = find_coverage_files(basename, input_dir)
        merged = merge_footprints(files)

    # Extract just the filename from basename if it contains a path
    basename_name = Path(basename).name
    output_path = Path(output_dir) / f"{basename_name}_coverage.json"

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    return str(output_path)


def merge_files_from_list(file_list_path: str, output_path: str):
    """Merge files listed in a text file"""
    with open(file_list_path) as f:
        files = [line.strip() for line in f if line.strip()]

    merged = merge_footprints(files)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge chatbot coverage files")
    parser.add_argument("basename", nargs="?", help="Base name prefix of files to merge")
    parser.add_argument("-i", "--input-dir", default=".", help="Directory containing the coverage files to merge")
    parser.add_argument("-o", "--output-dir", default=".", help="Directory to output merged file")
    parser.add_argument("-f", "--file-list", help="Path to file containing list of coverage files to merge")
    parser.add_argument("--output-file", help="Specific output file path")
    args = parser.parse_args()

    if args.file_list:
        if not args.output_file:
            print("Error: --output-file required when using --file-list")
            return 1
        output_file = merge_files_from_list(args.file_list, args.output_file)
    else:
        if not args.basename:
            print("Error: basename required when not using --file-list")
            return 1
        # If basename contains a path, use its directory as input_dir if not explicitly provided
        if args.input_dir == "." and "/" in args.basename:
            args.input_dir = str(Path(args.basename).parent)

        if args.output_file:
            # Use specific output file path
            files = find_coverage_files(args.basename, args.input_dir)
            merged = merge_footprints(files)
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, "w") as f:
                json.dump(merged, f, indent=2)
            output_file = args.output_file
        else:
            # Use output directory approach
            output_file = save_merged_coverage(args.basename, args.input_dir, args.output_dir)

    print(f"Merged coverage saved to: {output_file}")


if __name__ == "__main__":
    main()

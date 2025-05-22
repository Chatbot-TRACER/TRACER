import json
from pathlib import Path


class CoverageAnalyzer:
    """Analyze coverage data from coverage files."""

    def __init__(self, coverage_file: str):
        self.coverage_file = Path(coverage_file)
        self.data = self._load_coverage_data()

    def _load_coverage_data(self) -> dict:
        """Load coverage data from file."""
        if not self.coverage_file.exists():
            raise FileNotFoundError(
                f"Coverage file not found: {self.coverage_file}\n"
                "You may need to run coverage_merger.py first to generate it."
            )
        with open(self.coverage_file) as f:
            return json.load(f)

    def _detect_qa_modules(self) -> list[str]:
        """Detect QA modules by looking for question-like keys."""
        qa_modules = []
        specification = self.data.get("specification", {})

        for module_name, module_spec in specification.items():
            if module_name == "modules":
                continue

            if isinstance(module_spec, dict):
                # Check if this module has question-like keys (strings that end with ?)
                has_questions = any(isinstance(key, str) and key.endswith("?") for key in module_spec.keys())
                if has_questions:
                    qa_modules.append(module_name)

        return qa_modules

    def calculate_module_activation(self) -> dict[str, float]:
        """Calculate binary module activation (100% if any field covered, 0% otherwise)."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})
        activation = {}

        for module_name, module_spec in specification.items():
            if module_name == "modules":
                continue

            # Check if module has any activity in footprint
            module_footprint = footprint.get(module_name, {})

            # For empty modules, check if they appear in footprint at all
            if not module_spec:
                activation[module_name] = 100.0 if module_name in footprint else 0.0
                continue

            # For modules with fields, check if any field has values
            has_activity = False
            for field in module_spec.keys():
                if module_footprint.get(field):
                    has_activity = True
                    break

            activation[module_name] = 100.0 if has_activity else 0.0

        return activation

    def calculate_field_coverage(self) -> dict[str, dict]:
        """Calculate field coverage percentage for each module."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})
        qa_modules = self._detect_qa_modules()

        coverage = {}

        for module_name, module_spec in specification.items():
            if module_name == "modules":
                continue

            # Handle empty specification
            if not module_spec:
                # For empty modules, they're covered if they appear in footprint
                is_covered = module_name in footprint
                coverage[module_name] = {
                    "overall": 100.0 if is_covered else 0.0,
                    "fields": {},
                    "type": "empty",
                    "missing": [] if is_covered else [f"Module {module_name} not activated"],
                }
                continue

            module_footprint = footprint.get(module_name, {})
            field_details = {}
            covered_count = 0
            total_count = 0
            missing = []

            if module_name in qa_modules:
                # QA module: check if questions appear in their own lists
                for question, expected_answers in module_spec.items():
                    if question == "unknown":
                        continue
                    total_count += 1
                    question_responses = module_footprint.get(question, [])
                    # Check if the exact question appears in the response list
                    is_covered = question in question_responses
                    field_details[question] = 100.0 if is_covered else 0.0
                    if is_covered:
                        covered_count += 1
                    else:
                        missing.append(question)

            else:
                # Regular module: distinguish between constraints and expected values
                for field, spec_value in module_spec.items():
                    total_count += 1
                    field_values = module_footprint.get(field, [])

                    if isinstance(spec_value, list):
                        # List means these are expected values to match
                        expected_values_set = set(spec_value)
                        field_values_set = set(field_values)

                        if expected_values_set:
                            covered_values = field_values_set.intersection(expected_values_set)
                            field_coverage = len(covered_values) / len(expected_values_set) * 100
                            missing_values = expected_values_set - covered_values

                            field_details[field] = field_coverage
                            if field_coverage == 100.0:
                                covered_count += 1

                            # Add missing values to the missing list for this field
                            if missing_values:
                                missing.extend([f"{field}: {val}" for val in missing_values])
                        else:
                            # Empty list - just check if field has any values
                            is_covered = len(field_values) > 0
                            field_details[field] = 100.0 if is_covered else 0.0
                            if is_covered:
                                covered_count += 1
                            else:
                                missing.append(field)
                    else:
                        # String value means it's a constraint/type, not an expected value
                        # Just check if the field has any valid values
                        is_covered = len(field_values) > 0
                        field_details[field] = 100.0 if is_covered else 0.0
                        if is_covered:
                            covered_count += 1
                        else:
                            missing.append(field)

            # Calculate overall as average of all field coverage percentages
            if field_details:
                overall_coverage = sum(field_details.values()) / len(field_details)
            else:
                overall_coverage = 0.0

            coverage[module_name] = {
                "overall": overall_coverage,
                "fields": field_details,
                "type": "qa" if module_name in qa_modules else "regular",
                "missing": missing,
            }

        return coverage

    def generate_report(self) -> dict:
        """Generate coverage report."""
        module_usage = self.calculate_module_activation()
        parameter_coverage = self.calculate_field_coverage()

        # Calculate overall parameter coverage as average of all module parameter coverages
        # Only include modules that have specifications (not empty ones)
        coverage_percentages = []
        for module_name, module_data in parameter_coverage.items():
            if module_data["type"] != "empty":  # Skip empty modules for overall calculation
                coverage_percentages.append(module_data["overall"])

        overall_parameter_coverage = (
            sum(coverage_percentages) / len(coverage_percentages) if coverage_percentages else 0.0
        )

        # Calculate module usage percentage (binary: used or not)
        used_modules = sum(1 for usage in module_usage.values() if usage == 100.0)
        total_modules = len(module_usage)
        module_usage_overall = (used_modules / total_modules * 100) if total_modules > 0 else 0.0

        return {
            "overall_parameter_coverage": round(overall_parameter_coverage, 2),
            "module_usage": {
                "overall": round(module_usage_overall, 2),
                "modules": module_usage,
            },
            "parameter_coverage": parameter_coverage,
            "summary": {
                "total_modules": total_modules,
                "used_modules": used_modules,
                "modules_with_parameters": len([m for m in parameter_coverage.values() if m["type"] != "empty"]),
            },
        }

    def _get_output_path(self, output_file: str | None, extension: str) -> Path:
        """Generate output path with proper naming convention."""
        if output_file:
            return Path(output_file)

        # Use original filename but replace extension with _report.extension
        stem = self.coverage_file.stem
        if stem.endswith("_coverage"):
            # Replace _coverage with _report
            new_stem = stem[:-9] + "_report"
        else:
            # Just add _report
            new_stem = stem + "_report"

        return self.coverage_file.parent / f"{new_stem}.{extension}"

    def save_report(self, output_file: str | None = None) -> str:
        """Save report to file."""
        report = self.generate_report()
        out_path = self._get_output_path(output_file, "json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return str(out_path)

    def print_summary(self) -> None:
        """Print formatted coverage summary."""
        report = self.generate_report()
        summary = report["summary"]

        print("🤖 CHATBOT COVERAGE ANALYSIS")
        print("=" * 60)

        # High-level metrics
        print("\n📊 OVERALL METRICS")
        print(f"   Parameter Coverage: {report['overall_parameter_coverage']:.1f}%")
        print(
            f"   Module Usage:       {report['module_usage']['overall']:.1f}% ({summary['used_modules']}/{summary['total_modules']} modules)"
        )

        # Module usage breakdown
        print("\n🏗️  MODULE USAGE STATUS")
        unused_modules = []
        used_modules = []

        for module, usage in report["module_usage"]["modules"].items():
            if usage == 100.0:
                used_modules.append(module)
            else:
                unused_modules.append(module)

        if used_modules:
            print(f"   ✅ USED ({len(used_modules)}):")
            for module in sorted(used_modules):
                print(f"      • {module}")

        if unused_modules:
            print(f"   ❌ UNUSED ({len(unused_modules)}):")
            for module in sorted(unused_modules):
                print(f"      • {module}")

        # Parameter coverage details
        print("\n🎯 PARAMETER COVERAGE DETAILS")

        # Group by coverage level for better readability
        excellent = []  # 80%+
        good = []  # 50-79%
        poor = []  # 20-49%
        missing = []  # 0-19%

        for module, coverage_data in report["parameter_coverage"].items():
            if coverage_data["type"] == "empty":
                continue

            coverage = coverage_data["overall"]
            module_info = {
                "name": module,
                "coverage": coverage,
                "type": coverage_data["type"],
                "missing_count": len(coverage_data["missing"]),
            }

            if coverage >= 80:
                excellent.append(module_info)
            elif coverage >= 50:
                good.append(module_info)
            elif coverage >= 20:
                poor.append(module_info)
            else:
                missing.append(module_info)

        # Display by coverage level
        for category, modules, emoji in [
            ("EXCELLENT (80%+)", excellent, "🟢"),
            ("GOOD (50-79%)", good, "🟡"),
            ("POOR (20-49%)", poor, "🟠"),
            ("MISSING (0-19%)", missing, "🔴"),
        ]:
            if modules:
                print(f"\n   {emoji} {category}:")
                for mod in sorted(modules, key=lambda x: x["coverage"], reverse=True):
                    missing_text = f" ({mod['missing_count']} missing)" if mod["missing_count"] > 0 else ""
                    print(f"      • {mod['name']}: {mod['coverage']:.1f}%{missing_text}")

        # Show detailed missing items for critical modules
        critical_modules = [m for m in missing if m["coverage"] < 20]
        if critical_modules:
            print("\n🚨 MISSING PARAMETERS DETAILS:")
            for mod_info in critical_modules:
                module = mod_info["name"]
                coverage_data = report["parameter_coverage"][module]
                if coverage_data["missing"]:
                    print(f"   📌 {module}:")
                    # Show all missing items, no limit
                    for item in coverage_data["missing"]:
                        print(f"      • {item}")

    def save_readable_report(self, output_file: str | None = None) -> str:
        """Save a human-readable text report."""
        report = self.generate_report()
        out_path = self._get_output_path(output_file, "txt")

        # Capture the print output
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            self.print_summary()
            content = buffer.getvalue()
        finally:
            sys.stdout = old_stdout

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(out_path)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze coverage data")
    parser.add_argument("coverage_file", help="Path to coverage file to analyze")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (.json for JSON format, .txt for readable format)",
    )
    args = parser.parse_args()

    try:
        analyzer = CoverageAnalyzer(args.coverage_file)
        analyzer.print_summary()
        if args.output:
            if args.output.endswith(".json"):
                output_path = analyzer.save_report(args.output)
                print(f"\nJSON report saved to: {output_path}")
            else:
                output_path = analyzer.save_readable_report(args.output)
                print(f"\nReadable report saved to: {output_path}")
        else:
            # Save both formats by default
            json_path = analyzer.save_report()
            readable_path = analyzer.save_readable_report()
            print("\nReports saved:")
            print(f"  📊 JSON: {json_path}")
            print(f"  📝 Text: {readable_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nSuggested steps:\n"
            "1. First run coverage_merger.py to generate the coverage file:\n"
            "   python coverage_merger.py bikeshop\n"
            "2. Then analyze it:\n"
            "   python coverage_analyzer.py bikeshop_coverage.json",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

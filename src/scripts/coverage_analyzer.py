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
        modules_with_params = []
        for module_name, module_data in parameter_coverage.items():
            if module_data["type"] != "empty":
                coverage_percentages.append(module_data["overall"])
                modules_with_params.append(module_name)

        overall_parameter_coverage = (
            sum(coverage_percentages) / len(coverage_percentages) if coverage_percentages else 0.0
        )

        # Calculate module usage stats
        used_modules = [name for name, usage in module_usage.items() if usage == 100.0]
        unused_modules = [name for name, usage in module_usage.items() if usage == 0.0]
        total_modules = len(module_usage)
        module_usage_percentage = (len(used_modules) / total_modules * 100) if total_modules > 0 else 0.0

        # Calculate parameter stats at item level for consistency
        total_parameters = 0
        covered_parameters = 0
        missing_parameters = 0

        for name, data in parameter_coverage.items():
            if data["type"] != "empty":
                # Count individual items/values, not just fields
                module_spec = self.data.get("specification", {}).get(name, {})
                module_footprint = self.data.get("footprint", {}).get(name, {})

                if data["type"] == "qa":
                    # For QA modules, count questions
                    for question in module_spec.keys():
                        if question != "unknown":
                            total_parameters += 1
                            question_responses = module_footprint.get(question, [])
                            if question in question_responses:
                                covered_parameters += 1
                            else:
                                missing_parameters += 1
                else:
                    # For regular modules, count individual values
                    for field, spec_value in module_spec.items():
                        if isinstance(spec_value, list) and spec_value:
                            # Count each expected value
                            expected_values_set = set(spec_value)
                            field_values_set = set(module_footprint.get(field, []))
                            covered_values = field_values_set.intersection(expected_values_set)
                            missing_values = expected_values_set - covered_values

                            total_parameters += len(expected_values_set)
                            covered_parameters += len(covered_values)
                            missing_parameters += len(missing_values)
                        else:
                            # Constraint field - count as 1 item
                            total_parameters += 1
                            field_values = module_footprint.get(field, [])
                            if field_values:
                                covered_parameters += 1
                            else:
                                missing_parameters += 1

        # Calculate used items for each module
        parameters_with_used_items = {}
        for name, data in parameter_coverage.items():
            if data["type"] != "empty":
                used_items = []

                # Get the original specification to compare against
                module_spec = self.data.get("specification", {}).get(name, {})
                module_footprint = self.data.get("footprint", {}).get(name, {})

                if data["type"] == "qa":
                    # For QA modules, used items are questions that appear in their own response lists
                    for question in module_spec.keys():
                        if question != "unknown":
                            question_responses = module_footprint.get(question, [])
                            if question in question_responses:
                                used_items.append(question)
                else:
                    # For regular modules, find which specific values were used
                    for field, spec_value in module_spec.items():
                        field_values = module_footprint.get(field, [])
                        if isinstance(spec_value, list) and spec_value:
                            # List of expected values - find which ones were covered
                            expected_values_set = set(spec_value)
                            field_values_set = set(field_values)
                            covered_values = field_values_set.intersection(expected_values_set)
                            for value in covered_values:
                                used_items.append(f"{field}: {value}")
                        elif field_values:
                            # Field has values (constraint type)
                            used_items.append(field)

                parameters_with_used_items[name] = {
                    "overall_coverage": round(data["overall"], 2),
                    "total_fields": len(data["fields"]),
                    "covered_fields": sum(1 for cov in data["fields"].values() if cov == 100.0),
                    "missing_count": len(data["missing"]),
                    "used_count": len(used_items),
                    "field_details": {field: round(coverage, 2) for field, coverage in data["fields"].items()},
                    "used_items": sorted(used_items),
                    "missing_items": data["missing"],
                    "module_type": data["type"],
                }

        return {
            "summary": {
                "overall_parameter_coverage": round(overall_parameter_coverage, 2),
                "overall_module_usage": round(module_usage_percentage, 2),
                "total_modules": total_modules,
                "modules_used": len(used_modules),
                "total_parameters": total_parameters,
                "covered_parameters": covered_parameters,
                "missing_parameters": missing_parameters,
            },
            "modules": {"used": sorted(used_modules), "unused": sorted(unused_modules)},
            "parameters": {"coverage_by_module": parameters_with_used_items},
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
        print(f"   Parameter Coverage: {summary['overall_parameter_coverage']:.1f}%")
        print(
            f"   Module Usage:       {summary['overall_module_usage']:.1f}% ({len(report['modules']['used'])}/{summary['total_modules']} modules)"
        )
        print(
            f"   Parameters:         {summary['covered_parameters']}/{summary['total_parameters']} covered ({summary['missing_parameters']} missing)"
        )

        # Module usage breakdown
        print("\n🏗️ MODULE USAGE STATUS")
        used_modules = report["modules"]["used"]
        unused_modules = report["modules"]["unused"]

        if used_modules:
            print(f"   ✅ USED ({len(used_modules)}):")
            for module in used_modules:
                print(f"      • {module}")

        if unused_modules:
            print(f"   ❌ UNUSED ({len(unused_modules)}):")
            for module in unused_modules:
                print(f"      • {module}")

        # Parameter coverage details
        print("\n🎯 PARAMETER COVERAGE DETAILS")

        # Group by coverage level for better readability
        excellent = []  # 80%+
        good = []  # 50-79%
        poor = []  # 20-49%
        missing = []  # 0-19%

        for module_name, module_data in report["parameters"]["coverage_by_module"].items():
            coverage = module_data["overall_coverage"]
            module_info = {
                "name": module_name,
                "coverage": coverage,
                "type": module_data["module_type"],
                "missing_count": module_data["missing_count"],
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
                module_name = mod_info["name"]
                module_data = report["parameters"]["coverage_by_module"][module_name]
                if module_data["missing_items"]:
                    print(f"   📌 {module_name}:")
                    # Show all missing items, no limit
                    for item in module_data["missing_items"]:
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

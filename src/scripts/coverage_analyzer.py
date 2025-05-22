import json
from pathlib import Path
from typing import Dict, List, Optional


class CoverageAnalyzer:
    """Analyze coverage data from coverage files."""

    def __init__(self, coverage_file: str):
        self.coverage_file = Path(coverage_file)
        self.data = self._load_coverage_data()

    def _load_coverage_data(self) -> Dict:
        """Load coverage data from file."""
        if not self.coverage_file.exists():
            raise FileNotFoundError(
                f"Coverage file not found: {self.coverage_file}\n"
                "You may need to run coverage_merger.py first to generate it."
            )
        with open(self.coverage_file) as f:
            return json.load(f)

    def _detect_qa_modules(self) -> List[str]:
        """Detect QA modules by looking for question-like keys."""
        qa_modules = []
        specification = self.data.get("specification", {})

        for module_name, module_spec in specification.items():
            if module_name == "modules":
                continue

            if isinstance(module_spec, dict):
                # Check if this module has question-like keys (strings that end with ?)
                has_questions = any(
                    isinstance(key, str) and key.endswith("?")
                    for key in module_spec.keys()
                )
                if has_questions:
                    qa_modules.append(module_name)

        return qa_modules

    def calculate_module_activation(self) -> Dict[str, float]:
        """Calculate which modules from specification were actually tested."""
        # Use field coverage data for module activation to ensure consistency
        field_coverage = self.calculate_field_coverage()
        activation = {}

        for module, coverage_data in field_coverage.items():
            # Use the overall coverage percentage for activation
            activation[module] = coverage_data["overall"]

        return activation

    def calculate_field_coverage(self) -> Dict[str, Dict]:
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
                coverage[module_name] = {
                    "overall": 100.0,
                    "fields": {},
                    "type": "empty",
                    "missing": [],
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

                coverage[module_name] = {
                    "overall": (covered_count / total_count * 100)
                    if total_count > 0
                    else 100.0,
                    "fields": field_details,
                    "type": "qa",
                    "missing": missing,
                }
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
                            covered_values = field_values_set.intersection(
                                expected_values_set
                            )
                            field_coverage = (
                                len(covered_values) / len(expected_values_set) * 100
                            )
                            missing_values = expected_values_set - covered_values

                            field_details[field] = field_coverage
                            if field_coverage == 100.0:
                                covered_count += 1

                            # Add missing values to the missing list for this field
                            if missing_values:
                                missing.extend(
                                    [f"{field}: {val}" for val in missing_values]
                                )
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

                coverage[module_name] = {
                    "overall": (covered_count / total_count * 100)
                    if total_count > 0
                    else 100.0,
                    "fields": field_details,
                    "type": "regular",
                    "missing": missing,
                }

        return coverage

    def generate_report(self) -> Dict:
        """Generate coverage report."""
        module_activation = self.calculate_module_activation()
        field_coverage = self.calculate_field_coverage()

        # Calculate overall coverage as average of all module field coverages
        coverage_percentages = [
            module_data["overall"] for module_data in field_coverage.values()
        ]
        overall_coverage = (
            sum(coverage_percentages) / len(coverage_percentages)
            if coverage_percentages
            else 100.0
        )

        # Calculate module activation percentage
        activation_percentages = list(module_activation.values())
        module_activation_overall = (
            sum(activation_percentages) / len(activation_percentages)
            if activation_percentages
            else 100.0
        )

        return {
            "overall_coverage": round(overall_coverage, 2),
            "module_activation": {
                "overall": round(module_activation_overall, 2),
                "modules": module_activation,
            },
            "field_coverage": field_coverage,
        }

    def save_report(self, output_file: Optional[str] = None) -> str:
        """Save report to file."""
        report = self.generate_report()
        out_path = (
            Path(output_file)
            if output_file
            else self.coverage_file.with_suffix(".report.json")
        )
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return str(out_path)

    def print_summary(self) -> None:
        """Print formatted coverage summary."""
        report = self.generate_report()
        print(f"Coverage Report for {self.coverage_file}")
        print("=" * 50)

        print(f"\nOverall Coverage: {report['overall_coverage']:.2f}%")

        print(f"\nModule Activation: {report['module_activation']['overall']:.2f}%")
        for module, activation in report["module_activation"]["modules"].items():
            status = "✓" if activation == 100.0 else "✗"
            print(f"  {status} {module}: {activation:.0f}%")

        print("\nField Coverage by Module:")
        for module, coverage_data in report["field_coverage"].items():
            module_type = f"[{coverage_data['type']}]"
            print(f"  - {module}: {coverage_data['overall']:.2f}% {module_type}")

            # Show field details if not empty
            if coverage_data["fields"]:
                for field, percent in coverage_data["fields"].items():
                    status = "✓" if percent == 100.0 else "✗"
                    print(f"    {status} {field}: {percent:.0f}%")

            # Show missing items
            if coverage_data["missing"]:
                print(f"    Missing: {coverage_data['missing']}")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze coverage data")
    parser.add_argument("coverage_file", help="Path to coverage file to analyze")
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args()

    try:
        analyzer = CoverageAnalyzer(args.coverage_file)
        analyzer.print_summary()
        if args.output:
            output_path = analyzer.save_report(args.output)
            print(f"\nReport saved to: {output_path}")
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

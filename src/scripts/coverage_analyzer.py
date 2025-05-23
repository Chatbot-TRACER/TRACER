import json
from pathlib import Path


class CoverageAnalyzer:
    """Analyze coverage data from coverage files."""

    def __init__(self, coverage_file: str):
        self.coverage_file = Path(coverage_file)
        self.data = self._load_coverage_data()
        self.qa_modules = self._detect_qa_modules()
        # Calculate module activation status once and store it
        self.module_activation_status_data = self._calculate_module_activation_status()
        # Generate the full report data once
        self.report_data = self._generate_full_report_data()

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
                has_questions = any(isinstance(key, str) and key.endswith("?") for key in module_spec.keys())
                if has_questions:
                    qa_modules.append(module_name)
        return qa_modules

    def _calculate_module_activation_status(self) -> dict:
        """Calculate binary module activation status and lists."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})
        activation_map = {}
        used_modules_list = []
        unused_modules_list = []

        # Determine all module names that should be considered
        all_module_names_from_spec_keys = [m for m in specification.keys() if m != "modules"]
        all_module_names_from_modules_list = []
        if "modules" in specification and isinstance(specification["modules"], list):
            all_module_names_from_modules_list = specification["modules"]

        # Combine and uniqueify, preferring spec keys if available, then "modules" list
        combined_module_names = list(
            dict.fromkeys(all_module_names_from_spec_keys + all_module_names_from_modules_list)
        )
        if not combined_module_names:  # Fallback if spec is truly empty or malformed
            combined_module_names = list(footprint.keys())

        for module_name in combined_module_names:
            module_spec = specification.get(
                module_name, {}
            )  # Handles modules in "modules" list but without detailed spec
            module_footprint = footprint.get(module_name, {})
            is_activated = False

            if module_name in footprint:  # Basic check: if module appears in footprint at all
                if (
                    not isinstance(module_spec, dict) or not module_spec
                ):  # Empty or non-dict spec, activated if in footprint
                    is_activated = True
                else:  # Module has a dict spec, check if any specified field has activity
                    for field_name in module_spec.keys():
                        if module_footprint.get(field_name):
                            is_activated = True
                            break
                    if (
                        not is_activated and not module_spec.keys() and module_name in footprint
                    ):  # Spec is {} but module in footprint
                        is_activated = True

            activation_map[module_name] = 100.0 if is_activated else 0.0
            if is_activated:
                used_modules_list.append(module_name)
            else:
                unused_modules_list.append(module_name)

        return {
            "activation_map": activation_map,
            "used_modules": sorted(used_modules_list),
            "unused_modules": sorted(unused_modules_list),
            "total_modules": len(combined_module_names),
        }

    def _calculate_global_coverage_stats(self) -> dict:
        """Calculate global coverage statistics for modules, fields, and options."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})

        activated_count = len(self.module_activation_status_data["used_modules"])
        total_modules = self.module_activation_status_data["total_modules"]
        module_activation_percentage = (activated_count / total_modules * 100) if total_modules > 0 else 0.0

        total_defined_fields_count = 0
        total_used_fields_count = 0
        total_defined_options_count = 0
        total_covered_options_count = 0

        # Iterate through modules listed in the spec for field/option counting
        # This ensures we only count fields/options defined in the specification.
        for module_name, module_spec in specification.items():
            if module_name == "modules":  # Skip the "modules" list itself
                continue
            if not isinstance(
                module_spec, dict
            ):  # Skip if a module's spec is not a dictionary (e.g. "top-level": null)
                continue

            module_footprint = footprint.get(module_name, {})
            is_module_activated = self.module_activation_status_data["activation_map"].get(module_name, 0.0) == 100.0

            for field_name, spec_value in module_spec.items():
                total_defined_fields_count += 1
                # A field is "used" if the module was activated AND the field appears in the footprint with some value.
                if is_module_activated and field_name in module_footprint and module_footprint[field_name]:
                    total_used_fields_count += 1

                # Option coverage calculation
                if module_name not in self.qa_modules and isinstance(spec_value, list) and spec_value:
                    total_defined_options_count += len(spec_value)
                    if is_module_activated:  # Options can only be covered if the module itself was activated
                        footprint_values = set(module_footprint.get(field_name, []))
                        covered_options = set(spec_value).intersection(footprint_values)
                        total_covered_options_count += len(covered_options)
                elif module_name in self.qa_modules and isinstance(
                    spec_value, list
                ):  # QA "options" are the expected answers
                    # The "fields" are questions. The "options" are the expected answers within the spec_value list.
                    # This part of global option counting for QA might need refinement based on desired definition.
                    # Current model: QA fields are questions, their coverage is field coverage.
                    # If QA spec_value is a list of expected answers, they could be counted as options.
                    # For simplicity and alignment with current detailed report, let's assume QA "options" are not globally counted here,
                    # as their primary coverage is handled as "answered questions" (field coverage for QA).
                    # However, if the old report's "Option Coverage" for QA modules (e.g., 4/6 options) meant
                    # that 4 out of 6 questions were answered, then this needs adjustment.
                    # The previous structure had "Option Coverage: 66.7% (4/6 options)" for pizza_qa.
                    # This implies that for QA, "options" were synonymous with "questions".
                    total_defined_options_count += 1  # Each QA question is an "option item" in this global sense
                    if is_module_activated and field_name in module_footprint and module_footprint.get(field_name):
                        # If the question (field_name) was "used" (i.e., answered, present in footprint)
                        # For QA, the old model seemed to count a question as "covered" if it was present.
                        # The spec_value for QA is usually a list of example answers, not what's matched.
                        # Let's use the field_coverage logic for QA here for consistency.
                        if field_name in module_footprint.get(
                            field_name, []
                        ):  # Check if question itself is in its response list
                            total_covered_options_count += 1

        field_usage_percentage = (
            (total_used_fields_count / total_defined_fields_count * 100) if total_defined_fields_count > 0 else 0.0
        )
        option_value_percentage = (
            (total_covered_options_count / total_defined_options_count * 100)
            if total_defined_options_count > 0
            else 0.0
        )

        return {
            "module_activation_coverage": {
                "percentage": round(module_activation_percentage, 2),
                "activated_count": activated_count,
                "total_defined_modules": total_modules,
            },
            "field_usage_coverage": {  # This is "Fields Used" from spec
                "percentage": round(field_usage_percentage, 2),
                "used_field_count": total_used_fields_count,
                "total_defined_fields": total_defined_fields_count,
                "missing_field_count": total_defined_fields_count - total_used_fields_count,
            },
            "option_value_coverage": {  # This is "Options Covered" from spec lists
                "percentage": round(option_value_percentage, 2),
                "covered_option_count": total_covered_options_count,
                "total_defined_options": total_defined_options_count,
                "missing_option_count": total_defined_options_count - total_covered_options_count,
            },
        }

    def _calculate_detailed_module_coverage(self) -> dict:
        """Calculate detailed field and option coverage for each module."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})
        detailed_coverage = {}

        # Use the same source of module names as in _calculate_module_activation_status
        all_module_names_from_spec_keys = [m for m in specification.keys() if m != "modules"]
        all_module_names_from_modules_list = []
        if "modules" in specification and isinstance(specification["modules"], list):
            all_module_names_from_modules_list = specification["modules"]
        all_module_names = list(dict.fromkeys(all_module_names_from_spec_keys + all_module_names_from_modules_list))
        if not all_module_names:
            all_module_names = list(footprint.keys())

        for module_name in all_module_names:
            module_spec = specification.get(module_name, {})
            module_footprint = footprint.get(module_name, {})

            is_activated = self.module_activation_status_data["activation_map"].get(module_name, 0.0) == 100.0
            module_type = "regular"
            if module_name in self.qa_modules:
                module_type = "qa"
            elif isinstance(module_spec, dict) and not module_spec:  # Empty dict {}
                module_type = "empty"
            elif not isinstance(
                module_spec, dict
            ):  # e.g. "top-level": null, or module listed in "modules" but no spec entry
                module_type = "undefined_spec"  # Or some other indicator

            # Field Coverage for this module
            defined_field_names = list(module_spec.keys()) if isinstance(module_spec, dict) else []

            used_fields_list = []
            if is_activated and isinstance(module_spec, dict):
                for fn in defined_field_names:
                    if module_footprint.get(fn):
                        used_fields_list.append(fn)

            missing_fields_list = sorted(list(set(defined_field_names) - set(used_fields_list)))
            used_fields_list.sort()

            total_defined_fields_in_module = len(defined_field_names)
            used_fields_count_in_module = len(used_fields_list)

            field_coverage_percentage = 0.0
            if total_defined_fields_in_module > 0:
                field_coverage_percentage = used_fields_count_in_module / total_defined_fields_in_module * 100
            elif is_activated and (
                module_type == "empty" or module_type == "undefined_spec"
            ):  # Activated but no defined fields
                field_coverage_percentage = 100.0  # Considered fully covered in terms of its (zero) fields

            module_field_coverage = {
                "percentage": round(field_coverage_percentage, 2),
                "total_defined_in_module": total_defined_fields_in_module,
                "used_count_in_module": used_fields_count_in_module,
                "missing_count_in_module": total_defined_fields_in_module - used_fields_count_in_module,
                "used_fields_list": used_fields_list,
                "missing_fields_list": missing_fields_list,
            }
            if module_type == "qa":  # Add question lists for QA
                module_field_coverage["used_questions_list"] = used_fields_list  # Questions are fields for QA
                module_field_coverage["missing_questions_list"] = missing_fields_list

            # Option Coverage for this module (if applicable)
            module_option_coverage = None
            if module_type == "regular" and isinstance(module_spec, dict):
                details_per_option_field = {}
                module_total_defined_options = 0
                module_total_covered_options = 0
                module_used_options_summary = []
                module_missing_options_summary = []

                for field_name, spec_value in module_spec.items():
                    if isinstance(spec_value, list) and spec_value:
                        defined_options = set(spec_value)
                        field_footprint_values = set(module_footprint.get(field_name, []))

                        covered_options_for_field = set()
                        if is_activated:  # Options only covered if module is active
                            covered_options_for_field = defined_options.intersection(field_footprint_values)

                        missing_options_for_field = defined_options - covered_options_for_field

                        module_total_defined_options += len(defined_options)
                        module_total_covered_options += len(covered_options_for_field)

                        for opt in sorted(list(covered_options_for_field)):
                            module_used_options_summary.append(f"{field_name}: {opt}")
                        for opt in sorted(list(missing_options_for_field)):
                            module_missing_options_summary.append(f"{field_name}: {opt}")

                        details_per_option_field[field_name] = {
                            "percentage": round(len(covered_options_for_field) / len(defined_options) * 100, 2)
                            if defined_options
                            else 0.0,
                            "defined_options_count": len(defined_options),
                            "covered_options_count": len(covered_options_for_field),
                            "missing_options_count": len(missing_options_for_field),
                            "used_values": sorted(list(covered_options_for_field)),
                            "missing_values": sorted(list(missing_options_for_field)),
                        }

                option_percentage_for_module = 0.0
                if module_total_defined_options > 0:
                    option_percentage_for_module = module_total_covered_options / module_total_defined_options * 100
                elif is_activated:  # Activated but no defined option fields
                    option_percentage_for_module = 100.0

                module_option_coverage = {
                    "overall_percentage_for_module": round(option_percentage_for_module, 2),
                    "total_defined_options_in_module": module_total_defined_options,
                    "covered_options_in_module_count": module_total_covered_options,
                    "missing_options_in_module_count": module_total_defined_options - module_total_covered_options,
                    "used_options_summary_list": sorted(module_used_options_summary),
                    "missing_options_summary_list": sorted(module_missing_options_summary),
                    "details_per_option_field": details_per_option_field,
                }

            current_module_detail = {
                "module_type": module_type,
                "activated": is_activated,
                "field_coverage": module_field_coverage,
            }
            if module_option_coverage is not None:
                current_module_detail["option_coverage"] = module_option_coverage

            detailed_coverage[module_name] = current_module_detail

        return detailed_coverage

    def _generate_full_report_data(self) -> dict:
        """Generate the full coverage report data structure."""
        # module_activation_status is already calculated in __init__
        global_stats = self._calculate_global_coverage_stats()  # Depends on module_activation_status
        detailed_module_stats = self._calculate_detailed_module_coverage()  # Depends on module_activation_status

        report = {
            "global_summary": global_stats,
            "module_lists": {
                "used_modules": self.module_activation_status_data["used_modules"],
                "unused_modules": self.module_activation_status_data["unused_modules"],
                "total_modules": self.module_activation_status_data["total_modules"],
            },
            "module_details": detailed_module_stats,
        }
        return report

    def get_report(self) -> dict:
        """Return the generated report."""
        return self.report_data

    def _get_output_path(self, output_file: str | None, extension: str) -> Path:
        """Generate output path with proper naming convention."""
        if output_file:
            return Path(output_file)
        stem = self.coverage_file.stem
        new_stem = stem.replace("_coverage", "") + "_report" if "_coverage" in stem else stem + "_report"
        return self.coverage_file.parent / f"{new_stem}.{extension}"

    def save_report(self, output_file: str | None = None) -> str:
        """Save JSON report to file."""
        # report = self.generate_report() # Report is now generated in __init__
        out_path = self._get_output_path(output_file, "json")
        with open(out_path, "w") as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        return str(out_path)

    def print_summary(self) -> None:
        """Print formatted coverage summary with module overview and detailed breakdown."""
        gs = self.report_data["global_summary"]
        ml = self.report_data["module_lists"]
        md = self.report_data["module_details"]

        print("🤖 CHATBOT COVERAGE ANALYSIS")
        print("=" * 60)

        # 1. Global Summary
        print("\n📊 OVERALL METRICS")
        print(
            f"  • Module Activation: {gs['module_activation_coverage']['percentage']:.2f}% ({gs['module_activation_coverage']['activated_count']}/{gs['module_activation_coverage']['total_defined_modules']})"
        )
        print(
            f"  • Field Usage:       {gs['field_usage_coverage']['percentage']:.2f}% ({gs['field_usage_coverage']['used_field_count']}/{gs['field_usage_coverage']['total_defined_fields']})"
        )
        print(
            f"  • Option Coverage:   {gs['option_value_coverage']['percentage']:.2f}% ({gs['option_value_coverage']['covered_option_count']}/{gs['option_value_coverage']['total_defined_options']})"
        )

        # 2. Module Activation Status
        print("\n🏗️ MODULE ACTIVATION STATUS\n")
        if ml["used_modules"]:
            print(f"  ✅ USED ({len(ml['used_modules'])}):")
            for mod in ml["used_modules"]:
                print(f"       • {mod}")
        if ml["unused_modules"]:
            print(f"  ❌ UNUSED ({len(ml['unused_modules'])}):")
            for mod in ml["unused_modules"]:
                print(f"       • {mod}")

        # 3. Module Coverage Overview Grouped by Percentage
        print("\n🔍 MODULE COVERAGE OVERVIEW")
        grouped = {"EXCELLENT (80%+)": [], "GOOD (50-79%)": [], "POOR (20-49%)": [], "MISSING/LOW (0-19%)": []}
        for name in ml["used_modules"]:
            details = md[name]
            mod_type = details.get("module_type")
            if mod_type == "regular":
                pct = details["option_coverage"]["overall_percentage_for_module"]
                used = details["option_coverage"]["covered_options_in_module_count"]
                total = details["option_coverage"]["total_defined_options_in_module"]
                label = f"{name}: {pct:.2f}% ({used}/{total} options)"
            elif mod_type == "qa":
                pct = details["field_coverage"]["percentage"]
                used = details["field_coverage"]["used_count_in_module"]
                total = details["field_coverage"]["total_defined_in_module"]
                label = f"{name}: {pct:.2f}% ({used}/{total} questions)"
            else:
                pct = 100.0
                label = f"{name}: 100.00% (no spec)"

            if pct >= 80:
                grouped["EXCELLENT (80%+)"].append(label)
            elif pct >= 50:
                grouped["GOOD (50-79%)"].append(label)
            elif pct >= 20:
                grouped["POOR (20-49%)"].append(label)
            else:
                grouped["MISSING/LOW (0-19%)"].append(label)

        for category, modules in grouped.items():
            if modules:
                emoji = {
                    "EXCELLENT (80%+)": "🟢",
                    "GOOD (50-79%)": "🟡",
                    "POOR (20-49%)": "🟠",
                    "MISSING/LOW (0-19%)": "🔴",
                }[category]
                print(f"\n  {emoji} {category}:")
                for label in modules:
                    print(f"    • {label}")

        # 4. Detailed Breakdown per Module
        print("\n📝 DETAILED BREAKDOWN PER MODULE")
        for name in ml["used_modules"]:
            details = md[name]
            mod_type = details.get("module_type")
            if mod_type == "empty":
                emoji = "🧩"
            elif mod_type == "qa":
                emoji = "❓"
            elif mod_type == "regular":
                emoji = "📦"
            print(f"\n  {emoji} {mod_type.upper()} MODULE: {name}")
            if mod_type == "regular":
                oc = details["option_coverage"]["details_per_option_field"]
                for field, info in oc.items():
                    print(f"\n    🔹 {field}: {info['percentage']:.2f}%")
                    print("       ✅ Used:")
                    if info["used_values"]:
                        for v in info["used_values"]:
                            print(f"            • {v}")
                    else:
                        print("            • None")
                    print("       ❌ Missing:")
                    if info["missing_values"]:
                        for v in info["missing_values"]:
                            print(f"            • {v}")
                    else:
                        print("            • None")
            elif mod_type == "qa":
                fc = details["field_coverage"]
                print(f"\n    🔹 Questions answered: {fc['used_count_in_module']}/{fc['total_defined_in_module']}")
                print("       ✅ Answered:")
                if fc.get("used_questions_list"):
                    for q in fc["used_questions_list"]:
                        print(f"            • {q}")
                else:
                    print("            • None")
                print("       ❌ Unanswered:")
                if fc.get("missing_questions_list"):
                    for q in fc["missing_questions_list"]:
                        print(f"            • {q}")
                else:
                    print("            • None")
            else:
                print("\n    🔹 No detailed spec available.")

    def save_readable_report(self, output_file: str | None = None) -> str:
        """Save a human-readable text report."""
        out_path = self._get_output_path(output_file, "txt")
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

    parser = argparse.ArgumentParser(description="Analyze coverage data with refined structure.")
    parser.add_argument("coverage_file", help="Path to coverage file to analyze (e.g., merged_coverage.json)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. Extension determines format (.json or .txt). If not given, both are saved with default names.",
    )
    args = parser.parse_args()

    try:
        analyzer = CoverageAnalyzer(args.coverage_file)

        if args.output:
            if args.output.endswith(".json"):
                output_path = analyzer.save_report(args.output)
                print(f"\nJSON report saved to: {output_path}")
                print("\n--- CONSOLE SUMMARY ---")
                analyzer.print_summary()
            elif args.output.endswith(".txt"):
                output_path = analyzer.save_readable_report(args.output)
                print(f"\nReadable report saved to: {output_path}")
            else:
                print("Unsupported output file extension. Use .json or .txt.", file=sys.stderr)
                json_path = analyzer.save_report()
                readable_path = analyzer.save_readable_report()
                print("\nReports saved with default names due to unrecognized extension:")
                print(f"  📊 JSON: {json_path}")
                print(f"  📝 Text: {readable_path}")
                print("\n--- CONSOLE SUMMARY ---")
                analyzer.print_summary()
        else:
            analyzer.print_summary()
            json_path = analyzer.save_report()
            readable_path = analyzer.save_readable_report()
            print("\nReports saved:")
            print(f"  📊 JSON: {json_path}")
            print(f"  📝 Text: {readable_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

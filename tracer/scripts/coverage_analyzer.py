"""Coverage analyzer for TRACER project."""

import json
from pathlib import Path

# Coverage percentage thresholds
FULL_COVERAGE_THRESHOLD = 100.0
EXCELLENT_COVERAGE_THRESHOLD = 80
GOOD_COVERAGE_THRESHOLD = 50
POOR_COVERAGE_THRESHOLD = 20


class CoverageAnalyzer:
    """Analyze coverage data from coverage files."""

    def __init__(self, coverage_file: str) -> None:
        """Initialize the coverage analyzer.

        Args:
            coverage_file: Path to the coverage file to analyze.
        """
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
            error_msg = (
                f"Coverage file not found: {self.coverage_file}\n"
                "You may need to run coverage_merger.py first to generate it."
            )
            raise FileNotFoundError(error_msg)
        with self.coverage_file.open() as f:
            return json.load(f)

    def _detect_qa_modules(self) -> list[str]:
        """Detect QA modules by looking for question-like keys."""
        qa_modules = []
        specification = self.data.get("specification", {})
        for module_name, module_spec in specification.items():
            if module_name == "modules":
                continue
            if isinstance(module_spec, dict):
                has_questions = any(isinstance(key, str) and key.endswith("?") for key in module_spec)
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
        all_module_names_from_spec_keys = [m for m in specification if m != "modules"]
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
                    for field_name in module_spec:
                        if module_footprint.get(field_name):
                            is_activated = True
                            break
                    if (
                        not is_activated and not module_spec.keys() and module_name in footprint
                    ):  # Spec is {} but module in footprint
                        is_activated = True

            activation_map[module_name] = FULL_COVERAGE_THRESHOLD if is_activated else 0.0
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
        total_unknown_questions = 0

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
            is_module_activated = (
                self.module_activation_status_data["activation_map"].get(module_name, 0.0) == FULL_COVERAGE_THRESHOLD
            )

            # Count unknown questions for QA modules
            total_unknown_questions += self._count_unknown_questions_for_module(module_name, module_footprint)

            # Process field and option coverage
            fields_count, used_fields_count, options_count, covered_options_count = (
                self._process_field_and_option_coverage(
                    module_name, module_spec, module_footprint, is_module_activated=is_module_activated
                )
            )
            total_defined_fields_count += fields_count
            total_used_fields_count += used_fields_count
            total_defined_options_count += options_count
            total_covered_options_count += covered_options_count

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
            "unknown_questions": {
                "total_count": total_unknown_questions,
            },
        }

    def _count_unknown_questions_for_module(self, module_name: str, module_footprint: dict) -> int:
        """Count unknown questions for a QA module."""
        if module_name not in self.qa_modules:
            return 0
        unknown_questions = module_footprint.get("unknown", [])
        return len(unknown_questions)

    def _process_field_and_option_coverage(
        self, module_name: str, module_spec: dict, module_footprint: dict, *, is_module_activated: bool
    ) -> tuple[int, int, int, int]:
        """Process field and option coverage for a single module.

        Returns:
            tuple: (fields_count, used_fields_count, options_count, covered_options_count)
        """
        fields_count = 0
        used_fields_count = 0
        options_count = 0
        covered_options_count = 0

        for field_name, spec_value in module_spec.items():
            fields_count += 1
            # A field is "used" if the module was activated AND the field appears in the footprint with some value.
            if is_module_activated and field_name in module_footprint and module_footprint[field_name]:
                used_fields_count += 1

            # Option coverage calculation
            if module_name not in self.qa_modules and isinstance(spec_value, list) and spec_value:
                options_count += len(spec_value)
                if is_module_activated:  # Options can only be covered if the module itself was activated
                    footprint_values = set(module_footprint.get(field_name, []))
                    covered_options = set(spec_value).intersection(footprint_values)
                    covered_options_count += len(covered_options)
            elif module_name in self.qa_modules and isinstance(spec_value, list):
                options_count += 1
                if (
                    is_module_activated
                    and field_name in module_footprint
                    and module_footprint.get(field_name)
                    and field_name in module_footprint.get(field_name, [])
                ):
                    covered_options_count += 1

        return fields_count, used_fields_count, options_count, covered_options_count

    def _calculate_detailed_module_coverage(self) -> dict:
        """Calculate detailed field and option coverage for each module."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})
        detailed_coverage = {}

        # Use the same source of module names as in _calculate_module_activation_status
        all_module_names = self._get_all_module_names()

        for module_name in all_module_names:
            module_spec = specification.get(module_name, {})
            module_footprint = footprint.get(module_name, {})

            is_activated = (
                self.module_activation_status_data["activation_map"].get(module_name, 0.0) == FULL_COVERAGE_THRESHOLD
            )
            module_type = self._determine_module_type(module_spec, module_name)

            # Field Coverage for this module
            used_fields_list, missing_fields_list, field_coverage_percentage = self._calculate_module_field_coverage(
                module_name, module_spec, module_footprint, is_activated
            )

            module_field_coverage = {
                "percentage": round(field_coverage_percentage, 2),
                "total_defined_in_module": len(used_fields_list) + len(missing_fields_list),
                "used_count_in_module": len(used_fields_list),
                "missing_count_in_module": len(missing_fields_list),
                "used_fields_list": used_fields_list,
                "missing_fields_list": missing_fields_list,
            }
            if module_type == "qa":  # Add question lists for QA
                module_field_coverage["used_questions_list"] = used_fields_list  # Questions are fields for QA
                module_field_coverage["missing_questions_list"] = missing_fields_list

                # Add unknown questions for QA modules
                unknown_questions = module_footprint.get("unknown", [])
                module_field_coverage["unknown_questions_list"] = sorted(unknown_questions)
                module_field_coverage["unknown_questions_count"] = len(unknown_questions)

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

                        module_used_options_summary.extend(
                            f"{field_name}: {opt}" for opt in sorted(covered_options_for_field)
                        )
                        module_missing_options_summary.extend(
                            f"{field_name}: {opt}" for opt in sorted(missing_options_for_field)
                        )

                        details_per_option_field[field_name] = {
                            "percentage": round(len(covered_options_for_field) / len(defined_options) * 100, 2)
                            if defined_options
                            else 0.0,
                            "defined_options_count": len(defined_options),
                            "covered_options_count": len(covered_options_for_field),
                            "missing_options_count": len(missing_options_for_field),
                            "used_values": sorted(covered_options_for_field),
                            "missing_values": sorted(missing_options_for_field),
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

        return {
            "global_summary": global_stats,
            "module_lists": {
                "used_modules": self.module_activation_status_data["used_modules"],
                "unused_modules": self.module_activation_status_data["unused_modules"],
                "total_modules": self.module_activation_status_data["total_modules"],
            },
            "module_details": detailed_module_stats,
        }

    def get_report(self) -> dict:
        """Return the generated report."""
        return self.report_data

    def _get_output_path(self, output_file: str | None, extension: str) -> Path:
        """Generate output path with proper naming convention."""
        if output_file:
            return Path(output_file)
        stem = self.coverage_file.stem

        last_underscore_index = stem.rfind("_")

        if last_underscore_index != -1:
            base_name = stem[:last_underscore_index]
            new_stem = base_name + "_report"
        else:
            new_stem = stem + "_report"

        return self.coverage_file.parent / f"{new_stem}.{extension}"

    def save_report(self, output_file: str | None = None) -> str:
        """Save JSON report to file."""
        out_path = self._get_output_path(output_file, "json")
        with out_path.open("w") as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        return str(out_path)

    def print_summary(self) -> None:
        """Print formatted coverage summary with module overview and detailed breakdown."""
        self._print_global_summary()
        self._print_module_lists()
        self._print_module_overview()
        self._print_module_details()

        self._print_global_summary()
        self._print_module_lists()
        self._print_module_overview()
        self._print_module_details()

    def _print_global_summary(self) -> None:
        """Print the global coverage metrics section."""
        gs = self.report_data["global_summary"]

        print("🤖 CHATBOT COVERAGE ANALYSIS")
        print("=" * 60)

        # 1. Global Summary
        print("\n📊 OVERALL METRICS")
        print(
            f"  • Module Activation: {gs['module_activation_coverage']['percentage']:.2f}% ({gs['module_activation_coverage']['activated_count']}/{gs['module_activation_coverage']['total_defined_modules']})"
        )
        print(
            f"  • Field Usage: {gs['field_usage_coverage']['percentage']:.2f}% ({gs['field_usage_coverage']['used_field_count']}/{gs['field_usage_coverage']['total_defined_fields']})"
        )
        print(
            f"  • Option Coverage: {gs['option_value_coverage']['percentage']:.2f}% ({gs['option_value_coverage']['covered_option_count']}/{gs['option_value_coverage']['total_defined_options']})"
        )

        if gs["unknown_questions"]["total_count"] > 0:
            print(f"  • Unknown Questions: {gs['unknown_questions']['total_count']} found")

    def _print_module_lists(self) -> None:
        """Print the module activation status section."""
        ml = self.report_data["module_lists"]

        print("\n🏗️ MODULE ACTIVATION STATUS\n")
        if ml["used_modules"]:
            print(f"  ✅ USED ({len(ml['used_modules'])}):")
            for mod in ml["used_modules"]:
                print(f"       • {mod}")
        if ml["unused_modules"]:
            print(f"  ❌ UNUSED ({len(ml['unused_modules'])}):")
            for mod in ml["unused_modules"]:
                print(f"       • {mod}")

    def _get_coverage_groupings(self) -> dict[str, list[str]]:
        """Group modules by coverage level for overview section."""
        md = self.report_data["module_details"]

        grouped = {
            "EXCELLENT (80%+)": [],
            "GOOD (50-79%)": [],
            "POOR (20-49%)": [],
            "VERY POOR (0-19%)": [],
        }

        for module_name, module_info in md.items():
            if not module_info["activated"]:
                label = f"{module_name}: 0.00% (inactive)"
                grouped["VERY POOR (0-19%)"].append(label)
                continue

            field_pct = module_info["field_coverage"]["field_percentage"]
            if module_info["field_coverage"]["total_defined_fields"] == 0:
                label = f"{module_name}: 100.00% (no spec)"
            else:
                label = f"{module_name}: {field_pct:.2f}%"

            if field_pct >= EXCELLENT_COVERAGE_THRESHOLD:
                grouped["EXCELLENT (80%+)"].append(label)
            elif field_pct >= GOOD_COVERAGE_THRESHOLD:
                grouped["GOOD (50-79%)"].append(label)
            elif field_pct >= POOR_COVERAGE_THRESHOLD:
                grouped["POOR (20-49%)"].append(label)
            else:
                grouped["VERY POOR (0-19%)"].append(label)

        return grouped

    def _print_module_overview(self) -> None:
        """Print the module coverage overview section."""
        grouped = self._get_coverage_groupings()

        print("\n🔍 MODULE COVERAGE OVERVIEW")

        category_emojis = {
            "EXCELLENT (80%+)": "🌟",
            "GOOD (50-79%)": "👍",
            "POOR (20-49%)": "⚠️",
            "VERY POOR (0-19%)": "❌",
        }

        for category, modules in grouped.items():
            if modules:
                emoji = category_emojis.get(category, "📊")
                print(f"\n  {emoji} {category}:")
                for label in modules:
                    print(f"    • {label}")

    def _get_module_color_and_emoji(self, module_info: dict) -> tuple[str, str]:
        """Get color code and emoji for a module based on its coverage."""

        class Colors:
            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            ORANGE = "\033[38;5;208m"
            RED = "\033[91m"
            RESET = "\033[0m"

        if not module_info["activated"]:
            return Colors.RED, "❌"

        field_pct = module_info["field_coverage"]["field_percentage"]

        # Use field percentage for determining color unless it's a no-spec module
        pct_for_color = field_pct
        if module_info["field_coverage"]["total_defined_fields"] == 0:
            pct_for_color = 100.0  # Consider no-spec modules as 100%

        # Select color based on percentage
        color_code = Colors.RED  # Default to Red for 0% or undefined
        if pct_for_color >= EXCELLENT_COVERAGE_THRESHOLD:
            color_code = Colors.GREEN
        elif pct_for_color >= GOOD_COVERAGE_THRESHOLD:
            color_code = Colors.YELLOW
        elif pct_for_color >= POOR_COVERAGE_THRESHOLD:
            color_code = Colors.ORANGE

        # Select emoji
        if not module_info["activated"]:
            emoji = "❌"
        elif pct_for_color >= EXCELLENT_COVERAGE_THRESHOLD:
            emoji = "✅"
        elif pct_for_color >= GOOD_COVERAGE_THRESHOLD:
            emoji = "⚡"
        elif pct_for_color >= POOR_COVERAGE_THRESHOLD:
            emoji = "⚠️"
        else:
            emoji = "🔴"

        return color_code, emoji

    def _print_module_details(self) -> None:
        """Print the detailed breakdown per module section."""
        md = self.report_data["module_details"]

        class Colors:
            RESET = "\033[0m"

        print("\n📝 DETAILED BREAKDOWN PER MODULE")

        for module_name, module_info in md.items():
            color_code, emoji = self._get_module_color_and_emoji(module_info)

            # Module header
            field_pct = module_info["field_coverage"]["field_percentage"]
            if module_info["field_coverage"]["total_defined_fields"] == 0:
                module_header_text = f"{module_name}: 100.00% field coverage (no spec defined)"
            else:
                module_header_text = f"{module_name}: {field_pct:.2f}% field coverage"

            print(f"\n  {emoji} {color_code}{module_header_text}{Colors.RESET}")

            self._print_module_field_details(module_info)
            self._print_module_option_details(module_info)
            self._print_module_qa_details(module_name, module_info)

    def _print_module_field_details(self, module_info: dict) -> None:
        """Print field coverage details for a module."""
        field_info = module_info["field_coverage"]

        if field_info["total_defined_fields"] > 0:
            print(
                f"    📋 Fields: {field_info['used_fields_count']}/{field_info['total_defined_fields']} used ({field_info['field_percentage']:.2f}%)"
            )

    def _print_module_option_details(self, module_info: dict) -> None:
        """Print option coverage details for a module."""
        if "option_coverage" not in module_info:
            return

        option_info = module_info["option_coverage"]
        details_per_field = option_info.get("details_per_option_field", {})

        if details_per_field:
            for field, info in details_per_field.items():
                print(f"\n    🔹 {field}: {info['percentage']:.2f}%")
                if info.get("used_values"):
                    print("       ✅ Used:")
                    for v in info["used_values"]:
                        print(f"            • {v}")
                print("       ❌ Missing:")
                missing_values = info.get("missing_values", [])
                if missing_values:
                    for v in missing_values:
                        print(f"            • {v}")
                else:
                    print("            • None")

        print(
            f"    📊 Options: {option_info['covered_options_in_module_count']}/{option_info['total_defined_options_in_module']} covered ({option_info['overall_percentage_for_module']:.2f}%)"
        )

    def _get_answered_questions(self, module_footprint: dict) -> list[str]:
        """Get list of answered questions from module footprint."""
        answered_questions = []
        for key, value in module_footprint.items():
            if key != "unknown" and value:
                answered_questions.extend(value if isinstance(value, list) else [value])
        return answered_questions

    def _get_unanswered_questions(self, module_name: str, answered_questions: list[str]) -> list[str]:
        """Get list of unanswered questions for a QA module."""
        specification = self.data.get("specification", {})
        module_spec = specification.get(module_name, {})
        all_possible_questions = []
        if isinstance(module_spec, dict):
            for value in module_spec.values():
                if isinstance(value, list):
                    all_possible_questions.extend(value)
        return [q for q in all_possible_questions if q not in answered_questions]

    def _print_module_qa_details(self, module_name: str, module_info: dict) -> None:
        """Print QA-specific details for a module."""
        if module_name not in self.qa_modules:
            if module_info["field_coverage"]["total_defined_fields"] == 0:
                print("\n    🔹 No detailed spec available.")
            return

        # QA module specific logic
        footprint = self.data.get("footprint", {})
        module_footprint = footprint.get(module_name, {})

        unknown_questions = module_footprint.get("unknown", [])
        unknown_count = len(unknown_questions)

        if unknown_count > 0:
            print(f"    Unknown Questions Found: {unknown_count}")

        # Answered questions
        answered_questions = self._get_answered_questions(module_footprint)
        print("       ✅ Answered:")
        if answered_questions:
            for q in answered_questions:
                print(f"            • {q}")
        else:
            print("            • None")

        # Unanswered questions
        unanswered = self._get_unanswered_questions(module_name, answered_questions)
        print("       ❌ Unanswered:")
        if unanswered:
            for q in unanswered:
                print(f"            • {q}")
        else:
            print("            • None")

        # Unknown questions
        if unknown_questions:
            print("       ❓ Unknown Questions (not in spec):")
            for q in unknown_questions:
                print(f"            • {q}")

    def save_readable_report(self, output_file: str | None = None) -> str:
        """Save a human-readable text report, stripping ANSI color codes."""
        out_path = self._get_output_path(output_file, "txt")
        import io
        import re
        import sys

        # Remove the color codes from the txt
        ansi_escape_pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            self.print_summary()  # This will print with colors to the buffer
            content_with_colors = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        # Strip ANSI codes from the captured content
        content_without_colors = ansi_escape_pattern.sub("", content_with_colors)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(content_without_colors)
        return str(out_path)

    def _get_all_module_names(self) -> list[str]:
        """Get all module names from specification and footprint."""
        specification = self.data.get("specification", {})
        footprint = self.data.get("footprint", {})

        all_module_names_from_spec_keys = [m for m in specification if m != "modules"]
        all_module_names_from_modules_list = []
        if "modules" in specification and isinstance(specification["modules"], list):
            all_module_names_from_modules_list = specification["modules"]
        all_module_names = list(dict.fromkeys(all_module_names_from_spec_keys + all_module_names_from_modules_list))
        if not all_module_names:
            all_module_names = list(footprint.keys())
        return all_module_names

    def _determine_module_type(self, module_spec: dict, module_name: str) -> str:
        """Determine the type of module based on its specification."""
        if module_name in self.qa_modules:
            return "qa"
        if not module_spec:
            return "empty"
        if not isinstance(module_spec, dict):
            return "undefined_spec"
        return "regular"

    def _calculate_module_field_coverage(
        self, module_name: str, module_spec: dict, module_footprint: dict, *, is_activated: bool
    ) -> tuple[list[str], list[str], float]:
        """Calculate field coverage for a single module.

        Returns:
            tuple: (used_fields_list, missing_fields_list, field_coverage_percentage)
        """
        defined_field_names = list(module_spec.keys()) if isinstance(module_spec, dict) else []

        used_fields_list = []
        if is_activated and isinstance(module_spec, dict):
            used_fields_list = [fn for fn in defined_field_names if module_footprint.get(fn)]

        missing_fields_list = sorted(set(defined_field_names) - set(used_fields_list))
        used_fields_list.sort()

        total_defined_fields_in_module = len(defined_field_names)
        used_fields_count_in_module = len(used_fields_list)

        field_coverage_percentage = 0.0
        if total_defined_fields_in_module > 0 and used_fields_count_in_module > 0:
            field_coverage_percentage = used_fields_count_in_module / total_defined_fields_in_module * 100
        elif is_activated and module_name in {"empty", "undefined_spec"}:
            field_coverage_percentage = 100.0  # Considered fully covered in terms of its (zero) fields
        else:
            field_coverage_percentage = 0.0

        return used_fields_list, missing_fields_list, field_coverage_percentage

    def _calculate_module_option_coverage(
        self, module_name: str, module_spec: dict, module_footprint: dict, *, is_activated: bool
    ) -> dict | None:
        """Calculate option coverage for a single module.

        Returns:
            dict: Option coverage data, or None if module has no option fields
        """
        if not isinstance(module_spec, dict):
            return None

        module_total_defined_options = 0
        module_total_covered_options = 0
        module_used_options_summary = []
        module_missing_options_summary = []
        details_per_option_field = {}

        # For regular modules with list-based options
        if module_name not in self.qa_modules:
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

                    module_used_options_summary.extend(
                        f"{field_name}: {opt}" for opt in sorted(covered_options_for_field)
                    )
                    module_missing_options_summary.extend(
                        f"{field_name}: {opt}" for opt in sorted(missing_options_for_field)
                    )

                    details_per_option_field[field_name] = {
                        "percentage": round(len(covered_options_for_field) / len(defined_options) * 100, 2)
                        if defined_options
                        else 0.0,
                        "defined_options_count": len(defined_options),
                        "covered_options_count": len(covered_options_for_field),
                        "missing_options_count": len(missing_options_for_field),
                        "used_values": sorted(covered_options_for_field),
                        "missing_values": sorted(missing_options_for_field),
                    }

        # Calculate overall percentage
        option_percentage_for_module = 0.0
        if module_total_defined_options > 0:
            option_percentage_for_module = module_total_covered_options / module_total_defined_options * 100
        elif is_activated:  # Activated but no defined option fields
            option_percentage_for_module = 100.0

        return {
            "overall_percentage_for_module": round(option_percentage_for_module, 2),
            "total_defined_options_in_module": module_total_defined_options,
            "covered_options_in_module_count": module_total_covered_options,
            "missing_options_in_module_count": module_total_defined_options - module_total_covered_options,
            "used_options_summary_list": sorted(module_used_options_summary),
            "missing_options_summary_list": sorted(module_missing_options_summary),
            "details_per_option_field": details_per_option_field,
        }


def main() -> None:
    """Main function to run coverage analysis from command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze coverage data with refined structure.")
    parser.add_argument("coverage_file", help="Path to coverage file to analyze (e.g., merged_coverage.json)")
    parser.add_argument(
        "-o",
        "--output",
        help="Base name for the output report files. Saves both .json and .txt reports (e.g., 'my_report' creates 'my_report.json' and 'my_report.txt'). If an extension is provided, it will be used to determine the base name.",
    )
    args = parser.parse_args()

    try:
        analyzer = CoverageAnalyzer(args.coverage_file)

        # Always print the summary to the console first
        analyzer.print_summary()

        json_output_filename_arg = None
        txt_output_filename_arg = None
        save_message = "\nReports saved:"

        if args.output:
            user_provided_path = Path(args.output)
            output_dir = user_provided_path.parent
            base_name_stem = user_provided_path.stem

            # Ensure output directory exists
            if output_dir != Path():  # Avoid trying to create "." if no path is specified
                output_dir.mkdir(parents=True, exist_ok=True)

            json_output_filename_arg = str(output_dir / f"{base_name_stem}.json")
            txt_output_filename_arg = str(output_dir / f"{base_name_stem}.txt")
            save_message = f"\nReports saved based on name '{args.output}':"

        json_path = analyzer.save_report(json_output_filename_arg)
        readable_path = analyzer.save_readable_report(txt_output_filename_arg)

        print(save_message)
        print(f"  📊 JSON: {json_path}")
        print(f"  📝 Text: {readable_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError) as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

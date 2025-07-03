"""Utilities for generating markdown reports from chatbot exploration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from tracer.schemas.functionality_node_model import FunctionalityNode
from tracer.utils.logging_utils import get_logger

from .constants import MAX_DESCRIPTION_LENGTH, MAX_FUNCTIONS_PER_CATEGORY

logger = get_logger()


@dataclass
class ReportData:
    """Data structure for report generation."""

    structured_functionalities: list[FunctionalityNode]
    supported_languages: list[str]
    fallback_message: str | None
    token_usage: dict[str, Any] | None = None


def write_report(
    output_dir: str,
    report_data: ReportData,
) -> None:
    """Write analysis results to multiple report files.

    Args:
        output_dir: Directory to write the report files to
        report_data: Report data containing functionalities and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    write_main_report(
        output_path,
        report_data.structured_functionalities,
        report_data.supported_languages,
        report_data.fallback_message,
        report_data.token_usage,
    )
    write_json_data(output_path, report_data.structured_functionalities)


def write_main_report(
    output_path: Path,
    structured_functionalities: list[FunctionalityNode],
    supported_languages: list[str],
    fallback_message: str | None,
    token_usage: dict[str, Any] | None = None,
) -> None:
    """Write the main analysis report in Markdown format."""
    report_path = output_path / "README.md"
    try:
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# Chatbot Functionality Analysis\n\n")
            write_executive_summary(f, structured_functionalities, supported_languages)
            write_functionality_overview(f, structured_functionalities)
            write_technical_details(f, supported_languages, fallback_message)
            if token_usage:
                write_performance_stats(f, token_usage)
            write_files_reference(f)
        logger.info("Main report written to: %s", report_path)
    except OSError:
        logger.exception("Failed to write main report file.")


def write_category_overview(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
    """Write a category-based overview of main functions."""
    if not functionalities:
        f.write("No functionalities to overview.\n\n")
        return

    categories: dict[str, list[dict]] = {}

    def collect_by_category(nodes: list[FunctionalityNode]) -> None:
        for node in nodes:
            if isinstance(node, dict):
                category = node.get("suggested_category", "Uncategorized")
                if category not in categories:
                    categories[category] = []
                categories[category].append(node)
                collect_by_category(node.get("children", []))

    collect_by_category(functionalities)
    sorted_categories = get_sorted_categories(list(categories.keys()))
    write_category_sections(f, categories, sorted_categories)


def get_sorted_categories(category_names: list[str]) -> list[str]:
    """Sort categories alphabetically with Uncategorized last."""
    sorted_categories = sorted(category_names)
    if "Uncategorized" in sorted_categories:
        sorted_categories.remove("Uncategorized")
        sorted_categories.append("Uncategorized")
    return sorted_categories


def write_category_sections(f: TextIO, categories: dict[str, list[dict]], sorted_categories: list[str]) -> None:
    """Write the sections for each category."""
    for category in sorted_categories:
        nodes = categories[category]
        icon = "📂" if category != "Uncategorized" else "📄"
        f.write(f"**{icon} {category}** ({len(nodes)} functions)\n")
        write_category_functions(f, nodes)
        f.write("\n")


def write_category_functions(f: TextIO, nodes: list[dict]) -> None:
    """Write functions for a category with truncation if needed."""
    display_nodes = nodes[:MAX_FUNCTIONS_PER_CATEGORY]
    for node in display_nodes:
        name = node.get("name", "Unnamed").replace("_", " ").title()
        desc = node.get("description", "No description")
        if len(desc) > MAX_DESCRIPTION_LENGTH:  # Uses MAX_DESCRIPTION_LENGTH
            desc = desc[: MAX_DESCRIPTION_LENGTH - 3] + "..."
        f.write(f"- *{name}*: {desc}\n")
    remaining_count = len(nodes) - MAX_FUNCTIONS_PER_CATEGORY
    if remaining_count > 0:
        f.write(f"- *...and {remaining_count} more functions*\n")


def write_executive_summary(f: TextIO, functionalities: list[FunctionalityNode], languages: list[str]) -> None:
    """Write executive summary section."""
    f.write("## 📊 TRACER Report\n\n")
    if not functionalities:
        f.write("❌ **No functionalities discovered**\n\n")
        return

    total_functions = 0
    categories_set = set()

    def count_functions_recursive(nodes_list: list[FunctionalityNode]) -> None:
        nonlocal total_functions
        for item_node in nodes_list:
            if isinstance(item_node, dict):
                total_functions += 1
                if item_node.get("suggested_category"):
                    categories_set.add(item_node.get("suggested_category"))
                count_functions_recursive(item_node.get("children", []))

    count_functions_recursive(functionalities)
    f.write(f"✅ **{total_functions} functionalities** discovered across **{len(categories_set)} categories**\n\n")
    if languages:
        f.write(f"🌐 **Languages supported:** {', '.join(languages)}\n\n")
    f.write("### 🎯 Functionality Overview\n\n")
    write_category_overview(f, functionalities)


def write_functionality_overview(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
    """Write comprehensive functionality overview grouped by category with full details."""
    f.write("## 🗂️ Functionality Details\n\n")
    if not functionalities:
        f.write("No functionalities to categorize.\n\n")
        return

    categories_map: dict[str, list[FunctionalityNode]] = {}

    def collect_by_category_recursive(nodes_list: list[FunctionalityNode]) -> None:
        for item_node in nodes_list:
            if isinstance(item_node, dict):
                category = item_node.get("suggested_category", "Uncategorized")
                if category not in categories_map:
                    categories_map[category] = []
                categories_map[category].append(item_node)
                collect_by_category_recursive(item_node.get("children", []))

    collect_by_category_recursive(functionalities)
    sorted_categories_list = get_sorted_categories(list(categories_map.keys()))

    for category_name in sorted_categories_list:
        category_nodes = categories_map[category_name]
        icon = "📂" if category_name != "Uncategorized" else "📄"
        f.write(f"### {icon} {category_name} ({len(category_nodes)} functions)\n\n")
        for node_item in category_nodes:
            write_detailed_function_info(f, node_item)
        f.write("\n")


def write_detailed_function_info(f: TextIO, node: FunctionalityNode) -> None:
    """Write detailed information for a single function."""
    name = node.get("name", "Unnamed").replace("_", " ").title()
    desc = node.get("description", "No description")
    f.write(f"#### 🔧 {name}\n\n")
    f.write(f"**Description:** {desc}\n\n")
    write_function_parameters(f, node)
    write_function_outputs(f, node)
    write_function_relationships(f, node)
    f.write("---\n\n")


def write_function_parameters(f: TextIO, node: FunctionalityNode) -> None:
    """Write parameters section for a function."""
    parameters = node.get("parameters", [])
    if parameters and any(param for param in parameters if param is not None):
        f.write("**Parameters:**\n")
        for param in parameters:
            if param is not None and isinstance(param, dict):  # Ensure param is a dict
                param_name = param.get("name", "Unknown")
                param_desc = param.get("description", "No description")
                param_options = param.get("options", [])
                f.write(f"- `{param_name}`: {param_desc}")
                if param_options:
                    options_str = ", ".join(f"`{opt}`" for opt in param_options)
                    f.write(f" *Options: {options_str}*")
                f.write("\n")
            elif param is not None:  # Handle non-dict params if any (though schema implies dicts)
                f.write(f"- `{param!s}`\n")
        f.write("\n")
    else:
        f.write("**Parameters:** None\n\n")


def write_function_outputs(f: TextIO, node: FunctionalityNode) -> None:
    """Write outputs section for a function."""
    outputs = node.get("outputs", [])
    if outputs and any(output for output in outputs if output is not None):
        f.write("**Outputs:**\n")
        for output in outputs:
            if output is not None and isinstance(output, dict):  # Ensure output is a dict
                output_category = output.get("category", "Unknown")
                output_desc = output.get("description", "No description")
                f.write(f"- `{output_category}`: {output_desc}\n")
            elif output is not None:  # Handle non-dict outputs
                f.write(f"- {output!s}\n")
        f.write("\n")
    else:
        f.write("**Outputs:** None\n\n")


def write_function_relationships(f: TextIO, node: FunctionalityNode) -> None:
    """Write parent-child relationships for a function."""
    parent_names = node.get("parent_names", [])
    children = node.get("children", [])
    if parent_names:
        parents_str = ", ".join(f"`{parent.replace('_', ' ').title()}`" for parent in parent_names)
        f.write(f"**Parent Functions:** {parents_str}\n\n")
    if children:
        f.write("**Child Functions:**\n")
        for child in children:
            if isinstance(child, dict):
                child_name = child.get("name", "Unknown").replace("_", " ").title()
                child_desc = child.get("description", "No description")
                f.write(f"- `{child_name}`: {child_desc}\n")
        f.write("\n")


def write_technical_details(f: TextIO, languages: list[str], fallback_message: str | None) -> None:
    """Write technical details section."""
    f.write("## ⚙️ Technical Details\n\n")
    f.write("### 🌐 Language Support\n\n")
    if languages:
        for lang in languages:
            f.write(f"- {lang}\n")
    else:
        f.write("No specific language support detected.\n")
    f.write("\n")
    f.write("### 🔄 Fallback Behavior\n\n")
    if fallback_message:
        f.write(f"```\n{fallback_message}\n```\n\n")
    else:
        f.write("No fallback message detected.\n\n")


def write_performance_stats(f: TextIO, token_usage: dict[str, Any]) -> None:
    """Write performance statistics section."""
    f.write("## 📈 Performance Statistics\n\n")

    def fmt_num(num: float | str) -> str:
        return f"{num:,}" if isinstance(num, (int, float)) else str(num)

    f.write("### Overview\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Total LLM Calls | {fmt_num(token_usage.get('total_llm_calls', 'N/A'))} |\n")
    f.write(f"| Successful Calls | {fmt_num(token_usage.get('successful_llm_calls', 'N/A'))} |\n")
    f.write(f"| Failed Calls | {fmt_num(token_usage.get('failed_llm_calls', 'N/A'))} |\n")
    f.write(f"| Total Tokens | {fmt_num(token_usage.get('total_tokens_consumed', 'N/A'))} |\n")
    if "estimated_cost" in token_usage:
        f.write(f"| Estimated Cost | ${token_usage.get('estimated_cost', 0.0):.4f} USD |\n")
    if "total_application_execution_time" in token_usage:
        exec_time = token_usage["total_application_execution_time"]
        if isinstance(exec_time, dict) and "formatted" in exec_time:
            f.write(f"| Execution Time | {exec_time['formatted']} |\n")
    f.write("\n")

    f.write("### Phase Breakdown\n\n")
    phases = [
        ("Exploration", token_usage.get("exploration_phase", {})),
        ("Analysis", token_usage.get("analysis_phase", {})),
    ]
    f.write("| Phase | Prompt Tokens | Completion Tokens | Total Tokens | Cost |\n")
    f.write("|-------|---------------|-------------------|--------------|------|\n")
    for phase_name, phase_data in phases:
        prompt_tokens = fmt_num(phase_data.get("prompt_tokens", "N/A"))
        completion_tokens = fmt_num(phase_data.get("completion_tokens", "N/A"))
        total_tokens = fmt_num(phase_data.get("total_tokens", "N/A"))
        cost_val = phase_data.get("estimated_cost")
        cost = f"${cost_val:.4f}" if isinstance(cost_val, (float, int)) else "N/A"
        f.write(f"| {phase_name} | {prompt_tokens} | {completion_tokens} | {total_tokens} | {cost} |\n")
    f.write("\n")

    if token_usage.get("models_used"):
        f.write("### Models Used\n\n")
        for model in token_usage["models_used"]:
            f.write(f"- {model}\n")
        f.write("\n")


def write_files_reference(f: TextIO) -> None:
    """Write files reference section."""
    f.write("## 📁 Generated Files\n\n")
    f.write("This analysis generated the following files:\n\n")
    f.write("- **`README.md`** - This main report with comprehensive functionality analysis\n")
    f.write("- **`functionalities.json`** - Raw JSON data structure\n")
    f.write(
        "- **`workflow_graph.*`** - Visual graph of functionality relationships (format depends on configuration: PDF, PNG, SVG, or all formats)\n"
    )
    f.write("- **`profiles/`** - Directory containing user profile YAML files\n\n")


def write_json_data(output_path: Path, functionalities: list[FunctionalityNode]) -> None:
    """Write raw JSON data to separate file."""
    json_path = output_path / "functionalities.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(functionalities, f, indent=2, ensure_ascii=False)
        logger.info("JSON data written to: %s", json_path)
    except (TypeError, OSError):
        logger.exception("Failed to write JSON data")

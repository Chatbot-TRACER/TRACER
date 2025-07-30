"""CLI utility functions for handling command-line operations."""

import json
from typing import Any

from chatbot_connectors import ChatbotFactory

from tracer.utils.logging_utils import get_logger

logger = get_logger()


def handle_list_connector_params(technology: str) -> None:
    """Handle the --list-connector-params option.

    Args:
        technology: The chatbot technology to show parameters for

    Raises:
        ValueError: If the technology is invalid or parameters cannot be retrieved
    """
    try:
        params = ChatbotFactory.get_chatbot_parameters(technology)
        print(f"\nParameters for '{technology}' chatbot:")
        print("-" * 50)
        for param in params:
            print(f"  - Name: {param.name}")
            print(f"    Type: {param.type}")
            print(f"    Required: {param.required}")
            if param.default is not None:
                print(f"    Default: {param.default}")
            print(f"    Description: {param.description}")
            print()

        print_parameter_examples(params)

    except ValueError as e:
        available_types = ChatbotFactory.get_available_types()
        msg = f"Error: {e}. Available chatbot types: {', '.join(available_types)}"
        raise ValueError(msg) from e
    except (ImportError, AttributeError) as e:
        msg = f"Error retrieving parameters: {e}"
        raise RuntimeError(msg) from e


def print_parameter_examples(params: list) -> None:
    """Print usage examples for connector parameters.

    Args:
        params: List of parameter objects from the connector
    """
    print("Example usage:")
    example_params = {}
    for param in params:
        if param.required:
            if param.name == "base_url":
                example_params[param.name] = "http://localhost"
            elif param.name == "port":
                example_params[param.name] = 8080
            else:
                example_params[param.name] = f"<{param.name}>"

    if example_params:
        json_example = json.dumps(example_params, indent=2)
        kv_example = ",".join([f"{k}={v}" for k, v in example_params.items()])
        json_cleaned = json_example.replace("\n", "")
        print(f"  JSON format: --connector-params '{json_cleaned}'")
        print(f'  Key=Value format: --connector-params "{kv_example}"')


def handle_list_connectors() -> None:
    """Handle the --list-connectors option.

    Raises:
        RuntimeError: If connector information cannot be retrieved
    """
    try:
        available_types = ChatbotFactory.get_available_types()
        registered_connectors = ChatbotFactory.get_registered_connectors()

        print("\nAvailable Chatbot Connector Technologies:")
        print("=" * 50)

        if not available_types:
            print("No chatbot connectors are currently registered.")
        else:
            for connector_type in sorted(available_types):
                description = registered_connectors.get(connector_type, {}).get(
                    "description", "No description available"
                )
                print(f"  • {connector_type}")
                print(f"    Description: {description}")
                print(f"    Use: --technology {connector_type}")
                print(f"    Parameters: --list-connector-params {connector_type}")
                print()

            print(f"Total: {len(available_types)} connector(s) available")
            print("\nUsage:")
            print("  To see parameters for a specific connector:")
            print("    --list-connector-params <technology>")
            print("  To use a connector:")
            print("    --technology <technology> --connector-params <params>")

    except (ImportError, AttributeError) as e:
        msg = f"Error retrieving connector information: {e}"
        raise RuntimeError(msg) from e


def parse_connector_params(connector_params_str: str | None) -> dict[str, Any]:
    """Parse connector parameters from string input.

    Args:
        connector_params_str: JSON string or key=value pairs

    Returns:
        Dictionary of connector parameters

    Raises:
        ValueError: If parameter parsing fails
    """
    params = {}

    if connector_params_str:
        try:
            # Try to parse as JSON first
            if connector_params_str.strip().startswith("{"):
                params = json.loads(connector_params_str)
            else:
                # Parse as key=value pairs
                for pair in connector_params_str.split(","):
                    if "=" not in pair:
                        continue
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to convert to appropriate types
                    if value.lower() in ("true", "false"):
                        params[key] = value.lower() == "true"
                    elif value.isdigit():
                        params[key] = int(value)
                    else:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value

        except (json.JSONDecodeError, ValueError) as e:
            logger.exception("Failed to parse connector parameters: %s", connector_params_str)
            msg = f"Invalid connector parameters format: {e}"
            raise ValueError(msg) from e

    return params

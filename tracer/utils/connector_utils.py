"""Connector utility functions for instantiating and managing chatbot connections."""

from typing import Any

from chatbot_connectors import Chatbot, ChatbotFactory

from tracer.utils.logging_utils import get_logger
from tracer.utils.tracer_error import ConnectorError

logger = get_logger()


def instantiate_connector(technology: str, connector_params: dict[str, Any]) -> Chatbot:
    """Instantiate the appropriate chatbot connector based on the specified technology.

    Args:
        technology: The name of the chatbot technology platform
        connector_params: Dictionary of parameters for the connector

    Returns:
        An instance of the appropriate connector class

    Raises:
        ConnectorError: If the connector fails health check or has connectivity issues
    """
    logger.info("Instantiating connector for technology: %s", technology)

    if connector_params:
        logger.debug("Using connector parameters: %s", connector_params)

    try:
        # Create the chatbot using the factory with provided parameters
        chatbot = ChatbotFactory.create_chatbot(chatbot_type=technology, **connector_params)

        # Perform health check
        logger.info("Performing health check for chatbot connector...")
        chatbot.health_check()
        logger.info("Chatbot connector health check passed")

    except ValueError as e:
        logger.exception("Failed to instantiate connector for technology '%s'", technology)
        available_types = ChatbotFactory.get_available_types()
        logger.exception("Available chatbot types: %s", ", ".join(available_types))
        msg = f"Failed to instantiate connector for '{technology}'."
        raise ConnectorError(msg) from e

    except (TypeError, KeyError) as e:
        logger.exception("Invalid parameters for chatbot technology '%s'", technology)

        # Try to get parameter info to help the user
        try:
            required_params = ChatbotFactory.get_chatbot_parameters(technology)
            param_info = []
            for param in required_params:
                required_str = "Required" if param.required else "Optional"
                default_str = f" (default: {param.default})" if param.default is not None else ""
                param_info.append(f"  - {param.name} ({param.type}): {required_str}{default_str}")

            logger.exception("Expected parameters for '%s':\n%s", technology, "\n".join(param_info))
            logger.exception(
                "Use --list-connector-params %s to see detailed parameter information and examples", technology
            )
        except (ImportError, AttributeError):
            logger.exception("Use --list-connector-params %s to see required parameters", technology)

        if not connector_params:
            msg = f"No connector parameters provided for '{technology}'. Use --connector-params to specify required parameters or --list-connector-params {technology} for help."
        else:
            msg = f"Invalid parameters for chatbot '{technology}'. Check the connector parameters format."
        raise ConnectorError(msg) from e

    except ConnectorError:
        logger.exception("Connector health check failed for technology '%s'", technology)
        raise  # Re-raise the original ConnectorError to be caught by main

    except Exception as e:
        logger.exception("Unexpected error instantiating connector for technology '%s'", technology)
        msg = f"Unexpected error instantiating connector for '{technology}'."
        raise ConnectorError(msg) from e
    else:
        return chatbot

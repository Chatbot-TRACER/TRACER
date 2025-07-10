# Custom Exceptions in TRACER

This document outlines the custom exception hierarchy used in the TRACER application. These exceptions are designed to provide specific, actionable error information, making it easier to handle failures in different environments, such as a web interface or a CLI.

## Exception Hierarchy

The exceptions are organized in a hierarchy, with `TracerError` as the base class for all custom exceptions in the application.

- **`TracerError`**: The base class for all custom exceptions in TRACER. Catching this exception will catch any of the more specific exceptions listed below.

- **`GraphvizNotInstalledError`**: This exception is raised when the Graphviz `dot` executable is not found in the system's PATH. This indicates that Graphviz is either not installed or not configured correctly.

- **`ConnectorError`**: This is a base class for all errors related to chatbot connectors. It can be raised for a variety of connector-related issues, such as connection failures, invalid configurations, or authentication problems.

- **`LLMError`**: This exception is raised for any errors that occur while interacting with the Language Model (LLM) API. This could include issues with API keys, network problems, or other API-related errors.

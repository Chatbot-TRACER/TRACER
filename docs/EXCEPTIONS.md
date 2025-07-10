# Custom Exceptions in TRACER

This document outlines the custom exception hierarchy used in the TRACER application. These exceptions are designed to provide specific, actionable error information, making it easier to handle failures in different environments, such as a web interface or a CLI.

## Exception Hierarchy

The exceptions are organized in a hierarchy, with `TracerError` as the base class for all custom exceptions in the application.

- **`TracerError`**: The base class for all custom exceptions in TRACER. Catching this exception will catch any of the more specific exceptions listed below.

- **`GraphvizNotInstalledError`**: This exception is raised when the Graphviz `dot` executable is not found in the system's PATH. This indicates that Graphviz is either not installed or not configured correctly.

- **`ConnectorError`**: This is the base class for all errors related to chatbot connectors. More specific connector errors inherit from this class:
  - **`ConnectorConnectionError`**: Raised when the application is unable to establish a connection to the chatbot endpoint (e.g., network issues, incorrect URL, timeout).
  - **`ConnectorAuthenticationError`**: Raised when authentication with the chatbot connector fails (e.g., invalid API key, expired token).
  - **`ConnectorConfigurationError`**: Raised when the chatbot connector is configured incorrectly (e.g., missing required parameters, invalid settings).
  - **`ConnectorResponseError`**: Raised when the chatbot connector receives an invalid or unexpected response from the chatbot API (e.g., malformed JSON, unexpected status code).

- **`LLMError`**: This exception is raised for any errors that occur while interacting with the Language Model (LLM) API. This could include issues with API keys, network problems, or other API-related errors.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.7.0 - 2025-10-23

- Migrated from manually implementing the LLM initialization to using `init-chat-model`.

## 0.5.0 - 2025-10-13

### Added

- Updated `chabot-connectors` to include the BotLovers connector and Metro Madrid connector

## [0.3.0] - 2025-07-27

### Added

- **Separate Profile Model Support**: Introduced a new CLI argument `-pm/--profile-model` to specify a different model for profile generation. This allows users to use one model for exploration (e.g., a powerful model like `gpt-4o`) and a different model for profile generation (e.g., a cheaper model like `gpt-4o-mini`). If not specified, the profile model defaults to the same model used for exploration, maintaining backward compatibility.

### Changed

- **Model Configuration**: Updated the `-m/--model` argument description to clarify it's specifically for exploration. The configuration summary now displays both the exploration model and profile model for better transparency.
- **Internal Architecture**: Enhanced the analysis pipeline to support separate model configurations, with the profile model parameter propagated through the analysis phase to ensure profiles are generated with the specified model.

## [0.2.13] - 2025-07-11

### Fixed

- **LLM Error Handling**: Authentication and credential errors during LLM (OpenAI/Gemini) initialization are now correctly caught and raised as `LLMError`. This ensures that issues such as missing or invalid API keys, or lack of budget, are reported as LLM errors and can be handled appropriately by downstream consumers (e.g., web frontends).

### Changed

- **Exception Exposure for Web Integration**: All custom exception classes (`LLMError`, `ConnectorError`, etc.) are now re-exported in `tracer/__init__.py`. This allows web applications and other consumers to import and handle these exceptions directly from the `tracer` package, simplifying integration and error handling in external tools.

## [0.2.12] - 2025-07-10

### Fixed

- **Connector Stability**: Improved exception handling in chatbot connectors to prevent application hangs on connection errors. The application now raises a `ConnectorConnectionError` and exits gracefully.
- **Code Quality**: Addressed multiple linting issues (`TRY300`, `TRY003`, `EM102`, `RUF002`, `ANN204`, `D107`, `ANN401`) for improved code quality and consistency.

## [0.2.11] - 2025-07-10

### Added

- **Exception Hierarchy**: Introduced a structured exception hierarchy for Tracer, defining custom exceptions in `tracer/utils/tracer_error.py`.
- **Graphviz Checks**: Added early failure detection for Graphviz availability, raising `GraphvizNotInstalledError` if not found.

### Changed

- **Error Handling**: Propagated and handled specific exceptions (`ConnectorError`, `LLMError`, etc.) in `tracer/main.py` for improved error flow.
- **Connector Health Checks**: Updated connector health checks to raise granular `Connector*Error` types.
- **Documentation**: Documented the new exception hierarchy in `docs/EXCEPTIONS.md`.

## [0.2.10] - 2025-07-04

### Fixed

- **Logging**: Error and warning messages are now directed to standard error (stderr) while informational output remains on standard output (stdout). This prevents error traces from being mixed with regular logs and enables reliable shell redirection like `2>errors.log`.

## [0.2.9] - 2025-07-04

### Fixed

- **Execution Stability**: Implemented comprehensive error handling to prevent the application from hanging or crashing silently. `tracer` now provides clean, concise error messages and reliably exits with a non-zero status code upon failure.
- **Connection Errors**: Added a "fail-fast" health check to validate the chatbot URL at startup, preventing long timeouts on invalid connections.
- **Invalid Credentials**: Added a "fail-fast" health check for LLM API keys. The application now exits immediately with a clear error if a key is invalid.
- **API Timeouts**: Added request timeouts to OpenAI API calls to prevent the application from hanging indefinitely if the service is unresponsive.

## [0.2.8] - 2024-06-19

### Changed

- Moved `graph_state` from being a class attribute to a parameter passed through exploration sessions. This change simplifies state management by making it more explicit and less prone to side effects. The state is now initialized in `run_exploration` and passed to `_run_exploration_sessions`, which then manages it through each `run_exploration_session` call. This ensures that each exploration run starts with a clean, predictable state.

### Fixed

- Resolved an issue where `FunctionalityNode` instances were being shared across different exploration sessions. Previously, nodes were not being deep-copied, leading to unintended modifications and inconsistent state between sessions. Now, `copy.deepcopy()` is used in `_select_next_node` to ensure that each session works with an independent copy of the exploration graph, preventing state corruption and improving the reliability of the exploration process.

## [0.2.7] - 2024-06-19

### Fixed

- Fixed an issue where the `_detect_fallback` method in `ChatbotExplorationAgent` was not correctly handling exceptions during the fallback message detection process. The previous implementation had a broad exception catch that could mask underlying issues. The updated implementation now includes more specific exception handling and ensures that the `fallback_message` is always initialized, preventing potential `UnboundLocalError` exceptions.

## [0.2.6] - 2024-06-19

### Changed

- Updated the `_select_next_node` method in `ChatbotExplorationAgent` to improve the exploration strategy. Previously, the method would stop exploration if the pending nodes queue was empty. The new implementation allows for general exploration sessions even when no specific nodes are pending, ensuring that the exploration continues until the specified number of sessions is completed. This change enhances the thoroughness of the chatbot exploration process.

## [0.2.5] - 2024-06-19

### Added

- Implemented a feature to deduplicate functionality nodes before the analysis phase begins. This is achieved through the new `_aggressive_node_deduplication` method in the `ChatbotExplorationAgent`, which performs pairwise comparisons to merge similar nodes. This change reduces redundancy and improves the clarity of the final analysis by ensuring that the workflow structure is built from a concise and unique set of functionalities.

## [0.2.4] - 2024-06-13

### Changed

- The `is_duplicate_functionality` function has been updated to improve the accuracy of detecting duplicate nodes. The function now takes a `base_node` and a list of `other_nodes` as input, and it returns a boolean indicating if a duplicate is found, along with the name of the duplicate node. This change allows for more precise identification of redundant functionalities during the chatbot exploration analysis.

## [0.2.3] - 2024-06-13

### Added

- Introduced the `to_detailed_string` method in the `FunctionalityNode` class to provide a comprehensive, human-readable representation of a node's properties and its children. This feature is primarily for debugging purposes, offering a clear and structured way to inspect the state of the exploration graph at various stages.

## [0.2.2] - 2024-06-13

### Changed

- The `_process_node_group_for_merge` function has been refactored to enhance the node merging process. It now returns a list containing a single merged node upon success, rather than the node itself. This change standardizes the return type, making the function's output more predictable and easier to integrate into the broader analysis workflow.

## [0.2.1] - 2024-06-13

### Changed

- The `_check_and_merge_nodes` function has been renamed to `_group_and_merge_nodes` to more accurately reflect its purpose. This function is responsible for grouping nodes by name and then merging them, and the new name provides better clarity on its operation within the analysis graph.

## [0.2.0] - 2024-06-13

### Added

- Implemented a new workflow structuring graph, `build_structure_graph`, which replaces the previous `build_workflow_graph`. This new graph is designed to be more focused and efficient, responsible only for structuring the final output of the chatbot exploration. It processes the conversation history and discovered functionalities to build a coherent workflow, which is then passed to a separate graph for profile generation. This separation of concerns improves modularity and makes the analysis process easier to maintain and extend.

## [0.1.2] - 2024-06-13

### Changed

- The `build_workflow_graph` function has been updated to remove the final node that returned the graph's state. The state is now implicitly returned by the last operational node in the graph. This change simplifies the graph structure and makes the flow more intuitive.

## [0.1.1] - 2024-06-13

### Added

- Introduced a new parameter `nested_forward` in the `run_analysis` method and the `build_profile_generation_graph` function. When set to `True`, this parameter enables the use of nested `forward()` calls in the generated user profiles, allowing for more complex and realistic user interactions.

## [0.1.0] - 2024-06-12

### Changed

- Major refactoring of the analysis phase to use a two-step graph process. The system now first builds a `workflow_graph` to structure the functionalities and then uses a `profile_generation_graph` to create user profiles. This separation improves modularity and clarity.
- The `FunctionalityNode` model has been updated with a new `children` field, allowing for a hierarchical representation of chatbot functionalities.
- The `State` model for the analysis graphs has been updated to include `workflow_structure` and `conversation_goals`.

## [0.0.8] - 2024-06-11

### Fixed

- Fixed a bug where the `openai_api_key` was not being correctly passed to the `ChatOpenAI` constructor, causing authentication errors. The `_initialize_llm` method now correctly retrieves the API key from the environment and passes it to the model.

### Added

- Introduced a `TokenUsageTracker` callback to monitor and log the token consumption and estimated cost of LLM calls. The summary is now displayed at the end of the execution, providing better visibility into API usage.

## [0.0.7] - 2024-06-05

### Added

- This version introduces the `ChatbotExplorationAgent`, a new component that uses LangGraph to orchestrate the exploration and analysis of chatbots. This agent replaces the previous `ChatbotExplorer` and provides a more structured and extensible way to manage the different phases of the process.

## [0.t] - 2024-04-18

### Added

- Add `pyproject.toml` to standardize project configuration and dependency management. This change streamlines the setup process and ensures consistency across development environments.
- Introduce `ruff` as the code formatter and linter to enforce a consistent code style and identify potential issues early.
- Add `pre-commit` hooks to automate code formatting and linting before each commit, ensuring that all code committed to the repository adheres to the established standards.

## [0.0.6] - 2024-04-18

### Added

- This is the first version of the project where we are starting to use a changelog.

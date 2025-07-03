# TRACER

[![CI](https://github.com/Chatbot-TRACER/TRACER/actions/workflows/ci.yml/badge.svg)](https://github.com/Chatbot-TRACER/TRACER/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Chatbot-TRACER/TRACER)](https://github.com/Chatbot-TRACER/TRACER/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/chatbot-tracer)](https://pypi.org/project/chatbot-tracer/)

> Task Recognition and Chatbot ExploreR

A tool for automatically exploring and analyzing chatbots, generating a model of the functionalities and user profiles for testing.

## 1. Overview & Goals

TRACER is a tool designed to automatically interact with and analyze target chatbots. It uses Large Language Models (LLMs) to conduct multiple conversational sessions, identify the chatbot's core functionalities, limitations, and interaction flows, and generate structured outputs for testing and analysis.

The main goals of TRACER are:

- **Workflow Modeling:** Model the user's journey through the chatbot as a directed graph, capturing sequential dependencies, branching logic, and optional steps, adapt the modeling approach based on whether the chatbot is primarily transactional or informational.
- **Profile Generation:** Generate standardized YAML user profiles based on discovered functionalities and workflows, suitable for [Sensei](https://github.com/satori-chatbots/user-simulator).

## 2. Core Functionality

The system follows a multi-phase approach implemented via a LangGraph structure:

1. **Chatbot Interaction:** Connect to and converse with target chatbots (initially Taskyto, Ada-UAM) via provided connectors.
2. **Session Preparation:** Before starting the conversations the chatbot's language and fallback message are found by sending a few confusing messages.
3. **Exploration Sessions:**
   1. Conduct multiple conversational sessions to probe different aspects of the chatbot.
   2. If a fallback is received during the conversation, the LLM will rephrase the sentence and if it is received again, the topic will be changed.
   3. After each conversation the LLM tries to extract functionalities so that they can be further explored on the next sessions.
4. **Bot Classification:** After running all the sessions, the conversations and the found functionalities are passed to an LLM which will determine if the chatbot is transactional or informative.
   - **Transactional:** chatbots that allow you to perform actions, such as booking a flight or ordering food.
   - **Informational:** chatbots that provide information, such as answering questions or providing customer support.
   - The detected chatbot type is stored in the application state for use in subsequent processing steps.
5. **Functionality Analysis (LLM-based):** Depending on the chatbot's type a different prompt will be used, but in this section the LLM will receive conversations and functionalities and will try to merge functionalities that are the same, maybe find new ones, and find relationships between them. The output will be a structured representation of the discovered functionalities, including parent/child relationships and unique root nodes.
   - **Transactional:** The LLM will look for sequential dependencies, branching logic, and optional steps.
   - **Informational:** The LLM will look for independent topics and create separate root nodes for each topic.
6. **Profile Generation (LLM-based):** After the functionalities have been found and a workflow is created, the LLM will proceed to create the profiles for Sensei taking into account the discovered things. It is in done in different sections where different prompts will be creating the goals, context, parameters and so on.
7. **YAML Validation & Correction:** Validate generated YAML profiles with a script and if any error is found, pass it to the LLM to try to correct it.
8. **Output Generation:**

- Save validated YAML profiles to disk.
- Generate a text report (`report.txt`).
- Generate a visual workflow graph (`workflow_graph.png`) using Graphviz.

## 3. Workflow Graph Generation

One of the main outputs of this tool is a visual graph (`workflow_graph.png`) showing how users interact with the chatbot. Although, the primary focus of this tool is to make the profiles, this was added to help visualize the discovered functionalities and their relationships.

As it has been explained above, the system uses different approaches for transactional and informational chatbots.

**Example Desired Flow (Transactional - Pizza Bot):**

The goal is to capture flows like this: A user starts, sees menu items. This action leads to selecting a predefined pizza or customizing one. The user then orders drinks, and then the chatbot confirms the order.

```mermaid
graph LR
    Start((•)) --> F
    Start --> A
    Start --> E

    A[provide opening hours]
    E[provide price info]

    F[provide menu items] --> G[order predefined pizza
    Params: pizza_type];
    F --> H[order custom pizza
    Params: toppings, size];
    G --> I[order drinks
    Params: drink_type];
    H --> I;
    I --> D[provide order information];
```

**Example Desired Flow (Informational - Ada-UAM Bot):**

For an informational bot, the goal is to represent the different topics the user can inquire about independently. There are typically no required sequences between these topics. The structuring logic should default to creating separate root nodes.

```mermaid
graph LR
    Start((•)) --> A
    Start --> B
    Start --> C
    Start --> D
    Start --> E
    Start --> F
    Start --> G

    A[provide_contact_info];
    B[provide_opening_hours];
    C[explain_service_catalog];
    D[explain_ticketing_process];
    E[explain_wifi_setup];
    F[explain_software_access];
    G[handle_unclear_request];
```

> Note: The Mermaid diagrams above are illustrative of the desired logical flow. The actual implementation uses Graphviz.

## 4. Usage

### Installation

1. Ensure Python 3.11+ and Graphviz are installed.
2. Clone the repository:

    ```bash
    git clone https://github.com/Chatbot-TRACER/TRACER.git
    cd TRACER
    ```

3. Install the project:

    ```bash
    pip install .
    ```

4. Make sure to have the required environment variables set for OpenAI or Google Gemini models.

    ```bash
    export OPENAI_API_KEY=your_openai_api_key
    export GOOGLE_API_KEY=your_google_api_key
    ```

Execution:

```bash
TRACER --help
```

### Arguments

All arguments are optional.

- `-s, --sessions`: Number of exploration sessions (default: 3).
- `-n, --turns`: Maximum turns per session (default: 8).
- `-t, --technology`: Chatbot technology connector to use (default: `taskyto`). See available technologies below.
- `-u, --url`: Chatbot URL (default: `http://localhost:5000`). Only necessary for technologies like `taskyto` that require an explicit endpoint. Others may have the URL embedded in their connector.
- `-m, --model`: Model for analysis and generation (default: `gpt-4o-mini`). Supports both OpenAI models (e.g., `gpt-4o`) and Google Gemini models (e.g., `gemini-2.0-flash`).
- `-o, --output`: Output directory for generated files (default: `output`).
- `-v` or `-vv`: Verbosity level, none will show key information, `-v` will show the conversation and `-vv` will show be debug information.
- `-gfs`, `--graph-font-size`: Font size for the graph.
- `c`, `--compact`: Compact mode for the graph.
- `-td`, `--top-down`: Top-down layout for the graph.
- `--graph-format`: Export format for the graph (choices: pdf, png, svg, all). Default is pdf. Use 'all' to export in all formats.
- `-nf`, `--nested-forward`: All the variables will be nested, creates more exhaustive profiles but also the number of conversations grows exponentially.
- `-h, --help`: Show help message and exit.

### Supported Chatbot Technologies

- `taskyto`: Custom chatbot framework (requires self-hosting and initialization).
- `ada-uam`: MillionBot instance for Universidad Autónoma de Madrid (UAM).

### Environment Variables

- For OpenAI models: Set the `OPENAI_API_KEY` environment variable with your API key.
- For Gemini models: Set the `GOOGLE_API_KEY` environment variable with your API key from Google.

## 5. Input/Output

- **Input:**
  - Command-line arguments (see Usage).
  - Target chatbot accessible via its connector/URL.
- **Output (in the specified `--output` directory, organized by technology):**
  - Multiple `.yaml` files (one per generated user profile).
  - `report.txt` (structured text report summarizing findings).
  - `workflow_graph.png` (visual graph representation of the interaction flow).

### Example Commands

```bash
# Using OpenAI models
TRACER -t ada-uam -n 8 -s 12 -o generated_profiles/ada-uam -m gpt-4o-mini

# Using Gemini models
TRACER -t taskyto -n 10 -s 5 -o generated_profiles/taskyto -m gemini-2.0-flash
```

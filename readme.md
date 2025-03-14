# Chatbot Explorer

Project for my Master's Thesis.

A framework for automated exploration, analysis of conversational agents.

## Overview

Chatbot Explorer is a tool that automates the process of creating user profiles which will be later used to test the chatbot. It uses LLMs (Large Language Models) to interact with target chatbots, identify their features and limitations, and generate test profiles for systematic validation.

The system follows a three-phase approach:

1. **Exploration**: Automatically conducts conversations with the target chatbot to discover features
2. **Analysis**: Processes conversation data to identify functionalities and limitations
3. **Profile Generation**: Creates structured YAML profiles for systematic testing

## Architecture

The Lang Graph's structure is:

`[Entry] → explorer_node → analyzer_node → goal_generator_node → [Finish]`

- `explorer_node`: Conducts conversations with the target chatbot
- `analyzer_node`: Processes conversation data to extract features and limitations
- `goal_generator_node`: Creates user profiles and conversation goals

## Usage

```bash
python test.py [-h] [-s SESSIONS] [-n TURNS] [-t TECHNOLOGY] [-u URL] [-m MODEL] [-o OUTPUT]
```

### Arguments

All the arguments are optional.

- `-s, --sessions`: Number of exploration sessions (default: 3)
- `-n, --turns`: Maximum turns per session (default: 8)
- `-t, --technology`: Chatbot technology to use (default: taskyto)
- `-u, --url`: Chatbot URL to explore (default: <http://localhost:5000>, only necessary with Taskyto since the other technologies have the URL in their connector)
- `-m, --model`: OpenAI model to use (default: gpt-4o-mini)
- `-o, --output`: Output directory (default: output)

### Technology

- `taskyto`: Custom chatbot that must be self-hosted and initialized to a certain profile
- `ada-uam`: MillionBot chatbot for the UAM

### Output

The system generates:

- A text file with the discovered functionalities, limitations and language of the chatbot
- A list of YAML profiles ready to be used in the user-simulation

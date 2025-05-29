#!/bin/bash

# Tmux script for chatbot testing with taskyto and user-simulator
# Usage: ./tmux-chatbot-testing.sh [execution_number]
# Example: ./tmux-chatbot-testing.sh 1

EXECUTION=${1:-1}  # Default to execution_1 if no argument provided
SESSION_NAME="chatbot-testing-exec${EXECUTION}"

# Chatbot configurations
declare -A CHATBOTS=(
    ["bikeshop"]="5000"
    ["photography"]="5001"
    ["pizza-order"]="5002"
    ["veterinary"]="5003"
)

# Profile and connector mappings
# Note: These paths already use the correct TRACER directory names (e.g., pizzaorder)
declare -A PROFILES=(
    ["bikeshop"]="$HOME/miso/TRACER/results/bikeshop/execution_${EXECUTION}/profiles_logs/bikeshop_profile_${EXECUTION}"
    ["photography"]="$HOME/miso/TRACER/results/photography/execution_${EXECUTION}/profiles_logs/photography_profile_${EXECUTION}"
    ["pizza-order"]="$HOME/miso/TRACER/results/pizzaorder/execution_${EXECUTION}/profiles_logs/pizzaorder_profile_${EXECUTION}"
    ["veterinary"]="$HOME/miso/TRACER/results/veterinary/execution_${EXECUTION}/profiles_logs/veterinary_profile_${EXECUTION}"
)

declare -A CONNECTORS=(
    ["bikeshop"]="data/connectors/taskyto-5000.yml"
    ["photography"]="data/connectors/taskyto-5001.yml"
    ["pizza-order"]="data/connectors/taskyto-5002.yml"
    ["veterinary"]="data/connectors/taskyto-5003.yml"
)

# For user-simulator --project_path (uses 'bikeshop', 'pizzaorder')
declare -A PROJECTS=(
    ["bikeshop"]="examples/bikeshop"
    ["photography"]="examples/photography"
    ["pizza-order"]="examples/pizzaorder"
    ["veterinary"]="examples/veterinary"
)

# Mapping for taskyto --chatbot argument path segments (chatbot-llm specific)
declare -A TASKYTO_CHATBOT_YAML_PATHS=(
    ["bikeshop"]="bike-shop"
    ["photography"]="photography"
    ["pizza-order"]="pizza-order"
    ["veterinary"]="veterinary_center"
)

# ADDED: Mapping for TRACER directory names (consistent with PROFILES, PROJECTS values)
# This maps the logical chatbot keys to the directory names used in TRACER/results/
declare -A TRACER_DIR_NAMES=(
    ["bikeshop"]="bikeshop"
    ["photography"]="photography"
    ["pizza-order"]="pizzaorder" # Logical key "pizza-order" maps to "pizzaorder" dir
    ["veterinary"]="veterinary"
)

# MODIFIED: EXTRACTS array to point to the new TRACER results location
# This will be the value for sensei_chat.py --extract argument
declare -A EXTRACTS
for key in "${!CHATBOTS[@]}"; do
    tracer_name=${TRACER_DIR_NAMES[$key]}
    # Fallback, though TRACER_DIR_NAMES should cover all keys from CHATBOTS
    if [ -z "$tracer_name" ]; then tracer_name=$key; fi
    EXTRACTS["$key"]="$HOME/miso/TRACER/results/${tracer_name}/execution_${EXECUTION}/sensei_output"
done


# --- START OF CLEANING SECTION ---
echo "🧹 Cleaning up previous run data for execution_${EXECUTION}..."
for chatbot_key in "${!CHATBOTS[@]}"; do # Iterate using the logical chatbot keys
    tracer_dir_name=${TRACER_DIR_NAMES[$chatbot_key]}
    if [ -z "$tracer_dir_name" ]; then
        echo "  ⚠️ Warning: TRACER directory name not found for $chatbot_key. Skipping cleanup for its specific paths."
        continue
    fi

    # Path for taskyto profile logs (defined in PROFILES array, but we reconstruct for cleaning)
    # The actual profile *file* is named like .../profiles_logs/bikeshop_profile_1
    # The directory to clean is .../profiles_logs/
    profiles_logs_dir_base="$HOME/miso/TRACER/results/${tracer_dir_name}/execution_${EXECUTION}"
    profiles_logs_dir="${profiles_logs_dir_base}/profiles_logs"

    # Path for sensei_chat.py output (defined in EXTRACTS array)
    sensei_output_dir="${EXTRACTS[$chatbot_key]}" # This is already the full path to sensei_output

    # Path for profiles_coverage directory
    profiles_coverage_dir="${profiles_logs_dir_base}/profiles_coverage"

    echo "  - Cleaning taskyto profile logs in: $profiles_logs_dir"
    if [ -d "$profiles_logs_dir" ]; then
        rm -rf "${profiles_logs_dir:?}"/* # Clears content, keeps dir. :? for safety.
    else
        mkdir -p "$profiles_logs_dir" # Ensure it exists
    fi
    # If you want to remove and recreate the profiles_logs dir itself:
    # rm -rf "$profiles_logs_dir"
    # mkdir -p "$profiles_logs_dir"

    echo "  - Cleaning profiles coverage in: $profiles_coverage_dir"
    rm -rf "$profiles_coverage_dir" # Remove the directory entirely
    mkdir -p "$profiles_coverage_dir" # Recreate it empty

    echo "  - Cleaning sensei_chat.py output in: $sensei_output_dir"
    rm -rf "$sensei_output_dir" # Remove the directory entirely
    mkdir -p "$sensei_output_dir" # Recreate it empty
done
echo "✅ Cleanup complete."
# --- END OF CLEANING SECTION ---


# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME

echo "🚀 Setting up tmux session: $SESSION_NAME"
echo "📊 Testing execution_${EXECUTION} profiles"

# Window 1: Taskyto Servers
tmux rename-window -t $SESSION_NAME:0 "taskyto-servers"

# Create panes for each taskyto server (4 panes in window)
tmux split-window -h -t $SESSION_NAME:0
tmux split-window -v -t $SESSION_NAME:0.0
tmux split-window -v -t $SESSION_NAME:0.2

# Array to track pane assignments (uses logical chatbot keys)
PANES=("bikeshop" "photography" "pizza-order" "veterinary")

# Start taskyto servers in each pane
for i in "${!PANES[@]}"; do
    chatbot_key=${PANES[$i]} # e.g., "bikeshop", "pizza-order"
    port=${CHATBOTS[$chatbot_key]}
    profile_file_path=${PROFILES[$chatbot_key]} # Full path to the profile file
    taskyto_yaml_path_segment=${TASKYTO_CHATBOT_YAML_PATHS[$chatbot_key]}

    echo "🔧 Setting up taskyto server for $chatbot_key on port $port (using path: examples/yaml/$taskyto_yaml_path_segment)"

    tmux send-keys -t $SESSION_NAME:0.$i "cd $HOME/miso/chatbot-llm" C-m
    tmux send-keys -t $SESSION_NAME:0.$i "source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:0.$i "echo '🟢 Starting taskyto server for $chatbot_key on port $port'" C-m
    # taskyto-serve --profile expects the path to the profile *file*
    tmux send-keys -t $SESSION_NAME:0.$i "taskyto-serve --chatbot examples/yaml/$taskyto_yaml_path_segment --port $port --profile $profile_file_path" C-m
done

# Wait a moment for servers to start
sleep 2

# Window 2: User Simulators
tmux new-window -t $SESSION_NAME -n "user-simulators"

# Create panes for each user simulator (4 panes in window)
tmux split-window -h -t $SESSION_NAME:1
tmux split-window -v -t $SESSION_NAME:1.0
tmux split-window -v -t $SESSION_NAME:1.2

# Start user simulators in each pane
for i in "${!PANES[@]}"; do
    chatbot_key=${PANES[$i]}
    connector=${CONNECTORS[$chatbot_key]}
    project_path_segment=${PROJECTS[$chatbot_key]} # Relative path for --project_path, e.g., "examples/bikeshop"
    extract_path=${EXTRACTS[$chatbot_key]}         # Absolute path for --extract, e.g., "$HOME/miso/TRACER/results/bikeshop/execution_X/sensei_output"

    echo "🤖 Setting up user simulator for $chatbot_key"

    tmux send-keys -t $SESSION_NAME:1.$i "cd $HOME/miso/user-simulator" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "# Forcing pyenv initialization" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "export PYENV_ROOT=\"\$HOME/.pyenv\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "[[ -d \"\$PYENV_ROOT/bin\" ]] && export PATH=\"\$PYENV_ROOT/bin:\$PATH\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "eval \"\$(pyenv init - zsh)\"" C-m # Adjust 'zsh' if your tmux shell is different
    tmux send-keys -t $SESSION_NAME:1.$i "eval \"\$(pyenv virtualenv-init -)\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "pyenv activate user-sim" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "if [ -f .env ]; then echo 'Sourcing .env file from $HOME/miso/user-simulator'; set -a; source .env; set +a; else echo 'WARNING: .env file not found in $HOME/miso/user-simulator'; fi" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🔍 Python version:' && python --version" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🔍 Current directory:' && pwd" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🟡 Waiting for taskyto server...'" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "sleep 10" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🚀 Starting user simulator for $chatbot_key'" C-m
    # MODIFIED: Using project_path_segment for --project_path and extract_path for --extract
    tmux send-keys -t $SESSION_NAME:1.$i "python src/sensei_chat.py --technology taskyto --connector $connector --project_path $project_path_segment --user_profile execution_${EXECUTION} --extract \"$extract_path\" --verbose" C-m
done

# Window 3: Control Panel
tmux new-window -t $SESSION_NAME -n "control"
tmux send-keys -t $SESSION_NAME:2 "echo '🎛️  Control Panel - Execution ${EXECUTION}'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '📊 Current Status:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Taskyto servers: Window 0 (4 panes)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - User simulators: Window 1 (4 panes)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '🧹 Output directories (should be clean for this execution):'" C-m
for chatbot_key in "${!CHATBOTS[@]}"; do
    tracer_name=${TRACER_DIR_NAMES[$chatbot_key]:-$chatbot_key}
    tmux send-keys -t $SESSION_NAME:2 "echo \"  - $HOME/miso/TRACER/results/${tracer_name}/execution_${EXECUTION}/profiles_logs/\"" C-m
    tmux send-keys -t $SESSION_NAME:2 "echo \"  - $HOME/miso/TRACER/results/${tracer_name}/execution_${EXECUTION}/profile_coverage/\"" C-m
    tmux send-keys -t $SESSION_NAME:2 "echo \"  - ${EXTRACTS[$chatbot_key]}/\"" C-m
done
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '🔍 Monitoring commands:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Check server logs: Ctrl+s 0 (then navigate with Ctrl+s o)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Check simulator logs: Ctrl+s 1 (then navigate with Ctrl+s o)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Kill all: tmux kill-session -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '⚡ Quick restart script for next execution:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  ./tmux-chatbot-testing.sh $(($EXECUTION + 1))  # For next execution'" C-m


# Go to taskyto servers window
tmux select-window -t $SESSION_NAME:0

# Attach to session
echo ""
echo "✅ Session $SESSION_NAME created successfully!"
echo ""
echo "🔄 Starting order:"
echo "  1. Taskyto servers are starting..."
echo "  2. User simulators will connect after 10 second delay"
echo ""
echo "🎮 Controls:"
echo "  - Switch windows: Ctrl+s 0/1/2"
echo "  - Switch panes: Ctrl+s o"
echo "  - Kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "📊 To run different executions:"
echo "  - execution_2: ./tmux-chatbot-testing.sh 2"
echo "  - execution_3: ./tmux-chatbot-testing.sh 3"
echo ""
echo "🔗 Attaching to session..."

tmux attach-session -t $SESSION_NAME

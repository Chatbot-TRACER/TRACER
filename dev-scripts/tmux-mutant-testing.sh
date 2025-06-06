#!/bin/bash

# Tmux script for mutant testing with parallel execution
# Tests ONE mutant against 3 profile sets (execution_1, execution_2, execution_3) in parallel
# Usage: ./tmux-mutant-testing.sh <mutant_path>
# Example: ./tmux-mutant-testing.sh eval/mutants/bike-shop/ChangeRephrase_1

if [ $# -eq 0 ]; then
    echo "❌ Error: Mutant path required"
    echo "Usage: $0 <mutant_path>"
    echo "Examples:"
    echo "  $0 eval/mutants/bike-shop/ChangeRephrase_1"
    echo "  $0 /home/ivan/miso/papers/ICTSS.2025/eval/mutants/veterinary_center2/ChangeRephrase_1"
    echo "  $0 veterinary_center2/ChangeRephrase_1  # (searches in papers directory)"
    exit 1
fi

MUTANT_PATH="$1"

# Try to find the mutant path in multiple locations
MUTANT_FOUND=""
PAPERS_MUTANTS_DIR="$HOME/miso/papers/ICTSS.2025/eval/mutants"

# 1. Try the path as given (absolute or relative to current dir)
if [ -d "$MUTANT_PATH" ]; then
    MUTANT_FOUND="$MUTANT_PATH"
# 2. Try in the papers directory if it looks like a relative eval path
elif [[ "$MUTANT_PATH" == eval/mutants/* ]] && [ -d "$PAPERS_MUTANTS_DIR/${MUTANT_PATH#eval/mutants/}" ]; then
    MUTANT_FOUND="$PAPERS_MUTANTS_DIR/${MUTANT_PATH#eval/mutants/}"
    echo "🔍 Found mutant in papers directory: $MUTANT_FOUND"
# 3. Try in the papers directory if it looks like a domain/mutant path
elif [[ "$MUTANT_PATH" == */ChangeRephrase_* ]] || [[ "$MUTANT_PATH" == */Delete* ]] || [[ "$MUTANT_PATH" == */Swap* ]] || [[ "$MUTANT_PATH" =~ ^[^/]+/[^/]+$ ]]; then
    if [ -d "$PAPERS_MUTANTS_DIR/$MUTANT_PATH" ]; then
        MUTANT_FOUND="$PAPERS_MUTANTS_DIR/$MUTANT_PATH"
        echo "🔍 Found mutant in papers directory: $MUTANT_FOUND"
    fi
fi

# Validate mutant path exists
if [ -z "$MUTANT_FOUND" ]; then
    echo "❌ Error: Mutant path does not exist: $MUTANT_PATH"
    echo ""
    echo "Tried looking in:"
    echo "  - Current directory: $MUTANT_PATH"
    if [[ "$MUTANT_PATH" == eval/mutants/* ]]; then
        echo "  - Papers directory: $PAPERS_MUTANTS_DIR/${MUTANT_PATH#eval/mutants/}"
    fi
    echo "  - Papers directory: $PAPERS_MUTANTS_DIR/$MUTANT_PATH"
    echo ""
    echo "Available mutants:"
    if [ -d "$PAPERS_MUTANTS_DIR" ]; then
        for domain in "$PAPERS_MUTANTS_DIR"/*; do
            if [ -d "$domain" ]; then
                domain_name=$(basename "$domain")
                echo "  $domain_name/:"
                ls "$domain" | head -3 | sed 's/^/    /'
                total=$(ls "$domain" | wc -l)
                if [ $total -gt 3 ]; then
                    echo "    ... and $((total - 3)) more"
                fi
            fi
        done
    fi
    exit 1
fi

MUTANT_PATH="$MUTANT_FOUND"

# Parse mutant information
MUTANT_FULL_PATH=$(realpath "$MUTANT_PATH")
MUTANT_DOMAIN=$(basename $(dirname "$MUTANT_PATH"))
MUTANT_NAME=$(basename "$MUTANT_PATH")

echo "🧬 Mutant Testing Setup"
echo "   Mutant Path: $MUTANT_PATH"
echo "   Domain: $MUTANT_DOMAIN"
echo "   Mutant: $MUTANT_NAME"

# Session name based on mutant
SESSION_NAME="mutant-testing-${MUTANT_DOMAIN}-${MUTANT_NAME}"

# Domain name mappings (mutant dir → logical key → TRACER dir)
declare -A DOMAIN_MAPPINGS=(
    ["bike-shop"]="bikeshop"
    ["photography"]="photography"
    ["pizza_order"]="pizza-order"
    ["veterinary_center2"]="veterinary"
)

# TRACER directory names (logical key → TRACER results dir)
declare -A TRACER_DIR_NAMES=(
    ["bikeshop"]="bikeshop"
    ["photography"]="photography"
    ["pizza-order"]="pizzaorder"
    ["veterinary"]="veterinary"
)

# Project paths for user-simulator (logical key → examples path)
declare -A PROJECTS=(
    ["bikeshop"]="examples/bikeshop"
    ["photography"]="examples/photography"
    ["pizza-order"]="examples/pizzaorder"
    ["veterinary"]="examples/veterinary"
)

# Get logical key from mutant domain
LOGICAL_KEY=${DOMAIN_MAPPINGS[$MUTANT_DOMAIN]}
if [ -z "$LOGICAL_KEY" ]; then
    echo "❌ Error: Unknown mutant domain: $MUTANT_DOMAIN"
    echo "Supported domains: ${!DOMAIN_MAPPINGS[@]}"
    exit 1
fi

TRACER_DIR=${TRACER_DIR_NAMES[$LOGICAL_KEY]}
PROJECT_PATH=${PROJECTS[$LOGICAL_KEY]}

echo "   Logical Key: $LOGICAL_KEY"
echo "   TRACER Dir: $TRACER_DIR"
echo "   Project Path: $PROJECT_PATH"

# Port allocation for 3 parallel executions
declare -A PORTS=(
    ["execution_1"]="6000"
    ["execution_2"]="6001"
    ["execution_3"]="6002"
)

# Connector files for the 3 ports
declare -A CONNECTORS=(
    ["execution_1"]="data/connectors/taskyto-6000.yml"
    ["execution_2"]="data/connectors/taskyto-6001.yml"
    ["execution_3"]="data/connectors/taskyto-6002.yml"
)

# Profile file paths and output directories - MUTANT SPECIFIC
declare -A PROFILES
declare -A EXTRACTS
declare -A PROFILE_COVERAGE_DIRS

# Create mutant-specific directories to preserve results from different mutants
MUTANT_BASE_DIR="$HOME/miso/TRACER/results/${TRACER_DIR}/mutant_${MUTANT_NAME}"

for execution in "execution_1" "execution_2" "execution_3"; do
    execution_dir="${MUTANT_BASE_DIR}/${execution}"
    PROFILES[$execution]="${execution_dir}/profile_logs/${TRACER_DIR}_profile_${execution#execution_}"
    EXTRACTS[$execution]="${execution_dir}/sensei_output"
    PROFILE_COVERAGE_DIRS[$execution]="${execution_dir}/profile_coverage"
done

# --- START OF DIRECTORY CREATION SECTION ---
echo ""
echo "📁 Creating mutant-specific directories for ${MUTANT_DOMAIN}/${MUTANT_NAME}..."
echo "   Base directory: $MUTANT_BASE_DIR"

for execution in "execution_1" "execution_2" "execution_3"; do
    profile_logs_dir=$(dirname "${PROFILES[$execution]}")
    sensei_output_dir="${EXTRACTS[$execution]}"
    profile_coverage_dir="${PROFILE_COVERAGE_DIRS[$execution]}"

    echo "  - Setting up $execution directories:"
    echo "    Profile logs: $profile_logs_dir"
    mkdir -p "$profile_logs_dir"

    echo "    Sensei output: $sensei_output_dir"
    mkdir -p "$sensei_output_dir"

    echo "    Profile coverage: $profile_coverage_dir"
    mkdir -p "$profile_coverage_dir"
done

echo "✅ Mutant-specific directories created."
echo "💾 Results will be preserved separately for each mutant!"
# --- END OF DIRECTORY CREATION SECTION ---

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME

echo ""
echo "🚀 Setting up tmux session: $SESSION_NAME"
echo "🧬 Testing mutant: $MUTANT_DOMAIN/$MUTANT_NAME against 3 profile executions"

# Window 1: Taskyto Servers (3 panes for 3 executions)
tmux rename-window -t $SESSION_NAME:0 "taskyto-mutant"

# Create 3 panes for the 3 executions
tmux split-window -h -t $SESSION_NAME:0
tmux split-window -v -t $SESSION_NAME:0.0

# Array for execution order
EXECUTIONS=("execution_1" "execution_2" "execution_3")

# Start taskyto servers in each pane
for i in "${!EXECUTIONS[@]}"; do
    execution=${EXECUTIONS[$i]}
    port=${PORTS[$execution]}
    profile_file_path=${PROFILES[$execution]}

    echo "🔧 Setting up taskyto server for $execution on port $port"

    tmux send-keys -t $SESSION_NAME:0.$i "cd $HOME/miso/chatbot-llm" C-m
    tmux send-keys -t $SESSION_NAME:0.$i "source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:0.$i "echo '🟢 Starting taskyto server for mutant $MUTANT_NAME ($execution) on port $port'" C-m
    tmux send-keys -t $SESSION_NAME:0.$i "taskyto-serve --chatbot $MUTANT_FULL_PATH --port $port --profile $profile_file_path" C-m
done

# Wait for servers to start
sleep 2

# Window 2: User Simulators (3 panes for 3 executions)
tmux new-window -t $SESSION_NAME -n "sensei-mutant"

# Create 3 panes for the 3 simulators
tmux split-window -h -t $SESSION_NAME:1
tmux split-window -v -t $SESSION_NAME:1.0

# Start user simulators in each pane
for i in "${!EXECUTIONS[@]}"; do
    execution=${EXECUTIONS[$i]}
    connector=${CONNECTORS[$execution]}
    extract_path=${EXTRACTS[$execution]}
    execution_number=${execution#execution_}

    echo "🤖 Setting up user simulator for $execution"

    tmux send-keys -t $SESSION_NAME:1.$i "cd $HOME/miso/user-simulator" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "# Forcing pyenv initialization for $execution" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "export PYENV_ROOT=\"\$HOME/.pyenv\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "[[ -d \"\$PYENV_ROOT/bin\" ]] && export PATH=\"\$PYENV_ROOT/bin:\$PATH\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "eval \"\$(pyenv init - zsh)\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "eval \"\$(pyenv virtualenv-init -)\"" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "pyenv activate user-sim" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "if [ -f .env ]; then echo 'Sourcing .env file from $HOME/miso/user-simulator'; set -a; source .env; set +a; else echo 'WARNING: .env file not found in $HOME/miso/user-simulator'; fi" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🔍 Python version:' && python --version" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🔍 Current directory:' && pwd" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🟡 Waiting for taskyto server ($execution)...'" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "sleep 10" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "echo '🚀 Starting user simulator for mutant $MUTANT_NAME ($execution)'" C-m
    tmux send-keys -t $SESSION_NAME:1.$i "python src/sensei_chat.py --technology taskyto --connector $connector --project_path $PROJECT_PATH --user_profile $execution --extract \"$extract_path\" --verbose" C-m
done

# Window 3: Control Panel
tmux new-window -t $SESSION_NAME -n "control"
tmux send-keys -t $SESSION_NAME:2 "echo '🎛️  Mutant Testing Control Panel'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '🧬 Mutant Information:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo \"  - Domain: $MUTANT_DOMAIN\"" C-m
tmux send-keys -t $SESSION_NAME:2 "echo \"  - Mutant: $MUTANT_NAME\"" C-m
tmux send-keys -t $SESSION_NAME:2 "echo \"  - Path: $MUTANT_PATH\"" C-m
tmux send-keys -t $SESSION_NAME:2 "echo \"  - Logical Key: $LOGICAL_KEY\"" C-m
tmux send-keys -t $SESSION_NAME:2 "echo \"  - TRACER Dir: $TRACER_DIR\"" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '📊 Current Testing Status:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Taskyto servers: Window 0 (3 panes - execution_1/2/3)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - User simulators: Window 1 (3 panes - execution_1/2/3)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '🔌 Port Allocation:'" C-m
for execution in "execution_1" "execution_2" "execution_3"; do
    port=${PORTS[$execution]}
    tmux send-keys -t $SESSION_NAME:2 "echo \"  - $execution: Port $port\"" C-m
done
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '📁 Output Directories:'" C-m
for execution in "execution_1" "execution_2" "execution_3"; do
    tmux send-keys -t $SESSION_NAME:2 "echo \"  - $execution:\"" C-m
    tmux send-keys -t $SESSION_NAME:2 "echo \"    Profile logs: $(dirname "${PROFILES[$execution]}")\"" C-m
    tmux send-keys -t $SESSION_NAME:2 "echo \"    Sensei output: ${EXTRACTS[$execution]}\"" C-m
    tmux send-keys -t $SESSION_NAME:2 "echo \"    Profile coverage: ${PROFILE_COVERAGE_DIRS[$execution]}\"" C-m
done
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '🔍 Monitoring Commands:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Check server logs: Ctrl+s 0 (then navigate with Ctrl+s o)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Check simulator logs: Ctrl+s 1 (then navigate with Ctrl+s o)'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Kill all: tmux kill-session -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '🔄 Test Next Mutant:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  ./tmux-mutant-testing.sh eval/mutants/$MUTANT_DOMAIN/<next_mutant_name>'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '📈 Progress Monitoring:'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Watch output files grow in real-time'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '  - Check process completion in simulator panes'" C-m

# Go to taskyto servers window
tmux select-window -t $SESSION_NAME:0

# Display final information and attach
echo ""
echo "✅ Session $SESSION_NAME created successfully!"
echo ""
echo "🧬 Testing Configuration:"
echo "  - Mutant: $MUTANT_DOMAIN/$MUTANT_NAME"
echo "  - Parallel Executions: execution_1, execution_2, execution_3"
echo "  - Ports: 6000, 6001, 6002"
echo ""
echo "🔄 Starting Order:"
echo "  1. Taskyto servers starting for mutant $MUTANT_NAME..."
echo "  2. User simulators will connect after 10 second delay"
echo "  3. All 3 profile executions running in parallel"
echo ""
echo "🎮 Controls:"
echo "  - Switch windows: Ctrl+s 0/1/2"
echo "  - Switch panes: Ctrl+s o"
echo "  - Kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "🧬 To test other mutants:"
echo "  - Same domain: ./tmux-mutant-testing.sh eval/mutants/$MUTANT_DOMAIN/<other_mutant>"
echo "  - Other domains: ./tmux-mutant-testing.sh eval/mutants/<domain>/<mutant>"
echo ""
echo "🔗 Attaching to session..."

tmux attach-session -t $SESSION_NAME

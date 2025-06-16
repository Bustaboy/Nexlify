#!/bin/bash
# Enhanced Nexlify Startup Script for Linux/macOS
# Addresses all V3 improvements with graceful shutdown

# Enable strict error handling
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# PID file for tracking processes
PID_FILE="/tmp/nexlify_pids.txt"
STOP_FILE="EMERGENCY_STOP_ACTIVE"

# Trap for cleanup on exit
trap cleanup EXIT INT TERM

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Cleanup function
cleanup() {
    print_color "$YELLOW" "\n🛑 Initiating graceful shutdown..."
    
    # Create emergency stop file
    touch "$STOP_FILE"
    
    # Read PIDs and terminate processes
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            if [ ! -z "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                print_color "$CYAN" "Stopping process $pid..."
                kill -TERM "$pid" 2>/dev/null || true
                
                # Wait for graceful shutdown
                local count=0
                while kill -0 "$pid" 2>/dev/null && [ $count -lt 5 ]; do
                    sleep 1
                    ((count++))
                done
                
                # Force kill if needed
                if kill -0 "$pid" 2>/dev/null; then
                    print_color "$YELLOW" "Force stopping process $pid"
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done < "$PID_FILE"
        
        rm -f "$PID_FILE"
    fi
    
    # Remove emergency stop file
    rm -f "$STOP_FILE"
    
    print_color "$GREEN" "✓ Shutdown complete"
    print_color "$CYAN" "Logs saved to: logs/startup/"
}

# Banner
print_banner() {
    print_color "$CYAN" "╔══════════════════════════════════════════╗"
    print_color "$CYAN" "║       NEXLIFY TRADING PLATFORM           ║"
    print_color "$CYAN" "║         Night City Trader                ║"
    print_color "$CYAN" "║            v2.0.8                        ║"
    print_color "$CYAN" "╚══════════════════════════════════════════╝"
    echo
}

# Check Python version
check_python() {
    print_color "$CYAN" "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_color "$RED" "ERROR: Python 3 is not installed"
        print_color "$YELLOW" "Please install Python 3.11 or higher"
        exit 1
    fi
    
    # Get Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    # Check minimum version (3.11)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        print_color "$RED" "ERROR: Python 3.11 or higher required, found Python $PYTHON_VERSION"
        exit 1
    fi
    
    print_color "$GREEN" "✓ Python $PYTHON_VERSION detected"
}

# Check system requirements
check_system() {
    print_color "$CYAN" "Checking system requirements..."
    
    # Check available memory
    if command -v free &> /dev/null; then
        AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
        if [ "$AVAILABLE_MEM" -lt 2048 ]; then
            print_color "$YELLOW" "⚠ Warning: Low available memory (${AVAILABLE_MEM}MB)"
        fi
    fi
    
    # Check disk space
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_DISK" -lt 2 ]; then
        print_color "$YELLOW" "⚠ Warning: Low disk space (${AVAILABLE_DISK}GB)"
    fi
    
    # Check for display (for GUI)
    if [ -z "${DISPLAY:-}" ] && [ "$(uname)" != "Darwin" ]; then
        print_color "$YELLOW" "⚠ Warning: No DISPLAY variable set, GUI may not work"
        print_color "$YELLOW" "  For SSH: Use 'ssh -X' or set up X11 forwarding"
    fi
}

# Create log directories
setup_logs() {
    mkdir -p logs/startup
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="logs/startup"
    LAUNCHER_LOG="$LOG_DIR/launcher_${TIMESTAMP}.log"
    NEURAL_LOG="$LOG_DIR/neural_net_${TIMESTAMP}.log"
    GUI_LOG="$LOG_DIR/gui_${TIMESTAMP}.log"
}

# Wait for port to be ready
wait_for_port() {
    local port=$1
    local timeout=${2:-30}
    local elapsed=0
    
    print_color "$CYAN" "Waiting for API on port $port..."
    
    while [ $elapsed -lt $timeout ]; do
        if nc -z localhost $port 2>/dev/null; then
            print_color "$GREEN" "✓ API is ready"
            return 0
        fi
        
        sleep 1
        ((elapsed++))
        echo -ne "  Waiting... ${elapsed}s\r"
    done
    
    print_color "$YELLOW" "⚠ API took too long to start, continuing anyway"
    return 0
}

# Start component with logging
start_component() {
    local name=$1
    local script=$2
    local log_file=$3
    
    print_color "$CYAN" "Starting $name..."
    
    if [ ! -f "$script" ]; then
        print_color "$RED" "ERROR: $script not found"
        return 1
    fi
    
    # Start in background with output redirection
    nohup python3 -u "$script" > "$log_file" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "$pid" >> "$PID_FILE"
    
    print_color "$GREEN" "✓ $name started (PID: $pid)"
    return 0
}

# Monitor processes
monitor_processes() {
    print_color "$GREEN" "\n✓ Nexlify is running!"
    echo
    print_color "$GREEN" "Access Points:"
    print_color "$CYAN" "  • Trading API: http://localhost:8000"
    print_color "$CYAN" "  • GUI: Running in separate window"
    echo
    print_color "$YELLOW" "Commands:"
    print_color "$CYAN" "  • Press Ctrl+C for graceful shutdown"
    print_color "$CYAN" "  • Type 'stop' to shutdown"
    print_color "$CYAN" "  • Type 'status' to check components"
    print_color "$CYAN" "  • Type 'logs' to show log locations"
    echo
    
    # Monitor loop
    while true; do
        read -t 1 -n 100 user_input || true
        
        case "${user_input,,}" in
            stop|exit|quit)
                break
                ;;
            status)
                show_status
                ;;
            logs)
                show_logs
                ;;
            help)
                show_help
                ;;
        esac
        
        # Check if processes are still running
        if [ -f "$PID_FILE" ]; then
            while IFS= read -r pid; do
                if [ ! -z "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    print_color "$YELLOW" "⚠ Process $pid stopped unexpectedly"
                fi
            done < "$PID_FILE"
        fi
    done
}

# Show status
show_status() {
    echo
    print_color "$CYAN" "Component Status:"
    
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            if [ ! -z "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                # Get process info
                if command -v ps &> /dev/null; then
                    local info=$(ps -p "$pid" -o comm= 2>/dev/null || echo "Unknown")
                    print_color "$GREEN" "  • Process $pid: Running ($info)"
                else
                    print_color "$GREEN" "  • Process $pid: Running"
                fi
            else
                print_color "$RED" "  • Process $pid: Stopped"
            fi
        done < "$PID_FILE"
    fi
    
    # Check API port
    if nc -z localhost 8000 2>/dev/null; then
        print_color "$GREEN" "  • API Port 8000: Listening"
    else
        print_color "$RED" "  • API Port 8000: Not listening"
    fi
    echo
}

# Show logs
show_logs() {
    echo
    print_color "$CYAN" "Log Files:"
    ls -lh "$LOG_DIR"/*_${TIMESTAMP}.log 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'
    print_color "$CYAN" "\nLog directory: $LOG_DIR"
    echo
}

# Show help
show_help() {
    echo
    print_color "$CYAN" "Available Commands:"
    print_color "$YELLOW" "  • stop/exit/quit - Graceful shutdown"
    print_color "$YELLOW" "  • status - Show component status"
    print_color "$YELLOW" "  • logs - Show log file locations"
    print_color "$YELLOW" "  • help - Show this help message"
    echo
}

# Main execution
main() {
    print_banner
    
    # Checks
    check_python
    check_system
    setup_logs
    
    # Clear PID file
    > "$PID_FILE"
    
    # Remove any existing stop file
    rm -f "$STOP_FILE"
    
    # Check for smart_launcher.py
    if [ -f "smart_launcher.py" ]; then
        print_color "$CYAN" "Using smart_launcher.py for integrated startup..."
        start_component "Launcher" "smart_launcher.py" "$LAUNCHER_LOG"
        wait_for_port 8000
    else
        # Direct component startup
        print_color "$CYAN" "Starting components directly..."
        
        # Start Neural Net
        if [ -f "src/nexlify_neural_net.py" ]; then
            start_component "Neural Net" "src/nexlify_neural_net.py" "$NEURAL_LOG"
        elif [ -f "nexlify_neural_net.py" ]; then
            start_component "Neural Net" "nexlify_neural_net.py" "$NEURAL_LOG"
        else
            print_color "$RED" "ERROR: Neural net script not found"
            exit 1
        fi
        
        # Wait for API
        wait_for_port 8000
        
        # Start GUI
        if [ -f "cyber_gui.py" ]; then
            start_component "GUI" "cyber_gui.py" "$GUI_LOG"
        else
            print_color "$RED" "ERROR: GUI script not found"
            exit 1
        fi
    fi
    
    # Monitor
    monitor_processes
}

# Run main function
main

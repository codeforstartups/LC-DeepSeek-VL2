#!/usr/bin/env bash
set -euo pipefail

echo "🛑 Stopping all screen sessions..."

# Function to get active screen sessions
get_sessions() {
    screen -ls 2>/dev/null | grep -E "^\s*[0-9]+\." | awk '{print $1}' || true
}

# 1️⃣ Get all running screen sessions
sessions=$(get_sessions)

if [[ -z "$sessions" ]]; then
    echo "⚠️  No running screen sessions found."
    echo "🐳 Proceeding to Docker cleanup..."
else

echo "📋 Found screen sessions:"
echo "$sessions"
echo ""

# 2️⃣ Try graceful shutdown first
echo "🔄 Attempting graceful shutdown..."
for sess in $sessions; do
    echo "⏹ Gracefully stopping screen: $sess"

    # Send Ctrl-C to interrupt any running process
    screen -S "$sess" -X stuff $'\003' 2>/dev/null || true
    sleep 1

    # Send exit command
    screen -S "$sess" -X stuff "exit\n" 2>/dev/null || true
    sleep 1

    # Try to quit the screen session
    screen -S "$sess" -X quit 2>/dev/null || true
    sleep 0.5
done

# 3️⃣ Wait a bit and check what's still running
echo "⏳ Waiting for graceful shutdown..."
sleep 3

remaining_sessions=$(get_sessions)

if [[ -z "$remaining_sessions" ]]; then
    echo "✅ All screen sessions stopped gracefully!"
else

# 4️⃣ Force kill remaining sessions
echo "⚠️  Some sessions still running. Force terminating..."
echo "📋 Remaining sessions:"
echo "$remaining_sessions"
echo ""

for sess in $remaining_sessions; do
    echo "💥 Force killing screen: $sess"

    # Extract just the session ID (before the dot)
    session_id=$(echo "$sess" | cut -d'.' -f1)

    # Try multiple force termination methods
    screen -S "$sess" -X kill 2>/dev/null || true
    sleep 0.5

    # If screen session still exists, kill the process directly
    if screen -list 2>/dev/null | grep -q "$sess"; then
        echo "🔨 Killing process directly for session: $sess"
        kill -9 "$session_id" 2>/dev/null || true
    fi
done

# 5️⃣ Final check
echo "⏳ Final verification..."
sleep 2

final_sessions=$(get_sessions)

if [[ -z "$final_sessions" ]]; then
    echo "✅ All screen sessions have been terminated!"
else
    echo "❌ Some sessions may still be running:"
    echo "$final_sessions"
    echo ""
    echo "💡 You may need to manually kill these processes:"
    for sess in $final_sessions; do
        session_id=$(echo "$sess" | cut -d'.' -f1)
        echo "   kill -9 $session_id"
    done
    echo "⚠️  Continuing to Docker cleanup despite remaining sessions..."
fi

echo "🎉 Screen session cleanup completed!"
fi
fi
fi
echo ""
echo "🐳 Shutting down Docker services..."
docker compose -f ollama-compose.yml down

# Stop MindsDB container if it's running
if docker ps -q -f name=mindsdb | grep -q .; then
    echo "🧠 Stopping MindsDB container..."
    docker stop mindsdb
    echo "✅ MindsDB container stopped."
else
    echo "ℹ️  MindsDB container not running."
fi

echo "✅ Docker services stopped."

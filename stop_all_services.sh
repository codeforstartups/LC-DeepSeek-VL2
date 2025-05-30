#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Grab every "id.name" from screen -ls (skipping the header/footer lines)
sessions=$(screen -ls \
  | sed -n '2,$p' \
  | head -n -1 \
  | awk '{print $1}')

if [[ -z "$sessions" ]]; then
  echo "⚠️  No running screen sessions found."
  exit 0
fi

# 2️⃣ Loop through each session
for sess in $sessions; do
  echo "⏹ Stopping screen: $sess"

  # send Ctrl-C
  screen -S "$sess" -X stuff $'\003'
  sleep 1

  # then send 'exit' to close the shell
  screen -S "$sess" -X stuff "exit\n"
  sleep 0.5
done

echo "✅ All screen sessions have been signaled to stop."

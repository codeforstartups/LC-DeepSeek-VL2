#!/bin/sh
set -e

# Start Ollama server
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 10


if ! ollama list | grep -q 'deepseek-r1:1.5b'; then
  echo 'Pulling deepseek-r1:1.5b model...'
  ollama pull deepseek-r1:1.5b
else
  echo 'deepseek-r1:1.5b model already exists'
fi

echo "Ollama setup complete!"

# Keep the container running
wait $OLLAMA_PID

#!/bin/bash
set -e  # exit immediately if a command fails

echo "=== Step 1: Pulling Ollama models ==="
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text:v1.5

echo "=== Step 2: Installing Python requirements ==="
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found in current directory."
    exit 1
fi

echo "✅ Setup complete!"

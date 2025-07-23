#!/bin/bash

echo "Starting F5-TTS-THAI API Server..."

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found!"
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

if [ ! -f "api.py" ]; then
    echo "Error: api.py not found!"
    exit 1
fi

echo "Checking dependencies..."
python -c "import whisper, librosa, fastapi, uvicorn" 2>/dev/null || {
    echo "Installing missing dependencies..."
    pip install openai-whisper librosa fastapi uvicorn python-multipart
}

echo "Starting API server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python api.py
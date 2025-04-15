#!/bin/bash

set -e

CACHE_DIR="/cache"
MODEL_ID="nyrahealth/CrisperWhisper"
RUNPOD_SECRET_HF_TOKEN=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

# Check for Hugging Face token
if [ -z "$RUNPOD_SECRET_HF_TOKEN" ]; then
    echo "ERROR: RUNPOD_SECRET_HF_TOKEN environment variable not set!"
    echo "The CrisperWhisper model requires authentication. Please set this variable."
    exit 1
fi

echo "HF Token available with length: ${#RUNPOD_SECRET_HF_TOKEN}"

# Create cache directory
mkdir -p $CACHE_DIR

# Use Python and transformers to download the model
# This is more reliable than direct downloads for complex models
echo "Downloading CrisperWhisper model to cache..."
python3 -c "
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Get token from environment
hf_token = os.environ.get('RUNPOD_SECRET_HF_TOKEN')
cache_dir = '$CACHE_DIR'
model_id = '$MODEL_ID'

print(f'Downloading model {model_id} to {cache_dir}')

# Download model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    token=hf_token,
    local_files_only=False,
    use_safetensors=True
)

# Download processor
processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    token=hf_token,
    local_files_only=False
)

print('Model and processor downloaded successfully!')
"

if [ $? -eq 0 ]; then
    echo "CrisperWhisper model downloaded successfully to $CACHE_DIR"
else
    echo "Failed to download CrisperWhisper model"
    exit 1
fi
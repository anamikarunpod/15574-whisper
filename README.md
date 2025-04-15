[![RunPod](https://api.runpod.io/badge/kodxana/whisperx-worker)](https://www.runpod.io/console/hub/kodxana/whisperx-worker)

# CrisperWhisper Worker for RunPod

A serverless worker that provides high-quality speech transcription using CrisperWhisper on the RunPod platform.

## Features

- Automatic speech transcription with CrisperWhisper
- Support for different languages
- RunPod serverless compatibility
- GPU acceleration

## Input Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_file` | string | Yes | N/A | URL to the audio file for transcription |
| `language` | string | No | `null` | ISO code of the language spoken in the audio (e.g., 'en', 'fr') |
| `batch_size` | int | No | `16` | Batch size for processing |
| `temperature` | float | No | `0` | Temperature to use for sampling (higher = more random) |
| `debug` | bool | No | `false` | Whether to print debug information |

## Usage Example

```json
{
  "input": {
    "audio_file": "https://example.com/audio/sample.mp3",
    "language": "en",
    "debug": true
  }
}
```

## Output Format

The service returns a JSON object structured as follows:

```json
{
  "transcription": "The full transcribed text from the audio file.",
  "detected_language": "en"
}
```

## Secrets Setup

This worker requires a Hugging Face token to download the CrisperWhisper model. Add your token as a RunPod secret named `HF_TOKEN`, which will be available in the container as `RUNPOD_SECRET_HF_TOKEN`.

## Testing Locally

To test this worker locally:

1. Install Docker
2. Build the Docker image:
   ```
   docker build -t crisperwhisper-worker .
   ```
3. Run the container:
   ```
   docker run -e RUNPOD_SECRET_HF_TOKEN=your_hf_token -p 8000:8000 crisperwhisper-worker
   ```

## RunPod Deployment

1. Create a new Serverless template
2. Connect your GitHub repository
3. Add your Hugging Face token as a secret
4. Deploy the template

## License

This project is licensed under the Apache License, Version 2.0.

## Acknowledgments

- This project utilizes code from [WhisperX](https://github.com/m-bain/whisperX), licensed under the BSD-2-Clause license
- Special thanks to the RunPod team for the serverless platform

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

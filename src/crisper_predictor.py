from cog import BasePredictor, Input, Path, BaseModel
from pydub import AudioSegment
from typing import Any
import torch
import os
import gc
import time
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
compute_type = "float16"  # change to "int8" if low on GPU mem
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "nyrahealth/CrisperWhisper"

class Output(BaseModel):
    transcription: str
    detected_language: str = None  # CrisperWhisper might not return language detection

class CrisperPredictor(BasePredictor):
    def setup(self):
        print("Loading CrisperWhisper model...")
        hf_token = os.environ.get("RUNPOD_SECRET_HF_TOKEN")
        
        if not hf_token:
            print("WARNING: RUNPOD_SECRET_HF_TOKEN not set. Model download may fail if the model is private.")
        else:
            print(f"HF token available with length: {len(hf_token)}")
            
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=hf_token,
            cache_dir="/cache"  # Using RunPod's cache dir
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=hf_token,
            cache_dir="/cache"
        )
        print("CrisperWhisper model loaded successfully")

    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: str = Input(
                description="ISO code of the language spoken in the audio (e.g., 'en')",
                default=None),
            batch_size: int = Input(
                description="Batch size for parallelized processing",
                default=16),
            temperature: float = Input(
                description="Temperature for sampling",
                default=0),
            debug: bool = Input(
                description="Print debug information",
                default=False)
    ) -> Output:
        with torch.inference_mode():
            start_time = time.time_ns() / 1e6

            # Load audio file
            if debug:
                print(f"Loading audio file: {audio_file}")
            
            # Load audio using the processor
            # The processor may have its own audio loading functionality
            audio = self._load_audio(audio_file)
            
            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Audio loaded in {elapsed_time:.2f} ms")
                start_time = time.time_ns() / 1e6
            
            # Process audio with the processor
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            
            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Audio processed in {elapsed_time:.2f} ms")
                start_time = time.time_ns() / 1e6
            
            # Prepare generation config
            generation_config = {
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "max_length": 448,  # Adjust as needed
            }
            
            # Add language forcing if specified
            if language:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
                generation_config["forced_decoder_ids"] = forced_decoder_ids
            
            # Generate transcription
            predicted_ids = self.model.generate(
                input_features,
                **generation_config
            )
            
            # Decode the predicted ids
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Transcription generated in {elapsed_time:.2f} ms")
                print(f"Transcription: {transcription}")
            
            # Clean up to free memory
            gc.collect()
            torch.cuda.empty_cache()
            
            return Output(
                transcription=transcription,
                detected_language=language  # Just returning the input language as CrisperWhisper may not detect language
            )
    
    def _load_audio(self, file_path):
        """Load audio file and convert to appropriate format"""
        # Convert audio to mono WAV at 16kHz
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set to 16kHz
        
        # Export to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            audio.export(temp_wav.name, format='wav')
            temp_file_path = temp_wav.name
        
        # Load audio as numpy array - processor expects raw audio data
        import numpy as np
        import wave
        with wave.open(temp_file_path, 'rb') as wav_file:
            # Read wave file parameters
            n_channels, sampwidth, framerate, n_frames, _, _ = wav_file.getparams()
            # Read frames as bytes
            frames = wav_file.readframes(n_frames)
            # Convert bytes to numpy array
            audio_as_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Remove temporary file
        os.unlink(temp_file_path)
        
        return audio_as_np 
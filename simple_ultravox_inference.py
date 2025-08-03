#!/usr/bin/env python3

"""
Simple Ultravox inference script for local GPU inference.
This script shows how to run Ultravox on your local NVIDIA GPU for speech-to-text.
"""

import librosa
import numpy as np
from ultravox.inference.ultravox_infer import UltravoxInference


def main():
    print("Loading Ultravox model...")
    
    # Use the 1B model which should fit on your RTX 3060 12GB and hopefully not be gated
    model_path = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
    
    try:
        # You can adjust these parameters based on your hardware
        inference = UltravoxInference(
            model_path=model_path,
            device="cuda",  # Use GPU
            data_type="bfloat16",  # Use bfloat16 to save memory
            conversation_mode=False,  # Set to True if you want conversation history
        )
        
        print(f"Model loaded successfully on device: {inference.model.device}")
        
        # Example: Process text-only (no audio)
        print("\n=== Text-only example ===")
        text_response = inference.infer("What is the capital of France?")
        print(f"Response: {text_response}")
        
        # Example: Process audio file (you would replace this with your own audio file)
        print("\n=== To use with audio ===")
        print("To process audio, you would:")
        print("1. Load an audio file using librosa:")
        print("   audio, sr = librosa.load('your_audio_file.wav', sr=16000)")
        print("2. Then call:")
        print("   response = inference.infer('Transcribe this audio', audio)")
        
        print("\nModel is ready for inference!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nThis might be due to gated model access. You may need to:")
        print("1. Sign up for a HuggingFace account")
        print("2. Request access to the underlying model (e.g., Llama)")
        print("3. Login using: huggingface-cli login")


if __name__ == "__main__":
    main()

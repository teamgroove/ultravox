import os
import tempfile
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa
import numpy as np
from ultravox.inference.ultravox_infer import UltravoxInference

def predict(audio_file_path: str, question: str = "Transcribe this audio."):
    """
    Process audio file with Ultravox model
    """
    try:
        # Initialize Ultravox inference
        model_path = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
        
        inference = UltravoxInference(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            data_type="bfloat16"
        )
        
        # Load and process audio
        audio, sr = librosa.load(audio_file_path, sr=16000)
        
        # Run inference
        response = inference.infer(question, audio)
        
        return {
            "transcription": response,
            "model_used": model_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "model_used": model_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

def main():
    """Test the model with a simple text prompt"""
    print("Testing Ultravox model...")
    
    # First test without audio - just text
    try:
        model_path = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
        
        inference = UltravoxInference(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            data_type="bfloat16"
        )
        
        print(f"Model loaded on: {inference.model.device}")
        
        # Try a simple text-only inference
        print("\nTesting text-only inference...")
        
        # Create a simple test
        result = predict("test_audio.wav", "What is 2+2?")  # This will fail gracefully
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Test Ultravox v0.6 models using the ultravox library directly
"""

import os
import torch
from ultravox.inference.ultravox_infer import UltravoxInference

def test_v06_models():
    print("Testing Ultravox v0.6 models with authentication...")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set token
    hf_token = os.getenv("HF_TOKEN")
    print(f"HF Token set: {'Yes' if hf_token else 'No'}")
    
    # Try different v0.6 models, starting with smallest
    models_to_try = [
        "fixie-ai/ultravox-v0_6-gemma-3-27b",  # Try this first
        # These are too big for RTX 3060:
        # "fixie-ai/ultravox-v0_6-qwen-3-32b",   
        # "fixie-ai/ultravox-v0_6-llama-3_3-70b"
    ]
    
    for model_path in models_to_try:
        print(f"\n=== Trying {model_path} ===")
        
        try:
            # Initialize with lower precision and optimizations for RTX 3060
            inference = UltravoxInference(
                model_path=model_path,
                device="cuda",
                data_type="bfloat16",  # Use bfloat16 to save memory
                conversation_mode=False
            )
            
            print(f"Model loaded successfully on: {inference.model.device}")
            
            # Test text-only inference
            print("\n=== Testing text-only inference ===")
            try:
                response = inference.infer("What is 2 + 2?")
                print(f"Response: {response}")
                
                print("\nModel is working! You can now use it for speech inference.")
                print("Example usage:")
                print("import librosa")
                print("audio, sr = librosa.load('your_audio.wav', sr=16000)")
                print("response = inference.infer('Transcribe this audio', audio)")
                
                # Success, stop trying other models
                break
                
            except Exception as text_error:
                print(f"Text inference error: {text_error}")
                
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print("RTX 3060 doesn't have enough VRAM for this model size.")
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                print("Authentication issue - check HF token permissions.")
            else:
                print("Other error occurred.")
    
    print("\nIf all models failed due to memory, you might need to:")
    print("1. Use CPU inference (slower but works)")
    print("2. Try smaller v0.5 models")
    print("3. Use model sharding/offloading techniques")

if __name__ == "__main__":
    test_v06_models()

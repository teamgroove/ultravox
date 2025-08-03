#!/usr/bin/env python3

"""
Working Ultravox inference with v0.5 models that fit on RTX 3060
"""

import os
import torch
import librosa
import numpy as np
from ultravox.inference.ultravox_infer import UltravoxInference

def main():
    print("Testing Ultravox v0.5 models (smaller, should fit on RTX 3060)...")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set token
    hf_token = os.getenv("HF_TOKEN")
    print(f"HF Token set: {'Yes' if hf_token else 'No'}")
    
    # Try smaller v0.5 models that should fit on RTX 3060
    models_to_try = [
        "fixie-ai/ultravox-v0_5-llama-3_2-1b",    # Smallest, should definitely work
        "fixie-ai/ultravox-v0_5-llama-3_2-3b",    # Slightly larger but still manageable
        # "fixie-ai/ultravox-v0_5-llama-3_1-8b",  # Might be too big
    ]
    
    for model_path in models_to_try:
        print(f"\n=== Trying {model_path} ===")
        
        try:
            # Initialize with optimizations for RTX 3060
            inference = UltravoxInference(
                model_path=model_path,
                device="cuda",
                data_type="bfloat16",  # Use bfloat16 to save memory
                conversation_mode=False
            )
            
            print(f"✅ Model loaded successfully on: {inference.model.device}")
            
            # Test text-only inference
            print("\n=== Testing text-only inference ===")
            response = inference.infer("What is the capital of France?")
            print(f"📝 Text Response: {response}")
            
            # Test with simple math
            print("\n=== Testing simple math ===")
            response = inference.infer("What is 5 + 7?")
            print(f"🔢 Math Response: {response}")
            
            print("\n🎉 SUCCESS! Model is working!")
            print("\n📋 Usage examples:")
            print("# For speech transcription:")
            print("audio, sr = librosa.load('speech.wav', sr=16000)")
            print("response = inference.infer('Transcribe this audio', audio)")
            print("")
            print("# For speech question answering:")
            print("audio, sr = librosa.load('question.wav', sr=16000)")  
            print("response = inference.infer('Answer the question in the audio', audio)")
            print("")
            print("# For audio analysis:")
            print("audio, sr = librosa.load('audio.wav', sr=16000)")
            print("response = inference.infer('What sounds do you hear?', audio)")
            
            # Success, stop trying other models
            return True
                
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print("   → RTX 3060 doesn't have enough VRAM for this model size.")
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                print("   → Authentication issue - check HF token permissions.")
            else:
                print(f"   → Other error: {type(e).__name__}")
    
    print("\n❌ All models failed. Recommendations:")
    print("1. Try CPU inference: device='cpu' (slower but should work)")  
    print("2. Check HuggingFace token permissions")
    print("3. Try clearing GPU memory: torch.cuda.empty_cache()")
    return False

if __name__ == "__main__":
    main()

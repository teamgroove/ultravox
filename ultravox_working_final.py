#!/usr/bin/env python3

"""
Working Ultravox inference using the correct VoiceSample API
"""

import os
import torch
import librosa
import numpy as np
from ultravox.inference.ultravox_infer import UltravoxInference
from ultravox.data.data_sample import VoiceSample

def main():
    print("🚀 Testing Ultravox with correct API...")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set token
    hf_token = os.getenv("HF_TOKEN")
    print(f"HF Token set: {'Yes' if hf_token else 'No'}")
    
    # Use the 1B model that we know loads successfully
    model_path = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
    
    try:
        print(f"\n=== Loading {model_path} ===")
        
        # Initialize with optimizations for RTX 3060
        inference = UltravoxInference(
            model_path=model_path,
            device="cuda",
            data_type="bfloat16",  # Use bfloat16 to save memory
            conversation_mode=False
        )
        
        print(f"✅ Model loaded successfully on: {inference.model.device}")
        
        # Test text-only inference using VoiceSample
        print("\n=== Testing text-only inference ===")
        
        # Create a VoiceSample from prompt (no audio)
        sample = VoiceSample.from_prompt("What is the capital of France?")
        response = inference.infer(sample)
        print(f"📝 Text Response: {response.text}")
        
        # Test with simple math
        print("\n=== Testing simple math ===")
        sample = VoiceSample.from_prompt("What is 5 + 7?")
        response = inference.infer(sample)
        print(f"🔢 Math Response: {response.text}")
        
        # Test with a longer prompt
        print("\n=== Testing longer prompt ===")
        sample = VoiceSample.from_prompt("Tell me a short joke about programming.")
        response = inference.infer(sample)
        print(f"😄 Joke Response: {response.text}")
        
        print("\n🎉 SUCCESS! Ultravox is working on your RTX 3060!")
        print("\n📋 Usage examples for audio:")
        print("""
# For speech transcription:
import librosa
from ultravox.data.data_sample import VoiceSample

audio, sr = librosa.load('speech.wav', sr=16000)
sample = VoiceSample.from_prompt_and_raw('Transcribe this audio', audio, sr)
response = inference.infer(sample)
print(response.text)

# For speech question answering:
audio, sr = librosa.load('question.wav', sr=16000)  
sample = VoiceSample.from_prompt_and_raw('Answer the question in the audio', audio, sr)
response = inference.infer(sample)
print(response.text)

# For audio analysis:
audio, sr = librosa.load('audio.wav', sr=16000)
sample = VoiceSample.from_prompt_and_raw('What sounds do you hear?', audio, sr)
response = inference.infer(sample)
print(response.text)
        """)
        
        return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   → Error type: {type(e).__name__}")
        
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print("   → RTX 3060 doesn't have enough VRAM for this model size.")
        elif "401" in str(e) or "unauthorized" in str(e).lower():
            print("   → Authentication issue - check HF token permissions.")
        
        return False

if __name__ == "__main__":
    main()

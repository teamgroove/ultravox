#!/usr/bin/env python3

"""
Test script for Ultravox v0.6 models using transformers directly
Based on the examples from the Hugging Face model cards
"""

import os
import torch
import transformers
import numpy as np
import librosa
from transformers import AutoProcessor, AutoModelForCausalLM

def test_ultravox_v06():
    print("Testing Ultravox v0.6 models...")
    
    # Set HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Try the smallest model first (27B might still be too big for RTX 3060)
    model_name = "fixie-ai/ultravox-v0_6-gemma-3-27b"
    
    try:
        print(f"\nLoading model: {model_name}")
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(
            model_name, 
            token=hf_token,
            trust_remote_code=True
        )
        
        # Try loading with reduced precision to fit in memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            device_map="auto",  # Automatically distribute across available devices
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully!")
        print(f"Model device: {model.device}")
        
        # Test text-only first
        print("\n=== Testing text-only inference ===")
        text_prompt = "What is the capital of France?"
        
        inputs = processor(text=text_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        print("\nModel is ready for audio inference!")
        print("To use with audio, you would:")
        print("1. Load audio: audio, sr = librosa.load('file.wav', sr=16000)")
        print("2. Process: inputs = processor(text='<|audio|> Transcribe this', audio=audio, return_tensors='pt')")
        print("3. Generate: outputs = model.generate(**inputs)")
        
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print("\nRTX 3060 might not have enough VRAM for the 27B model.")
            print("Unfortunately, the v0.6 models are all quite large:")
            print("- gemma-3-27b: ~27B parameters")
            print("- llama-3_3-70b: ~70B parameters") 
            print("- qwen-3-32b: ~32B parameters")
            print("\nYou might need to use a smaller v0.5 model or run with CPU inference.")

if __name__ == "__main__":
    test_ultravox_v06()

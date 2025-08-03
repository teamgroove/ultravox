#!/usr/bin/env python3

"""
Local Ultravox Web Server
FastAPI server that provides a web interface for your local Ultravox model
"""

import os
import io
import base64
import asyncio
from typing import Optional
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ultravox.inference.ultravox_infer import UltravoxInference
from ultravox.data.data_sample import VoiceSample

# Global model instance
inference_model = None

class ChatRequest(BaseModel):
    message: str
    audio_base64: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

def initialize_model():
    """Initialize the Ultravox model"""
    global inference_model
    
    print("🚀 Initializing Ultravox model...")
    
    try:
        # Set HF token
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️  Warning: HF_TOKEN not set")
        
        # Use the working 1B model
        model_path = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
        
        inference_model = UltravoxInference(
            model_path=model_path,
            device="cuda",
            data_type="bfloat16",
            conversation_mode=False
        )
        
        print(f"✅ Model loaded successfully on: {inference_model.model.device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Initialize FastAPI app
app = FastAPI(title="Local Ultravox Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = initialize_model()
    if not success:
        print("❌ Failed to initialize model. Server will still start but won't work properly.")

@app.get("/")
async def root():
    """Serve the main web interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Ultravox Chat</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto max-w-4xl p-6">
        <h1 class="text-3xl font-bold text-center mb-8">🎤 Local Ultravox Chat</h1>
        
        <div id="status" class="mb-4 p-4 bg-blue-100 rounded-lg">
            <span class="font-semibold">Status:</span> <span id="status-text">Ready</span>
        </div>
        
        <!-- Chat Container -->
        <div id="chat-container" class="bg-white rounded-lg shadow-lg p-6 mb-6 h-96 overflow-y-auto">
            <div id="messages"></div>
        </div>
        
        <!-- Input Form -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <div class="flex flex-col space-y-4">
                <!-- Text Input -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Text Message:</label>
                    <input type="text" id="text-input" placeholder="Type your message here..." 
                           class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                </div>
                
                <!-- Audio Input -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Audio File:</label>
                    <input type="file" id="audio-input" accept="audio/*" 
                           class="w-full p-3 border border-gray-300 rounded-lg">
                </div>
                
                <!-- Buttons -->
                <div class="flex space-x-4">
                    <button onclick="sendMessage()" 
                            class="flex-1 bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg">
                        Send Message
                    </button>
                    <button onclick="startRecording()" id="record-btn"
                            class="bg-red-500 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg">
                        🎤 Record
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        function addMessage(sender, message, isError = false) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `mb-4 p-3 rounded-lg ${sender === 'user' ? 'bg-blue-100 ml-8' : isError ? 'bg-red-100 mr-8' : 'bg-gray-100 mr-8'}`;
            messageDiv.innerHTML = `
                <div class="font-semibold ${sender === 'user' ? 'text-blue-800' : isError ? 'text-red-800' : 'text-gray-800'}">
                    ${sender === 'user' ? 'You' : isError ? 'Error' : 'Ultravox'}
                </div>
                <div class="mt-1">${message}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function setStatus(status) {
            document.getElementById('status-text').textContent = status;
        }

        async function sendMessage() {
            const textInput = document.getElementById('text-input');
            const audioInput = document.getElementById('audio-input');
            const message = textInput.value.trim();
            
            if (!message && !audioInput.files[0]) {
                alert('Please enter a message or select an audio file');
                return;
            }

            setStatus('Processing...');
            
            try {
                const formData = new FormData();
                if (message) {
                    formData.append('message', message);
                    addMessage('user', message);
                }
                
                if (audioInput.files[0]) {
                    formData.append('audio_file', audioInput.files[0]);
                    addMessage('user', '🎵 Audio file uploaded');
                }

                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.status === 'success') {
                    addMessage('assistant', result.response);
                } else {
                    addMessage('error', result.response || 'Unknown error', true);
                }
                
            } catch (error) {
                addMessage('error', 'Failed to connect to server: ' + error.message, true);
            }
            
            // Clear inputs
            textInput.value = '';
            audioInput.value = '';
            setStatus('Ready');
        }

        async function startRecording() {
            const recordBtn = document.getElementById('record-btn');
            
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const formData = new FormData();
                        formData.append('audio_file', audioBlob, 'recording.wav');
                        formData.append('message', 'Transcribe this audio');
                        
                        setStatus('Processing recording...');
                        addMessage('user', '🎤 Voice recording');
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                body: formData
                            });

                            const result = await response.json();
                            
                            if (result.status === 'success') {
                                addMessage('assistant', result.response);
                            } else {
                                addMessage('error', result.response || 'Unknown error', true);
                            }
                        } catch (error) {
                            addMessage('error', 'Failed to process recording: ' + error.message, true);
                        }
                        
                        setStatus('Ready');
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    recordBtn.textContent = '⏹️ Stop';
                    recordBtn.className = 'bg-gray-500 hover:bg-gray-700 text-white font-bold py-3 px-6 rounded-lg';
                } catch (error) {
                    addMessage('error', 'Could not access microphone: ' + error.message, true);
                }
            } else {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                recordBtn.textContent = '🎤 Record';
                recordBtn.className = 'bg-red-500 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg';
            }
        }

        // Enter key to send message
        document.getElementById('text-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Initial status check
        fetch('/status').then(r => r.json()).then(data => {
            setStatus(data.status);
            if (data.status !== 'Ready') {
                addMessage('system', 'Model is initializing...', false);
            }
        }).catch(() => {
            setStatus('Server Error');
        });
    </script>
</body>
</html>
    """)

@app.get("/status")
async def get_status():
    """Get server status"""
    if inference_model is None:
        return {"status": "Model not loaded", "ready": False}
    return {"status": "Ready", "ready": True}

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    """Handle chat requests with optional audio"""
    
    if inference_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        audio_data = None
        
        # Process audio file if provided
        if audio_file:
            print(f"📁 Processing audio file: {audio_file.filename}")
            
            # Read audio file
            audio_bytes = await audio_file.read()
            
            # Convert to numpy array using librosa
            try:
                # Use librosa to load audio from bytes
                audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                print(f"🎵 Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
            except Exception as e:
                print(f"❌ Error loading audio: {e}")
                return ChatResponse(response=f"Error processing audio file: {str(e)}", status="error")
        
        # Create VoiceSample
        if audio_data is not None:
            sample = VoiceSample.from_prompt_and_raw(message, audio_data, 16000)
        else:
            sample = VoiceSample.from_prompt(message)
        
        print(f"💭 Processing: {message}")
        
        # Run inference
        response = inference_model.infer(sample)
        
        print(f"✅ Response: {response.text}")
        
        return ChatResponse(response=response.text)
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_json")
async def chat_json_endpoint(request: ChatRequest):
    """Handle JSON chat requests (alternative endpoint)"""
    
    if inference_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        audio_data = None
        
        # Process base64 audio if provided
        if request.audio_base64:
            try:
                audio_bytes = base64.b64decode(request.audio_base64)
                audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            except Exception as e:
                return ChatResponse(response=f"Error processing audio: {str(e)}", status="error")
        
        # Create VoiceSample
        if audio_data is not None:
            sample = VoiceSample.from_prompt_and_raw(request.message, audio_data, 16000)
        else:
            sample = VoiceSample.from_prompt(request.message)
        
        # Run inference
        response = inference_model.infer(sample)
        
        return ChatResponse(response=response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Local Ultravox Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🎯 API docs will be available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

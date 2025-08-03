# Local Ultravox Web Interface

A web-based chat interface for your local Ultravox model running on RTX 3060.

## Features

✅ **Text Chat**: Chat with Ultravox using text messages  
✅ **Audio Upload**: Upload audio files for transcription and analysis  
✅ **Voice Recording**: Record audio directly in the browser  
✅ **Real-time Processing**: All processing happens locally on your GPU  
✅ **No API Keys**: No external services required  

## Quick Start

1. **Start the server:**
   ```bash
   poetry run python start_ultravox_web.py
   ```

2. **Open your browser:**
   - Main interface: http://localhost:8080
   - API documentation: http://localhost:8080/docs

3. **Start chatting!**
   - Type text messages and get responses
   - Upload audio files (.wav, .mp3, etc.)
   - Use the microphone button to record audio

## Usage Examples

### Text Chat
- "What is the capital of France?"
- "Tell me a joke about programming"
- "Explain quantum computing in simple terms"

### Audio Processing
- Upload a speech file with prompt: "Transcribe this audio"
- Upload a song with prompt: "What genre is this music?"
- Record your voice asking: "What's the weather like?"

## API Endpoints

- `GET /` - Web interface
- `GET /status` - Server status
- `POST /chat` - Submit text/audio (form data)
- `POST /chat_json` - Submit text/audio (JSON)

## Technical Details

- **Model**: `fixie-ai/ultravox-v0_5-llama-3_2-1b`
- **Device**: CUDA (RTX 3060)
- **Precision**: bfloat16 (optimized for GPU memory)
- **Framework**: FastAPI + Uvicorn
- **Audio**: Processed with librosa at 16kHz

## Troubleshooting

**Port already in use?**
- Change the port in `start_ultravox_web.py` from 8080 to another port

**Model not loading?**
- Make sure HF_TOKEN environment variable is set
- Check GPU memory availability
- Verify poetry environment is activated

**Audio issues?**
- Supported formats: WAV, MP3, OGG, FLAC
- Files are automatically resampled to 16kHz
- Maximum file size depends on available memory

## Performance Tips

- First request may be slower (model warmup)
- Subsequent requests are much faster
- GPU memory usage: ~3-4GB for the 1B model
- CPU usage is minimal during inference

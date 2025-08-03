#!/usr/bin/env python3

"""
Startup script for Local Ultravox Web Server
"""

import uvicorn
from local_ultravox_server import app

if __name__ == "__main__":
    print("🚀 Starting Local Ultravox Web Server...")
    print("📍 Web interface: http://localhost:8080")
    print("🎯 API docs: http://localhost:8080/docs")
    print("⏹️  Press CTRL+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8080,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")

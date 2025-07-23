# fastapi_socket_wrapper.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import socket
import struct
import tempfile
import wave
import numpy as np
import os
import uvicorn
import glob

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown - cleanup temporary files
    temp_files = glob.glob("/tmp/tmp*.wav")
    for file in temp_files:
        try:
            os.unlink(file)
        except:
            pass

app = FastAPI(title="F5-TTS Socket API", version="1.0.0", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    max_length: int = 1000

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_duration: float = None

@app.post("/tts", response_class=FileResponse)
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from text using F5-TTS socket server"""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    if len(request.text) > request.max_length:
        raise HTTPException(status_code=400, detail=f"Text too long (max {request.max_length} chars)")
    
    try:
        # Connect to F5-TTS socket server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30)  # 30 second timeout
            s.connect(('localhost', int(9998)))
            s.sendall(request.text.encode('utf-8'))
            
            # Collect audio chunks
            audio_chunks = []
            while True:
                data = s.recv(4096)
                if data == b'END':
                    break
                if len(data) > 0:
                    # Unpack float32 audio data
                    chunk_size = len(data) // 4
                    if chunk_size > 0:
                        audio_chunk = struct.unpack(f'{chunk_size}f', data)
                        audio_chunks.extend(audio_chunk)
            
            if not audio_chunks:
                raise HTTPException(status_code=500, detail="No audio data received")
            
            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                audio_data = np.array(audio_chunks, dtype=np.float32)
                # Normalize and convert to int16
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Calculate duration
            duration = len(audio_chunks) / 24000
            
            return FileResponse(
                temp_file.name,
                media_type='audio/wav',
                filename=f'tts_output_{hash(request.text)}.wav',
                headers={
                    'X-Audio-Duration': str(duration),
                    'X-Audio-Length': str(len(audio_chunks))
                }
            )
            
    except socket.timeout:
        raise HTTPException(status_code=504, detail="TTS server timeout")
    except ConnectionRefusedError:
        raise HTTPException(status_code=503, detail="TTS server not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest):
    """Stream TTS audio in real-time"""
    def generate_audio():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(30)
                s.connect(('localhost', int(9998)))
                s.sendall(request.text.encode('utf-8'))
                
                while True:
                    data = s.recv(4096)
                    if data == b'END':
                        break
                    if len(data) > 0:
                        yield data
        except Exception as e:
            yield f"Error: {str(e)}".encode()
    
    return StreamingResponse(
        generate_audio(),
        media_type='audio/wav',
        headers={'Transfer-Encoding': 'chunked'}
    )

@app.get("/health")
async def health_check():
    """Check if the TTS socket server is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(('localhost', int(9998)))
            return {"status": "healthy", "tts_server": "available"}
    except:
        return {"status": "unhealthy", "tts_server": "unavailable"}

@app.post("/tts/info", response_model=TTSResponse)
async def tts_info(request: TTSRequest):
    """Get TTS generation info without generating audio"""
    return TTSResponse(
        success=True,
        message=f"Text length: {len(request.text)} chars",
        audio_duration=len(request.text) * 0.1  # Rough estimate
    )

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "F5-TTS Socket API", 
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /tts": "Generate TTS audio file",
            "POST /tts/stream": "Stream TTS audio",
            "POST /tts/info": "Get TTS info",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_socket_wrapper:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
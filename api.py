# api.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tempfile
import wave
import numpy as np
import os
import uvicorn
import glob
import sys
sys.path.append('./src')
from f5_tts.socket_server import TTSStreamingProcessor
from huggingface_hub import hf_hub_download
import whisper
import librosa

# Global instances
tts_processor = None
whisper_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Initialize TTS processor and Whisper model
    global tts_processor, whisper_model
    try:
        ckpt_file = str(hf_hub_download(repo_id="VIZINTZOR/F5-TTS-THAI", filename="model_1000000.pt"))
        vocab_file = str(hf_hub_download(repo_id="VIZINTZOR/F5-TTS-THAI", filename="vocab.txt"))
        print(f"Downloaded Thai vocab file: {vocab_file}")
        ref_audio = "src/f5_tts/infer/examples/thai_examples/tts_gen_1.wav"
        ref_text = ""  # Let it auto-transcribe the reference audio
        
        tts_processor = TTSStreamingProcessor(
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ref_audio=ref_audio,
            ref_text=ref_text,
            device=None,  # Will auto-detect best device
        )
        print("TTS processor initialized successfully")
        
        # Initialize Whisper model for speech-to-text
        whisper_model = whisper.load_model("base")
        print("Whisper model initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        tts_processor = None
        whisper_model = None
    
    yield
    
    # Shutdown - cleanup temporary files
    temp_files = glob.glob("/tmp/tmp*.wav")
    for file in temp_files:
        try:
            os.unlink(file)
        except:
            pass

app = FastAPI(title="F5-TTS API", version="1.0.0", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    max_length: int = 1000

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_duration: float = None

class STTResponse(BaseModel):
    success: bool
    text: str
    language: str = None
    confidence: float | None = None

@app.post("/tts", response_class=FileResponse)
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from text using F5-TTS directly"""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    if len(request.text) > request.max_length:
        raise HTTPException(status_code=400, detail=f"Text too long (max {request.max_length} chars)")
    
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    try:
        # Generate audio using TTS processor
        from f5_tts.infer.utils_infer import infer_batch_process, chunk_text
        
        # Process text similar to generate_stream method
        text_batches = chunk_text(request.text, max_chars=tts_processor.max_chars)
        if tts_processor.first_package:
            text_batches = chunk_text(text_batches[0], max_chars=tts_processor.few_chars) + text_batches[1:]
            text_batches = chunk_text(text_batches[0], max_chars=tts_processor.min_chars) + text_batches[1:]
            tts_processor.first_package = False

        audio_stream = infer_batch_process(
            (tts_processor.audio, tts_processor.sr),
            tts_processor.ref_text,
            text_batches,
            tts_processor.model,
            tts_processor.vocoder,
            progress=None,
            device=tts_processor.device,
            streaming=True,
            chunk_size=2048,
        )
        
        audio_chunks = []
        for audio_chunk, _ in audio_stream:
            if len(audio_chunk) > 0:
                audio_chunks.extend(audio_chunk)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio data generated")
        
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest):
    """Stream TTS audio in real-time"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    def generate_audio():
        try:
            from f5_tts.infer.utils_infer import infer_batch_process, chunk_text
            
            # Process text similar to generate_stream method
            text_batches = chunk_text(request.text, max_chars=tts_processor.max_chars)
            if tts_processor.first_package:
                text_batches = chunk_text(text_batches[0], max_chars=tts_processor.few_chars) + text_batches[1:]
                text_batches = chunk_text(text_batches[0], max_chars=tts_processor.min_chars) + text_batches[1:]
                tts_processor.first_package = False

            audio_stream = infer_batch_process(
                (tts_processor.audio, tts_processor.sr),
                tts_processor.ref_text,
                text_batches,
                tts_processor.model,
                tts_processor.vocoder,
                progress=None,
                device=tts_processor.device,
                streaming=True,
                chunk_size=2048,
            )
            
            for audio_chunk, _ in audio_stream:
                if len(audio_chunk) > 0:
                    # Convert float32 to bytes for streaming
                    audio_data = np.array(audio_chunk, dtype=np.float32)
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
        except Exception as e:
            yield f"Error: {str(e)}".encode()
    
    return StreamingResponse(
        generate_audio(),
        media_type='audio/wav',
        headers={'Transfer-Encoding': 'chunked'}
    )

@app.post("/stt", response_model=STTResponse)
async def speech_to_text(audio_file: UploadFile = File(...)):
    """Convert speech audio to text using Whisper"""
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Speech-to-text model not initialized")
    
    # Validate file type
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load and process audio
        audio, sr = librosa.load(temp_file_path, sr=16000)  # Whisper expects 16kHz
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(audio, language='th')  # Specify Thai language
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return STTResponse(
            success=True,
            text=result["text"].strip(),
            language=result.get("language", "th"),
            confidence=None  # Whisper doesn't provide confidence directly
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the TTS processor and STT model are available"""
    tts_status = "available" if tts_processor is not None else "not_initialized"
    stt_status = "available" if whisper_model is not None else "not_initialized"
    
    overall_status = "healthy" if (tts_processor is not None and whisper_model is not None) else "partial"
    
    return {
        "status": overall_status,
        "tts_processor": tts_status,
        "stt_model": stt_status
    }

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
        "message": "F5-TTS API", 
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /tts": "Generate TTS audio file",
            "POST /tts/stream": "Stream TTS audio",
            "POST /tts/info": "Get TTS info",
            "POST /stt": "Convert speech to text",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
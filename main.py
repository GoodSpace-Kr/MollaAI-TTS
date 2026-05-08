import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tts import KokoroTTS


logger = logging.getLogger("molla.tts")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(default="af_heart", description="Kokoro voice id")
    lang_code: str = Field(default="a", description="Kokoro language code")
    sample_rate: int = Field(default=24000, ge=8000, le=48000)

app = FastAPI(title="Molla TTS Server", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/tts/stream")
def stream_tts(payload: TTSRequest):
    try:
        tts = KokoroTTS(
            lang_code=payload.lang_code,
            voice=payload.voice,
            sample_rate=payload.sample_rate,
            output_dir=os.getenv("TTS_OUTPUT_DIR", "tts_out"),
        )
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
        return StreamingResponse(
            tts.stream_wav_bytes(payload.text),
            media_type="audio/wav",
            headers=headers,
        )
    except Exception as exc:
        logger.exception(
            "tts_stream_failed voice=%s lang_code=%s sample_rate=%s text_len=%s",
            payload.voice,
            payload.lang_code,
            payload.sample_rate,
            len(payload.text),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

import os
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tts import TTSRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("molla.tts")
tts_registry = TTSRegistry(output_dir=os.getenv("TTS_OUTPUT_DIR", "tts_out"))


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
    request_started_at = time.perf_counter()
    try:
        logger.info(
            "tts_request_received voice=%s lang_code=%s sample_rate=%s text_len=%s text=%r",
            payload.voice,
            payload.lang_code,
            payload.sample_rate,
            len(payload.text),
            payload.text[:120],
        )
        tts = tts_registry.get_tts(
            lang_code=payload.lang_code,
            voice=payload.voice,
            sample_rate=payload.sample_rate,
        )
        logger.info(
            "tts_pipeline_ready init_ms=%s voice=%s lang_code=%s sample_rate=%s",
            int((time.perf_counter() - request_started_at) * 1000),
            payload.voice,
            payload.lang_code,
            payload.sample_rate,
        )
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
        logger.info(
            "tts_stream_response_opened elapsed_ms=%s text_len=%s",
            int((time.perf_counter() - request_started_at) * 1000),
            len(payload.text),
        )
        return StreamingResponse(
            tts.stream_wav_bytes(payload.text, request_started_at=request_started_at),
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

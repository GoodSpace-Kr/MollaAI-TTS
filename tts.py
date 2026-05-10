import io
import logging
import os
import platform
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from kokoro import KPipeline

logger = logging.getLogger("molla.tts")


@dataclass(slots=True)
class PipelineHandle:
    pipeline: KPipeline
    lock: threading.Lock


class TTSRegistry:
    def __init__(self, output_dir: str = "tts_out") -> None:
        self.output_dir = output_dir
        self._handles: dict[str, PipelineHandle] = {}
        self._registry_lock = threading.Lock()
        os.makedirs(self.output_dir, exist_ok=True)

    def get_tts(self, *, lang_code: str, voice: str, sample_rate: int) -> "KokoroTTS":
        handle = self._get_or_create_handle(lang_code)
        return KokoroTTS(
            pipeline=handle.pipeline,
            pipeline_lock=handle.lock,
            lang_code=lang_code,
            voice=voice,
            sample_rate=sample_rate,
            output_dir=self.output_dir,
        )

    def _get_or_create_handle(self, lang_code: str) -> PipelineHandle:
        existing = self._handles.get(lang_code)
        if existing is not None:
            logger.info("tts_pipeline_cache_hit lang_code=%s", lang_code)
            return existing

        with self._registry_lock:
            existing = self._handles.get(lang_code)
            if existing is not None:
                logger.info("tts_pipeline_cache_hit lang_code=%s", lang_code)
                return existing

            started_at = time.perf_counter()
            handle = PipelineHandle(
                pipeline=KPipeline(lang_code=lang_code),
                lock=threading.Lock(),
            )
            self._handles[lang_code] = handle
            logger.info(
                "tts_pipeline_cache_miss lang_code=%s init_ms=%s",
                lang_code,
                int((time.perf_counter() - started_at) * 1000),
            )
            return handle


class KokoroTTS:
    def __init__(
        self,
        *,
        pipeline: KPipeline,
        pipeline_lock: threading.Lock,
        lang_code="a",
        voice="af_heart",
        sample_rate=24000,
        output_dir="tts_out",
    ):
        self.pipeline = pipeline
        self.pipeline_lock = pipeline_lock
        self.lang_code = lang_code
        self.voice = voice
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def text_to_wav(self, text: str, filename="reply.wav") -> str:
        final_audio = None
        for audio in self.synthesize_chunks(text):
            final_audio = audio

        if final_audio is None:
            raise RuntimeError("오디오 생성 실패")

        wav_path = os.path.join(self.output_dir, filename)
        sf.write(wav_path, final_audio, self.sample_rate)
        return wav_path

    def synthesize_chunks(self, text: str):
        wait_started_at = time.perf_counter()
        with self.pipeline_lock:
            logger.info(
                "tts_pipeline_lock_acquired lang_code=%s voice=%s wait_ms=%s text_len=%s",
                self.lang_code,
                self.voice,
                int((time.perf_counter() - wait_started_at) * 1000),
                len(text),
            )
            for i, (_, _, audio) in enumerate(self.pipeline(text, voice=self.voice)):
                print(f"[TTS chunk {i}]")
                yield audio

    def text_to_wav_bytes(self, text: str) -> bytes:
        wav_buffer = io.BytesIO()
        chunk_found = False

        with sf.SoundFile(
            wav_buffer,
            mode="w",
            samplerate=self.sample_rate,
            channels=1,
            format="WAV",
            subtype="PCM_16",
        ) as wav_file:
            for audio in self.synthesize_chunks(text):
                chunk_found = True
                wav_file.write(audio)

        if not chunk_found:
            raise RuntimeError("오디오 생성 실패")
        return wav_buffer.getvalue()

    def stream_wav_bytes(self, text: str, request_started_at: float | None = None):
        stream_started_at = time.perf_counter()
        base_started_at = request_started_at if request_started_at is not None else stream_started_at
        logger.info(
            "tts_stream_generator_started elapsed_ms=%s text_len=%s voice=%s",
            int((stream_started_at - base_started_at) * 1000),
            len(text),
            self.voice,
        )
        yield self._wav_header()
        logger.info(
            "tts_header_sent elapsed_ms=%s text_len=%s voice=%s",
            int((time.perf_counter() - base_started_at) * 1000),
            len(text),
            self.voice,
        )

        chunk_found = False
        first_chunk_logged = False
        total_audio_bytes = 0
        for audio in self.synthesize_chunks(text):
            chunk_found = True
            pcm_bytes = self._audio_to_pcm16_bytes(audio)
            total_audio_bytes += len(pcm_bytes)
            if not first_chunk_logged:
                first_chunk_logged = True
                logger.info(
                    "tts_first_audio_chunk_ready elapsed_ms=%s chunk_bytes=%s text_len=%s voice=%s",
                    int((time.perf_counter() - base_started_at) * 1000),
                    len(pcm_bytes),
                    len(text),
                    self.voice,
                )
            yield pcm_bytes

        if not chunk_found:
            raise RuntimeError("오디오 생성 실패")
        logger.info(
            "tts_stream_completed elapsed_ms=%s total_audio_bytes=%s text_len=%s voice=%s",
            int((time.perf_counter() - base_started_at) * 1000),
            total_audio_bytes,
            len(text),
            self.voice,
        )

    def play_wav(self, wav_path: str):
        if shutil.which("ffplay"):
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if platform.system() == "Darwin" and shutil.which("afplay"):
            subprocess.run(
                ["afplay", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if shutil.which("mpv"):
            subprocess.run(
                ["mpv", "--no-video", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if shutil.which("aplay"):
            subprocess.run(
                ["aplay", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        raise RuntimeError("오디오 플레이어가 없습니다. aplay, ffplay, mpv 중 하나를 설치하세요.")


    def speak(self, text: str, filename="reply.wav") -> str:
        import time
        start_time = time.time()
        wav_path = self.text_to_wav(text, filename=filename)
        print("TTS 소요 시간: ", time.time() - start_time)
        self.play_wav(wav_path)
        return wav_path

    def _audio_to_pcm16_bytes(self, audio) -> bytes:
        pcm = np.asarray(audio, dtype=np.float32)
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def _wav_header(self) -> bytes:
        channels = 1
        bits_per_sample = 16
        byte_rate = self.sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        unknown_length = 0xFFFFFFFF

        return b"".join(
            [
                b"RIFF",
                struct.pack("<I", unknown_length),
                b"WAVE",
                b"fmt ",
                struct.pack("<I", 16),
                struct.pack("<H", 1),
                struct.pack("<H", channels),
                struct.pack("<I", self.sample_rate),
                struct.pack("<I", byte_rate),
                struct.pack("<H", block_align),
                struct.pack("<H", bits_per_sample),
                b"data",
                struct.pack("<I", unknown_length),
            ]
        )

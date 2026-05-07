import io
import os
import platform
import shutil
import struct
import subprocess

import numpy as np
import soundfile as sf
from kokoro import KPipeline

class KokoroTTS:
    def __init__(self, lang_code="a", voiceㄴ="af_heart", sample_rate=24000, output_dir="tts_out"):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def text_to_wav(self, text: str, filename="reply.wav") -> str:
        generator = self.pipeline(text, voice=self.voice)

        final_audio = None
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"[TTS chunk {i}]")
            final_audio = audio

        if final_audio is None:
            raise RuntimeError("오디오 생성 실패")

        wav_path = os.path.join(self.output_dir, filename)
        sf.write(wav_path, final_audio, self.sample_rate)
        return wav_path

    def synthesize_chunks(self, text: str):
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

    def stream_wav_bytes(self, text: str):
        # Use an open-ended WAV header so the client can start consuming audio
        # before total length is known.
        yield self._wav_header()

        chunk_found = False
        for audio in self.synthesize_chunks(text):
            chunk_found = True
            yield self._audio_to_pcm16_bytes(audio)

        if not chunk_found:
            raise RuntimeError("오디오 생성 실패")

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

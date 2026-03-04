"""Wyoming STT proxy for Qwen3-ASR (vLLM OpenAI-compatible backend).

Listens on TCP (default port 10300) and translates Wyoming audio events
into HTTP requests to the Qwen3-ASR vLLM transcription API, then returns
the transcript as a Wyoming Transcript event.

Compatible with Home Assistant's Wyoming integration.
"""

import argparse
import asyncio
import io
import logging
import struct
import wave
from functools import partial

import aiohttp
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, AsrModel, AsrProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.asr import Transcribe, Transcript

_LOGGER = logging.getLogger(__name__)


def _make_wav(pcm_data: bytes, rate: int, width: int, channels: int) -> bytes:
    """Wrap raw PCM data in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class QwenASRWyomingHandler(AsyncEventHandler):
    """Handle Wyoming events for a single client connection."""

    def __init__(
        self,
        wyoming_info: Info,
        asr_url: str,
        model_name: str,
        language: str | None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.asr_url = asr_url.rstrip("/")
        self.model_name = model_name
        self.language = language

        # Audio accumulation state
        self._rate: int = 16_000
        self._width: int = 2
        self._channels: int = 1
        self._audio_chunks: list[bytes] = []
        self._recording: bool = False

    async def handle_event(self, event: Event) -> bool:
        from wyoming.info import Describe

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Transcribe.is_type(event.type):
            # Reset state for new utterance
            self._audio_chunks = []
            self._recording = False
            return True

        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self._rate = audio_start.rate
            self._width = audio_start.width
            self._channels = audio_start.channels
            self._audio_chunks = []
            self._recording = True
            _LOGGER.debug(
                "AudioStart: rate=%d width=%d channels=%d",
                self._rate, self._width, self._channels,
            )
            return True

        if AudioChunk.is_type(event.type):
            if self._recording:
                chunk = AudioChunk.from_event(event)
                self._audio_chunks.append(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            self._recording = False
            if not self._audio_chunks:
                _LOGGER.warning("AudioStop received with no audio data")
                await self.write_event(Transcript(text="").event())
                return True

            pcm_data = b"".join(self._audio_chunks)
            _LOGGER.debug(
                "AudioStop: collected %d bytes of PCM (%.1f s)",
                len(pcm_data),
                len(pcm_data) / (self._rate * self._width * self._channels),
            )

            wav_data = _make_wav(pcm_data, self._rate, self._width, self._channels)
            text = await self._transcribe(wav_data)
            _LOGGER.info("Transcript: %r", text)
            await self.write_event(Transcript(text=text).event())
            return True

        return True

    async def _transcribe(self, wav_data: bytes) -> str:
        """Send audio to Qwen3-ASR vLLM transcription API and return text."""
        try:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_data,
                filename="audio.wav",
                content_type="audio/wav",
            )
            form.add_field("model", self.model_name)
            if self.language:
                form.add_field("language", self.language)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.asr_url}/v1/audio/transcriptions",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("text", "").strip()
        except Exception as exc:
            _LOGGER.error("Qwen3-ASR transcription request failed: %s", exc)
            return ""


async def wait_for_service(url: str, max_attempts: int = 60) -> None:
    """Poll the /health or /v1/models endpoint until the server is ready."""
    _LOGGER.info("Waiting for Qwen3-ASR at %s …", url)
    for attempt in range(max_attempts):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        _LOGGER.info("Qwen3-ASR server is ready")
                        return
        except Exception:
            pass
        await asyncio.sleep(5)
    _LOGGER.warning("Qwen3-ASR not reachable after %ds; continuing anyway", max_attempts * 5)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wyoming STT proxy for Qwen3-ASR (vLLM backend)"
    )
    parser.add_argument(
        "--asr-url",
        default="http://qwen-asr:8000",
        help="Base URL of the Qwen3-ASR vLLM service",
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10300",
        help="Wyoming server URI (e.g. tcp://0.0.0.0:10300)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Model name to pass to the transcription API",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force transcription language (e.g. 'English'). Omit for auto-detect.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asr_url = args.asr_url.rstrip("/")

    await wait_for_service(asr_url)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="qwen3-asr",
                description="Qwen3-ASR speech recognition (vLLM backend)",
                attribution=Attribution(
                    name="Qwen",
                    url="https://github.com/QwenLM/Qwen3-ASR",
                ),
                installed=True,
                version="0.6B",
                models=[
                    AsrModel(
                        name="Qwen3-ASR-0.6B",
                        description="Qwen3-ASR 0.6B — 52-language ASR",
                        attribution=Attribution(
                            name="Qwen",
                            url="https://huggingface.co/Qwen/Qwen3-ASR-0.6B",
                        ),
                        installed=True,
                        languages=["en", "zh", "fr", "de", "es", "ja", "ko", "ru",
                                   "pt", "it", "ar", "hi", "nl", "sv", "da", "fi",
                                   "pl", "cs", "tr", "vi", "th", "id", "ms", "el",
                                   "hu", "ro", "mk", "fa", "fil", "yue"],
                        version="0.6B",
                    )
                ],
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(
        "Wyoming Qwen3-ASR proxy ready  uri=%s  backend=%s",
        args.uri,
        asr_url,
    )

    await server.run(
        partial(
            QwenASRWyomingHandler,
            wyoming_info,
            asr_url,
            args.model,
            args.language,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

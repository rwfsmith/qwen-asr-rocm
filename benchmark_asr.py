#!/usr/bin/env python3
"""
Benchmark: Qwen3-ASR ONNX vs Faster-Whisper (Wyoming)

Compares two STT backends by sending the same audio samples to both and
measuring latency, throughput and transcript quality (WER if reference text
is provided).

Endpoints tested:
  1. Qwen3-ASR ONNX  — OpenAI-compatible /v1/audio/transcriptions HTTP API
  2. Faster-Whisper   — Wyoming TCP protocol (the existing HA integration)

Usage:
  # Minimal — uses built-in TTS to generate test audio:
  python benchmark_asr.py

  # Point at your endpoints:
  python benchmark_asr.py \
      --onnx-url http://<truenas-ip>:8001 \
      --wyoming-host <ha-ip> --wyoming-port 10300 \
      --rounds 5

  # Use your own WAV files:
  python benchmark_asr.py --audio-dir ./test_audio/

  # Skip one endpoint (only test what's available):
  python benchmark_asr.py --skip-wyoming
  python benchmark_asr.py --skip-onnx

Requirements:
  pip install aiohttp wyoming
  # Optional for WER: pip install jiwer
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import statistics
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

try:
    import aiohttp
except ImportError:
    sys.exit("ERROR: pip install aiohttp")

# Wyoming is optional — only needed if testing faster-whisper endpoint
try:
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.asr import Transcribe, Transcript
    from wyoming.event import Event, async_read_event, async_write_event
    from wyoming.info import Describe

    HAS_WYOMING = True
except ImportError:
    HAS_WYOMING = False

# jiwer is optional — only needed for WER computation
try:
    from jiwer import wer as compute_wer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


# ── Test audio samples ──────────────────────────────────────────────────────

# Short sentences covering common HA voice-assistant commands
BUILTIN_SAMPLES: list[dict[str, str]] = [
    {"text": "Turn on the living room lights.", "lang": "en"},
    {"text": "What is the temperature outside?", "lang": "en"},
    {"text": "Set a timer for five minutes.", "lang": "en"},
    {"text": "Lock the front door.", "lang": "en"},
    {"text": "Play some music in the kitchen.", "lang": "en"},
]


def generate_sine_wav(
    text: str, duration_s: float = 2.0, rate: int = 16_000
) -> bytes:
    """Generate a silent WAV file (zero-amplitude) as a placeholder.

    Real benchmarks should use actual speech recordings.  This lets you
    verify connectivity and measure baseline latency without needing
    pre-recorded audio.
    """
    n_samples = int(rate * duration_s)
    # Near-silent noise so the encoder has *something* to process
    import random

    random.seed(hash(text) & 0xFFFFFFFF)
    samples = bytes(
        struct.pack("<h", random.randint(-10, 10)) for _ in range(n_samples)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples)
    return buf.getvalue()


def load_test_audio(audio_dir: str | None) -> list[dict]:
    """Load test cases — either from a directory of WAV files or built-in."""
    samples: list[dict] = []

    if audio_dir:
        p = Path(audio_dir)
        if not p.is_dir():
            sys.exit(f"ERROR: --audio-dir {audio_dir} is not a directory")

        for wav_path in sorted(p.glob("*.wav")):
            ref_path = wav_path.with_suffix(".txt")
            ref_text = ref_path.read_text().strip() if ref_path.exists() else None
            samples.append(
                {
                    "name": wav_path.stem,
                    "wav": wav_path.read_bytes(),
                    "reference": ref_text,
                }
            )
        if not samples:
            sys.exit(f"ERROR: no .wav files found in {audio_dir}")
        print(f"Loaded {len(samples)} audio file(s) from {audio_dir}")
    else:
        print("No --audio-dir given — using built-in placeholder audio")
        print("  (for meaningful quality comparison, provide real speech WAVs)")
        for i, s in enumerate(BUILTIN_SAMPLES):
            samples.append(
                {
                    "name": f"sample_{i+1}",
                    "wav": generate_sine_wav(s["text"]),
                    "reference": s["text"],
                }
            )

    return samples


# ── Results ──────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    name: str
    transcript: str
    reference: str | None
    latency_ms: float
    audio_duration_s: float
    error: str | None = None

    @property
    def rtf(self) -> float:
        """Real-time factor: latency / audio_duration (lower = faster)."""
        if self.audio_duration_s > 0:
            return (self.latency_ms / 1000.0) / self.audio_duration_s
        return float("inf")


@dataclass
class BenchmarkSummary:
    endpoint: str
    results: list[RunResult] = field(default_factory=list)

    @property
    def successes(self) -> list[RunResult]:
        return [r for r in self.results if r.error is None]

    @property
    def failures(self) -> list[RunResult]:
        return [r for r in self.results if r.error is not None]

    def latency_stats(self) -> dict:
        times = [r.latency_ms for r in self.successes]
        if not times:
            return {}
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) >= 2 else times[0],
        }

    def avg_rtf(self) -> float:
        rtfs = [r.rtf for r in self.successes if r.rtf != float("inf")]
        return statistics.mean(rtfs) if rtfs else float("inf")

    def wer(self) -> float | None:
        if not HAS_JIWER:
            return None
        refs = []
        hyps = []
        for r in self.successes:
            if r.reference:
                refs.append(r.reference)
                hyps.append(r.transcript)
        if not refs:
            return None
        return compute_wer(refs, hyps)


def wav_duration(wav_data: bytes) -> float:
    """Return duration in seconds of a WAV file."""
    buf = io.BytesIO(wav_data)
    with wave.open(buf, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ── ONNX endpoint (OpenAI-compatible HTTP) ───────────────────────────────────


async def transcribe_onnx(
    session: aiohttp.ClientSession,
    url: str,
    wav_data: bytes,
    model: str,
) -> tuple[str, float]:
    """Send WAV to /v1/audio/transcriptions, return (text, latency_ms)."""
    form = aiohttp.FormData()
    form.add_field("file", wav_data, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", model)

    t0 = time.perf_counter()
    async with session.post(
        f"{url}/v1/audio/transcriptions",
        data=form,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000
    return data.get("text", "").strip(), latency_ms


async def run_onnx_benchmark(
    url: str, model: str, samples: list[dict], rounds: int
) -> BenchmarkSummary:
    summary = BenchmarkSummary(endpoint=f"ONNX ({url})")
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("  Warming up ONNX endpoint...")
        try:
            await transcribe_onnx(session, url, samples[0]["wav"], model)
        except Exception as e:
            print(f"  WARNING: warmup failed: {e}")

        for round_num in range(rounds):
            for sample in samples:
                name = f"r{round_num+1}_{sample['name']}"
                try:
                    text, latency = await transcribe_onnx(
                        session, url, sample["wav"], model
                    )
                    summary.results.append(
                        RunResult(
                            name=name,
                            transcript=text,
                            reference=sample.get("reference"),
                            latency_ms=latency,
                            audio_duration_s=wav_duration(sample["wav"]),
                        )
                    )
                except Exception as e:
                    summary.results.append(
                        RunResult(
                            name=name,
                            transcript="",
                            reference=sample.get("reference"),
                            latency_ms=0,
                            audio_duration_s=wav_duration(sample["wav"]),
                            error=str(e),
                        )
                    )
    return summary


# ── Wyoming endpoint (TCP protocol) ─────────────────────────────────────────


async def transcribe_wyoming(
    host: str, port: int, wav_data: bytes
) -> tuple[str, float]:
    """Send audio via Wyoming protocol, return (text, latency_ms)."""
    reader, writer = await asyncio.open_connection(host, port)

    try:
        # Parse WAV header
        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            channels = wf.getnchannels()
            pcm = wf.readframes(wf.getnframes())

        t0 = time.perf_counter()

        # Send Transcribe
        await async_write_event(Transcribe().event(), writer)

        # Send AudioStart
        await async_write_event(
            AudioStart(rate=rate, width=width, channels=channels).event(),
            writer,
        )

        # Send audio in chunks (~100ms each)
        chunk_size = rate * width * channels // 10  # 100ms
        for offset in range(0, len(pcm), chunk_size):
            chunk = pcm[offset : offset + chunk_size]
            await async_write_event(
                AudioChunk(
                    audio=chunk, rate=rate, width=width, channels=channels
                ).event(),
                writer,
            )

        # Send AudioStop
        await async_write_event(AudioStop().event(), writer)

        # Wait for Transcript response
        while True:
            event = await asyncio.wait_for(
                async_read_event(reader), timeout=120
            )
            if event is None:
                break
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                latency_ms = (time.perf_counter() - t0) * 1000
                return transcript.text or "", latency_ms

        latency_ms = (time.perf_counter() - t0) * 1000
        return "", latency_ms
    finally:
        writer.close()
        await writer.wait_closed()


async def run_wyoming_benchmark(
    host: str, port: int, samples: list[dict], rounds: int
) -> BenchmarkSummary:
    summary = BenchmarkSummary(endpoint=f"Wyoming ({host}:{port})")

    # Warmup
    print("  Warming up Wyoming endpoint...")
    try:
        await transcribe_wyoming(host, port, samples[0]["wav"])
    except Exception as e:
        print(f"  WARNING: warmup failed: {e}")

    for round_num in range(rounds):
        for sample in samples:
            name = f"r{round_num+1}_{sample['name']}"
            try:
                text, latency = await transcribe_wyoming(
                    host, port, sample["wav"]
                )
                summary.results.append(
                    RunResult(
                        name=name,
                        transcript=text,
                        reference=sample.get("reference"),
                        latency_ms=latency,
                        audio_duration_s=wav_duration(sample["wav"]),
                    )
                )
            except Exception as e:
                summary.results.append(
                    RunResult(
                        name=name,
                        transcript="",
                        reference=sample.get("reference"),
                        latency_ms=0,
                        audio_duration_s=wav_duration(sample["wav"]),
                        error=str(e),
                    )
                )
    return summary


# ── Reporting ────────────────────────────────────────────────────────────────


def print_summary(summary: BenchmarkSummary) -> None:
    print(f"\n{'='*60}")
    print(f"  {summary.endpoint}")
    print(f"{'='*60}")

    n_ok = len(summary.successes)
    n_fail = len(summary.failures)
    print(f"  Requests:  {n_ok} succeeded, {n_fail} failed")

    if not summary.successes:
        print("  No successful transcriptions — skipping stats")
        return

    stats = summary.latency_stats()
    print(f"\n  Latency:")
    print(f"    Min:     {stats['min_ms']:>8.1f} ms")
    print(f"    Max:     {stats['max_ms']:>8.1f} ms")
    print(f"    Mean:    {stats['mean_ms']:>8.1f} ms")
    print(f"    Median:  {stats['median_ms']:>8.1f} ms")
    print(f"    Stdev:   {stats['stdev_ms']:>8.1f} ms")
    print(f"    P95:     {stats['p95_ms']:>8.1f} ms")

    rtf = summary.avg_rtf()
    print(f"\n  Real-time factor (RTF): {rtf:.3f}x")
    if rtf < 1.0:
        print(f"    → {1/rtf:.1f}x faster than real-time ✓")
    else:
        print(f"    → {rtf:.1f}x slower than real-time ✗")

    wer_val = summary.wer()
    if wer_val is not None:
        print(f"\n  Word Error Rate (WER): {wer_val*100:.1f}%")

    # Show sample transcripts
    print(f"\n  Sample transcripts:")
    shown = set()
    for r in summary.successes:
        base = r.name.split("_", 1)[1] if "_" in r.name else r.name
        if base in shown:
            continue
        shown.add(base)
        ref_str = f" (ref: {r.reference})" if r.reference else ""
        print(f'    [{base}] "{r.transcript}"{ref_str}')
        if len(shown) >= 5:
            break

    if summary.failures:
        print(f"\n  Errors:")
        for r in summary.failures[:3]:
            print(f"    [{r.name}] {r.error}")


def print_comparison(summaries: list[BenchmarkSummary]) -> None:
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")

    headers = ["Metric"] + [s.endpoint.split("(")[0].strip() for s in summaries]
    rows: list[list[str]] = []

    # Latency
    for label, key in [
        ("Mean latency", "mean_ms"),
        ("Median latency", "median_ms"),
        ("P95 latency", "p95_ms"),
    ]:
        row = [label]
        for s in summaries:
            stats = s.latency_stats()
            if stats:
                row.append(f"{stats[key]:.0f} ms")
            else:
                row.append("N/A")
        rows.append(row)

    # RTF
    row = ["RTF"]
    for s in summaries:
        rtf = s.avg_rtf()
        row.append(f"{rtf:.3f}x" if rtf != float("inf") else "N/A")
    rows.append(row)

    # WER
    if HAS_JIWER:
        row = ["WER"]
        for s in summaries:
            w = s.wer()
            row.append(f"{w*100:.1f}%" if w is not None else "N/A")
        rows.append(row)

    # Success rate
    row = ["Success rate"]
    for s in summaries:
        total = len(s.results)
        ok = len(s.successes)
        row.append(f"{ok}/{total}")
    rows.append(row)

    # Print table
    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  " + "  │  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "──┼──".join("─" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# ── Main ─────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-ASR ONNX vs Faster-Whisper (Wyoming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with real audio files (put .wav files and optional .txt reference files):
  python benchmark_asr.py --audio-dir ./test_audio/ --rounds 3

  # Test only the ONNX endpoint:
  python benchmark_asr.py --skip-wyoming --onnx-url http://192.168.1.100:8001

  # Test only the Wyoming endpoint:
  python benchmark_asr.py --skip-onnx --wyoming-host 192.168.1.100 --wyoming-port 10300
""",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory of .wav files (with optional .txt reference transcripts)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of times to run each sample (default: 3)",
    )
    parser.add_argument(
        "--onnx-url",
        default="http://localhost:8001",
        help="Qwen3-ASR ONNX endpoint URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--onnx-model",
        default="andrewleech/qwen3-asr-0.6b-onnx-int8",
        help="Model name to pass to ONNX endpoint",
    )
    parser.add_argument(
        "--wyoming-host",
        default="localhost",
        help="Faster-Whisper Wyoming host (default: localhost)",
    )
    parser.add_argument(
        "--wyoming-port",
        type=int,
        default=10300,
        help="Faster-Whisper Wyoming port (default: 10300)",
    )
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX test")
    parser.add_argument(
        "--skip-wyoming", action="store_true", help="Skip Wyoming test"
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write detailed results to a JSON file",
    )

    args = parser.parse_args()

    if args.skip_onnx and args.skip_wyoming:
        sys.exit("ERROR: cannot skip both endpoints")

    if not args.skip_wyoming and not HAS_WYOMING:
        sys.exit(
            "ERROR: wyoming package required for Wyoming test.\n"
            "  pip install wyoming\n"
            "  Or use --skip-wyoming to skip."
        )

    samples = load_test_audio(args.audio_dir)
    summaries: list[BenchmarkSummary] = []

    print(f"\nBenchmark: {len(samples)} sample(s) × {args.rounds} round(s)\n")

    # --- ONNX ---
    if not args.skip_onnx:
        print(f"Testing ONNX endpoint: {args.onnx_url}")
        onnx_summary = await run_onnx_benchmark(
            args.onnx_url, args.onnx_model, samples, args.rounds
        )
        summaries.append(onnx_summary)
        print_summary(onnx_summary)

    # --- Wyoming ---
    if not args.skip_wyoming:
        print(f"\nTesting Wyoming endpoint: {args.wyoming_host}:{args.wyoming_port}")
        wyoming_summary = await run_wyoming_benchmark(
            args.wyoming_host, args.wyoming_port, samples, args.rounds
        )
        summaries.append(wyoming_summary)
        print_summary(wyoming_summary)

    # --- Comparison ---
    if len(summaries) == 2:
        print_comparison(summaries)

    # --- JSON output ---
    if args.output_json:
        output = {
            "config": {
                "rounds": args.rounds,
                "n_samples": len(samples),
                "onnx_url": args.onnx_url if not args.skip_onnx else None,
                "wyoming": f"{args.wyoming_host}:{args.wyoming_port}"
                if not args.skip_wyoming
                else None,
            },
            "results": {},
        }
        for s in summaries:
            key = s.endpoint.split("(")[0].strip().lower()
            output["results"][key] = {
                "stats": s.latency_stats(),
                "rtf": s.avg_rtf(),
                "wer": s.wer(),
                "successes": len(s.successes),
                "failures": len(s.failures),
                "runs": [
                    {
                        "name": r.name,
                        "transcript": r.transcript,
                        "reference": r.reference,
                        "latency_ms": r.latency_ms,
                        "audio_duration_s": r.audio_duration_s,
                        "rtf": r.rtf,
                        "error": r.error,
                    }
                    for r in s.results
                ],
            }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"\nDetailed results written to {args.output_json}")

    print()


if __name__ == "__main__":
    asyncio.run(main())

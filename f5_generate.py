#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 F5-TTS SES ÜRETİCİ (f5-env içinde çalışır)
================================================================================
 Ana pipeline bunu SUBPROCESS olarak çağırır. Bir videonun tüm segmentlerini
 TEK model yüklemesiyle üretir (verimli), timeline'a hizalar, tek WAV yazar.

 Girdi: --job <job.json>  içinde:
   {
     "segments": [{"start":0.0,"end":2.5,"text":"..."}],
     "total_duration": 26.2,
     "ref_audio": "/path/ref_voice.wav",
     "ref_text": "referans transkripti",
     "ckpt_file": "/path/f5_tts_turkish.safetensors",
     "vocab_file": "/path/vocab.txt",
     "output": "/path/3_turkish_audio.wav",
     "nfe_step": 32,
     "max_speedup": 1.35
   }

 Kullanım (pipeline tarafından):
   f5-env/bin/python f5_generate.py --job /tmp/job.json
================================================================================
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def run_ffmpeg(args):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr.strip()[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    segments = [s for s in job["segments"] if s["text"].strip()]
    if not segments:
        print("HATA: segment yok", file=sys.stderr)
        sys.exit(1)

    # F5-TTS modelini TEK kez yükle
    from f5_tts.api import F5TTS
    print(f"[f5] Model yükleniyor: {Path(job['ckpt_file']).name}", flush=True)
    f5 = F5TTS(
        model="F5TTS_Base",
        ckpt_file=job["ckpt_file"],
        vocab_file=job["vocab_file"],
        device="cuda",
    )

    ref_audio = job["ref_audio"]
    ref_text = job.get("ref_text", "")
    nfe = int(job.get("nfe_step", 32))

    sr_out = 24000  # F5-TTS çıktı örnekleme hızı
    total_ms = int(job["total_duration"] * 1000)
    timeline = np.zeros(int(total_ms / 1000 * sr_out), dtype=np.float32)

    tmpdir = Path(tempfile.mkdtemp(prefix="f5seg_"))
    max_speedup = float(job.get("max_speedup", 1.35))

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        print(f"[f5] segment {i+1}/{len(segments)}: {text[:40]}...", flush=True)
        try:
            result = f5.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                nfe_step=nfe,
            )
            # F5-TTS sürümüne göre 2 veya 3 değer döndürebilir
            # (audio, sample_rate) veya (audio, sample_rate, spectrogram)
            audio = result[0]
            sr = result[1]
        except Exception as e:
            print(f"[f5] segment {i} üretilemedi: {e}", file=sys.stderr)
            continue

        audio = np.asarray(audio, dtype=np.float32)
        # Örnekleme hızını normalize et (F5 genelde 24k döndürür)
        if sr != sr_out:
            seg_wav = tmpdir / f"seg_{i}.wav"
            sf.write(seg_wav, audio, sr)
            fixed = tmpdir / f"seg_{i}_24k.wav"
            run_ffmpeg(["-i", str(seg_wav), "-ar", str(sr_out), "-ac", "1",
                        str(fixed)])
            audio, _ = sf.read(fixed, dtype="float32")

        # Segment süresine sığdır (gerekirse hızlandır)
        slot_sec = seg["end"] - seg["start"]
        clip_sec = len(audio) / sr_out
        if clip_sec > slot_sec and slot_sec > 0.2:
            speed = min(clip_sec / slot_sec, max_speedup)
            if speed > 1.02:
                seg_wav = tmpdir / f"seg_{i}_pre.wav"
                sf.write(seg_wav, audio, sr_out)
                fast = tmpdir / f"seg_{i}_fast.wav"
                run_ffmpeg(["-i", str(seg_wav), "-filter:a",
                            f"atempo={speed:.3f}", str(fast)])
                audio, _ = sf.read(fast, dtype="float32")

        # Timeline'a yerleştir
        start_idx = int(seg["start"] * sr_out)
        end_idx = start_idx + len(audio)
        if end_idx > len(timeline):
            audio = audio[:len(timeline) - start_idx]
            end_idx = len(timeline)
        timeline[start_idx:end_idx] += audio

    # Kırpma (clipping) önle
    peak = np.max(np.abs(timeline)) if len(timeline) else 1.0
    if peak > 1.0:
        timeline = timeline / peak * 0.98

    sf.write(job["output"], timeline, sr_out)
    print(f"[f5] BİTTİ -> {job['output']}", flush=True)


if __name__ == "__main__":
    main()

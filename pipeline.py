#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ANIME-REELS — ÇOK-GPU BATCH PIPELINE (v2)
================================================================================
 Yabancı Reels videolarını TOPLU olarak Türkçe seslendirir + anime stiline çevirir.

 MİMARİ:
   FAZ 1 (ses)   : Tüm videoların sesi CUDA:2'de sırayla (Whisper + edge-tts)
   FAZ 2 (anime) : Videolar 3 GPU'ya DAĞITILIR — her kart kendi videolarını
                   paralel işler (AnimeGANv3 ONNX). 3 video = 3 kat hız.
   FAZ 3 (mux)   : Anime video + Türkçe ses birleştirme

 AnimateDiff/ComfyUI YOK. Titremesiz, oran korumalı, sürüm-savaşsız.

--------------------------------------------------------------------------------
 KURULUM: (bkz. önceki pipeline.py başlığı — aynı venv)
   onnxruntime-gpu==1.19.2, opencv-python-headless, faster-whisper, edge-tts,
   google-genai, pydub, requests, tqdm + nvidia-*-cu12 kütüphaneleri
   Model: models/AnimeGANv3_Shinkai_37.onnx (veya Hayao)
--------------------------------------------------------------------------------
 KULLANIM:
   export GEMINI_API_KEY="..."

   # Tüm klasör, Shinkai stili, 3 GPU paralel:
   python pipeline.py --input-dir inputs --model models/AnimeGANv3_Shinkai_37.onnx

   # Kullanılacak GPU'ları seç (varsayılan 0,1,2):
   python pipeline.py --input-dir inputs --gpus 0,1,2

   # Belirli adımlar:
   python pipeline.py --input-dir inputs --steps transcribe,translate,voice
   python pipeline.py --input-dir inputs --steps anime,mux
================================================================================
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests


# ==============================================================================
# AYARLAR
# ==============================================================================

CONFIG = {
    "WHISPER_DEVICE_INDEX": 2,
    "WHISPER_MODEL": "large-v3",
    "WHISPER_COMPUTE_TYPE": "float16",

    "GEMINI_MODEL": "gemini-2.5-flash",
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),

    "TTS_ENGINE": "f5",              # "edge" | "f5"
    "EDGE_TTS_VOICE": "tr-TR-EmelNeural",

    # --- F5-TTS (klonlama) ayarları ---
    "F5_PYTHON": "/home/atos/anime-reels/run_f5.sh",              # f5-env'in python'u
    "F5_GENERATE_SCRIPT": "/home/atos/anime-reels/f5_generate.py",
    "F5_CKPT": "f5_models/f5_tts_turkish.safetensors",
    "F5_VOCAB": "f5_models/vocab.txt",
    "F5_REF_AUDIO": "f5_models/ref_voice.wav",
    "F5_REF_TEXT": ("İşte bu harika bir ses baylar. Yerinizde olsam bunu "
                    "yapmazdım. Beş kilometre çapında her şeyi buhar eder. "
                    "Bazıları yapmaz. Simsiyah yanmış."),
    "F5_NFE_STEP": 32,                             # 16 hızlı, 32 iyi, 64 en iyi
    "TTS_MAX_SPEEDUP": 1.35,
    "VOICE_SAMPLE_MIN_SEC": 3.0,
    "VOICE_SAMPLE_MAX_SEC": 5.0,

    "ANIME_MODEL": "models/whitebox_cartoon.onnx",
    "ANIME_GPUS": [0, 1, 2],       # Render için kullanılacak GPU'lar

    "VIDEO_EXTS": [".mp4", ".mov", ".mkv", ".webm", ".avi"],
    "WORK_DIR": "work",
    "OUTPUT_DIR": "output",
}

ALL_STEPS = ["transcribe", "translate", "voice", "anime", "mux"]

# Thread'ler log yazarken karışmasın diye kilit
_log_lock = threading.Lock()


# ==============================================================================
# LOGLAMA
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("anime-reels")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ==============================================================================
# FFMPEG
# ==============================================================================

def run_ffmpeg(args: list, logger: logging.Logger):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg hatası: {r.stderr.strip()[:400]}")


def video_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


# ==============================================================================
# FAZ 1a — TRANSKRİPSİYON
# ==============================================================================

class Transcriber:
    def __init__(self, logger):
        self.logger = logger
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self.logger.info("Whisper %s yükleniyor (CUDA:%d)...",
                             CONFIG["WHISPER_MODEL"], CONFIG["WHISPER_DEVICE_INDEX"])
            self._model = WhisperModel(
                CONFIG["WHISPER_MODEL"], device="cuda",
                device_index=CONFIG["WHISPER_DEVICE_INDEX"],
                compute_type=CONFIG["WHISPER_COMPUTE_TYPE"])
        return self._model

    def run(self, video_path: Path, work_dir: Path) -> dict:
        out = work_dir / "1_transcript.json"
        if out.exists():
            self.logger.info("  [%s] transcript var, atlanıyor.", video_path.stem)
            return json.loads(out.read_text(encoding="utf-8"))
        audio = work_dir / "original_audio.wav"
        run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                    str(audio)], self.logger)
        model = self._get_model()
        seg_iter, info = model.transcribe(str(audio), beam_size=5, vad_filter=True)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": s.text.strip()} for s in seg_iter]
        result = {"language": info.language, "duration": round(info.duration, 2),
                  "segments": segments}
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        self.logger.info("  [%s] transkript: %d segment, dil=%s",
                         video_path.stem, len(segments), info.language)
        return result

    def cleanup(self):
        if self._model is not None:
            del self._model
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass


# ==============================================================================
# FAZ 1b — ÇEVİRİ
# ==============================================================================

def translate(transcript, work_dir, logger, tag=""):
    out = work_dir / "2_translation.json"
    if out.exists():
        logger.info("  [%s] çeviri var, atlanıyor.", tag)
        return json.loads(out.read_text(encoding="utf-8"))
    if not CONFIG["GEMINI_API_KEY"]:
        raise RuntimeError("GEMINI_API_KEY tanımlı değil!")
    segs = transcript["segments"]
    if not segs:
        raise RuntimeError("Transkript boş.")
    system_prompt = (
        "Sen profesyonel bir video lokalizasyon uzmanısın. Zaman damgalı Reels "
        "transkriptini Türkçeye çevir.\nKURALLAR:\n"
        "1. start/end ASLA değişmez.\n2. Segment sayısı korunur.\n"
        "3. Çeviri segment süresine sığsın; gerekirse kısalt.\n"
        "4. Doğal, akıcı, günlük Türkçe.\n"
        "5. SADECE geçerli JSON. Markdown/```/açıklama YOK.\n"
        'FORMAT: {"segments":[{"start":0.0,"end":2.5,"text":"..."}]}')
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{CONFIG['GEMINI_MODEL']}:generateContent")
    body = {"system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(
                {"segments": segs}, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0.4,
                                 "response_mime_type": "application/json"}}
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = requests.post(url, params={"key": CONFIG["GEMINI_API_KEY"]},
                                 json=body, timeout=120)
            resp.raise_for_status()
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = raw.replace("```json", "").replace("```", "").strip()
            tr = json.loads(raw)
            if len(tr.get("segments", [])) != len(segs):
                raise ValueError("Segment sayısı uyuşmuyor.")
            out.write_text(json.dumps(tr, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            logger.info("  [%s] çeviri: %d segment", tag, len(segs))
            return tr
        except (requests.RequestException, KeyError, ValueError,
                json.JSONDecodeError) as e:
            last_err = e
            logger.warning("  [%s] Gemini denemesi %d/3: %s", tag, attempt, e)
            time.sleep(3 * attempt)
    raise RuntimeError(f"Gemini çevirisi başarısız: {last_err}")


# ==============================================================================
# FAZ 1c — SESLENDİRME (edge-tts)
# ==============================================================================

def _extract_sample(work_dir, transcript, logger):
    audio = work_dir / "original_audio.wav"
    mn, mx = CONFIG["VOICE_SAMPLE_MIN_SEC"], CONFIG["VOICE_SAMPLE_MAX_SEC"]
    cands = sorted(transcript["segments"], key=lambda s: s["end"] - s["start"],
                   reverse=True)
    best = next((s for s in cands if s["end"] - s["start"] >= mn),
                cands[0] if cands else None)
    if best is None:
        raise RuntimeError("Ses örneği için segment yok.")
    sample = work_dir / "3_voice_sample.wav"
    run_ffmpeg(["-i", str(audio), "-ss", f"{best['start']:.2f}", "-t",
                f"{min(best['end']-best['start'], mx):.2f}", "-ac", "1",
                "-ar", "22050", str(sample)], logger)
    return sample


def _tts_edge(segments, seg_dir, logger):
    import edge_tts

    async def synth(text, out):
        await edge_tts.Communicate(text, CONFIG["EDGE_TTS_VOICE"]).save(str(out))

    files = []
    for i, seg in enumerate(segments):
        mp3 = seg_dir / f"seg_{i:03d}.mp3"
        wav = seg_dir / f"seg_{i:03d}.wav"
        asyncio.run(synth(seg["text"], mp3))
        run_ffmpeg(["-i", str(mp3), "-ar", "22050", "-ac", "1", str(wav)], logger)
        mp3.unlink(missing_ok=True)
        files.append(wav)
    return files


def _voice_f5(segs, transcript, video_path, work_dir, out, logger, tag):
    """F5-TTS ile klonlanmış ses üretir (f5-env'de subprocess olarak)."""
    total = video_duration(video_path)
    job = {
        "segments": segs,
        "total_duration": total,
        "ref_audio": str(Path(CONFIG["F5_REF_AUDIO"]).resolve()),
        "ref_text": CONFIG["F5_REF_TEXT"],
        "ckpt_file": str(Path(CONFIG["F5_CKPT"]).resolve()),
        "vocab_file": str(Path(CONFIG["F5_VOCAB"]).resolve()),
        "output": str(out.resolve()),
        "nfe_step": CONFIG["F5_NFE_STEP"],
        "max_speedup": CONFIG["TTS_MAX_SPEEDUP"],
    }
    job_file = work_dir / "f5_job.json"
    job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

    f5_python = str(Path(CONFIG["F5_PYTHON"]).resolve())
    f5_script = str(Path(CONFIG["F5_GENERATE_SCRIPT"]).resolve())
    logger.info("  [%s] F5-TTS klonlama başlıyor (%d segment)...", tag, len(segs))

    # F5 python'u TAM İZOLE çağır:
    #  1) -E : PYTHON* ortam değişkenlerini yoksay
    #  2) -s : user site-packages'ı yoksay
    #  3) env: PYTHONPATH/PYTHONHOME/VIRTUAL_ENV temizlenmiş kopya ortam
    # Böylece ana venv'in hiçbir etkisi f5-env'i bozamaz.
    import os as _os
    clean_env = {k: v for k, v in _os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
                              "PYTHONSTARTUP", "PYTHONNOUSERSITE")}
    proc = subprocess.run([f5_python, "--job",
                           str(job_file)],
                          capture_output=True, text=True, env=clean_env)
    if proc.returncode != 0:
        raise RuntimeError(f"F5-TTS hatası: {proc.stderr.strip()[:500]}")
    if not out.exists():
        raise RuntimeError("F5-TTS çıktı üretmedi.")


def voice(translation, transcript, video_path, work_dir, logger, tag=""):
    out = work_dir / "3_turkish_audio.wav"
    if out.exists():
        logger.info("  [%s] Türkçe ses var, atlanıyor.", tag)
        return out
    segs = [s for s in translation["segments"] if s["text"].strip()]

    if CONFIG["TTS_ENGINE"] == "f5":
        _voice_f5(segs, transcript, video_path, work_dir, out, logger, tag)
    else:
        from pydub import AudioSegment
        seg_dir = work_dir / "tts_segments"
        seg_dir.mkdir(exist_ok=True)
        _extract_sample(work_dir, transcript, logger)
        seg_files = _tts_edge(segs, seg_dir, logger)
        total = video_duration(video_path)
        timeline = AudioSegment.silent(duration=int(total * 1000),
                                       frame_rate=22050)
        for seg, wav in zip(segs, seg_files):
            clip = AudioSegment.from_wav(wav)
            slot = (seg["end"] - seg["start"]) * 1000
            if len(clip) > slot and slot > 200:
                speed = min(len(clip) / slot, CONFIG["TTS_MAX_SPEEDUP"])
                if speed > 1.02:
                    fast = wav.with_name(wav.stem + "_fit.wav")
                    run_ffmpeg(["-i", str(wav), "-filter:a",
                                f"atempo={speed:.3f}", str(fast)], logger)
                    clip = AudioSegment.from_wav(fast)
            timeline = timeline.overlay(clip, position=int(seg["start"] * 1000))
        timeline.export(out, format="wav")

    run_ffmpeg(["-i", str(video_path), "-i", str(out), "-map", "0:v", "-map",
                "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(work_dir / "3_preview_turkish.mp4")], logger)
    logger.info("  [%s] Türkçe ses üretildi.", tag)
    return out


# ==============================================================================
# FAZ 2 — ANIME FİLTRE (GPU başına bir worker)
# ==============================================================================

class AnimeWorker:
    """
    Belirli bir GPU'da çalışan stilizasyon ONNX oturumu.
    İki model ailesini otomatik algılar:
      - AnimeGANv3 : NHWC girdi, dinamik boyut (32'nin katı)
      - White-box  : NCHW girdi [1,3,720,720], NHWC çıktı, sabit boyut
    """

    def __init__(self, model_path: str, gpu: int, logger):
        import onnxruntime as ort
        self.logger = logger
        self.gpu = gpu
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = [("CUDAExecutionProvider", {"device_id": gpu}),
                         "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            logger.warning("  [GPU%d] CUDA yok — CPU modu (yavaş)", gpu)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=providers)
        in_meta = self.sess.get_inputs()[0]
        self.inp = in_meta.name
        self.outp = self.sess.get_outputs()[0].name
        shape = in_meta.shape  # ör. [1,3,720,720] (NCHW) veya [1,H,W,3] (NHWC)

        # Model tipini şekle bakarak algıla
        # NCHW ise 2. eleman (index 1) kanal sayısı (3) olur
        self.is_nchw = (len(shape) == 4 and shape[1] == 3)
        # Sabit boyutlu mu? (720 gibi tam sayı) yoksa dinamik mi?
        if self.is_nchw:
            self.fixed_h = shape[2] if isinstance(shape[2], int) else None
            self.fixed_w = shape[3] if isinstance(shape[3], int) else None
        else:
            self.fixed_h = shape[1] if isinstance(shape[1], int) else None
            self.fixed_w = shape[2] if isinstance(shape[2], int) else None

        with _log_lock:
            logger.info("  [GPU%d] Model formatı: %s, sabit boyut: %sx%s",
                        gpu, "NCHW" if self.is_nchw else "NHWC",
                        self.fixed_w, self.fixed_h)

    @staticmethod
    def _letterbox(bgr, tw, th):
        """Oranı koruyarak tw x th kareye sığdırır; boşlukları siyahla doldurur.
        Geri dönüşüm için ölçek ve offset bilgisini de döndürür."""
        oh, ow = bgr.shape[:2]
        scale = min(tw / ow, th / oh)
        nw, nh = int(round(ow * scale)), int(round(oh * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=bgr.dtype)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas, scale, ox, oy, nw, nh

    def _frame(self, bgr):
        oh, ow = bgr.shape[:2]

        if self.fixed_h and self.fixed_w:
            # SABİT boyut (White-box 720x720): letterbox ile oran koru
            canvas, scale, ox, oy, nw, nh = self._letterbox(
                bgr, self.fixed_w, self.fixed_h)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = rgb / 127.5 - 1.0
            if self.is_nchw:
                inp = np.transpose(rgb, (2, 0, 1))[None, ...]  # HWC->CHW, batch
            else:
                inp = rgb[None, ...]
            out = self.sess.run([self.outp], {self.inp: inp})[0]
            out = np.squeeze(out)
            # Çıktı NHWC beklenir (White-box öyle); değilse çevir
            if out.ndim == 3 and out.shape[0] == 3:
                out = np.transpose(out, (1, 2, 0))
            out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # Letterbox'ı geri sök: dolgu alanını at, orijinal boyuta büyüt
            out = out[oy:oy + nh, ox:ox + nw]
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC)
            return out

        # DİNAMİK boyut (AnimeGANv3): 32'nin katına yuvarla, NHWC
        ph, pw = (oh // 32) * 32, (ow // 32) * 32
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 127.5 - 1.0)[None, ...]
        out = self.sess.run([self.outp], {self.inp: img})[0]
        out = np.clip((np.squeeze(out) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if (out.shape[1], out.shape[0]) != (ow, oh):
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC)
        return out

    def process(self, video_path: Path, work_dir: Path) -> Path:
        out = work_dir / "4_anime_noaudio.mp4"
        if out.exists():
            with _log_lock:
                self.logger.info("  [GPU%d] %s: anime var, atlanıyor.",
                                 self.gpu, video_path.stem)
            return out
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with _log_lock:
            self.logger.info("  [GPU%d] %s: %dx%d, %.0f fps, %d kare -> başladı",
                             self.gpu, video_path.stem, w, h, fps, total)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        t0, n = time.time(), 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(self._frame(fr))
            n += 1
        cap.release()
        writer.release()
        with _log_lock:
            self.logger.info("  [GPU%d] %s: %d kare bitti (%.1f kare/sn)",
                             self.gpu, video_path.stem, n,
                             n / (time.time() - t0) if n else 0)
        return out


def anime_phase(videos, work_dirs, model_path, gpus, logger):
    """Videoları GPU havuzuna dağıtarak paralel anime render eder."""
    # Her GPU için bir worker (kendi ONNX oturumu)
    workers = {}
    for g in gpus:
        logger.info("  [GPU%d] AnimeGANv3 oturumu açılıyor...", g)
        workers[g] = AnimeWorker(model_path, g, logger)

    # Videoları GPU'lara round-robin dağıt
    tasks = []  # (gpu, video, work_dir)
    for i, v in enumerate(videos):
        g = gpus[i % len(gpus)]
        tasks.append((g, v, work_dirs[v]))

    def do_task(task):
        g, v, wd = task
        return workers[g].process(v, wd)

    # Her GPU aynı anda 1 video işler -> max_workers = GPU sayısı
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {pool.submit(do_task, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            g, v, _ = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error("  [GPU%d] %s BAŞARISIZ: %s", g, v.stem, e)


# ==============================================================================
# FAZ 3 — MUX
# ==============================================================================

def mux(anime_video, audio, video_path, out_dir, base, logger):
    out_dir.mkdir(exist_ok=True)
    final = out_dir / f"{base}_anime_TR.mp4"
    if audio and audio.exists():
        run_ffmpeg(["-i", str(anime_video), "-i", str(audio), "-map", "0:v",
                    "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-preset",
                    "medium", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(final)], logger)
    else:
        run_ffmpeg(["-i", str(anime_video), "-i", str(video_path), "-map",
                    "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                    "-preset", "medium", "-c:a", "aac", "-shortest",
                    str(final)], logger)
    logger.info("  mux -> %s", final.name)
    return final


# ==============================================================================
# ORKESTRASYON
# ==============================================================================

def collect_videos(args):
    vids = []
    if args.input_dir:
        for p in sorted(Path(args.input_dir).iterdir()):
            if p.suffix.lower() in CONFIG["VIDEO_EXTS"]:
                vids.append(p.resolve())
    for f in args.input or []:
        p = Path(f).resolve()
        if p.exists():
            vids.append(p)
    seen, uniq = set(), []
    for v in vids:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def main():
    ap = argparse.ArgumentParser(description="Anime-Reels çok-GPU batch pipeline")
    ap.add_argument("--input-dir")
    ap.add_argument("--input", nargs="*")
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    ap.add_argument("--model", default=CONFIG["ANIME_MODEL"])
    ap.add_argument("--gpus", default=",".join(map(str, CONFIG["ANIME_GPUS"])),
                    help="Anime render GPU'ları, ör: 0,1,2")
    args = ap.parse_args()

    videos = collect_videos(args)
    if not videos:
        print("HATA: video yok (--input-dir veya --input ver).")
        sys.exit(1)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if set(steps) - set(ALL_STEPS):
        print(f"HATA: Geçersiz adım. Geçerli: {ALL_STEPS}")
        sys.exit(1)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    work_root = Path(CONFIG["WORK_DIR"])
    work_root.mkdir(exist_ok=True)
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    logger = setup_logging(work_root / "pipeline.log")

    work_dirs = {}
    for v in videos:
        d = work_root / v.stem
        d.mkdir(parents=True, exist_ok=True)
        work_dirs[v] = d

    logger.info("=" * 60)
    logger.info("ANIME-REELS v2: %d video | GPU'lar: %s | Adımlar: %s",
                len(videos), gpus, steps)
    logger.info("Model: %s", Path(args.model).name)
    logger.info("=" * 60)

    t0 = time.time()
    try:
        # ---------- FAZ 1: SES (sıralı, CUDA:2) ----------
        if any(s in steps for s in ("transcribe", "translate", "voice")):
            logger.info(">>> FAZ 1: SES (%d video, CUDA:%d) <<<",
                        len(videos), CONFIG["WHISPER_DEVICE_INDEX"])
            transcriber = Transcriber(logger)
            for i, v in enumerate(videos, 1):
                d = work_dirs[v]
                logger.info("--- Ses %d/%d: %s ---", i, len(videos), v.stem)
                try:
                    tr = ts = None
                    if "transcribe" in steps:
                        ts = transcriber.run(v, d)
                    if "translate" in steps:
                        ts = ts or json.loads(
                            (d / "1_transcript.json").read_text(encoding="utf-8"))
                        tr = translate(ts, d, logger, v.stem)
                    if "voice" in steps:
                        ts = ts or json.loads(
                            (d / "1_transcript.json").read_text(encoding="utf-8"))
                        tr = tr or json.loads(
                            (d / "2_translation.json").read_text(encoding="utf-8"))
                        voice(tr, ts, v, d, logger, v.stem)
                except Exception as e:
                    logger.error("Ses başarısız (%s): %s — sonraki videoya",
                                 v.stem, e)
            transcriber.cleanup()
            logger.info(">>> FAZ 1 bitti. <<<")

        # ---------- FAZ 2: ANIME (3 GPU paralel) ----------
        if "anime" in steps:
            logger.info(">>> FAZ 2: ANIME (%d GPU paralel) <<<", len(gpus))
            anime_phase(videos, work_dirs, args.model, gpus, logger)
            logger.info(">>> FAZ 2 bitti. <<<")

        # ---------- FAZ 3: MUX ----------
        if "mux" in steps:
            logger.info(">>> FAZ 3: MUX <<<")
            for v in videos:
                d = work_dirs[v]
                anime_vid = d / "4_anime_noaudio.mp4"
                audio = d / "3_turkish_audio.wav"
                if not anime_vid.exists():
                    logger.warning("  mux atlandı (%s): anime yok.", v.stem)
                    continue
                mux(anime_vid, audio if audio.exists() else None, v, out_dir,
                    v.stem, logger)
            logger.info(">>> FAZ 3 bitti. <<<")

        logger.info("=" * 60)
        logger.info("BİTTİ. Toplam: %.1f dk | Çıktılar: %s/",
                    (time.time() - t0) / 60, out_dir)

    except KeyboardInterrupt:
        logger.warning("İptal edildi.")
        sys.exit(130)
    except Exception as e:
        logger.error("HATA: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

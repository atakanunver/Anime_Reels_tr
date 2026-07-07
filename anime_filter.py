#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ANIME FİLTRE MODÜLÜ (AnimeGANv3 + ONNX Runtime)
================================================================================
 AnimateDiff/ComfyUI'a ALTERNATİF. Videoyu yeniden ÜRETMEZ, üstüne anime STİLİ
 uygular. Bu sayede:
   - Titreme YOK (her kare aynı deterministik modelden geçer)
   - En-boy oranı KORUNUR (kırpma yok)
   - Senkron BOZULMAZ (kare sayısı değişmez)
   - Yazılar büyük ölçüde KORUNUR
   - torch/transformers sürüm çakışması YOK (ONNX Runtime bağımsız çalışır)

--------------------------------------------------------------------------------
 KURULUM (pipeline venv'inde ya da AYRI bir venv'de):
--------------------------------------------------------------------------------
   python -m pip install onnxruntime-gpu opencv-python-headless numpy tqdm

   # ONNX Runtime, torch'tan BAĞIMSIZ kendi CUDA'sını kullanır.
   # GPU görünmezse (CUDAExecutionProvider yoksa) otomatik CPU'ya düşer;
   # CPU çok yavaştır, GPU şart.

   # MODEL İNDİR (Hayao / Miyazaki stili - en popüler):
   mkdir -p ~/reels/anime_models
   cd ~/reels/anime_models
   wget -c https://github.com/TachibanaYoshino/AnimeGANv3/raw/master/deploy/AnimeGANv3_Hayao_36.onnx

   # Alternatif stiller (aynı klasöre indirip --model ile seç):
   #   AnimeGANv3_Shinkai_37.onnx     (Makoto Shinkai - canlı renkler)
   #   AnimeGANv3_Arcane.onnx         (Arcane dizisi stili)
   #   AnimeGANv3_JP_face_v1.0.onnx   (Japon anime yüz)

--------------------------------------------------------------------------------
 KULLANIM:
--------------------------------------------------------------------------------
   # Tek video:
   python anime_filter.py --input video.mp4 --model ~/reels/anime_models/AnimeGANv3_Hayao_36.onnx

   # Türkçe sesle birleştir (pipeline'ın ürettiği sesi kullan):
   python anime_filter.py --input video.mp4 \
       --model ~/reels/anime_models/AnimeGANv3_Hayao_36.onnx \
       --audio work/video/3_turkish_audio.wav

   # Klasördeki tüm videolar:
   python anime_filter.py --input-dir ~/reels/inputs \
       --model ~/reels/anime_models/AnimeGANv3_Hayao_36.onnx --use-pipeline-audio

   # GPU seç (varsayılan 0):
   python anime_filter.py --input video.mp4 --model ... --gpu 1
================================================================================
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ==============================================================================
# LOGLAMA
# ==============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("anime")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


# ==============================================================================
# ANIMEGANV3 ONNX MOTORU
# ==============================================================================

class AnimeGANv3:
    """AnimeGANv3 ONNX modelini tek kare üzerinde çalıştırır."""

    def __init__(self, model_path: str, gpu: int, logger: logging.Logger):
        import onnxruntime as ort
        self.logger = logger

        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = [("CUDAExecutionProvider", {"device_id": gpu}),
                         "CPUExecutionProvider"]
            logger.info("ONNX Runtime GPU modu (CUDA:%d)", gpu)
        else:
            providers = ["CPUExecutionProvider"]
            logger.warning("CUDA sağlayıcısı yok — CPU modu (ÇOK YAVAŞ). "
                           "'pip install onnxruntime-gpu' kurulu mu?")

        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        # AnimeGANv3 modelleri boyutun 32'nin katı olmasını ister
        logger.info("Model yüklendi: %s", Path(model_path).name)

    @staticmethod
    def _to_32x(h: int, w: int) -> tuple:
        """Boyutu 32'nin katına yuvarlar (AnimeGANv3 gereksinimi)."""
        return (h // 32) * 32, (w // 32) * 32

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Tek BGR kareyi anime stiline çevirir, orijinal boyuta geri döndürür."""
        orig_h, orig_w = frame_bgr.shape[:2]
        proc_h, proc_w = self._to_32x(orig_h, orig_w)

        # BGR -> RGB, boyut ayarla, [-1, 1] normalize
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

        # Inference
        out = self.sess.run([self.out_name], {self.inp_name: img})[0]

        # [-1, 1] -> [0, 255], RGB -> BGR, orijinal boyuta geri
        out = (np.squeeze(out) + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if (out.shape[1], out.shape[0]) != (orig_w, orig_h):
            out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        return out


# ==============================================================================
# VİDEO İŞLEME
# ==============================================================================

def get_video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def run_ffmpeg(args: list, logger: logging.Logger):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg hatası: {r.stderr.strip()[:400]}")


def stylize_video(engine: AnimeGANv3, video_path: Path, out_dir: Path,
                  audio_path: Path, logger: logging.Logger) -> Path:
    """Videoyu kare kare anime stiline çevirir, sesi (varsa) birleştirir."""
    from tqdm import tqdm

    info = get_video_info(video_path)
    logger.info("Video: %dx%d, %.1f fps, %d kare",
                info["width"], info["height"], info["fps"], info["frames"])

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_video = out_dir / f"{video_path.stem}_anime_noaudio.mp4"

    # OpenCV VideoWriter (mp4v). Ses YOK — sonra ffmpeg ile eklenecek.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, info["fps"],
                             (info["width"], info["height"]))

    cap = cv2.VideoCapture(str(video_path))
    t0 = time.time()
    n = 0
    with tqdm(total=info["frames"], desc="Anime stili", unit="kare") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            styled = engine.process_frame(frame)
            writer.write(styled)
            n += 1
            bar.update(1)
    cap.release()
    writer.release()

    fps_proc = n / (time.time() - t0) if n else 0
    logger.info("%d kare işlendi (%.1f kare/sn)", n, fps_proc)

    # Ses birleştir
    final = out_dir / f"{video_path.stem}_anime.mp4"
    if audio_path and audio_path.exists():
        logger.info("Ses birleştiriliyor: %s", audio_path.name)
        run_ffmpeg(["-i", str(temp_video), "-i", str(audio_path),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(final)], logger)
        temp_video.unlink(missing_ok=True)
    else:
        # Ses yoksa orijinal videonun sesini koru (varsa)
        logger.info("Harici ses yok — orijinal ses (varsa) korunuyor.")
        run_ffmpeg(["-i", str(temp_video), "-i", str(video_path),
                    "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-c:a", "aac", "-shortest", str(final)], logger)
        temp_video.unlink(missing_ok=True)

    logger.info("BİTTİ -> %s", final)
    return final


# ==============================================================================
# ANA
# ==============================================================================

VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".webm", ".avi"]


def main():
    ap = argparse.ArgumentParser(description="AnimeGANv3 anime filtre (ONNX)")
    ap.add_argument("--input", help="Tek video dosyası")
    ap.add_argument("--input-dir", help="Video klasörü")
    ap.add_argument("--model", required=True, help="AnimeGANv3 .onnx yolu")
    ap.add_argument("--audio", help="Birleştirilecek ses (tek video için)")
    ap.add_argument("--use-pipeline-audio", action="store_true",
                    help="work/<video>/3_turkish_audio.wav sesini otomatik kullan")
    ap.add_argument("--out-dir", default="output_anime", help="Çıktı klasörü")
    ap.add_argument("--gpu", type=int, default=0, help="GPU indeksi (varsayılan 0)")
    args = ap.parse_args()

    logger = setup_logging()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model bulunamadı: %s", model_path)
        logger.error("İndir: cd ~/reels/anime_models && wget -c "
                     "https://github.com/TachibanaYoshino/AnimeGANv3/raw/"
                     "master/deploy/AnimeGANv3_Hayao_36.onnx")
        sys.exit(1)

    # Video listesi
    videos = []
    if args.input_dir:
        for p in sorted(Path(args.input_dir).iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append(p.resolve())
    if args.input:
        p = Path(args.input).resolve()
        if p.exists():
            videos.append(p)
    if not videos:
        logger.error("İşlenecek video yok (--input veya --input-dir ver).")
        sys.exit(1)

    engine = AnimeGANv3(str(model_path), args.gpu, logger)
    out_dir = Path(args.out_dir)

    t0 = time.time()
    for i, v in enumerate(videos, 1):
        logger.info("=" * 55)
        logger.info("Video %d/%d: %s", i, len(videos), v.name)
        logger.info("=" * 55)

        # Ses kaynağını belirle
        audio = None
        if args.audio:
            audio = Path(args.audio)
        elif args.use_pipeline_audio:
            cand = Path("work") / v.stem / "3_turkish_audio.wav"
            if cand.exists():
                audio = cand
            else:
                logger.warning("Pipeline sesi yok (%s) — orijinal ses kullanılacak.",
                               cand)

        try:
            stylize_video(engine, v, out_dir, audio, logger)
        except Exception as e:
            logger.error("Video başarısız (%s): %s", v.name, e)

    logger.info("TÜM İŞLER BİTTİ. Toplam: %.1f dk", (time.time() - t0) / 60)
    logger.info("Çıktılar: %s/", out_dir)


if __name__ == "__main__":
    main()

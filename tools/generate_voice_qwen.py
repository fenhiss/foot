#!/usr/bin/env python3
"""
Озвучка комментатора нейросетью Qwen3-TTS (https://github.com/QwenLM/Qwen3-TTS, Apache-2.0).

Голос комментатора («низкий, тёплый, расслабленный мужской голос») создан моделью
Qwen3-TTS-12Hz-1.7B-VoiceDesign по текстовому описанию. Это синтетический голос,
а не копия реального человека. Образцы лежат в tools/voice_ref/:
  calm.flac    — спокойная подача (обычные реплики)
  excited.flac — тот же голос на эмоциях (голы, удары, победа)
Все фразы озвучиваются моделью Qwen3-TTS-12Hz-1.7B-Base, которая клонирует этот голос,
поэтому тембр одинаковый во всех репликах.

Каждая фраза проверяется распознаванием речи (faster-whisper). Если слова съедены
или фраза получилась слишком длинной (зависание), она перегенерируется.

Запуск (CPU, ~30–40 минут; на GPU в разы быстрее):
    pip install qwen-tts faster-whisper
    python3 tools/generate_voice_qwen.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --whisper Systran/faster-whisper-small
"""
import argparse
import base64
import difflib
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
from generate_voice import build_items  # noqa: E402  (общий список фраз)

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = {
    "calm": (os.path.join(HERE, "voice_ref", "calm.flac"),
             "Добрый вечер, друзья! Мы на стадионе, трибуны полны, и через секунду начинается матч. "
             "Нортсайд против Ривергейта — будет жарко!"),
    "excited": (os.path.join(HERE, "voice_ref", "excited.flac"),
                "Гол! Гол! Какой удар! Мяч в сетке, стадион просто взрывается! Это невероятно, друзья!"),
}
STYLE_REF = {"calm": "calm", "lively": "calm", "excited": "excited"}
NUMWORDS = {"ноль": "0", "один": "1", "два": "2", "три": "3", "четыре": "4", "пять": "5", "шесть": "6"}


def norm(t):
    t = t.lower().replace("ё", "е")
    for w, d in NUMWORDS.items():
        t = re.sub(rf"\b{w}\b", d, t)
    return re.sub(r"[^a-zа-я0-9]+", "", t)


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def to_mp3(wav, sr):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    af = ",".join([
        "highpass=f=60",
        "silenceremove=start_periods=1:start_threshold=-45dB",
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse",
        "acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:makeup=3",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ])
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "wav", "-i", "-", "-af", af, "-ac", "1", "-ar", "24000",
         "-c:a", "libmp3lame", "-b:a", "56k", "-f", "mp3", "-"],
        input=buf.getvalue(), check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--whisper", default="Systran/faster-whisper-small")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "voice", "voice-pack.js"))
    ap.add_argument("--mp3-dir", help="дополнительно сохранить mp3 сюда (для прослушивания)")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--only", help="регулярка: генерировать только эти id (остальные берутся из --out)")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel
    from faster_whisper import WhisperModel

    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=device, dtype=dtype)
    asr = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    prompts = {k: tts.create_voice_clone_prompt(ref_audio=p, ref_text=t) for k, (p, t) in REFS.items()}

    pack = {}
    if args.only and os.path.exists(args.out):
        src = open(args.out, encoding="utf-8").read()
        pack = json.loads(src[src.index("=") + 1:src.rstrip().rindex(";")])

    items = build_items()
    if args.only:
        items = [it for it in items if re.search(args.only, it[0])]
    for n, (cid, style, text) in enumerate(items, 1):
        best = None
        for attempt in range(args.tries):
            wavs, sr = tts.generate_voice_clone(text=text, language="Russian",
                                                voice_clone_prompt=prompts[STYLE_REF[style]])
            wav = np.asarray(wavs[0], dtype=np.float32)
            dur = len(wav) / sr
            w16 = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                                  "-ar", "16000", "-f", "f32le", "-"], input=wav.tobytes(),
                                 check=True, capture_output=True).stdout
            segs, _ = asr.transcribe(np.frombuffer(w16, np.float32), language="ru", beam_size=5)
            heard = " ".join(s.text.strip() for s in segs)
            sim = similarity(text, heard)
            too_long = dur > 1.5 + 0.12 * len(text)
            score = sim - (0.5 if too_long else 0)
            if best is None or score > best[0]:
                best = (score, wav, sr, heard, dur)
            if sim >= 0.8 and not too_long:
                break
        score, wav, sr, heard, dur = best
        mp3 = to_mp3(wav, sr)
        pack[cid] = base64.b64encode(mp3).decode()
        if args.mp3_dir:
            os.makedirs(args.mp3_dir, exist_ok=True)
            open(os.path.join(args.mp3_dir, cid + ".mp3"), "wb").write(mp3)
        flag = "OK " if score >= 0.8 else "?? "
        print(f"{flag}[{n}/{len(items)}] {cid} ({dur:.1f}s, {score:.2f}) {text}  =>  {heard}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("/* Сгенерировано tools/generate_voice_qwen.py (Qwen3-TTS, синтетический голос из tools/voice_ref). "
                "Не редактировать вручную. */\n")
        f.write("window.VOICE_PACK=" + json.dumps(pack, separators=(",", ":")) + ";\n")
    print("OK ->", os.path.abspath(args.out), f"{os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

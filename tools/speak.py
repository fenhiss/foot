#!/usr/bin/env python3
"""
Озвучка произвольного текста голосом комментатора (Qwen3-TTS, клонирование из tools/voice_ref).

Понимает теги в стиле ElevenLabs v3 в начале фразы или абзаца:
  [excited] [shouting] [happy] [energetic]   -> эмоциональный образец голоса
  [calm] [neutral] [serious] [whispers] ...  -> спокойный образец
Тег действует до следующего тега. Прочие теги ([laughs], [sighs]...) вырезаются.

    python3 tools/speak.py input.txt out.mp3 --model <Base> --whisper <faster-whisper>
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from generate_voice_qwen import REFS, similarity  # noqa: E402

EXCITED = {"excited", "shouting", "shouts", "happy", "energetic", "enthusiastic", "cheering", "joyful", "loud"}


def segments(text):
    style = "calm"
    out = []
    for part in re.split(r"(\[[^\]]+\])", text):
        m = re.fullmatch(r"\[([^\]]+)\]", part.strip())
        if m:
            tag = m.group(1).strip().lower()
            if tag in EXCITED:
                style = "excited"
            elif tag in {"calm", "neutral", "serious", "whispers", "sad", "soft", "relaxed", "warm"}:
                style = "calm"
            continue
        # Разбиваем на предложения, короткие склеиваем до ~220 символов
        sents = re.findall(r"[^.!?…]+[.!?…]*", part.replace("\n", " "))
        buf = ""
        for s in (x.strip() for x in sents):
            if not s:
                continue
            if buf and len(buf) + len(s) > 220:
                out.append((style, buf)); buf = ""
            buf = (buf + " " + s).strip()
        if buf:
            out.append((style, buf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--whisper", default="Systran/faster-whisper-small")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--pause", type=float, default=0.35, help="пауза между фразами, сек")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel
    from faster_whisper import WhisperModel
    torch.set_num_threads(os.cpu_count() or 4)
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    asr = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    prompts = {k: tts.create_voice_clone_prompt(ref_audio=p, ref_text=t) for k, (p, t) in REFS.items()}

    segs = segments(open(args.input, encoding="utf-8").read())
    pieces, sr = [], 24000
    for i, (style, text) in enumerate(segs, 1):
        best = None
        for _ in range(args.tries):
            wavs, sr = tts.generate_voice_clone(text=text, language="Russian", voice_clone_prompt=prompts[style])
            wav = np.asarray(wavs[0], dtype=np.float32)
            w16 = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                                  "-ar", "16000", "-f", "f32le", "-"], input=wav.tobytes(),
                                 check=True, capture_output=True).stdout
            heard = " ".join(s.text.strip() for s in asr.transcribe(np.frombuffer(w16, np.float32),
                                                                     language="ru", beam_size=5)[0])
            too_long = len(wav) / sr > 1.5 + 0.12 * len(text)
            score = similarity(text, heard) - (0.5 if too_long else 0)
            if best is None or score > best[0]:
                best = (score, wav, heard)
            if score >= 0.85:
                break
        score, wav, heard = best
        print(f"{'OK' if score >= 0.8 else '??'} [{i}/{len(segs)}] ({style}, {score:.2f}) {text}  =>  {heard}", flush=True)
        pieces += [wav, np.zeros(int(sr * args.pause), np.float32)]

    audio = np.concatenate(pieces)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-af", "highpass=f=60,acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:makeup=3,"
                           "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "44100", "-b:a", "128k", args.output], input=audio.tobytes(), check=True)
    print("OK ->", args.output, f"{len(audio) / sr:.1f} s")


if __name__ == "__main__":
    main()

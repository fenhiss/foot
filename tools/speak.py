#!/usr/bin/env python3
"""
Озвучка произвольного текста голосом комментатора (Qwen3-TTS, клонирование из tools/voice_ref).

Голос по умолчанию (--voice energetic): энергичный рассказчик в стиле шортсов/YouTube —
низкий голос с живыми, широкими интонациями (образец voice_ref/excited.flac) + ускорение темпа.
--voice classic — прежний спокойный низкий голос.

Понимает теги в стиле ElevenLabs v3, тег действует до следующего:
  [serious] [tense] [curious] [calm] ...    -> основной голос рассказчика
  [gruff]                                   -> хриплый голос тренера (персонаж)
  [pause] [short pause] [long pause]        -> паузы
Прочие теги ([laughs], [sighs]...) вырезаются.

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

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = dict(REFS)
VOICES = {  # стиль из тегов -> образец голоса
    "energetic": {"base": "excited", "excited": "excited", "gruff": "gruff"},
    "classic": {"base": "calm", "excited": "excited", "gruff": "gruff"},
}
REFS["gruff"] = (os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_ref", "gruff.flac"),
                 "Видишь? От борца не убегают. Борца встречают. Он устал. Теперь твоё время. Жди, когда он нырнёт.")

EXCITED = {"excited", "shouting", "shouts", "happy", "energetic", "enthusiastic", "cheering", "joyful", "loud",
           "hyped", "triumphant"}
CALM = {"calm", "neutral", "serious", "whispers", "whispering", "sad", "soft", "relaxed", "warm", "warmly",
        "curious", "deadpan", "sarcastic", "thoughtful", "tense", "determined", "intense", "urgent"}
GRUFF = {"gruff", "rough", "coach"}
PAUSES = {"pause": 0.35, "short pause": 0.2, "long pause": 0.8}  # плотный темп, без провисаний


def segments(text):
    style = "base"
    out = []
    for part in re.split(r"(\[[^\]]+\])", text):
        m = re.fullmatch(r"\[([^\]]+)\]", part.strip())
        if m:
            tag = m.group(1).strip().lower()
            if tag in PAUSES:
                out.append(("pause", PAUSES[tag]))
            elif tag in EXCITED:
                style = "excited"
            elif tag in GRUFF:
                style = "gruff"
            elif tag in CALM:
                style = "base"
            continue  # прочие теги ([laughs] и т.п.) пропускаем
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
    ap.add_argument("--pause", type=float, default=0.12, help="пауза между фразами, сек")
    ap.add_argument("--voice", choices=list(VOICES), default="energetic")
    ap.add_argument("--tempo", type=float, default=1.06, help="итоговое ускорение без изменения тона")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel
    from faster_whisper import WhisperModel
    torch.set_num_threads(os.cpu_count() or 4)
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    asr = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    vmap = VOICES[args.voice]
    prompts = {st: tts.create_voice_clone_prompt(ref_audio=REFS[r][0], ref_text=REFS[r][1]) for st, r in vmap.items()}

    segs = segments(open(args.input, encoding="utf-8").read())
    pieces, sr = [], 24000
    for i, (style, text) in enumerate(segs, 1):
        if style == "pause":
            pieces.append(np.zeros(int(sr * text), np.float32))
            continue
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
        np.save(args.output + ".partial.npy", np.concatenate(pieces))

    audio = np.concatenate(pieces)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-af", f"rubberband=tempo={args.tempo}:pitchq=quality:formant=preserved,"
                           "highpass=f=60,equalizer=f=3000:t=q:w=1.2:g=2,"
                           "acompressor=threshold=-24dB:ratio=3:attack=5:release=100:makeup=4,"
                           "loudnorm=I=-14:TP=-1.5:LRA=8",
                    "-ar", "44100", "-b:a", "128k", args.output], input=audio.tobytes(), check=True)
    print("OK ->", args.output, f"{len(audio) / sr:.1f} s")


if __name__ == "__main__":
    main()

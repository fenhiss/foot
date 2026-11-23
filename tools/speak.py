#!/usr/bin/env python3
"""
Озвучка произвольного текста голосом комментатора (Qwen3-TTS, клонирование из tools/voice_ref).

Голос по умолчанию (--voice casual): лёгкий, расслабленный разговорный рассказчик для шортсов —
живые широкие интонации, «с улыбкой» (синтетический голос из Qwen3-TTS VoiceDesign, voice_ref/casual.flac),
студийное звучание (эквалайзер + «воздух»). --voice energetic — низкий напористый вариант.
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
REFS["casual"] = (os.path.join(HERE, "voice_ref", "casual.flac"),
                  "Вы выбрали — ещё один бой! А Артур написал: ещё один бой, и только потом в профессиональную лигу. "
                  "Так и будет. Шутка. Просто похожи. Утром — алгебра. Вечером — лоу-кики. Что делаешь? Пиши в комментариях!")
VOICES = {  # стиль из тегов -> образец голоса
    "casual": {"base": "casual", "excited": "casual", "gruff": "gruff"},
    "energetic": {"base": "excited", "excited": "excited", "gruff": "gruff"},
    "classic": {"base": "calm", "excited": "excited", "gruff": "gruff"},
}
# Финальная обработка для каждого голоса: тон, темп, эквалайзер.
# Для casual — эквалайзер подогнан по усреднённому спектру референса в манере «casual, relaxed, light»:
# меньше гула в басу, больше разборчивой середины и «воздуха» сверху (эксайтер), как у студийной озвучки.
CASUAL_EQ = "entry(63,-3.3);entry(80,-5.2);entry(100,-5.9);entry(125,-6.1);entry(160,-6.4);entry(200,-5.4);entry(250,-2.9);entry(315,-0.9);entry(400,1.7);entry(500,4.2);entry(630,3.0);entry(800,0.8);entry(1000,2.2);entry(1250,5.4);entry(1600,5.8);entry(2000,4.7);entry(2500,5.1);entry(3150,4.2);entry(4000,1.0);entry(5000,-1.2);entry(6300,-2.6);entry(8000,-2.4);entry(10000,5.0);entry(12500,7.0);entry(16000,7.0)"
MASTER = {
    "casual": ("rubberband=pitch=0.93:tempo=1.08:formant=preserved:pitchq=quality,aresample=44100,"
               "aexciter=amount=1.5:drive=6:blend=0:freq=6000:ceil=16000,"
               f"firequalizer=gain_entry='{CASUAL_EQ}',"
               "acompressor=threshold=-24dB:ratio=3:attack=5:release=100:makeup=3,loudnorm=I=-16:TP=-1.5:LRA=9"),
    "energetic": ("rubberband=tempo=1.06:pitchq=quality:formant=preserved,highpass=f=60,equalizer=f=3000:t=q:w=1.2:g=2,"
                  "acompressor=threshold=-24dB:ratio=3:attack=5:release=100:makeup=4,loudnorm=I=-14:TP=-1.5:LRA=8"),
    "classic": ("highpass=f=60,acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:makeup=3,"
                "loudnorm=I=-16:TP=-1.5:LRA=11"),
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
    ap.add_argument("--voice", choices=list(VOICES), default="casual")
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
            wavs, sr = tts.generate_voice_clone(text=text, language="Russian", voice_clone_prompt=prompts[style],
                                                temperature=0.95)
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
                    "-af", MASTER[args.voice],
                    "-ar", "44100", "-b:a", "128k", args.output], input=audio.tobytes(), check=True)
    print("OK ->", args.output, f"{len(audio) / sr:.1f} s")


if __name__ == "__main__":
    main()

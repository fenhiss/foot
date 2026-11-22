#!/usr/bin/env python3
"""
Генератор озвучки комментатора для Football Career.

Использует Piper (https://github.com/rhasspy/piper) — открытый нейросетевой TTS,
русский голос из https://huggingface.co/rhasspy/piper-voices.
Все фразы синтезируются заранее, проходят обработку в ffmpeg
(компрессия, «эфирный» тембр, ускорение/подъём тона для эмоциональных реплик)
и упаковываются в voice/voice-pack.js (base64 MP3), чтобы игра работала
даже при открытии HTML-файла двойным кликом, без сервера.

Запуск:
    pip install piper-tts
    python3 tools/generate_voice.py --model path/to/ru_RU-ruslan-medium.onnx
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tempfile
import wave

from piper import PiperVoice, SynthesisConfig

# Названия команд по-русски (латиницу русский голос читает плохо)
TEAMS = {
    "RIV": "Ривергейт", "OLD": "Олдкасл", "EAS": "Истборо", "KIN": "Кингспорт",
    "WES": "Вестхейвен", "RED": "Редмонт", "LAK": "Лейксайд", "STO": "Стоунбридж",
    "HIG": "Хаймур", "SOU": "Саутпорт",
}
NUM = ["ноль", "один", "два", "три", "четыре", "пять", "шесть"]

# Стили подачи: скорость (length_scale < 1 — быстрее), выразительность,
# темп и подъём тона после синтеза.
STYLES = {
    # Голос ruslan от природы говорит быстро, поэтому length_scale ближе к 1.
    # Тон поднимается совсем немного и с сохранением форманты, чтобы голос
    # оставался низким и мужским даже в эмоциональных репликах.
    "calm":    dict(length=1.00, noise=0.667, noise_w=0.80, tempo=1.00, pitch=1.00, gain=0.0),
    "lively":  dict(length=0.95, noise=0.72,  noise_w=0.85, tempo=1.02, pitch=1.00, gain=1.0),
    "excited": dict(length=0.88, noise=0.78,  noise_w=0.90, tempo=1.05, pitch=1.03, gain=2.0),
}

# id-группы -> (стиль, [варианты])
PHRASES = {
    "kickoff": ("lively", [
        "Добрый вечер, друзья! Мяч в игре, матч начался!",
        "Стартовый свисток! Поехали!",
        "Судья даёт свисток, и мы начинаем!",
        "Трибуны гудят, команды на поле. Начали!",
    ]),
    "attack": ("lively", [
        "Нортсайд идёт в атаку!",
        "Проход по флангу, скорость высокая!",
        "Обыгрывает одного, идёт дальше!",
        "Опасно! Подбирается к штрафной!",
        "Смелый дриблинг, защита в панике!",
        "Пространство есть, можно бить!",
    ]),
    "box": ("excited", [
        "Он в штрафной! Бей!",
        "Выход один на один!",
        "Момент! Нужно завершать!",
    ]),
    "tackle": ("lively", [
        "Отбор! Защитник успел.",
        "Мяч потерян, жёсткий подкат.",
        "Не прошёл! Соперник забирает мяч.",
        "Чисто сыграно в отборе.",
        "Ай-яй-яй, потеря мяча!",
    ]),
    "shot": ("excited", [
        "Удар!",
        "Бьёт!",
        "Удар по воротам!",
        "Пробивает!",
    ]),
    "miss": ("lively", [
        "Мимо! Совсем чуть-чуть!",
        "Вратарь на месте, не прошло.",
        "Выше ворот! Обидно.",
        "Отбили! Голом не стал этот удар.",
    ]),
    "goal": ("excited", [
        "Го-о-ол! Какой удар! Мяч в сетке!",
        "Гол! Гол! Нортсайд забивает!",
        "Да-а-а! Это гол! Стадион взрывается!",
        "Невероятно! Мяч в воротах!",
        "Вот это класс! Гол!",
    ]),
    "concede": ("lively", [
        "Пропускаем. Соперник забивает.",
        "Гол в наши ворота. Неприятно.",
        "Соперник наказывает за ошибку.",
        "Ой, пропустили. Надо отыгрываться!",
    ]),
    "late": ("lively", [
        "Последние минуты матча!",
        "Время уходит, остаются считанные минуты!",
        "Добавленное время! Всё решится сейчас!",
    ]),
    "pause": ("calm", [
        "Небольшая пауза.",
        "Игра остановлена.",
    ]),
    "resume": ("calm", [
        "Продолжаем!",
        "Игра возобновлена.",
    ]),
    "win": ("excited", [
        "Финальный свисток! Победа Нортсайда!",
        "Всё! Три очка в копилке! Отличная игра!",
        "Матч окончен! Заслуженная победа!",
    ]),
    "draw": ("calm", [
        "Финальный свисток. Ничья.",
        "Матч окончен, очки поделены.",
        "Ничья. Борьба была равной.",
    ]),
    "loss": ("calm", [
        "Финальный свисток. Сегодня поражение.",
        "Матч окончен. Не наш день.",
        "Уступаем. Нужно работать дальше.",
    ]),
    "sim": ("calm", [
        "Быстрая симуляция матча.",
        "Смотрим обзор матча.",
    ]),
    "season_end": ("calm", [
        "Сезон завершён! Спасибо, что были с нами!",
    ]),
}


def build_items():
    items = []  # (id, style, text)
    for group, (style, variants) in PHRASES.items():
        for i, text in enumerate(variants):
            items.append((f"{group}_{i}", style, text))
    for code, name in TEAMS.items():
        items.append((f"vs_{code}", "lively", f"Сегодня Нортсайд принимает команду {name}!"))
    for h in range(len(NUM)):
        for a in range(len(NUM)):
            items.append((f"score_{h}_{a}", "calm", f"Счёт: {NUM[h]} — {NUM[a]}."))
    return items


def synth(voice, text, st):
    cfg = SynthesisConfig(length_scale=st["length"], noise_scale=st["noise"], noise_w_scale=st["noise_w"])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=cfg)
    return buf.getvalue()


def process(wav_bytes, st):
    # Тёплый «радийный» мужской тембр: плотный низ (грудь), мягкая середина,
    # без резких верхов; эмоциональный подъём тона — через rubberband с сохранением форманты.
    chain = []
    if st["pitch"] != 1.0 or st["tempo"] != 1.0:
        chain.append(f"rubberband=pitch={st['pitch']}:tempo={st['tempo']}:formant=preserved:pitchq=quality")
    chain += [
        "highpass=f=70",
        "lowshelf=f=150:g=3.5",
        "equalizer=f=400:t=q:w=1.2:g=-1.5",
        "equalizer=f=2500:t=q:w=1.2:g=1.5",
        "highshelf=f=7000:g=-3",
        "acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=4",
        f"volume={st['gain']}dB",
        "silenceremove=start_periods=1:start_threshold=-45dB",
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse",
        "alimiter=limit=0.95",
    ]
    af = ",".join(chain)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        src = f.name
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", src, "-af", af, "-ac", "1", "-ar", "22050",
             "-c:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "-"],
            check=True, capture_output=True).stdout
    finally:
        os.unlink(src)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="путь к .onnx модели Piper")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "voice", "voice-pack.js"))
    ap.add_argument("--wav-dir", help="дополнительно сохранить mp3-файлы сюда (для прослушивания)")
    args = ap.parse_args()

    voice = PiperVoice.load(args.model)
    pack = {}
    items = build_items()
    for n, (cid, style, text) in enumerate(items, 1):
        mp3 = process(synth(voice, text, STYLES[style]), STYLES[style])
        pack[cid] = base64.b64encode(mp3).decode()
        if args.wav_dir:
            os.makedirs(args.wav_dir, exist_ok=True)
            with open(os.path.join(args.wav_dir, cid + ".mp3"), "wb") as f:
                f.write(mp3)
        print(f"[{n}/{len(items)}] {cid}: {text}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("/* Сгенерировано tools/generate_voice.py (Piper TTS, голос: "
                + os.path.basename(args.model) + "). Не редактировать вручную. */\n")
        f.write("window.VOICE_PACK=" + json.dumps(pack, separators=(",", ":")) + ";\n")
    print("OK ->", os.path.abspath(args.out), f"{os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

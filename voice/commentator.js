/*
 * Живой комментатор для Football Career.
 * - Голос: заранее записанные нейросетевые фразы (Piper TTS) из voice-pack.js.
 * - Без наложений и «мямления»: одна реплика за раз, приоритеты, кулдауны,
 *   важные события (гол, финальный свисток) перебивают второстепенные.
 * - Атмосфера стадиона: шум трибун, рёв на гол, «у-у-у» на промах, свисток.
 *   Всё синтезируется через Web Audio, отдельные файлы не нужны.
 */
(function () {
  "use strict";
  var PACK = window.VOICE_PACK || {};
  var STORE_KEY = "fcCommentatorOn";

  // Приоритеты: 3 — перебивает всё, 2 — встаёт в очередь, 1 — только если тихо.
  var PRIORITY = { goal: 3, concede: 3, win: 3, draw: 3, loss: 3, season_end: 3, kickoff: 3, sim: 3,
    shot: 2, miss: 2, tackle: 2, box: 2, late: 2, pause: 2, resume: 2, attack: 1 };
  // Минимальный интервал между репликами одной группы (мс), чтобы не повторяться.
  var COOLDOWN = { attack: 9000, box: 7000, tackle: 3500, shot: 1200, miss: 3000, late: 60000 };

  var enabled = true;
  try { enabled = localStorage.getItem(STORE_KEY) !== "0"; } catch (e) {}

  var ctx = null, master, voiceBus, crowdBus, crowdDuck, crowdLevel, decoded = {}, decoding = null;
  var history = [], current = null, currentPri = 0, queue = [], lastVariant = {}, lastAt = {};

  function groupVariants(group) {
    var out = [];
    for (var i = 0; PACK[group + "_" + i]; i++) out.push(group + "_" + i);
    return out;
  }

  function pick(group) {
    var v = groupVariants(group);
    if (!v.length) return null;
    if (v.length === 1) return v[0];
    var id;
    do { id = v[Math.floor(Math.random() * v.length)]; } while (id === lastVariant[group]);
    lastVariant[group] = id;
    return id;
  }

  function b64ToBuf(b64) {
    var bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  function decodeAll() {
    if (decoding) return decoding;
    var ids = Object.keys(PACK);
    decoding = Promise.all(ids.map(function (id) {
      return new Promise(function (res) {
        try {
          ctx.decodeAudioData(b64ToBuf(PACK[id]), function (buf) { decoded[id] = buf; res(); }, function () { res(); });
        } catch (e) { res(); }
      });
    }));
    return decoding;
  }

  function init() {
    if (ctx) { if (ctx.state === "suspended") ctx.resume(); return; }
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();
    master = ctx.createGain(); master.gain.value = 0.9; master.connect(ctx.destination);

    // Голос: лёгкая компрессия, чтобы звучал плотно, как в эфире.
    var comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18; comp.ratio.value = 3; comp.attack.value = 0.004; comp.release.value = 0.15;
    voiceBus = ctx.createGain(); voiceBus.gain.value = 1.0;
    voiceBus.connect(comp); comp.connect(master);

    // Трибуны: петля из шума, отфильтрованная под «гул стадиона».
    crowdBus = ctx.createGain(); crowdBus.gain.value = 0;      // общий уровень (вкл/выкл матча)
    crowdLevel = ctx.createGain(); crowdLevel.gain.value = 0.06; // напряжение момента
    crowdDuck = ctx.createGain(); crowdDuck.gain.value = 1;     // приглушение под голос
    crowdLevel.connect(crowdDuck); crowdDuck.connect(crowdBus); crowdBus.connect(master);
    startCrowdLoop();
    decodeAll();
  }

  function startCrowdLoop() {
    var sec = 4, sr = ctx.sampleRate, n = sec * sr;
    var buf = ctx.createBuffer(2, n, sr);
    for (var ch = 0; ch < 2; ch++) {
      var d = buf.getChannelData(ch), last = 0;
      for (var i = 0; i < n; i++) {
        // «Коричневый» шум + медленные волны громкости = живой гул толпы.
        last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02;
        var swell = 0.75 + 0.25 * Math.sin(i / sr * 2 * Math.PI * (0.35 + ch * 0.1));
        d[i] = last * 3.2 * swell;
      }
      // Плавная склейка краёв петли.
      var fade = Math.floor(sr * 0.25);
      for (var j = 0; j < fade; j++) { var k = j / fade; d[n - fade + j] = d[n - fade + j] * (1 - k) + d[j] * k; }
    }
    var src = ctx.createBufferSource(); src.buffer = buf; src.loop = true; src.loopStart = 0.25; src.loopEnd = sec;
    var bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.frequency.value = 700; bp.Q.value = 0.6;
    var lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 2200;
    src.connect(bp); bp.connect(lp); lp.connect(crowdLevel);
    src.start();
  }

  function ramp(param, value, time) {
    var t = ctx.currentTime;
    param.cancelScheduledValues(t);
    param.setValueAtTime(param.value, t);
    param.linearRampToValueAtTime(value, t + time);
  }

  function setCrowd(level, time) { if (ctx) ramp(crowdLevel.gain, level, time || 0.6); }

  function roar() { // рёв трибун на гол
    if (!ctx) return;
    var t = ctx.currentTime, g = crowdLevel.gain;
    g.cancelScheduledValues(t); g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(0.55, t + 0.25); g.linearRampToValueAtTime(0.35, t + 2.5);
    g.linearRampToValueAtTime(0.07, t + 6);
  }

  function ooh() { // вздох трибун на промах
    if (!ctx) return;
    var t = ctx.currentTime, g = crowdLevel.gain;
    g.cancelScheduledValues(t); g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(0.25, t + 0.3); g.linearRampToValueAtTime(0.07, t + 2.2);
  }

  function whistle(times) { // судейский свисток
    if (!ctx || !enabled) return;
    var t = ctx.currentTime + 0.05;
    var lens = times === 3 ? [0.25, 0.25, 0.9] : [0.55];
    lens.forEach(function (len) {
      var o = ctx.createOscillator(), lfo = ctx.createOscillator(), lg = ctx.createGain(), g = ctx.createGain();
      o.type = "square"; o.frequency.value = 2650;
      lfo.frequency.value = 38; lg.gain.value = 90; lfo.connect(lg); lg.connect(o.frequency);
      var f = ctx.createBiquadFilter(); f.type = "bandpass"; f.frequency.value = 2700; f.Q.value = 4;
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.18, t + 0.02);
      g.gain.setValueAtTime(0.18, t + len - 0.05); g.gain.linearRampToValueAtTime(0, t + len);
      o.connect(f); f.connect(g); g.connect(master);
      o.start(t); lfo.start(t); o.stop(t + len + 0.02); lfo.stop(t + len + 0.02);
      t += len + 0.12;
    });
  }

  function stopCurrent() {
    if (current) { try { current.onended = null; current.stop(); } catch (e) {} }
    current = null; currentPri = 0;
    if (ctx) ramp(crowdDuck.gain, 1, 0.4);
  }

  function playIds(ids, pri) {
    if (!ids.length) { stopCurrent(); next(); return; }
    var id = ids[0], buf = decoded[id];
    if (!buf) { // ещё декодируется — подождать
      decodeAll().then(function () { if (decoded[id]) playIds(ids, pri); else playIds(ids.slice(1), pri); });
      return;
    }
    var src = ctx.createBufferSource(); src.buffer = buf;
    src.playbackRate.value = 0.98 + Math.random() * 0.05; // микро-вариативность, чтобы не звучало «по шаблону»
    src.connect(voiceBus);
    ramp(crowdDuck.gain, 0.45, 0.15);
    current = src; currentPri = pri;
    history.push(id); if (history.length > 30) history.shift();
    src.onended = function () { if (current === src) { current = null; playIds(ids.slice(1), pri); } };
    src.start();
  }

  function next() {
    if (current || !queue.length) return;
    var item = queue.shift();
    if (performance.now() - item.at > 2500) { next(); return; } // устаревшая реплика — не говорим невпопад
    playIds(item.ids, item.pri);
  }

  /**
   * say("goal") или say(["goal", "score_1_0"]) — группа или конкретный id.
   */
  function say(parts, opts) {
    if (!enabled || !ctx) return;
    opts = opts || {};
    parts = [].concat(parts);
    var group = parts[0];
    var pri = opts.priority || PRIORITY[group] || 1;
    var now = performance.now();
    if (COOLDOWN[group] && lastAt[group] && now - lastAt[group] < COOLDOWN[group]) return;
    var ids = parts.map(function (p) { return PACK[p] ? p : pick(p); }).filter(Boolean);
    if (!ids.length) return;
    lastAt[group] = now;

    if (!current) { playIds(ids, pri); return; }
    if (pri > currentPri || pri === 3) { queue = []; stopCurrent(); playIds(ids, pri); return; }
    if (pri >= 2) { queue = [{ ids: ids, pri: pri, at: now }]; } // одна реплика в очереди, без накопления
  }

  function scoreId(h, a) { var id = "score_" + h + "_" + a; return PACK[id] ? id : null; }

  var api = {
    get enabled() { return enabled; },
    init: init,
    history: history, // последние произнесённые фразы (для отладки)
    say: say,
    whistle: whistle,
    roar: roar,
    ooh: ooh,
    setCrowd: setCrowd,
    toggle: function () {
      enabled = !enabled;
      try { localStorage.setItem(STORE_KEY, enabled ? "1" : "0"); } catch (e) {}
      if (enabled) init(); else { queue = []; stopCurrent(); }
      if (ctx) ramp(master.gain, enabled ? 0.9 : 0, 0.2);
      return enabled;
    },
    // --- события матча ---
    matchStart: function (teamCode) {
      init(); if (!ctx) return;
      ramp(crowdBus.gain, enabled ? 1 : 0, 1.5); setCrowd(0.08, 1);
      whistle(1);
      setTimeout(function () { say(["kickoff", "vs_" + teamCode], { priority: 3 }); }, 700);
    },
    goal: function (h, a) { roar(); var s = scoreId(h, a); say(s ? ["goal", s] : ["goal"]); },
    concede: function (h, a) { ooh(); var s = scoreId(h, a); say(s ? ["concede", s] : ["concede"]); },
    shot: function () { setCrowd(0.18, 0.2); say("shot"); },
    miss: function () { ooh(); say("miss"); },
    tackle: function () { setCrowd(0.07, 0.8); say("tackle"); },
    attack: function () { setCrowd(0.12, 1.2); say("attack"); },
    box: function () { setCrowd(0.2, 0.6); say("box"); },
    late: function () { say("late"); },
    pause: function () { setCrowd(0.04, 0.5); say("pause"); },
    resume: function () { setCrowd(0.08, 0.5); say("resume"); },
    matchEnd: function (h, a, simulated, seasonOver) {
      init(); if (!ctx) return;
      var res = h > a ? "win" : h === a ? "draw" : "loss", s = scoreId(h, a);
      var parts = simulated ? ["sim", res] : [res];
      if (s) parts.push(s);
      if (seasonOver) parts.push("season_end_0");
      whistle(3);
      if (res === "win") roar();
      setTimeout(function () { say(parts, { priority: 3 }); }, simulated ? 200 : 1500);
      setTimeout(function () { if (ctx) ramp(crowdBus.gain, 0, 4); }, 7000);
    }
  };

  // Браузеры разрешают звук только после касания/клика — включаем при первом взаимодействии.
  var unlock = function () { if (enabled) init(); window.removeEventListener("pointerdown", unlock); window.removeEventListener("keydown", unlock); };
  window.addEventListener("pointerdown", unlock);
  window.addEventListener("keydown", unlock);

  window.Commentator = api;
})();

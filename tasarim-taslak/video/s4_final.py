"""v2 final: seslendirme (Ahmet, onaylı telaffuz) + v2 görsel kayıtlar (kayit/sahne_XX.mp4) → final videolar.

Çıktı: iris-tanitim-v2.mp4, iris-tanitim-v2-altyazili.mp4 (1080p30, CRF 17, AAC 192k),
       iris-tanitim-v2-720.mp4, iris-tanitim-v2-altyazili-720.mp4 (≤ 10 MiB, H.264 main, yuv420p, +faststart),
       iris-tanitim-v2.srt, zamanlama_v2.json, ses_v2/
Görüntüye uydurma: zorunlu kesimler (S02 ikinci kaydırma, S03 ROC durağı, S05 grafik gezintisi) → ses
görüntüden kısaysa önce animasyon sonrası durma (≥1,5 sn kalır), sonra sahne sonu durması kısaltılır; ses uzunsa
animasyon sonrası durmaya kesintisiz "durma" eklenir (hareketsiz karede; son kare dondurulmaz). Vurgu halkaları kesilmez.
Çalıştırma: sistem Python (edge-tts, imageio-ffmpeg, numpy).
"""
import asyncio
import json
import re
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import imageio_ffmpeg
import numpy as np

from senaryo import SCENES_V2, VOICE_V2

HERE = Path(__file__).parent
KAYIT, BANT, SES = HERE / "kayit", HERE / "bant", HERE / "ses_v2"
SES.mkdir(exist_ok=True)
TMP = Path(tempfile.gettempdir()) / "iris-final-v2"
TMP.mkdir(exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
VOICE, RATE = "tr-TR-AhmetNeural", "+5%"
LEAD, GAP, TAIL = 0.4, 0.45, 0.7
FPS = 30
MIN_HOLD, MIN_TAIL = 1.5, 0.6
# Zorunlu kesimler (sahne içi saniye, ön izleme kareleriyle doğrulandı)
CUTS = {"02": [(12.0, None)],        # ikinci kaydırma + son durma (~4,1 sn)
        "03": [(7.5, 9.6)],          # ROC durağı (hareketsiz, kesintisiz)
        "05": [(7.5, 10.77)],        # güvenilirlik grafiği gezintisi (sayfa aynı, yalnız imleç atlar)
        "06": [(9.0, 10.0)],         # şelale üzerinde boş bekleme (hareketsiz 8,37–10,37; vurgu yok)
        "08": [(5.5, 6.1)]}          # yaş değişiminden sonra hareketsiz bekleme (4,83–6,27; vurgu yok)
CARD_MIN = {"01": 7.2, "11": 6.5}    # kart animasyonları bitmeden kesme; kapanışta isim satırı okunabilsin
CLOSE_FADE = 0.6


def ff(*a, timeout=1800):
    subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *a], check=True, timeout=timeout)


def duration(p):
    e = subprocess.run([FF, "-nostdin", "-hide_banner", "-i", str(p)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", e).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------- 1) Ses ----------
async def _tts(text, out):
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))


def sentence_wav(sid, k, spoken):
    wav, txt = SES / f"{sid}_{k}.wav", SES / f"{sid}_{k}.txt"
    if wav.exists() and txt.exists() and txt.read_text(encoding="utf-8") == f"{VOICE}|{RATE}|{spoken}":
        return wav
    mp3 = SES / f"{sid}_{k}.mp3"
    asyncio.run(_tts(spoken, mp3))
    t = "silenceremove=start_periods=1:start_threshold=-45dB"
    ff("-i", str(mp3), "-af", f"{t},areverse,{t},areverse", "-ar", "48000", "-ac", "1", str(wav))
    txt.write_text(f"{VOICE}|{RATE}|{spoken}", encoding="utf-8")
    return wav


def voice():
    out = {}
    for s in SCENES_V2:
        sid, t, parts = s["id"], LEAD, []
        for k, (shown, spoken) in enumerate(VOICE_V2[sid], 1):
            w = sentence_wav(sid, k, spoken or shown)
            d = duration(w)
            parts.append({"wav": str(w), "at": round(t, 3), "dur": d, "shown": shown, "spoken": spoken or shown})
            t += d + GAP
        out[sid] = {"parts": parts, "need": round(t - GAP + TAIL, 3)}
    return out


# ---------- 2) Görüntü planı ----------
def static_runs(path):
    """Gerçekten hareketsiz aralıklar (imleç dahil): 480×270 gri karelerde ardışık en büyük fark < 10."""
    p = subprocess.run([FF, "-nostdin", "-loglevel", "error", "-i", str(path), "-vf", "scale=480:270,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, 270, 480).astype(np.int16)
    still = np.concatenate([[False], np.abs(np.diff(a, axis=0)).reshape(len(a) - 1, -1).max(axis=1) < 10])
    runs, i = [], 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            if (j - i) / FPS >= 0.4:
                runs.append([(i - 1) / FPS, j / FPS])
            i = j
        else:
            i += 1
    return runs, len(a) / FPS


def subtract(keep, a, b):
    out = []
    for x, y in keep:
        if b <= x or a >= y:
            out.append([x, y])
        else:
            if a > x:
                out.append([x, a])
            if b < y:
                out.append([b, y])
    return out


def plan(sid, kind, need):
    """Dönüş: parçalar [("v", a, b) | ("p", t, x)], sahne süresi."""
    src = KAYIT / f"sahne_{sid}.mp4"
    if kind == "kart":
        v = duration(src)
        d = max(need, CARD_MIN[sid])
        if d <= v:
            return [("v", 0.0, d)], d, "kart sonu kısaltıldı" if d < v else ""
        return [("v", 0.0, v), ("p", v - 0.1, d - v)], d, f"kart sonu {d - v:.1f} sn uzatıldı (durağan)"
    runs, v = static_runs(src)
    keep = [[0.0, v]]
    for a, b in CUTS.get(sid, []):
        keep = subtract(keep, a, v if b is None else b)
    length = lambda k: sum(y - x for x, y in k)  # noqa: E731
    cur = length(keep)
    note = []
    first = next((r for r in runs if r[0] < 5.0 and r[1] - r[0] >= 1.0), None)   # animasyon sonrası durma
    last = runs[-1] if runs and runs[-1][1] >= v - 0.1 else None
    if CUTS.get(sid) and CUTS[sid][0][1] is None:
        end = CUTS[sid][0][0]
        last = next((r for r in reversed(runs) if r[1] >= end - 0.15 and r[0] < end), None)
        if last:
            last = [last[0], min(last[1], end)]
    if cur > need:   # görüntü uzun: durmaları kısalt
        for r, mn, name in ((first, MIN_HOLD, "açılış durması"), (last, MIN_TAIL, "son durma")):
            if not r or cur <= need:
                continue
            spare = (r[1] - r[0]) - mn
            take = min(spare, cur - need)
            if take > 0.05:
                a = r[0] + mn if name == "açılış durması" else r[1] - take
                keep = subtract(keep, a, a + take)
                cur = length(keep)
                note.append(f"{name} −{take:.1f} sn")
    pieces = [("v", x, y) for x, y in keep]
    if cur < need:   # ses uzun: animasyon sonrası durmaya kesintisiz durma ekle
        x = need - cur
        t = (first[0] + first[1]) / 2 if first else (keep[-1][1] - 1 / FPS)
        out = []
        for kind_, a, b in pieces:
            if kind_ == "v" and a <= t < b:
                out += [("v", a, t), ("p", t, x), ("v", t, b)]
            else:
                out.append((kind_, a, b))
        pieces = out
        note.append(f"durma +{x:.1f} sn @ {t:.1f}")
        cur = need
    return pieces, cur, "; ".join(note)


def render_scene(s, pieces, dur, audio_parts):
    sid = s["id"]
    src = str(KAYIT / f"sahne_{sid}.mp4")
    chains, labels = [], []
    for i, (kind, a, b) in enumerate(pieces):
        if kind == "v":
            chains.append(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[s{i}]")
        else:   # durma: hareketsiz tek kare, b saniye
            # (tpad tek karede kare hızını bozuyordu → loop + açık zaman damgası)
            n = max(int(round(b * FPS)), 1)
            chains.append(f"[0:v]trim=start={max(a - 0.2, 0):.4f}:duration=0.2,setpts=PTS-STARTPTS,"
                          f"trim=end_frame=1,loop=loop={n - 1}:size=1:start=0,setpts=N/{FPS}/TB[s{i}]")
        labels.append(f"[s{i}]")
    # eksik kalan son kare(ler) son kareyle tamamlanır (yuvarlama payı, en çok birkaç kare)
    chains.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0,fps={FPS},settb=1/{FPS},"
                  f"tpad=stop_mode=clone:stop=6[cv]")
    inputs = ["-i", src]
    last = "[cv]"
    if s["band"]:
        inputs += ["-loop", "1", "-t", f"{dur + 1:.3f}", "-i", str(BANT / f"bant{sid}.png")]  # kare sayısı -frames:v ile
        chains.append(f"[1:v]format=rgba,fade=t=in:st=0.2:d=0.4:alpha=1[b];[cv][b]overlay=0:0:shortest=1[ov]")
        last = "[ov]"
    if sid == SCENES_V2[0]["id"]:
        chains.append(f"{last}fade=t=in:st=0:d=0.6[fi]")
        last = "[fi]"
    if sid == SCENES_V2[-1]["id"]:
        chains.append(f"{last}fade=t=out:st={dur - CLOSE_FADE:.2f}:d={CLOSE_FADE}[fo]")
        last = "[fo]"
    chains.append(f"{last}format=yuv420p[out]")
    out = TMP / f"v_{sid}.mp4"
    ff(*inputs, "-filter_complex", ";".join(chains), "-map", "[out]", "-an", "-frames:v", str(round(dur * FPS)),
       "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-color_range", "tv",
       "-r", str(FPS), str(out))
    # ses: cümleler kendi yerinde, sahne süresine sessizlikle tamamlanır
    ain, af = [], []
    for i, p in enumerate(audio_parts):
        ain += ["-i", p["wav"]]
        af.append(f"[{i}:a]aresample=48000,adelay={int(p['at'] * 1000)}:all=1[a{i}]")
    af.append(f"{''.join(f'[a{i}]' for i in range(len(audio_parts)))}amix=inputs={len(audio_parts)}:normalize=0,"
              f"apad,aformat=sample_rates=48000:channel_layouts=stereo[aout]")
    wav = TMP / f"a_{sid}.wav"
    ff(*ain, "-filter_complex", ";".join(af), "-map", "[aout]", "-t", f"{dur:.3f}", str(wav))
    return out, wav


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def split_cue(text, a, b, limit=84):
    if len(text) <= limit:
        return [(a, b, text)]
    cands = [m.start() for m in re.finditer(r"[,:] ", text)] or [m.start() for m in re.finditer(" ", text)]
    cut = min(cands, key=lambda i: abs(i - len(text) / 2)) + 1
    x, y = text[:cut].strip(), text[cut:].strip()
    mid = a + (b - a) * len(x) / (len(x) + len(y))
    return [(a, mid, x), (mid, b, y)]


def verify(path):
    e = subprocess.run([FF, "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    return e.returncode == 0 and not e.stderr.strip(), e.stderr.strip()[:300]


def encode_720(src, out, total, limit_mib=10.0):
    kb_total = limit_mib * 0.92 * 1024 * 1024 * 8 / total / 1000
    ab = 64
    vb = int(kb_total - ab)
    log = str(TMP / f"p720_{out.stem}")
    base = ["-i", str(src), "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-profile:v", "main",
            "-level:v", "3.1", "-preset", "slow", "-b:v", f"{vb}k", "-maxrate", f"{int(vb * 1.6)}k",
            "-bufsize", f"{vb * 3}k", "-pix_fmt", "yuv420p", "-r", str(FPS), "-passlogfile", log]
    ff(*base, "-pass", "1", "-an", "-f", "mp4", "NUL")
    ff(*base, "-pass", "2", "-c:a", "aac", "-b:a", f"{ab}k", "-ac", "1", "-ar", "48000",
       "-movflags", "+faststart", str(out))
    return vb


def main():
    vo = voice()
    timeline, vparts, aparts, cues, t0 = [], [], [], [], 0.0
    for s in SCENES_V2:
        sid = s["id"]
        pieces, dur, note = plan(sid, s["kind"], vo[sid]["need"])
        dur = round(dur * FPS) / FPS   # tam kare katı: görüntü ve ses aynı uzunlukta, sahneler arasında kayma birikmez
        v, a = render_scene(s, pieces, dur, vo[sid]["parts"])
        vparts.append(v)
        aparts.append(a)
        for p in vo[sid]["parts"]:
            cues += split_cue(p["shown"], t0 + p["at"], t0 + p["at"] + p["dur"])
        timeline.append({"id": sid, "start": round(t0, 2), "dur": round(dur, 2), "ses": round(vo[sid]["need"], 2),
                         "band": s["band"], "not": note})
        print(f"sahne {sid:>3}: {dur:6.2f} sn (ses {vo[sid]['need']:.2f}) {note}", flush=True)
        t0 += dur
    lst = TMP / "v.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in vparts), encoding="utf-8")
    alst = TMP / "a.txt"
    alst.write_text("".join(f"file '{p.as_posix()}'\n" for p in aparts), encoding="utf-8")
    full_a = TMP / "ses_tam.wav"
    ff("-f", "concat", "-safe", "0", "-i", str(alst), "-c", "copy", str(full_a))
    master = HERE / "iris-tanitim-v2.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(full_a), "-map", "0:v", "-map", "1:a",
       "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(master))
    srt = HERE / "iris-tanitim-v2.srt"
    with open(srt, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")
    style = ("FontName=Segoe UI Semibold,FontSize=12,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=1.6,Shadow=0,"
             "OutlineColour=&H285F3A1E,BackColour=&H285F3A1E,MarginV=30,MarginL=40,MarginR=40,Alignment=2")
    sub = HERE / "iris-tanitim-v2-altyazili.mp4"
    subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", master.name, "-vf",
                    f"subtitles={srt.name}:force_style='{style}'", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", sub.name],
                   check=True, cwd=HERE, timeout=3600)
    res = {}
    for src, name in ((master, "iris-tanitim-v2-720.mp4"), (sub, "iris-tanitim-v2-altyazili-720.mp4")):
        vb = encode_720(src, HERE / name, t0)
        res[name] = vb
    report = {"toplam": round(t0, 2), "ses": VOICE, "hiz": RATE, "sahneler": timeline, "dosyalar": {}}
    for p in (master, sub, HERE / "iris-tanitim-v2-720.mp4", HERE / "iris-tanitim-v2-altyazili-720.mp4"):
        ok, err = verify(p)
        report["dosyalar"][p.name] = {"MiB": round(p.stat().st_size / 2 ** 20, 2), "sure": round(duration(p), 2),
                                      "cozumleme_hatasiz": ok, "hata": err}
        print(p.name, report["dosyalar"][p.name], flush=True)
    (HERE / "zamanlama_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOPLAM {t0:.2f} sn")


if __name__ == "__main__":
    main()

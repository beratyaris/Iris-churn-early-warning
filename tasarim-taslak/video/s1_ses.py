"""Adım 1: Seslendirme (edge-tts) + sahne süreleri + altyazı (.srt).
Çalıştırma: python s1_ses.py   (sistem Python; edge-tts, imageio-ffmpeg gerekli)
Çıktı: ses/cumle_XX_Y.mp3, ses/sahneXX.wav, zamanlama.json, iris-tanitim.srt
"""
import asyncio
import json
import re
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

from senaryo import GAP, LEAD, RATE, SCENES, TAIL, VOICE

HERE = Path(__file__).parent
SES = HERE / "ses"
SES.mkdir(exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()


def duration(path):
    err = subprocess.run([FF, "-nostdin", "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


async def tts(text, out):
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def split_cue(text, start, end, limit=84):
    """Uzun cümleyi ortaya en yakın virgülden iki altyazıya böler (süre karakter oranında)."""
    if len(text) <= limit:
        return [(start, end, text)]
    cands = [m.start() for m in re.finditer(r", ", text)] or [m.start() for m in re.finditer(r" ", text)]
    cut = min(cands, key=lambda i: abs(i - len(text) / 2)) + 1
    a, b = text[:cut].strip(), text[cut:].strip()
    mid = start + (end - start) * len(a) / (len(a) + len(b))
    return [(start, mid, a), (mid, end, b)]


def main():
    timeline, cues, t0 = [], [], 0.0
    for sc in SCENES:
        parts, t = [], LEAD
        for k, (shown, spoken) in enumerate(sc["sentences"], 1):
            raw = SES / f"cumle_{sc['id']:02d}_{k}.mp3"
            asyncio.run(tts(spoken or shown, raw))
            # TTS dosyasının başındaki/sonundaki sessizliği kırp (aralar senaryo.GAP ile verilir)
            mp3 = SES / f"cumle_{sc['id']:02d}_{k}.wav"
            trim = "silenceremove=start_periods=1:start_threshold=-45dB"
            subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-af",
                            f"{trim},areverse,{trim},areverse", "-ar", "48000", str(mp3)], check=True)
            d = duration(mp3)
            parts.append((mp3, t))
            cues += split_cue(shown, t0 + t, t0 + t + d)
            t += d + GAP
        total = round(max(t - GAP + TAIL, sc.get("min_dur", 0)), 2)
        # Sahne sesi: cümleler kendi başlangıç zamanlarında, arası sessizlik (48 kHz stereo wav)
        wav = SES / f"sahne{sc['id']:02d}.wav"
        inputs, filt = [], []
        for i, (mp3, st) in enumerate(parts):
            inputs += ["-i", str(mp3)]
            filt.append(f"[{i}:a]aresample=48000,adelay={int(st * 1000)}:all=1[a{i}]")
        mix = "".join(f"[a{i}]" for i in range(len(parts)))
        filt.append(f"{mix}amix=inputs={len(parts)}:normalize=0,apad,aformat=channel_layouts=stereo[out]")
        subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt),
                        "-map", "[out]", "-t", str(total), str(wav)], check=True, timeout=120)
        timeline.append({"id": sc["id"], "start": round(t0, 2), "dur": total, "band": sc["band"], "kind": sc["kind"]})
        print(f"Sahne {sc['id']:2d}: {total:5.2f} sn")
        t0 += total
    print(f"TOPLAM: {t0:.2f} sn ({int(t0 // 60)}:{t0 % 60:05.2f})")
    (HERE / "zamanlama.json").write_text(json.dumps({"voice": VOICE, "rate": RATE, "total": round(t0, 2),
                                                     "scenes": timeline}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(HERE / "iris-tanitim.srt", "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")


if __name__ == "__main__":
    main()

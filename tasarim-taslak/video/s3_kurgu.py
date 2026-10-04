"""Adım 3: Kurgu (imageio-ffmpeg). Sahne klipleri (görüntü + bant + ses) → birleştirme → altyazılı/altyazısız MP4.
Çıktı: klipler/XX_sahne.mp4, iris-tanitim.mp4, iris-tanitim-altyazili.mp4 (H.264 1080p 25 fps, AAC 48 kHz)
"""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
T = json.loads((HERE / "zamanlama.json").read_text(encoding="utf-8"))
M = json.loads((HERE / "kayit" / "isaretler.json").read_text(encoding="utf-8"))
KLIP = HERE / "klipler"
KLIP.mkdir(exist_ok=True)
FPS = 25
VENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]
AENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]


def ff(*args):
    subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *args], check=True, cwd=HERE)


def main():
    parts, last = [], T["scenes"][-1]["id"]
    for s in T["scenes"]:
        sid, dur = s["id"], s["dur"]
        out = KLIP / f"{sid:02d}_sahne.mp4"
        wav = HERE / "ses" / f"sahne{sid:02d}.wav"
        fades = []
        if sid == 1:
            fades.append("fade=t=in:st=0:d=0.6")
        if sid == last:
            fades.append(f"fade=t=out:st={dur - 1.0:.2f}:d=1.0")
        if s["kind"] == "kart":
            src, start = HERE / "kayit" / f"kart{sid:02d}.webm", M[f"kart{sid:02d}_lead"]
            vf = ",".join(["scale=1920:1080", f"fps={FPS}", *fades]) + "[v]"
            ff("-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-i", str(wav),
               "-filter_complex", f"[0:v]{vf}", "-map", "[v]", "-map", "1:a", *VENC, *AENC, "-t", f"{dur:.3f}", str(out))
        else:
            src, start = HERE / "kayit" / "ekran.webm", M["ekran"][str(sid)]
            band = HERE / "bant" / f"bant{sid:02d}.png"
            chain = ",".join([f"fps={FPS}", *fades]) if fades else f"fps={FPS}"
            ff("-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-loop", "1", "-t", f"{dur:.3f}", "-i", str(band),
               "-i", str(wav), "-filter_complex",
               f"[1:v]format=rgba,fade=t=in:st=0.15:d=0.35:alpha=1[b];[0:v][b]overlay=0:0:shortest=1,{chain}[v]",
               "-map", "[v]", "-map", "2:a", *VENC, *AENC, "-t", f"{dur:.3f}", str(out))
        parts.append(out)
        print("klip", out.name, dur)
    lst = KLIP / "liste.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", "iris-tanitim.mp4")
    # Gömülü altyazı: SRT ölçeği 288 satır (1080/288 = 3,75): FontSize 12 ≈ 45 px, MarginV 30 ≈ 112 px (bandın üstü)
    style = ("FontName=Segoe UI Semibold,FontSize=12,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=1.6,Shadow=0,"
             "OutlineColour=&H285F3A1E,BackColour=&H285F3A1E,MarginV=30,MarginL=40,MarginR=40,Alignment=2")
    ff("-i", "iris-tanitim.mp4", "-vf", f"subtitles=iris-tanitim.srt:force_style='{style}'", *VENC, "-c:a", "copy",
       "-movflags", "+faststart", "iris-tanitim-altyazili.mp4")
    print("tamam")


if __name__ == "__main__":
    main()

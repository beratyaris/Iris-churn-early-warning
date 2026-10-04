"""v2 sessiz önizleme: kayit/sahne_XX.mp4 (1080p) + bant/bantXX.png → onizleme-v2-sessiz.mp4 (720p, ses yok, ≤ 15 MB).
Sahne süreleri zamanlama_v2_taslak.json'a yazılır (ses metni onaylanınca final kurguda sese göre ayarlanacak).
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from senaryo import SCENES_V2

HERE = Path(__file__).parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
META = json.loads((HERE / "kayit" / "isaretler_v2.json").read_text(encoding="utf-8"))
OUT = HERE / "onizleme-v2-sessiz.mp4"
LIMIT_MB = 15.0
TMP = Path(tempfile.gettempdir()) / "iris-onizleme"
TMP.mkdir(exist_ok=True)


def ff(*a):
    subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *a], check=True, timeout=1200)


def main():
    parts, timeline, t = [], [], 0.0
    last = SCENES_V2[-1]["id"]
    for s in SCENES_V2:
        sid, dur = s["id"], META[s["id"]]["dur"]
        src, out = HERE / "kayit" / f"sahne_{sid}.mp4", TMP / f"p_{sid}.mp4"
        fades = (["fade=t=in:st=0:d=0.5"] if sid == SCENES_V2[0]["id"] else []) + \
                ([f"fade=t=out:st={dur - 0.9:.2f}:d=0.9"] if sid == last else [])
        tail = ",".join(["scale=1280:720:flags=lanczos:out_range=tv", *fades, "format=yuv420p"])
        if s["band"]:
            ff("-i", str(src), "-loop", "1", "-t", f"{dur:.3f}", "-i", str(HERE / "bant" / f"bant{sid}.png"),
               "-filter_complex", f"[1:v]format=rgba,fade=t=in:st=0.2:d=0.4:alpha=1[b];"
                                  f"[0:v][b]overlay=0:0:shortest=1,{tail}[v]",
               "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-r", "30", str(out))
        else:
            ff("-i", str(src), "-vf", tail, "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-r", "30", str(out))
        parts.append(out)
        timeline.append({"id": sid, "start": round(t, 2), "dur": dur, "band": s["band"],
                         "segments": META[sid]["segments"]})
        t += dur
    lst = TMP / "liste.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    joined = TMP / "birlesik.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined))
    # 2 geçiş, hedef boyutun %94'ü (kapsayıcı payı)
    kbps = int(LIMIT_MB * 0.94 * 1024 * 1024 * 8 / t / 1000)
    log = str(TMP / "pass")
    base = ["-i", str(joined), "-an", "-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k",
            "-maxrate", f"{int(kbps * 1.8)}k", "-bufsize", f"{kbps * 3}k", "-pix_fmt", "yuv420p", "-color_range", "tv", "-passlogfile", log]
    ff(*base, "-pass", "1", "-f", "mp4", "NUL" if os.name == "nt" else "/dev/null")
    ff(*base, "-pass", "2", "-movflags", "+faststart", str(OUT))
    size = OUT.stat().st_size / 2 ** 20
    (HERE / "zamanlama_v2_taslak.json").write_text(json.dumps(
        {"not": "Sessiz önizleme taslağı; final süreler onaylı seslendirmeye göre yeniden ayarlanacak.",
         "toplam": round(t, 2), "sahneler": timeline}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{OUT.name}: {size:.2f} MiB, {t:.2f} sn, video {kbps} kb/s")


if __name__ == "__main__":
    main()

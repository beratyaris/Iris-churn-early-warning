"""Telaffuz örnekleri (tr-TR-AhmetNeural, +5%). edge-tts 7.x özel SSML/phoneme kabul etmez (metni kaçışlar);
bu yüzden okunuş düz metinde yazıldı. Çıktı: NN_terim_x.mp3 + telaffuz-hepsi.mp3 (etiketli, 0,7 sn aralı)."""
import asyncio
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

VOICE, RATE = "tr-TR-AhmetNeural", "+5%"
HERE = Path(__file__).parent
TMP = HERE / "_tmp"
TMP.mkdir(exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()

TERMS = [  # (no, ad, sözlü ad, cümle kalıbı, [(harf, okunuş)])
    # Kullanıcı İngilizce okunuşu seçti: iki yazım varyantı (doğallık dinlenerek seçilecek)
    (1, "iris", "Ayris, kesinleşen okunuş. İki yazım", "Bu {}: bir erken uyarı sistemi.",
     [("1", "Ayris"), ("2", "Ay-ris")]),
    (2, "xgboost", "İkinci terim", "En iyi sonucu {} modeli verdi.",
     [("a", "eks ci bust"), ("b", "iks ci bust"), ("c", "eks-ci-boost")]),
    (3, "smotenc", "Üçüncü terim", "Eğitimde {} yöntemini kullandık.",
     [("a", "smot en si"), ("b", "smot en ce"), ("c", "esmot en si")]),
    (4, "mcnemar", "Dördüncü terim", "Farkları {} testiyle sınadık.",
     [("a", "mek nimar"), ("b", "mak nemar"), ("c", "mık nimar")]),
    (5, "shap", "Beşinci terim", "{} ile nedenini gösteriyoruz.",
     [("a", "şep"), ("b", "şap")]),
    (6, "randomforest", "Altıncı terim", "{} modelini de denedik.",
     [("a", "rendım forest"), ("b", "random forest")]),
    (7, "lojistikregresyon", "Yedinci terim, kontrol", "{} modeliyle başladık.",
     [("a", "Lojistik regresyon")]),
]


async def say(text, out):
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))


def ff(*a):
    subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *a], check=True)


def trimmed(mp3, wav):
    t = "silenceremove=start_periods=1:start_threshold=-45dB"
    ff("-i", str(mp3), "-af", f"{t},areverse,{t},areverse", "-ar", "24000", "-ac", "1", str(wav))
    return wav


def main():
    seq, rows = [], []
    for no, name, spoken_no, tpl, opts in TERMS:
        lab = TMP / f"lab{no}.mp3"
        asyncio.run(say(spoken_no + ".", lab))
        seq.append(trimmed(lab, TMP / f"lab{no}.wav"))
        for letter, reading in opts:
            sentence = tpl.format(reading)
            sentence = sentence[0].upper() + sentence[1:]
            out = HERE / f"{no:02d}_{name}_{letter}.mp3"
            asyncio.run(say(sentence, out))
            rows.append((out.name, sentence))
            if len(opts) > 1:
                tag = TMP / f"tag{no}{letter}.mp3"
                label = f"Yazım {'bir' if letter == '1' else 'iki'}." if letter.isdigit() else f"Seçenek {letter.upper()}."
                asyncio.run(say(label, tag))
                seq.append(trimmed(tag, TMP / f"tag{no}{letter}.wav"))
            seq.append(trimmed(out, TMP / f"s{no}{letter}.wav"))
    sil = TMP / "sil.wav"
    ff("-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.7", str(sil))
    lst = TMP / "liste.txt"
    lst.write_text("".join(f"file '{p.name}'\nfile 'sil.wav'\n" for p in seq), encoding="utf-8")
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1",
       str(HERE / "telaffuz-hepsi.mp3"))
    for r in rows:
        print(" | ".join(r))


if __name__ == "__main__":
    main()

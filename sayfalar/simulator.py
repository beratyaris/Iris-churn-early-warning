""""Model ne derdi?" simülatörü: müşteri özellikleri değişince modelin TAHMİNİ nasıl değişir?

- Girdiler eğitim verisindeki gerçek aralıklarla sınırlı (iris_data.input_bounds); kategorik
  seçenekler veriden. Form, Müşteri Risk Skoru sayfasıyla ortak (iris_musteri.customer_form).
- Nadirlik kontrolü (iris_karar.rarity_check): tek değişken grubu eğitim setinin %1'inden azsa ya
  da iki değişkenin birlikte görüldüğü müşteri sayısı 10'dan azsa "çok nadir", 0 ise "yok".
- Tek ölçek: göstergede kalibre olasılık; uyarı durumu ayrı satırda model skoru ≥ eşik.
- Dil nedensel değildir: bu sayfa iki profil için modelin tahminlerini karşılaştırır; bir
  müdahalenin (ör. müşteriyi aktif yapmanın) etkisini göstermez.
"""
import streamlit as st

import iris_musteri as cm
import iris_ui as ui
from iris_data import SHAP_UNIT, current_settings, is_calibrated, load_artifacts, rarity_index
from iris_karar import (COUNTRY_TR, GENDER_TR, GROUP_LABELS, RARE_PAIR_N, RARE_SINGLE_SHARE, describe,
                        rarity_check)
from train import MODEL_NAMES

art = load_artifacts()
key, thr = current_settings(art)
preds = art["preds"]
score_col = f"proba_{key}"
calibrated = is_calibrated(art, key)
ss = st.session_state

ui.page_header("Karar Destek", "Model ne derdi?",
               "Müşteri özelliklerini değiştirin; modelin bu profil için tahmini ve gerekçesi (SHAP) "
               "anında güncellenir.",
               chip=f"{MODEL_NAMES[key]} · eşik {ui.fmt_num(thr, 2)}")

ui.html(f'<div class="iris-callout" style="--tone:{ui.AMBER}"><div class="c-title">'
        "Bu bir tahmin karşılaştırmasıdır; müdahalenin etkisini göstermez.</div>"
        "Ekrandaki sayılar “bu özelliklere sahip bir müşteri için model şunu tahmin eder” diye "
        "okunmalıdır. Örneğin bir müşteriyi aktif üye yapmak, modelin aktif üyeler için verdiği "
        "tahmini o müşteriye kazandırmaz.</div>")

# ---------- Başlangıç profili ----------
ss.setdefault("sim_base", {"label": "Varsayılan profil", "profile": dict(cm.DEFAULT_PROFILE),
                           "clipped": []})
ids = preds.sort_values(score_col, ascending=False)["customer_id"].tolist()


def _load_test_customer():
    cid = ss.get("sim_start")
    if cid is None:
        return
    raw = cm.profile_from_test_row(preds.loc[preds["customer_id"] == cid])
    profile, clipped = cm.clamp_profile(raw)
    ss["sim_base"] = {"label": f"Test müşterisi {cid}", "profile": profile, "clipped": clipped}
    cm.set_form("s", profile)


def _load_default():
    ss["sim_base"] = {"label": "Varsayılan profil", "profile": dict(cm.DEFAULT_PROFILE), "clipped": []}
    ss["sim_start"] = None
    cm.set_form("s", cm.DEFAULT_PROFILE)


def _reset():
    cm.set_form("s", ss["sim_base"]["profile"])


def score_label(c):
    s = float(preds.loc[preds["customer_id"] == c, score_col].iloc[0])
    return f"{c} · skor {ui.fmt_num(s, 2)}"


left, right = st.columns([1.15, 1], gap="medium")

with left, st.container(key="card-sim-form"):
    ui.card_title("Müşteri profili",
                  "Değerler eğitim verisindeki gerçek aralıklarla sınırlıdır (alanların yanındaki "
                  "ⓘ işaretinde). Kategorik seçenekler veriden gelir.")
    c1, c2 = st.columns([1.6, 1], vertical_alignment="bottom")
    c1.selectbox("Başlangıç profili: test setinden müşteri (isteğe bağlı)", ids, index=None,
                 key="sim_start", format_func=score_label, on_change=_load_test_customer,
                 placeholder="Müşteri no yazın ya da seçin",
                 help="Seçilen test müşterisinin özellikleri forma yüklenir ve karşılaştırma için "
                      "başlangıç olarak saklanır. Liste model skoruna göre sıralıdır.")
    c2.button("Varsayılan profil", key="sim_default", on_click=_load_default, width="stretch")
    profile = cm.customer_form("s", sliders=True, initial=ss["sim_base"]["profile"])
    base = ss["sim_base"]
    changed = [c for c in cm.DEFAULT_PROFILE if profile[c] != base["profile"][c]]
    st.button("Başlangıca dön", key="sim_reset", on_click=_reset, disabled=not changed)
    if base["clipped"]:
        st.caption("Not: başlangıç müşterisinin " + ", ".join(GROUP_LABELS[c].lower() for c in base["clipped"])
                   + " değeri eğitim aralığının dışındaydı; aralığın sınırına çekildi.")

# ---------- Tahminler ----------
x_now = cm.features(art, profile)
x_base = cm.features(art, base["profile"])
s_now, p_now = cm.score(art, key, x_now)
s_base, p_base = cm.score(art, key, x_base)
rare = rarity_check(rarity_index(), profile)


def fmt_val(col, v):
    if col == "country":
        return COUNTRY_TR[v]
    if col == "gender":
        return GENDER_TR[v]
    if col in ("credit_card", "active_member"):
        return "Evet" if v else "Hayır"
    return ui.fmt_num(v)


ui.html(cm.ALERT_CSS)
with right:
    with st.container(key="card-sim-gauge"):
        ui.card_title("Modelin tahmini",
                      "Kalibre olasılık: model, bu profildeki 100 müşteriden kaçının gitmesini bekliyor"
                      if calibrated else "Model skoru (kalibrasyon yok)")
        warn = ("" if rare["level"] == "ok" else
                '<div class="iris-status"><span class="iris-pill warn">Tahmin güvenilir değil: '
                f'profil eğitim verisinde {"yok" if rare["level"] == "yok" else "çok nadir"}</span></div>')
        diff_pp = (p_now - p_base) * 100
        in_now, in_base = s_now >= thr, s_base >= thr
        alert_change = ("" if in_now == in_base else
                        f' · uyarı listesi: {"Evet" if in_base else "Hayır"} → <b>{"Evet" if in_now else "Hayır"}</b>')
        moves = "".join(f"<li>{GROUP_LABELS[c]}: {fmt_val(c, base['profile'][c])} → "
                        f"<b>{fmt_val(c, profile[c])}</b></li>" for c in changed)
        compare = (f'<div class="iris-card-sub" style="margin-top:.7rem">Başlangıç ({base["label"]}): '
                   f'model tahmini <b>{ui.fmt_pct(p_base)}</b> → şimdi <b>{ui.fmt_pct(p_now)}</b> '
                   f'{ui.delta_badge(diff_pp)}{alert_change}</div>'
                   + (f'<ul class="iris-card-sub" style="text-align:left;margin:.3rem 0 0 1.1rem;padding:0">'
                      f"{moves}</ul>" if moves else
                      '<div class="iris-card-sub">Profil başlangıçla aynı.</div>'))
        ui.html(cm.gauge_html(p_now, s_now, thr, calibrated, caption="modelin tahmin ettiği gitme olasılığı",
                              extra=warn + compare), js=True)

    with st.container(key="card-sim-rarity"):
        ui.card_title("Bu profil eğitim verisinde var mı?",
                      f"Eğitim seti: {ui.fmt_num(rare['n_train'])} müşteri. Kural: tek değişken grubu eğitim "
                      f"setinin %{ui.fmt_num(RARE_SINGLE_SHARE * 100)}'inden azsa ya da iki değişken birlikte "
                      f"{RARE_PAIR_N} müşteriden azında görülüyorsa <b>çok nadir</b>, hiç görülmüyorsa <b>yok</b>.")
        if rare["level"] == "ok":
            ui.html(f'<div class="iris-callout" style="--tone:{ui.TURQ}"><div class="c-title">'
                    "Profil eğitim verisinde temsil ediliyor.</div>En küçük grup: "
                    f"{rare['min_desc']} · <b>{ui.fmt_num(rare['min_n'])}</b> müşteri.</div>")
        else:
            word = "yok" if rare["level"] == "yok" else "çok nadir"
            lines = []
            for d in rare["issues"][:4]:
                ctx = ("" if len(d["cols"]) == 1 else
                       f" ({GROUP_LABELS[d['ref_col']]} = "
                       f"{d['groups'][d['cols'].index(d['ref_col'])]}: {ui.fmt_num(d['ref'])} müşteri)")
                lines.append(f"{describe(d)}: eğitimde <b>{ui.fmt_num(d['n'])}</b> müşteri{ctx}")
            more = len(rare["issues"]) - len(lines)
            ui.html(ui.callout(f"Bu profil eğitim verisinde {word}; tahmin güvenilir değil.", lines,
                               tone=ui.CORAL,
                               note=(f"ve {more} nadir grup daha. " if more > 0 else "")
                               + "Model bu bölgede neredeyse hiç örnek görmedi; gösterilen tahmin "
                                 "eğitim verisinin dışına taşma (ekstrapolasyon) olabilir."))

# ---------- SHAP ----------
with st.container(key="card-sim-shap"):
    unit = SHAP_UNIT[key]
    ui.card_title("Modelin gerekçesi (SHAP)",
                  "Başlangıç: modelin ortalama çıktısı; her çubuk bir değişkenin bu profilde model "
                  "çıktısını ne kadar <b>artırdığını</b> (mercan) ya da <b>azalttığını</b> (turkuaz) gösterir.")
    w = cm.shap_waterfall(art, key, x_now, chart_key="sim-shap-wf", height=420)
    ui.html(f'<div class="iris-card-sub"><b>Okuma ipucu:</b> Değerler {unit} biriminde; artı riski '
            "artırır, eksi düşürür. SHAP, kalibrasyon öncesi model çıktısını açıklar "
            f"(bu profilde {ui.fmt_num(w['fx'], w['dec'])} → model skoru {ui.fmt_num(s_now, 3)}); "
            "değişkenlerin modeldeki payını gösterir, gerçek dünyadaki etkisini değil.</div>")

"""Adım 2 (v2): Görüntü kaydı — CDP screencast (JPEG q90, 1920×1080) + ffmpeg ile yüksek kaliteli sahne klipleri.
v1 (Playwright record_video, düşük bit hızı) kayit_v1/ altında saklı (betik: kayit_v1/s2_kayit_v1.py).

Ön koşul: Iris headless çalışıyor (varsayılan http://localhost:8662). Uygulama koduna dokunmaz.
Çalıştırma: python s2_kayit.py            → bant/ + kayit/sahne_XX.mp4 (1080p 30 fps, CRF 16) + kayit/isaretler_v2.json
            python s2_kayit.py kartlar    → yalnız kartlar + bantlar
Kareler OneDrive dışında tutulur: %TEMP%/iris-video-kareler (yüzlerce MB; IRIS_KARELER ile değiştirilebilir).

Animasyon kuralı: Iris'in sayfa giriş animasyonu, sayfa oturumda ilk kez açılınca oynar (iris_ui.begin_page).
Her sahne yeni bir sayfaya gidilerek başlar; segment, yeni sayfanın başlığı DOM'a girdiği karede (animasyonun
ilk karesi) başlar → boş yükleme anı kırpılır, animasyon başı kırpılmaz. Ardından HOLD sn hiçbir şey yapılmaz.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

from senaryo import CLOSING_CARD, SCENES_V2

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
LOGO_DIR = ROOT / "logo" / "final"
URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8662")
ONLY_CARDS = "kartlar" in sys.argv[1:]
KAYIT, BANT = HERE / "kayit", HERE / "bant"
FRAMES = Path(os.environ.get("IRIS_KARELER", Path(tempfile.gettempdir()) / "iris-video-kareler"))
SIZE = {"width": 1920, "height": 1080}
FF = imageio_ffmpeg.get_ffmpeg_exe()
HOLD = 3.3   # sayfa açılışından sonra bekleme: giriş animasyonu (~1,5–1,8 sn) + en az 1,5 sn durma
FONT = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">'


def svg_inner(name):
    """SVG dosyasının <defs>…</svg> arası — kartta satır içi canlandırmak için."""
    s = (LOGO_DIR / name).read_text(encoding="utf-8")
    return s[s.index("<defs>"):s.rindex("</svg>")]


def data_uri(name):
    return "data:image/svg+xml;base64," + base64.b64encode((LOGO_DIR / name).read_bytes()).decode()


# ---------- Açılış / kapanış kartları: logo çizilerek gelir ----------
# Yay: pathLength=1 + dashoffset 1→0; degrade soldan sağa olduğu için renk turkuaz → amber → mercan ilerler.
# Sonra göz bebeği (beyaz daire + lacivert nokta) belirir, "Iris" harfleri sırayla çizilir, i'nin noktası düşer.
CARD = """<!doctype html><html><head><meta charset="utf-8">{font}<style>
html,body{{margin:0;width:1920px;height:1080px;overflow:hidden;font-family:Inter,'Segoe UI',sans-serif}}
body{{background:radial-gradient(1100px 700px at 78% 112%, rgba(20,184,166,.26), transparent 60%),
  radial-gradient(900px 600px at 8% -12%, rgba(45,212,191,.12), transparent 60%), #1E3A5F;
  display:flex;flex-direction:column;align-items:center;justify-content:center;color:#F7FAFC}}
svg.logo{{width:{lw}px;height:auto;overflow:visible;position:relative}}
svg.logo circle{{transform-box:fill-box;transform-origin:center;transform:scale(0)}}
.arc{{stroke-dasharray:1 2;stroke-dashoffset:1.001;opacity:0;animation:draw {arc}s cubic-bezier(.55,.1,.35,1) {t0}s forwards}}
svg.logo > circle{{animation:pop .55s cubic-bezier(.3,1.6,.5,1) {t1}s forwards}}
svg.logo > circle + circle{{animation-delay:{t1b}s}}
.w1,.w2,.w3,.w4{{stroke-dasharray:1 2;stroke-dashoffset:1.001;opacity:0}}  /* çizim başlayana dek gizli: yuvarlak uç nokta gibi görünmesin */
.w1{{animation:draw {wd}s ease-out {w1}s forwards}} .w2{{animation:draw {wd}s ease-out {w2}s forwards}}
.w3{{animation:draw {wd}s ease-out {w3}s forwards}} .w4{{animation:draw {wd}s ease-out {w4}s forwards}}
svg.logo g circle{{animation:pop .45s cubic-bezier(.3,1.8,.5,1) {w5}s forwards}}
.glow{{position:absolute;width:760px;height:380px;border-radius:50%;filter:blur(70px);opacity:0;
  background:radial-gradient(closest-side, rgba(20,184,166,.35), rgba(245,158,11,.12), transparent);
  animation:glow 2.4s ease {t0}s forwards}}
.bar{{width:0;height:6px;border-radius:3px;margin:44px 0 30px;
  background:linear-gradient(90deg,#14B8A6,#F59E0B 50%,#FF7A59);animation:bar 1.1s cubic-bezier(.2,.7,.2,1) {b0}s forwards}}
.t{{font-size:50px;font-weight:750;letter-spacing:-.012em;opacity:0;animation:up .8s ease {b1}s forwards;text-align:center}}
.s{{font-size:28px;color:rgba(247,250,252,.80);margin-top:16px;opacity:0;animation:up .8s ease {b2}s forwards;
  text-align:center;max-width:1500px;line-height:1.5}}
.s b{{color:#2DD4BF;font-weight:600}}
.c{{position:absolute;bottom:72px;font-size:22px;color:rgba(247,250,252,.62);letter-spacing:.02em;opacity:0;
  animation:up .8s ease {b3}s forwards}}
body.wait *{{animation-play-state:paused !important}}  /* font hazır olana dek bekle */
@keyframes draw{{from{{opacity:1}}to{{stroke-dashoffset:0;opacity:1}}}} @keyframes pop{{to{{transform:scale(1)}}}}
@keyframes bar{{to{{width:440px}}}} @keyframes glow{{40%{{opacity:1}}to{{opacity:.55}}}}
@keyframes up{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
</style></head><body class="wait"><div class="glow"></div>
<svg class="logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 159 64">{logo}</svg>
<div class="bar"></div><div class="t">{t}</div><div class="s">{s}</div><div class="c">{c}</div></body></html>"""


def card_html(title, sub, credit, fast=False):
    k = 0.72 if fast else 1.0
    logo = svg_inner("iris-lockup-b-dark.svg")
    for old, cls in (('<path d="M10 43', "arc"), ('<path d="M7.25', "w1"), ('<path d="M17.25', "w2"),
                     ('<path d="M41 ', "w3"), ('<path d="M71.5', "w4")):
        assert old in logo, old
        logo = logo.replace(old, old.replace("<path ", f'<path class="{cls}" pathLength="1" '), 1)
    t0 = 0.45
    r = lambda v: round(t0 + v * k, 2)  # noqa: E731
    return CARD.format(font=FONT, logo=logo, lw=560, t=title, s=sub, c=credit, arc=round(1.6 * k, 2), t0=t0,
                       t1=r(1.45), t1b=r(1.7), wd=round(0.5 * k, 2), w1=r(2.0), w2=r(2.35), w3=r(2.7), w4=r(2.95),
                       w5=r(3.4), b0=r(3.6), b1=r(4.0), b2=r(4.5), b3=r(5.1))


BAND = """<!doctype html><html><head><meta charset="utf-8">{font}<style>
html,body{{margin:0;background:transparent;font-family:Inter,'Segoe UI',sans-serif}}
.b{{position:absolute;left:0;right:0;bottom:0;height:64px;display:flex;align-items:center;gap:18px;padding:0 40px;
  background:linear-gradient(90deg,rgba(30,58,95,.97),rgba(30,58,95,.90) 70%,rgba(30,58,95,.82));
  border-top:3px solid #14B8A6;color:#F7FAFC;font-size:27px;font-weight:600;letter-spacing:.005em}}
.n{{display:inline-flex;width:34px;height:34px;border-radius:50%;background:#14B8A6;color:#fff;font-size:18px;
  font-weight:800;align-items:center;justify-content:center}}
.logo{{margin-left:auto;height:40px;width:40px}}
</style></head><body><div class="b"><span class="n">{n}</span>{t}<img class="logo" src="{logo}"></div></body></html>"""

# Sahte imleç + vurgu halkası + yumuşak kaydırma (yalnız kayıt tarayıcısında; uygulama koduna dokunmaz)
HELPERS_JS = r"""
(() => { const add = () => { if (document.getElementById('vc')) return;
  const s = document.createElement('style'); s.textContent = `
  #vc{position:fixed;left:-60px;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
    background:rgba(255,255,255,.94);border:3px solid #1E3A5F;box-shadow:0 3px 10px rgba(14,30,52,.35);
    z-index:2147483647;pointer-events:none;transition:transform .12s ease}
  #vc.dn{transform:scale(.78)}
  .vr{position:fixed;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;border:3px solid #14B8A6;
    z-index:2147483646;pointer-events:none;animation:vr .6s ease-out forwards}
  @keyframes vr{to{transform:scale(3.2);opacity:0}}
  .vring{position:fixed;border:3px solid #14B8A6;border-radius:18px;pointer-events:none;z-index:2147483645;
    box-shadow:0 0 0 7px rgba(20,184,166,.16),0 12px 34px rgba(20,184,166,.22);
    opacity:0;transform:scale(1.035);transition:opacity .35s ease,transform .45s cubic-bezier(.2,.7,.2,1)}
  .vring.on{opacity:1;transform:scale(1)}`;
  document.head.appendChild(s);
  const c = document.createElement('div'); c.id = 'vc'; document.body.appendChild(c);
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', e => { c.classList.add('dn'); const r = document.createElement('div'); r.className = 'vr';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px'; document.body.appendChild(r);
    setTimeout(() => r.remove(), 700); }, true);
  addEventListener('mouseup', () => c.classList.remove('dn'), true); };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add);
  window.__ring = (x, y, w, h, ms, pad) => { const r = document.createElement('div'); r.className = 'vring';
    Object.assign(r.style, {left: (x - pad) + 'px', top: (y - pad) + 'px', width: (w + 2 * pad) + 'px', height: (h + 2 * pad) + 'px'});
    document.body.appendChild(r); requestAnimationFrame(() => requestAnimationFrame(() => r.classList.add('on')));
    setTimeout(() => { r.classList.remove('on'); setTimeout(() => r.remove(), 450); }, ms); };
  const scroller = el => { while (el && el !== document.body) { const st = getComputedStyle(el);
      if (/(auto|scroll)/.test(st.overflowY) && el.scrollHeight > el.clientHeight + 2) return el; el = el.parentElement; }
    return document.scrollingElement; };
  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const anim = (sc, to, ms) => new Promise(res => { const from = sc.scrollTop, t0 = performance.now();
    to = Math.max(0, Math.min(to, sc.scrollHeight - sc.clientHeight));
    const step = now => { const k = Math.min((now - t0) / ms, 1); sc.scrollTop = from + (to - from) * ease(k);
      k < 1 ? requestAnimationFrame(step) : res(true); }; requestAnimationFrame(step); });
  window.__scrollTo = (sel, offset, ms) => { const el = document.querySelector(sel); if (!el) return Promise.resolve(false);
    const sc = scroller(el); return anim(sc, sc.scrollTop + el.getBoundingClientRect().top - offset, ms); };
})();
"""


class Screencast:
    def __init__(self, ctx, page, tag):
        self.cdp, self.frames, self.dir = ctx.new_cdp_session(page), [], FRAMES / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cdp.on("Page.screencastFrame", self._on)

    def _on(self, f):
        path = self.dir / f"{len(self.frames):06d}.jpg"
        path.write_bytes(base64.b64decode(f["data"]))
        self.frames.append((f["metadata"]["timestamp"], str(path)))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": f["sessionId"]})

    def start(self):
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": 1920,
                                               "maxHeight": 1080, "everyNthFrame": 1})

    def stop(self):
        self.cdp.send("Page.stopScreencast")


class Rec:
    def __init__(self, page):
        self.p, self.x, self.y = page, 960, 540
        self.segs, self.cur = {}, None

    def wait(self, s):
        self.p.wait_for_timeout(int(s * 1000))

    def seg_start(self, sid):
        self.cur = sid
        self.segs.setdefault(sid, []).append([time.time() - 0.03, None])
        print(f"sahne {sid} segment başladı", flush=True)

    def seg_end(self):
        self.segs[self.cur][-1][1] = time.time()

    def move(self, x, y, s=0.7):
        self.p.mouse.move(x, y, steps=max(int(s * 45), 2))
        self.x, self.y = x, y

    def box(self, loc):
        return loc.bounding_box(timeout=20000)

    def to(self, loc, s=0.7, fx=0.5, fy=0.5):
        b = self.box(loc)
        self.move(b["x"] + b["width"] * fx, b["y"] + b["height"] * fy, s)

    def click(self, loc, s=0.7):
        self.to(loc, s)
        self.wait(0.12)
        self.p.mouse.down()
        self.wait(0.08)
        self.p.mouse.up()

    def ring(self, loc, s=1.8, pad=8, hover=True):
        """Önemli sayı/kart: turkuaz çerçeve (hafif ölçekle girer), imleç yanında; s sn durur."""
        b = self.box(loc)
        if hover:
            self.move(b["x"] + b["width"] * 0.82, b["y"] + b["height"] * 0.78, 0.6)
        self.p.evaluate("([x,y,w,h,ms,p]) => window.__ring(x,y,w,h,ms,p)",
                        [b["x"], b["y"], b["width"], b["height"], int(s * 1000), pad])
        self.wait(s + 0.45)

    def scroll_to(self, sel, offset=150, s=2.0):
        """Yumuşak kaydırma: hedef kartın üstü ekranda `offset` px'e gelir (üst 3/4'te kalsın diye)."""
        self.p.evaluate("([q,o,ms]) => window.__scrollTo(q,o,ms)", [sel, offset, int(s * 1000)])
        self.wait(0.25)

    def open(self, sid, click_loc, title):
        """Sayfa geçişi: segment, yeni sayfanın başlığının DOM'a girdiği karede başlar."""
        self.click(click_loc, 0.8)
        self.p.wait_for_function(
            "t => [...document.querySelectorAll('.iris-title')].some(e => e.textContent.includes(t))",
            arg=title, polling="raf", timeout=60000)
        self.seg_start(sid)
        self.wait(HOLD)

    def nav(self, sid, label, title):
        self.open(sid, self.p.locator('[data-testid="stSidebarNav"] a').filter(has_text=label).first, title)


def record_cards(b):
    out = {}
    for sid, title, sub, credit, dur, fast in [
        ("01", "Müşteri Terki Erken Uyarı Sistemi",
         "Banka müşterilerinin ayrılma riskini önceden görmek, açıklamak ve <b>maliyete göre karar vermek</b> için.",
         "Çanakkale Onsekiz Mart Üniversitesi · İstatistik Bölümü · Bitirme Projesi", 11.5, False),
        ("11", "Veriden karara.", CLOSING_CARD, "", 7.0, True)]:
        ctx = b.new_context(viewport=SIZE)
        pg = ctx.new_page()
        sc = Screencast(ctx, pg, f"kart{sid}")
        pg.set_content("<body style='margin:0;background:#1E3A5F'></body>")
        sc.start()
        pg.set_content(card_html(title, sub, credit, fast))   # animasyonlar duraklatılmış yüklenir
        pg.evaluate("() => document.fonts.ready")
        pg.wait_for_timeout(600)
        t0 = time.time()
        pg.evaluate("() => document.body.classList.remove('wait')")
        pg.wait_for_timeout(int((dur + 0.3) * 1000))
        sc.stop()
        ctx.close()
        out[sid] = {"frames": sc.frames, "segs": [[t0, t0 + dur]]}
        print(f"kart {sid}: {len(sc.frames)} kare", flush=True)
    return out


def make_bands(b):
    BANT.mkdir(exist_ok=True)
    pg = b.new_page(viewport=SIZE)
    n = 0
    for s in SCENES_V2:
        if s["band"]:
            n += 1
            pg.set_content(BAND.format(font=FONT, logo=data_uri("iris-icon-dark.svg"), n=n, t=s["band"]))
            pg.wait_for_timeout(500)
            pg.screenshot(path=str(BANT / f"bant{s['id']}.png"), omit_background=True)
    pg.close()


def record_app(b):
    ctx = b.new_context(viewport=SIZE, color_scheme="light", locale="tr-TR")
    ctx.add_init_script(HELPERS_JS)
    page = ctx.new_page()
    sc = Screencast(ctx, page, "ekran")
    sc.start()
    r = Rec(page)
    P = page.locator
    kpis = P(".iris-kpi")

    # 01b · Giriş ekranı: ilk açılış animasyonu (gösterge yayı + iğne, sayılar), butonlar üzerinde gezinme
    page.goto(URL + "/", wait_until="domcontentloaded")
    # Giriş CSS'i uygulanınca başla (öncesinde kenar çubuklu iskelet bir kare boyanıyor)
    page.wait_for_function("""() => { const h = document.querySelector('.iris-hero .h-value'),
        sb = document.querySelector('[data-testid="stSidebar"]'), hub = document.querySelector('.st-key-hub');
        return h && hub && (!sb || getComputedStyle(sb).display === 'none'); }""", polling="raf", timeout=90000)
    r.seg_start("01b")
    r.wait(HOLD + 0.3)
    for grp in (".st-key-hub-karar a", ".st-key-hub-analiz a"):
        for loc in P(grp).all():
            r.to(loc, 0.42, fx=0.35)
            r.wait(0.22)
    r.ring(P(".st-key-hub-sunum a"), 1.5, pad=6)
    r.seg_end()

    # 02 · Genel Bakış
    r.open("02", P(".st-key-hub-karar a").filter(has_text="Genel Bakış"), "Genel Bakış")
    r.to(kpis.nth(0), 0.6)
    r.wait(0.6)
    r.ring(kpis.filter(has_text="Uyarı listesi").first, 1.7)
    r.move(1200, 640, 0.5)
    r.scroll_to("[class*='st-key-card-']", 150, 2.2)
    r.wait(2.0)
    r.scroll_to(".st-key-card-age-dist", 150, 2.2)
    r.wait(1.8)
    r.seg_end()

    # 03 · Model Karşılaştırma
    r.nav("03", "Model Karşılaştırma", "Model Karşılaştırma")
    r.to(P(".st-key-card-metrics tr").filter(has_text="XGBoost").first, 0.7, fx=0.3)
    r.wait(1.0)
    r.move(1200, 640, 0.5)
    r.scroll_to(".st-key-card-roc", 150, 2.2)
    r.wait(1.8)
    r.scroll_to(".st-key-card-mcnemar", 160, 2.2)
    r.ring(P(".st-key-card-mcnemar"), 2.0, hover=False)
    r.wait(0.6)
    r.seg_end()

    # 04 · Maliyet ve Eşik: recall %82, uyarı listesi 769, eşik 0,21
    r.nav("04", "Maliyet ve Eşik", "Maliyet")
    r.ring(kpis.filter(has_text="Recall").first, 1.8)
    r.ring(kpis.filter(has_text="Uyarı listesi").first, 1.5)
    r.ring(P(".iris-chip").first, 1.5, pad=6)
    r.move(1200, 640, 0.5)
    r.scroll_to(".st-key-card-cost", 140, 2.2)
    plot = P(".st-key-card-cost .js-plotly-plot").first
    r.to(plot, 0.9, fx=0.29, fy=0.55)
    r.wait(1.2)
    r.to(plot, 0.9, fx=0.5, fy=0.45)
    r.wait(1.4)
    r.seg_end()

    # 05 · Modele güvenebilir miyiz?: Brier 0,1041 → 0,1011
    r.nav("05", "Modele güvenebilir", "Modele güvenebilir")
    r.move(1100, 300, 0.7)
    r.wait(1.0)
    r.move(1200, 640, 0.5)
    r.scroll_to(".st-key-card-guven-rel", 130, 2.0)
    rel = P(".st-key-card-guven-rel .js-plotly-plot").first
    for fx, fy in ((0.12, 0.8), (0.35, 0.58), (0.6, 0.42)):
        r.to(rel, 0.75, fx=fx, fy=fy)
        r.wait(0.45)
    r.scroll_to(".iris-kpi", 300, 2.0)
    r.ring(kpis.filter(has_text="SMOTE'lu").first, 1.6)
    r.ring(kpis.filter(has_text="Kalibre").first, 1.8)
    r.wait(0.6)
    r.seg_end()

    # 06 · Risk Skoru (gösterge + şelale) → Açıklanabilirlik (beeswarm)
    r.nav("06", "Müşteri Risk Skoru", "Müşteri Risk Skoru")
    r.to(P(".st-key-card-gauge"), 0.7, fy=0.55)
    r.wait(1.0)
    r.move(1200, 700, 0.5)
    r.scroll_to(".st-key-card-shap", 110, 2.2)
    r.to(P(".st-key-card-shap .js-plotly-plot").first, 0.9, fx=0.45, fy=0.25)
    r.wait(2.0)
    r.seg_end()
    r.nav("06", "Açıklanabilirlik", "Açıklanabilirlik")
    bee = P(".st-key-card-beeswarm .js-plotly-plot").first
    r.to(bee, 0.9, fx=0.45, fy=0.13)
    r.wait(1.2)
    r.to(bee, 0.8, fx=0.4, fy=0.27)
    r.wait(1.4)
    r.seg_end()

    # 07 · Confounding: Almanya OR
    r.nav("07", "Confounding", "Confounding")
    r.to(P(".st-key-card-strat"), 0.8, fy=0.55)
    r.wait(1.2)
    r.move(1200, 640, 0.5)
    r.scroll_to(".st-key-card-or", 140, 2.4)
    r.ring(P(".st-key-card-or"), 2.0, hover=False)
    r.to(P(".st-key-card-or table").first, 0.7, fx=0.3, fy=0.6)
    r.wait(1.4)
    r.seg_end()

    # 08 · Simülatör: yaş 40→~55, Almanya (bakiye zaten 0) → "eğitimde yok" uyarısı üst 3/4'te
    r.nav("08", "Model ne derdi?", "Model ne derdi?")
    thumb = P(".st-key-s_a input[type=range]").locator("xpath=../..")
    r.to(thumb, 0.6)
    r.p.mouse.down()
    r.move(r.x + 60, r.y, 0.9)
    r.p.mouse.up()
    r.wait(1.8)
    r.click(P(".st-key-s_c [data-testid='stSelectbox']"), 0.7)
    r.wait(0.6)
    r.click(page.get_by_role("option", name="Almanya"), 0.5)
    r.wait(1.0)
    r.to(P(".st-key-s_b input"), 0.6, fx=0.3)
    r.wait(1.0)
    r.move(1500, 700, 0.5)
    r.scroll_to(".st-key-card-sim-rarity", 210, 1.8)
    r.ring(P(".st-key-card-sim-rarity"), 2.2, hover=False)
    r.wait(0.8)
    r.seg_end()

    # 09 · Kampanya Planlayıcı: bütçe 769 → 360 (en kârlı), net fayda + eğri
    r.nav("09", "Kampanya Planlayıcı", "Kampanya Planlayıcı")
    r.click(P(".st-key-camp_budget input"), 0.7)
    page.keyboard.press("Control+A")
    page.keyboard.type("360", delay=220)
    page.keyboard.press("Enter")
    r.wait(2.0)
    r.ring(kpis.filter(has_text="Net fayda").first, 1.6)
    r.move(1200, 640, 0.5)
    r.scroll_to(".st-key-card-camp-curve", 130, 2.0)
    r.to(P(".st-key-card-camp-curve .js-plotly-plot").first, 0.8, fx=0.22, fy=0.2)
    r.wait(1.6)
    r.seg_end()

    # 10 · Sunum Modu: iki adım
    r.nav("10", "Sunum Modu", "Sunum Modu")
    for _ in range(2):
        r.click(P(".st-key-sunum_next button"), 0.6)
        r.wait(2.3)
    r.seg_end()

    sc.stop()
    ctx.close()
    return {"frames": sc.frames, "segs": r.segs}


def first_landing_frame(frames, a):
    """Giriş ekranı: sağ panelin lacivert (giriş CSS'i uygulanmış) olduğu ilk kare. Öncesinde Streamlit'in
    kenar çubuklu iskeleti bir-iki kare boyanıyor; DOM koşulu bunu yakalayamıyor (kare zamanı ≠ DOM zamanı)."""
    from PIL import Image
    for t, p in frames:
        if t >= a - 0.2:
            r, g, b_ = Image.open(p).convert("RGB").getpixel((1700, 980))
            if 0.299 * r + 0.587 * g + 0.114 * b_ < 90:
                return t
    return a


def build_clip(frames, segs, out, landing=False):
    """Segment karelerini gerçek zaman damgalarıyla (concat demuxer, kare başına süre) 30 fps H.264'e çevirir."""
    lines, total = [], 0.0
    for a, b in segs:
        if landing:
            a = first_landing_frame(frames, a)
        # Kareler yalnız boyama olunca gelir: işaretten sonraki ilk kare (≤0,6 sn) segmentin başı olur;
        # böylece önceki sayfanın/iskeletin bayat karesi taşınmaz, animasyonun ilk boyanan karesi baştadır.
        first = next((t for t, _ in frames if a <= t <= a + 0.6), None)
        if first is not None:
            a = first
        idx = [i for i, (t, _) in enumerate(frames) if t <= a]
        i = idx[-1] if idx else 0
        seq = [(a, frames[i][1])] + [(t, p) for t, p in frames[i + 1:] if a < t < b]
        for k, (t, p) in enumerate(seq):
            nxt = seq[k + 1][0] if k + 1 < len(seq) else b
            lines.append(f"file '{Path(p).as_posix()}'\nduration {max(nxt - t, 0.001):.4f}\n")
        total += b - a
    lines.append(f"file '{Path(seq[-1][1]).as_posix()}'\n")
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(lines), encoding="utf-8")
    subprocess.run([FF, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-vf", "fps=30,scale=out_range=tv,format=yuv420p", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "16", "-color_range", "tv", "-t", f"{total:.3f}", str(out)], check=True, timeout=900)
    lst.unlink()
    return round(total, 2)


def main():
    KAYIT.mkdir(exist_ok=True)
    raw = FRAMES / "kayit_ham.json"   # kare listesi + segmentler: `python s2_kayit.py klip` kayıtsız klip üretir
    if "klip" in sys.argv[1:]:
        rec = json.loads(raw.read_text(encoding="utf-8"))
    else:
        with sync_playwright() as pw:
            b = pw.chromium.launch(channel="msedge", headless=True)
            make_bands(b)
            rec = record_cards(b)
            if not ONLY_CARDS:
                rec["ekran"] = record_app(b)
            b.close()
        old = json.loads(raw.read_text(encoding="utf-8")) if raw.exists() else {}
        raw.write_text(json.dumps({**old, **rec}), encoding="utf-8")
    meta_path = KAYIT / "isaretler_v2.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for sid in ("01", "11"):
        meta[sid] = {"dur": build_clip(rec[sid]["frames"], rec[sid]["segs"], KAYIT / f"sahne_{sid}.mp4"),
                     "segments": 1}
    if "ekran" in rec:
        print(f"ekran: {len(rec['ekran']['frames'])} kare", flush=True)
        only = [a for a in sys.argv[1:] if a.startswith("sahne=")]  # klip modunda: sahne=01b,02
        want = only[0][6:].split(",") if only else None
        for sid, segs in rec["ekran"]["segs"].items():
            if want and sid not in want:
                continue
            meta[sid] = {"dur": build_clip(rec["ekran"]["frames"], segs, KAYIT / f"sahne_{sid}.mp4",
                                           landing=sid == "01b"), "segments": len(segs)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for k in sorted(meta):
        print(k, meta[k])


if __name__ == "__main__":
    main()

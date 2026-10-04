"""Adım 2: Görüntü kaydı (Playwright + headless Edge, 1920×1080, açık tema).
Ön koşul: s1_ses.py çalışmış (zamanlama.json) ve Iris headless çalışıyor (varsayılan http://localhost:8662).
Çıktı: kayit/kart01.webm, kayit/kart11.webm, kayit/ekran.webm + kayit/isaretler.json, bant/bantXX.png
"""
import base64
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from senaryo import CLOSING_CARD

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
# set_content sayfası file:// yükleyemez → logo data URI olarak gömülür
LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    (ROOT / "logo" / "final" / "iris-logo-header-dark.svg").read_bytes()).decode()
URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8662")
ONLY_CARDS = "kartlar" in sys.argv[1:]   # python s2_kayit.py kartlar → yalnız kartlar + bantlar
KAYIT, BANT = HERE / "kayit", HERE / "bant"
KAYIT.mkdir(exist_ok=True)
BANT.mkdir(exist_ok=True)
T = json.loads((HERE / "zamanlama.json").read_text(encoding="utf-8"))
DUR = {s["id"]: s["dur"] for s in T["scenes"]}
SIZE = {"width": 1920, "height": 1080}
FONT = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">'

# ---------- Açılış / kapanış kartları ----------
CARD = """<!doctype html><html><head><meta charset="utf-8">{font}<style>
html,body{{margin:0;width:1920px;height:1080px;overflow:hidden;font-family:Inter,'Segoe UI',sans-serif}}
body{{background:radial-gradient(1100px 700px at 75% 110%, rgba(20,184,166,.28), transparent 60%),
  radial-gradient(900px 600px at 10% -10%, rgba(45,212,191,.14), transparent 60%), #1E3A5F;
  display:flex;flex-direction:column;align-items:center;justify-content:center;color:#F7FAFC}}
img{{height:150px;opacity:0;transform:scale(.92);animation:in 1.1s cubic-bezier(.2,.7,.2,1) .25s forwards}}
.bar{{width:0;height:6px;border-radius:3px;margin:46px 0 34px;
  background:linear-gradient(90deg,#14B8A6,#F59E0B 50%,#FF7A59);animation:bar 1.2s ease .9s forwards}}
.t{{font-size:46px;font-weight:700;letter-spacing:-.01em;opacity:0;animation:up .9s ease 1.3s forwards;text-align:center}}
.s{{font-size:28px;color:rgba(247,250,252,.78);margin-top:18px;opacity:0;animation:up .9s ease 1.7s forwards;text-align:center;max-width:1500px;line-height:1.5}}
.s b{{color:#2DD4BF;font-weight:600}}
@keyframes in{{to{{opacity:1;transform:scale(1)}}}} @keyframes bar{{to{{width:420px}}}}
@keyframes up{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
</style></head><body><img src="{logo}"><div class="bar"></div><div class="t">{t}</div><div class="s">{s}</div></body></html>"""

BAND = """<!doctype html><html><head><meta charset="utf-8">{font}<style>
html,body{{margin:0;background:transparent;font-family:Inter,'Segoe UI',sans-serif}}
.b{{position:absolute;left:0;right:0;bottom:0;height:64px;display:flex;align-items:center;gap:18px;padding:0 44px;
  background:linear-gradient(90deg,rgba(30,58,95,.97),rgba(30,58,95,.90) 70%,rgba(30,58,95,.80));
  border-top:3px solid #14B8A6;color:#F7FAFC;font-size:27px;font-weight:600;letter-spacing:.005em}}
.n{{display:inline-flex;width:34px;height:34px;border-radius:50%;background:#14B8A6;color:#fff;font-size:18px;
  font-weight:800;align-items:center;justify-content:center}}
.logo{{margin-left:auto;height:30px;opacity:.95}}
</style></head><body><div class="b"><span class="n">{n}</span>{t}<img class="logo" src="{logo}"></div></body></html>"""

# Sahte imleç (headless'ta imleç çizilmez): fare olaylarını izleyen daire + tıklama halkası
CURSOR_JS = r"""
(() => { const add = () => { if (document.getElementById('vc')) return;
  const s = document.createElement('style'); s.textContent = `
  #vc{position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
    background:rgba(255,255,255,.92);border:3px solid #1E3A5F;box-shadow:0 3px 10px rgba(14,30,52,.35);
    z-index:2147483647;pointer-events:none;transition:transform .12s ease}
  #vc.dn{transform:scale(.78)}
  .vr{position:fixed;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;border:3px solid #14B8A6;
    z-index:2147483646;pointer-events:none;animation:vr .6s ease-out forwards}
  @keyframes vr{to{transform:scale(3.2);opacity:0}}`;
  document.head.appendChild(s);
  const c = document.createElement('div'); c.id = 'vc'; c.style.left = '-50px'; document.body.appendChild(c);
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', e => { c.classList.add('dn'); const r = document.createElement('div'); r.className = 'vr';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px'; document.body.appendChild(r);
    setTimeout(() => r.remove(), 700); }, true);
  addEventListener('mouseup', () => c.classList.remove('dn'), true); };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add); })();
"""


class Rec:
    def __init__(self, page):
        self.p, self.x, self.y = page, 960, 540
        self.t0 = time.monotonic()
        self.marks = {}

    def now(self):
        return time.monotonic() - self.t0

    def wait(self, s):
        self.p.wait_for_timeout(int(s * 1000))

    def move(self, x, y, s=0.7):
        steps = max(int(s * 40), 2)
        self.p.mouse.move(x, y, steps=steps)
        self.x, self.y = x, y

    def box(self, loc):
        b = loc.bounding_box(timeout=15000)
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    def to(self, loc, s=0.7, dx=0, dy=0):
        x, y = self.box(loc)
        self.move(x + dx, y + dy, s)

    def click(self, loc, s=0.7, dx=0, dy=0):
        self.to(loc, s, dx, dy)
        self.wait(0.12)
        self.p.mouse.down()
        self.wait(0.08)
        self.p.mouse.up()

    def scroll(self, dy, s):
        """Yavaş kaydırma: s saniyede dy piksel (tekerlek adımlarıyla)."""
        n = max(int(s / 0.03), 1)
        for _ in range(n):
            self.p.mouse.wheel(0, dy / n)
            self.p.wait_for_timeout(30)

    def ready(self, text):
        self.p.locator(".iris-title").filter(has_text=text).first.wait_for(timeout=60000)
        self.p.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60000)

    def nav(self, label, header):
        self.click(self.p.locator('[data-testid="stSidebarNav"] a').filter(has_text=label).first, 0.8)
        self.ready(header)

    def scene(self, sid):
        self.marks[sid] = round(self.now(), 3)
        print(f"sahne {sid} başladı @ {self.marks[sid]:.2f}", flush=True)

    def until(self, sid, extra=0.6):
        """Sahne en az sesi kadar sürsün (+ kesim payı)."""
        left = self.marks[sid] + DUR[sid] + extra - self.now()
        if left > 0:
            self.wait(left)
        else:
            print(f"  ! sahne {sid} {-left:.1f} sn uzun sürdü", flush=True)


def record_card(b, name, title, sub, dur):
    ctx = b.new_context(viewport=SIZE, record_video_dir=str(KAYIT / "tmp"), record_video_size=SIZE)
    pg = ctx.new_page()
    t0 = time.monotonic()
    pg.set_content(CARD.format(font=FONT, logo=LOGO, t=title, s=sub))
    lead = time.monotonic() - t0
    pg.wait_for_timeout(int((dur + 0.8) * 1000))
    path = pg.video.path()
    ctx.close()
    out = KAYIT / f"{name}.webm"
    Path(path).replace(out)
    return round(lead, 3)


def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True)
        # Bant görselleri (şeffaf PNG, 1920×1080; alt 64 px)
        pg = b.new_page(viewport=SIZE)
        for s in T["scenes"]:
            if s["band"]:
                pg.set_content(BAND.format(font=FONT, logo=LOGO, n=s["id"] - 1, t=s["band"]))
                pg.wait_for_timeout(600)
                pg.screenshot(path=str(BANT / f"bant{s['id']:02d}.png"), omit_background=True)
        pg.close()
        marks = {"kart01_lead": record_card(b, "kart01", "Müşteri Terki Erken Uyarı Sistemi",
                                            "Banka müşterilerinin ayrılma riskini önceden görmek, açıklamak ve "
                                            "<b>maliyete göre karar vermek</b> için.", DUR[1]),
                 "kart11_lead": record_card(b, "kart11", "Veriden karara.", CLOSING_CARD, DUR[11])}
        if ONLY_CARDS:
            old = json.loads((KAYIT / "isaretler.json").read_text(encoding="utf-8"))
            (KAYIT / "isaretler.json").write_text(json.dumps({**old, **marks}, indent=2), encoding="utf-8")
            print("kartlar + bantlar yenilendi")
            return

        ctx = b.new_context(viewport=SIZE, color_scheme="light", locale="tr-TR",
                            record_video_dir=str(KAYIT / "tmp"), record_video_size=SIZE)
        ctx.add_init_script(CURSOR_JS)
        page = ctx.new_page()
        r = Rec(page)
        page.goto(URL + "/", wait_until="domcontentloaded")
        page.locator(".iris-hub-foot").wait_for(timeout=60000)
        r.wait(4.0)

        # Süreler zamanlama.json'daki ses sürelerine göre (S2 14,5 · S3 12,4 · S4 14,8 · S5 13,2 · S6 11,3 ·
        # S7 10,9 · S8 10,4 · S9 10,0 · S10 7,0 sn). Fazlası kesilir; eksik kalırsa until() bekler.
        # 2 · Genel Bakış (girişten tıklayarak)
        r.scene(2)
        r.click(page.locator(".st-key-hub-karar a").filter(has_text="Genel Bakış"), 0.9)
        r.ready("Genel Bakış")
        r.wait(1.2)
        kpis = page.locator(".iris-kpi")
        for i in range(min(kpis.count(), 2)):
            r.to(kpis.nth(i), 0.5)
            r.wait(0.6)
        r.move(1150, 700, 0.5)
        r.scroll(520, 2.6)
        r.wait(0.8)
        r.scroll(420, 2.0)
        r.until(2)

        # 3 · Model Karşılaştırma
        r.scene(3)
        r.nav("Model Karşılaştırma", "Model Karşılaştırma")
        r.wait(1.2)
        rows = page.locator("table.iris-table tr")
        for i in (1, 3, 5):
            if rows.count() > i:
                r.to(rows.nth(i), 0.55)
                r.wait(0.6)
        r.move(1150, 650, 0.5)
        r.scroll(650, 3.0)
        r.until(3)

        # 4 · Maliyet ve Eşik (eğri)
        r.scene(4)
        r.nav("Maliyet ve Eşik", "Maliyet")
        r.wait(1.8)
        r.move(1150, 650, 0.6)
        r.scroll(420, 2.6)
        r.wait(0.6)
        plots = page.locator(".js-plotly-plot")
        if plots.count():
            bx = plots.first.bounding_box()
            if bx:
                for fx in (0.15, 0.22, 0.3, 0.22):
                    r.move(bx["x"] + bx["width"] * fx, bx["y"] + bx["height"] * 0.45, 0.8)
                    r.wait(0.5)
        r.until(4)

        # 5 · Modele güvenebilir miyiz?
        r.scene(5)
        r.nav("Modele güvenebilir", "Modele güvenebilir")
        r.wait(1.6)
        r.move(1100, 330, 0.7)
        r.wait(0.9)
        r.move(1150, 700, 0.6)
        r.scroll(340, 2.2)
        r.wait(0.4)
        rel = page.locator(".st-key-card-guven-rel .js-plotly-plot")
        if rel.count():
            bx = rel.first.bounding_box()
            if bx:
                for fx, fy in ((0.1, 0.85), (0.3, 0.6), (0.55, 0.45)):
                    r.move(bx["x"] + bx["width"] * fx, bx["y"] + bx["height"] * fy, 0.7)
                    r.wait(0.4)
        r.until(5)

        # 6 · Risk Skoru (gösterge + şelale) → Açıklanabilirlik (beeswarm)
        r.scene(6)
        r.nav("Müşteri Risk Skoru", "Müşteri Risk Skoru")
        r.wait(1.2)
        r.move(1150, 750, 0.5)
        r.scroll(520, 2.0)
        r.wait(0.8)
        r.nav("Açıklanabilirlik", "Açıklanabilirlik")
        r.wait(1.2)
        bee = page.locator(".st-key-card-beeswarm .js-plotly-plot")
        if bee.count():
            bx = bee.first.bounding_box()
            if bx:
                r.move(bx["x"] + bx["width"] * 0.45, bx["y"] + bx["height"] * 0.12, 0.8)
        r.until(6)

        # 7 · Confounding
        r.scene(7)
        r.nav("Confounding", "Confounding")
        r.wait(1.0)
        r.move(1150, 650, 0.6)
        r.scroll(600, 3.0)
        r.until(7)

        # 8 · Simülatör: yaş kaydırıcısı → Almanya (bakiye zaten 0) → uyarı
        r.scene(8)
        r.nav("Model ne derdi?", "Model ne derdi?")
        r.wait(1.0)
        thumb = page.locator(".st-key-s_a input[type=range]").locator("xpath=../..")  # kaydırıcı tutamacı
        r.to(thumb, 0.5)
        r.p.mouse.down()
        r.move(r.x + 120, r.y, 0.9)
        r.p.mouse.up()
        r.wait(0.9)
        r.click(page.locator(".st-key-s_c [data-testid='stSelectbox']"), 0.6)
        r.wait(0.4)
        r.click(page.get_by_role("option", name="Almanya"), 0.45)
        r.wait(0.7)
        r.move(1500, 800, 0.5)
        r.scroll(380, 1.4)
        r.until(8)

        # 9 · Kampanya Planlayıcı: bütçeyi değiştir → net fayda eğrisi
        r.scene(9)
        r.nav("Kampanya Planlayıcı", "Kampanya Planlayıcı")
        r.wait(1.1)
        inp = page.locator(".st-key-camp_budget input")
        r.click(inp, 0.6)
        page.keyboard.press("Control+A")
        page.keyboard.type("360", delay=170)
        page.keyboard.press("Enter")
        r.wait(1.3)
        r.move(1150, 750, 0.5)
        r.scroll(380, 1.6)
        cp = page.locator(".js-plotly-plot")
        if cp.count():
            bx = cp.first.bounding_box()
            if bx:
                for fx in (0.14, 0.22):
                    r.move(bx["x"] + bx["width"] * fx, bx["y"] + bx["height"] * 0.3, 0.6)
                    r.wait(0.3)
        r.until(9)

        # 10 · Sunum Modu: 2 adım ileri
        r.scene(10)
        r.nav("Sunum Modu", "Sunum Modu")
        r.wait(0.8)
        for _ in range(2):
            r.click(page.locator(".st-key-sunum_next button"), 0.5)
            r.wait(1.1)
        r.until(10)

        path = page.video.path()
        ctx.close()
        b.close()
    Path(path).replace(KAYIT / "ekran.webm")
    (KAYIT / "isaretler.json").write_text(json.dumps({"ekran": r.marks, **marks}, indent=2), encoding="utf-8")
    print("kayıt tamam")


if __name__ == "__main__":
    main()

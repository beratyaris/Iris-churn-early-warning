"""Iris tasarım sistemi: renkler, tema CSS'i, Plotly şablonu ve ortak bileşenler.

Tüm sayfalar bu modülü kullanır; renk/boyut değişikliği tek yerden yapılır.
Bileşenler yalnız HTML/SVG üretir, sayıları çağıran sayfa gerçek veriden hesaplar.
Animasyon ilkesi: yalnız ilk yüklemede (sayı sayma, çubukların büyümesi, ibre dönüşü);
sürekli hareket yok, `prefers-reduced-motion` açıksa hepsi kapanır.
"""
import base64
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).parent
LOGO = ROOT / "logo" / "final"

# ---------- Renkler ----------
TURQ = "#14B8A6"    # birincil, düşük risk
NAVY = "#1E3A5F"    # metin, paneller
CORAL = "#FF7A59"   # vurgu, yüksek risk
AMBER = "#F59E0B"   # orta risk
BG = "#F7FAFC"
SURFACE = "#FFFFFF"
BORDER = "#E3EAF1"
GRID = "#EAF0F5"
MUTED = "#5B6B80"   # ikincil metin (beyaz üzerinde AA)
SUBTLE = "#8494A7"
NEUTRAL = "#A9B8C8"  # "kalan müşteri" gibi risksiz karşılaştırma serisi
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

# Churn ORANI için risk bantları (portföy ortalaması %20,4 ortadaki bantta kalır)
RATE_BANDS = [0.15, 0.25]
BAND_COLORS = [TURQ, AMBER, CORAL]
BAND_LABELS = ["Düşük < %15", "Orta %15–25", "Yüksek ≥ %25"]


# ---------- Risk rengi (OKLab'da ara değer: turkuaz→amber→mercan) ----------
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255


def _to_oklab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ np.cbrt(m1 @ lin)


def _from_oklab(lab):
    m2i = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
    m1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    lin = m1i @ ((m2i @ lab) ** 3)
    lin = np.clip(lin, 0, 1)
    rgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(c * 255)):02X}" for c in np.clip(rgb, 0, 1))


_STOPS = [_to_oklab(_hex_to_rgb(c)) for c in (TURQ, AMBER, CORAL)]


def risk_color(t):
    """t ∈ [0, 1] → turkuaz (düşük) … amber (orta) … mercan (yüksek)."""
    t = float(np.clip(t, 0, 1))
    if t <= 0.5:
        a, b, k = _STOPS[0], _STOPS[1], t / 0.5
    else:
        a, b, k = _STOPS[1], _STOPS[2], (t - 0.5) / 0.5
    return _from_oklab(a + (b - a) * k)


def rate_band(rate):
    """Churn oranı bandı: 0 düşük, 1 orta, 2 yüksek."""
    return int(np.searchsorted(RATE_BANDS, rate, side="right"))


def rate_color(rate):
    """Churn oranı (0-1) için renk. Sürekli ara tonlar turkuaz ile amber arasında
    haki/yeşilimsi bir renge düştüğü için çubuklarda üç net bant kullanılır."""
    return BAND_COLORS[rate_band(rate)]


def proba_color(p, thr):
    """Churn olasılığı için renk: eşik altı turkuaz, eşik–0,5 amber, 0,5 ve üstü mercan."""
    return TURQ if p < thr else (AMBER if p < 0.5 else CORAL)


# ---------- Sayı biçimleri (Türkçe) ----------
def fmt_num(v, dec=0):
    s = f"{v:,.{dec}f}"
    return s.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


def fmt_pct(v, dec=1):
    """0.2037 → '%20,4' (Türkçede yüzde işareti başta)."""
    return "%" + fmt_num(v * 100, dec)


# ---------- Plotly şablonu ----------
def _register_template():
    axis = dict(showgrid=False, zeroline=False, showline=False, ticks="",
                tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=MUTED, size=12)),
                automargin=True)
    pio.templates["iris"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, color=NAVY, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[NAVY, TURQ, CORAL, AMBER, SUBTLE],
        separators=",.",
        xaxis=axis,
        yaxis={**axis, "showgrid": True, "gridcolor": GRID, "gridwidth": 1},
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER,
                        font=dict(family=FONT, color=NAVY, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=MUTED, size=12), title=None),
        margin=dict(l=4, r=4, t=12, b=4),
        bargap=0.38, hovermode="closest",
    ))


_register_template()
PLOTLY_CONFIG = {"displayModeBar": False, "locale": "tr"}


def plot(fig, height=260, key=None):
    """Tüm Plotly grafikleri aynı şablon/ayarla çizilir."""
    fig.update_layout(template="plotly_white+iris", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


# ---------- CSS ----------
_BASE_CSS = f"""
<style>
:root {{
  --iris-turq:{TURQ}; --iris-navy:{NAVY}; --iris-coral:{CORAL}; --iris-amber:{AMBER};
  --iris-bg:{BG}; --iris-surface:{SURFACE}; --iris-border:{BORDER}; --iris-muted:{MUTED};
  --iris-subtle:{SUBTLE};
  --iris-shadow: 0 1px 2px rgba(30,58,95,.04), 0 10px 28px -16px rgba(30,58,95,.22);
  --iris-grad: linear-gradient(90deg,{TURQ} 0%,{AMBER} 50%,{CORAL} 100%);
}}
.stApp {{ background: var(--iris-bg); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem; }}

/* Kenar çubuğu */
[data-testid="stSidebar"] {{ background: var(--iris-surface); border-right: 1px solid var(--iris-border); }}
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] {{
  text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; font-weight: 700;
  color: var(--iris-subtle); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(20,184,166,.12); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: var(--iris-navy); font-weight: 600; }}
.iris-side-label {{ text-transform: uppercase; letter-spacing: .08em; font-size: .7rem;
  font-weight: 700; color: var(--iris-subtle); margin: .25rem 0 -.25rem; }}

/* Sayfa başlığı */
.iris-head {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
  flex-wrap:wrap; margin-bottom:.25rem; animation: iris-fadeup .5s ease-out both; }}
.iris-eyebrow {{ font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700;
  color: var(--iris-turq); }}
.iris-title {{ font-size: 2rem; font-weight: 750; color: var(--iris-navy); line-height:1.15;
  margin:.15rem 0 .2rem; letter-spacing:-.01em; }}
.iris-sub {{ color: var(--iris-muted); font-size:.95rem; }}
.iris-chip {{ display:inline-flex; align-items:center; gap:.45rem; background: var(--iris-surface);
  border:1px solid var(--iris-border); border-radius:999px; padding:.35rem .8rem; font-size:.82rem;
  color: var(--iris-navy); box-shadow: var(--iris-shadow); }}
.iris-chip i {{ width:.55rem; height:.55rem; border-radius:50%; background: var(--iris-turq);
  display:inline-block; }}

/* Kartlar: st.container(key="card-...") */
[class*="st-key-card-"], .st-key-filterbar {{
  background: var(--iris-surface); border: 1px solid var(--iris-border); border-radius: 16px;
  padding: 16px 18px 10px; box-shadow: var(--iris-shadow);
  animation: iris-fadeup .55s ease-out both; }}
.st-key-filterbar {{ padding: 12px 18px 6px; }}
.iris-card-title {{ font-weight: 650; color: var(--iris-navy); font-size: 1rem; }}
.iris-card-sub {{ color: var(--iris-muted); font-size: .8rem; margin-top: .1rem; }}

/* KPI kartları */
.iris-kpis {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 16px; }}
@media (max-width: 1100px) {{ .iris-kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
@media (max-width: 640px) {{ .iris-kpis {{ grid-template-columns: 1fr; }} }}
.iris-kpi {{ background: var(--iris-surface); border:1px solid var(--iris-border); border-radius:16px;
  padding: 16px 18px 14px; box-shadow: var(--iris-shadow); position:relative; overflow:hidden;
  animation: iris-fadeup .5s ease-out both; }}
.iris-kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  background: var(--tone, var(--iris-turq)); }}
.iris-kpi .k-label {{ color: var(--iris-muted); font-size:.82rem; font-weight:600; }}
.iris-kpi .k-row {{ display:flex; align-items:flex-end; justify-content:space-between; gap:.75rem;
  margin:.35rem 0 .45rem; }}
.iris-kpi .k-value {{ font-size: 2.05rem; font-weight: 750; color: var(--iris-navy);
  letter-spacing:-.02em; line-height:1; font-variant-numeric: tabular-nums; white-space:nowrap; }}
.iris-kpi .k-value small {{ font-size: 1rem; font-weight:600; color: var(--iris-muted); }}
.iris-kpi .k-mini {{ flex: 0 0 auto; }}
.iris-kpi .k-mini svg {{ display:block; }}
.iris-kpi .k-sub {{ color: var(--iris-muted); font-size:.8rem; line-height:1.35; }}
.iris-kpi .k-sub b {{ color: var(--iris-navy); font-weight:650; }}
.iris-kpi .k-cap {{ color: var(--iris-subtle); font-size:.7rem; margin-top:.3rem; }}
.iris-delta {{ display:inline-flex; align-items:center; gap:.2rem; font-weight:650;
  border-radius:999px; padding:.05rem .45rem; font-size:.76rem; }}
.iris-delta.up {{ color:#B5462B; background: rgba(255,122,89,.14); }}
.iris-delta.down {{ color:#0F766E; background: rgba(20,184,166,.14); }}
.iris-delta.flat {{ color: var(--iris-muted); background: #EEF2F6; }}

.iris-section {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
  flex-wrap:wrap; margin: .6rem 0 -.2rem; }}

/* Risk ölçeği göstergesi */
.iris-scale {{ display:flex; align-items:center; gap:.6rem; color: var(--iris-muted); font-size:.78rem; }}
.iris-scale .bar {{ width:140px; height:8px; border-radius:99px; background: var(--iris-grad); }}
.iris-scale .sw {{ display:inline-flex; align-items:center; gap:.35rem; }}
.iris-scale .sw i {{ width:.7rem; height:.7rem; border-radius:3px; display:inline-block; }}
.iris-kpi .k-mini img {{ display:block; }}

/* Animasyonlar (yalnız ilk görünüm) */
@keyframes iris-fadeup {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}
@keyframes iris-grow {{ from {{ transform: scaleY(0); }} to {{ transform: scaleY(1); }} }}
@keyframes iris-draw {{ from {{ stroke-dashoffset: 1; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes iris-fade {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
.stPlotlyChart .barlayer .point path {{ transform-box: fill-box; transform-origin: 50% 100%;
  animation: iris-grow .9s cubic-bezier(.2,.8,.2,1) both; }}
.stPlotlyChart .barlayer .point text, .stPlotlyChart .shapelayer, .stPlotlyChart .annotation {{
  animation: iris-fade .6s .55s ease-out both; }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""

# Sayıları 0'dan hedefe sayar (bir kez; öğe değişirse yeniden). JS kapalıysa HTML'deki
# son değer zaten doğru olduğu için hiçbir şey bozulmaz.
_COUNTUP_JS = """
<script>
(function () {
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmt = (v, d) => v.toLocaleString('tr-TR', {minimumFractionDigits: d, maximumFractionDigits: d});
  document.querySelectorAll('[data-countup]:not([data-cu-done])').forEach(function (el) {
    el.setAttribute('data-cu-done', '1');
    if (reduce) return;
    const to = parseFloat(el.dataset.countup), d = +(el.dataset.dec || 0);
    const pre = el.dataset.pre || '', suf = el.dataset.suf || '', dur = +(el.dataset.dur || 1100);
    const t0 = performance.now();
    function step(t) {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = pre + fmt(to * e, d) + suf;
      if (k < 1) requestAnimationFrame(step);
    }
    el.textContent = pre + fmt(0, d) + suf;
    requestAnimationFrame(step);
  });
})();
</script>
"""


def inject_base_css():
    st.html(_BASE_CSS)


def html(body, js=False):
    """st.html; js=True ise sayı sayma betiği eklenir."""
    if js:
        st.html(body + _COUNTUP_JS, unsafe_allow_javascript=True)
    else:
        st.html(body)


# ---------- Bileşenler ----------
def count(value, dec=0, pre="", suf="", dur=1100):
    """Sayan sayı: HTML'de son değer yazılıdır, JS 0'dan sayar."""
    text = pre + fmt_num(value, dec) + suf
    return (f'<span data-countup="{value:.6f}" data-dec="{dec}" data-pre="{escape(pre)}" '
            f'data-suf="{escape(suf)}" data-dur="{dur}">{escape(text)}</span>')


def page_header(eyebrow, title, subtitle="", chip=""):
    chip_html = f'<div class="iris-chip"><i></i>{chip}</div>' if chip else ""
    st.html(f'<div class="iris-head"><div><div class="iris-eyebrow">{escape(eyebrow)}</div>'
            f'<div class="iris-title">{escape(title)}</div>'
            f'<div class="iris-sub">{subtitle}</div></div>{chip_html}</div>')


def card_title(title, subtitle=""):
    sub = f'<div class="iris-card-sub">{subtitle}</div>' if subtitle else ""
    st.html(f'<div class="iris-card-title">{escape(title)}</div>{sub}')


def kpi_card(label, value_html, sub="", mini="", tone=TURQ, cap=""):
    cap_html = f'<div class="k-cap">{cap}</div>' if cap else ""
    return (f'<div class="iris-kpi" style="--tone:{tone}"><div class="k-label">{escape(label)}</div>'
            f'<div class="k-row"><div class="k-value">{value_html}</div>'
            f'<div class="k-mini">{mini}</div></div><div class="k-sub">{sub}</div>{cap_html}</div>')


def kpi_row(cards):
    html('<div class="iris-kpis">' + "".join(cards) + "</div>", js=True)


def delta_badge(diff_pp, better_when_lower=True):
    """Puan farkı rozeti; churn için yükseliş mercan, düşüş turkuaz."""
    if abs(diff_pp) < 0.05:
        return '<span class="iris-delta flat">± 0,0 puan</span>'
    up = diff_pp > 0
    cls = ("up" if up else "down") if better_when_lower else ("down" if up else "up")
    arrow = "▲" if up else "▼"
    return f'<span class="iris-delta {cls}">{arrow} {fmt_num(abs(diff_pp), 1)} puan</span>'


def svg_img(svg, width=None, height=None, alt="", cls=""):
    """SVG'yi <img> (data URI) olarak döndürür. st.html satır içi SVG'yi temizlediği için
    tüm SVG'ler böyle gömülür; SVG içindeki CSS animasyonları img içinde de çalışır."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    size = (f' width="{width}"' if width else "") + (f' height="{height}"' if height else "")
    return f'<img class="{cls}" src="{uri}"{size} alt="{escape(alt)}" title="{escape(alt)}">'


_SVG_ANIM = """<style>
@keyframes g {{ from {{ transform: scaleY(0); }} to {{ transform: scaleY(1); }} }}
@keyframes d {{ from {{ stroke-dashoffset: 1; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes f {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
rect.b {{ transform-box: fill-box; transform-origin: 50% 100%; animation: g .8s cubic-bezier(.2,.8,.2,1) both; }}
.l {{ stroke-dasharray: 1; animation: d 1.2s .15s cubic-bezier(.3,.7,.2,1) both; }}
.a, circle, line {{ animation: f .6s .9s ease-out both; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>"""


def sparkline(values, color, width=112, height=40, title=""):
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    span = (hi - lo) or 1.0
    xs = np.linspace(3, width - 5, len(v))
    ys = height - 5 - (v - lo) / span * (height - 12)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    line = "M" + pts.replace(" ", " L")
    area = f"{line} L{xs[-1]:.1f},{height} L{xs[0]:.1f},{height} Z"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">{_SVG_ANIM.format()}'
           f'<path class="a" d="{area}" fill="{color}" fill-opacity=".14"/>'
           f'<path class="l" d="{line}" pathLength="1" fill="none" stroke="{color}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
           f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" fill="{color}" stroke="#fff" '
           f'stroke-width="1.5"/></svg>')
    return svg_img(svg, width, height, title)


def minibars(values, colors, width=112, height=40, title="", marker=None):
    """Küçük çubuklar (dağılım). marker: 0-1 arası x konumunda dikey eşik çizgisi."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    gap = 2
    bw = (width - gap * (n - 1)) / n
    top = v.max() or 1.0
    parts = []
    for i, (val, col) in enumerate(zip(v, colors)):
        h = max(val / top * (height - 4), 1.5 if val > 0 else 0)
        x = i * (bw + gap)
        parts.append(f'<rect class="b" x="{x:.1f}" y="{height - h:.1f}" width="{bw:.1f}" '
                     f'height="{h:.1f}" rx="{min(2, bw / 2):.1f}" fill="{col}" '
                     f'style="animation-delay:{i * 0.025:.3f}s"/>')
    if marker is not None:
        mx = min(max(marker * width, 1), width - 1)
        parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="0" y2="{height}" stroke="{NAVY}" '
                     f'stroke-width="1.5" stroke-dasharray="3 2"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">{_SVG_ANIM.format()}{"".join(parts)}</svg>')
    return svg_img(svg, width, height, title)


def band_legend():
    """Churn oranı bantları için renk anahtarı."""
    items = "".join(f'<span class="sw"><i style="background:{c}"></i>{lab}</span>'
                    for c, lab in zip(BAND_COLORS, BAND_LABELS))
    return f'<div class="iris-scale">{items}</div>'


# ---------- Giriş sayfası göstergesi ----------
def gauge_svg(share):
    """Iris sembolünün yayı = hız göstergesi. share ∈ [0,1] ibre konumunu verir.
    İbre yalnız ilk yüklemede -90°'den hedef açıya döner (SVG içi CSS)."""
    angle = -90 + 180 * float(np.clip(share, 0, 1))
    cx, cy, r = 160, 172, 118
    ticks = []
    for i in range(11):
        a = np.pi * (1 - i / 10)
        r1, r2 = r + 32, r + (44 if i % 5 == 0 else 39)
        x1, y1 = cx + r1 * np.cos(a), cy - r1 * np.sin(a)
        x2, y2 = cx + r2 * np.cos(a), cy - r2 * np.sin(a)
        w = 2.4 if i % 5 == 0 else 1.5
        ticks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{NAVY}" stroke-opacity=".4" stroke-width="{w}" stroke-linecap="round"/>')
    font = "font-family=\"'Segoe UI', Inter, sans-serif\" font-size=\"12\" font-weight=\"600\" " \
           f"fill=\"{NAVY}\" fill-opacity=\".6\" text-anchor=\"middle\""
    labels = (f'<text x="{cx - r}" y="{cy + 44}" {font}>%0 · düşük</text>'
              f'<text x="{cx}" y="{cy - r - 52}" {font}>%50</text>'
              f'<text x="{cx + r}" y="{cy + 44}" {font}>%100 · yüksek</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -8 320 236">
<style>
.n {{ transform-box: view-box; transform-origin: {cx}px {cy}px; transform: rotate({angle:.2f}deg);
      animation: n 1.8s .3s cubic-bezier(.3,1.35,.45,1) both; }}
@keyframes n {{ from {{ transform: rotate(-90deg); }} to {{ transform: rotate({angle:.2f}deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .n {{ animation: none; }} }}
</style>
<defs>
  <linearGradient id="gg" x1="{cx - r}" y1="0" x2="{cx + r}" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="{TURQ}"/><stop offset=".5" stop-color="{AMBER}"/>
    <stop offset="1" stop-color="{CORAL}"/>
  </linearGradient>
  <filter id="gs" x="-20%" y="-20%" width="140%" height="170%">
    <feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="{NAVY}" flood-opacity=".18"/>
  </filter>
</defs>
<path d="M{cx - r} {cy} A{r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="#FFFFFF"
      stroke-opacity=".6" stroke-width="54" stroke-linecap="round"/>
<path d="M{cx - r} {cy} A{r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="url(#gg)"
      stroke-width="44" stroke-linecap="round" filter="url(#gs)"/>
{''.join(ticks)}{labels}
<g class="n"><path d="M{cx - 5.5} {cy} L{cx} {cy - r + 6} L{cx + 5.5} {cy} Z" fill="{NAVY}"
   stroke="{NAVY}" stroke-width="3" stroke-linejoin="round"/></g>
<circle cx="{cx}" cy="{cy}" r="27" fill="{NAVY}" filter="url(#gs)"/>
<circle cx="{cx - 8.5}" cy="{cy - 8.5}" r="7.5" fill="#FFFFFF"/>
</svg>"""
    return svg_img(svg, alt=f"Portföy risk göstergesi: {fmt_pct(share)}", cls="iris-gauge")


_LANDING_CSS = f"""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarHeader"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding: 0 !important; max-width: 100% !important; }}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:first-of-type {{ gap: 0 !important; }}
.st-key-hero, .st-key-hub {{ min-height: 100vh; justify-content: center; }}
.st-key-hero {{
  background:
    radial-gradient(1200px 600px at 20% 10%, rgba(255,255,255,.75), transparent 60%),
    radial-gradient(900px 700px at 90% 100%, rgba(20,184,166,.28), transparent 60%),
    linear-gradient(160deg, #EFFBF9 0%, #CFF1EC 45%, #A7E3DA 100%);
  padding: 48px 56px; align-items: center; }}
.st-key-hub {{ background: {NAVY}; padding: 44px 56px; color: #F7FAFC;
  box-shadow: -18px 0 40px -24px rgba(8,20,36,.55); }}
.iris-hero {{ width: 100%; max-width: 640px; margin: 0 auto; text-align: center; }}
.iris-gauge {{ width: 100%; max-width: 560px; height: auto; display:block; margin: 0 auto; }}
.iris-hero .h-kicker {{ font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
  font-weight:700; color:#0F766E; margin-top: .5rem; }}
.iris-hero .h-value {{ font-size: 4.2rem; font-weight: 800; color: {NAVY}; line-height:1;
  letter-spacing:-.03em; margin:.35rem 0 .4rem; font-variant-numeric: tabular-nums; }}
.iris-hero .h-band {{ display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .9rem;
  border-radius:999px; background: rgba(255,255,255,.7); color:{NAVY}; font-weight:650;
  font-size:.92rem; box-shadow: 0 6px 18px -10px rgba(30,58,95,.35); }}
.iris-hero .h-band i {{ width:.65rem; height:.65rem; border-radius:50%; display:inline-block; }}
.iris-hero .h-def {{ color:#34506F; font-size:.84rem; line-height:1.5; margin: 1.1rem auto 0;
  max-width: 520px; }}
.iris-hub-logo img {{ height: 44px; width: auto; display:block; }}
.iris-hub-title {{ font-size: 2.05rem; font-weight: 750; line-height: 1.18; letter-spacing:-.01em;
  margin: 1.6rem 0 .45rem; color:#FFFFFF; }}
.iris-hub-sub {{ color: rgba(247,250,252,.66); font-size: .92rem; line-height:1.5; }}
.iris-stats {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin: 1.5rem 0 .6rem; }}
.iris-stat {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.09);
  border-radius: 14px; padding: 12px 14px; }}
.iris-stat .s-val {{ font-size: 1.55rem; font-weight: 750; color:#FFFFFF; line-height:1.1;
  font-variant-numeric: tabular-nums; }}
.iris-stat .s-lab {{ font-size: .74rem; color: rgba(247,250,252,.66); margin-top:.25rem; }}
.iris-stat .s-dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%;
  margin-right:.35rem; vertical-align: middle; }}
.iris-group {{ font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700;
  color: {TURQ}; margin: 1.1rem 0 -.2rem; }}
.st-key-hub [data-testid="stPageLink"] a {{
  display:flex; align-items:center; gap: 14px; width: 100%; box-sizing: border-box;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
  border-radius: 999px; padding: 8px 22px 8px 10px !important; text-decoration: none;
  transition: background .18s ease, border-color .18s ease, transform .18s ease; }}
.st-key-hub [data-testid="stPageLink"] a:hover {{ background: rgba(20,184,166,.16);
  border-color: rgba(20,184,166,.55); transform: translateX(3px); }}
.st-key-hub [data-testid="stPageLink"] a::after {{ content: "→"; margin-left: auto;
  color: rgba(247,250,252,.45); font-size: 1.1rem; }}
.st-key-hub [data-testid="stPageLink"] a p,
.st-key-hub [data-testid="stPageLink"] a span {{ color: #FFFFFF !important; font-size: 1.04rem !important;
  font-weight: 600; }}
.st-key-hub [data-testid="stPageLink"] [data-testid="stIconMaterial"] {{
  width: 42px; height: 42px; min-width: 42px; border-radius: 50%; display:inline-flex;
  align-items:center; justify-content:center; background: rgba(20,184,166,.16);
  color: {TURQ} !important; font-size: 1.35rem !important; margin: 0 !important; }}
.st-key-hub [data-testid="stPageLink"] a > span:first-child {{ width: 42px !important;
  min-width: 42px; height: 42px; display: inline-flex; margin: 0 !important; }}
.st-key-hub-analiz [data-testid="stPageLink"] [data-testid="stIconMaterial"] {{
  background: rgba(255,122,89,.15); color: {CORAL} !important; }}
.iris-hub-foot {{ color: rgba(247,250,252,.45); font-size:.74rem; margin-top: 1.2rem; }}
@media (max-width: 900px) {{
  .st-key-hero, .st-key-hub {{ min-height: auto; padding: 32px 20px; }}
  .iris-stats {{ grid-template-columns: 1fr 1fr 1fr; }}
}}
</style>
"""


def inject_landing_css():
    st.html(_LANDING_CSS)


def logo_img(name, alt="Iris"):
    """logo/final içindeki bir SVG'yi <img> olarak döndürür."""
    return svg_img((LOGO / name).read_text(encoding="utf-8"), alt=alt)

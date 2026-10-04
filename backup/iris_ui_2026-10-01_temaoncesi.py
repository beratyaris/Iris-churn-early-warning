"""Iris tasarım sistemi: renkler, açık/koyu tema, CSS, Plotly şablonu ve ortak bileşenler.

Tüm sayfalar bu modülü kullanır; renk/boyut değişikliği tek yerden yapılır.
Bileşenler yalnız HTML/SVG üretir, sayıları çağıran sayfa gerçek veriden hesaplar.
Animasyon ilkesi: yalnız ilk yüklemede (sayı sayma, çubukların büyümesi, ibre dönüşü);
sürekli hareket yok, `prefers-reduced-motion` açıksa hepsi kapanır.
Tema: Streamlit'in açık/koyu seçimi `st.context.theme.type` ile okunur; CSS değişkenleri,
Plotly şablonu, gösterge ve logo buna göre seçilir.
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

# ---------- Marka renkleri (iki temada aynı) ----------
TURQ = "#14B8A6"    # birincil, düşük risk
NAVY = "#1E3A5F"    # marka laciverti
CORAL = "#FF7A59"   # vurgu, yüksek risk
AMBER = "#F59E0B"   # orta / ortalamaya yakın
BG = "#F7FAFC"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

# ---------- Tema jetonları ----------
THEMES = {
    "light": dict(
        bg=BG, surface="#FFFFFF", border="#E3EAF1", grid="#EAF0F5", ink=NAVY,
        muted="#5B6B80", subtle="#8494A7", neutral="#A9B8C8", soft="#EEF2F6",
        accent_text="#0F766E", up_text="#B5462B", down_text="#0F766E", amber_text="#92400E",
        shadow="0 1px 2px rgba(30,58,95,.04), 0 10px 28px -16px rgba(30,58,95,.22)",
        hub="#1E3A5F", france="#3E6A9E"),
    "dark": dict(
        bg="#0D1929", surface="#14253A", border="#24374F", grid="#1E3047", ink="#E8EFF6",
        muted="#A7B6C8", subtle="#8799AF", neutral="#5E718A", soft="#1C2F46",
        accent_text="#2DD4BF", up_text="#FFA58E", down_text="#5EEAD4", amber_text="#FBBF24",
        shadow="0 1px 2px rgba(0,0,0,.35), 0 12px 28px -16px rgba(0,0,0,.7)",
        hub="#0A1522", france="#7FA7D6"),
}


def mode():
    """Etkin tema: 'light' ya da 'dark' (bilinmiyorsa açık)."""
    try:
        t = st.context.theme.type
    except Exception:  # AppTest / eski sürüm
        t = None
    return "dark" if t == "dark" else "light"


def tok(name):
    """Etkin temadaki renk jetonu (ör. tok('ink'))."""
    return THEMES[mode()][name]


# Geri uyumluluk: eski sabit adlar (açık tema değerleri)
SURFACE, BORDER, GRID = "#FFFFFF", "#E3EAF1", "#EAF0F5"
MUTED, SUBTLE, NEUTRAL = "#5B6B80", "#8494A7", "#A9B8C8"


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


# ---------- Ortalamaya göre renk (Wilson %95 GA kuralı) ----------
REL_COLORS = {"below": TURQ, "near": AMBER, "above": CORAL}
REL_LABELS = {"below": "Ortalamanın altında", "near": "Ortalamadan ayırt edilemez",
              "above": "Ortalamanın üstünde"}


def rel_class(lo, hi, avg):
    """Grubun %95 GA'sı [lo, hi] portföy ortalamasını dışarıda bırakıyorsa altında/üstünde,
    içeriyorsa 'ayırt edilemez'."""
    if hi < avg:
        return "below"
    if lo > avg:
        return "above"
    return "near"


def rel_color(lo, hi, avg):
    return REL_COLORS[rel_class(lo, hi, avg)]


def proba_color(p, thr):
    """Churn skoru için renk: eşik altı turkuaz, eşik ve üstü (uyarı listesi) mercan."""
    return TURQ if p < thr else CORAL


# ---------- Sayı biçimleri (Türkçe) ----------
def fmt_num(v, dec=0):
    s = f"{v:,.{dec}f}"
    return s.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


def fmt_pct(v, dec=1):
    """0.2037 → '%20,4' (Türkçede yüzde işareti başta)."""
    return "%" + fmt_num(v * 100, dec)


def fmt_p(v):
    """p-değeri: '< 0,001' ya da '0,034'."""
    return "< 0,001" if v < 0.001 else fmt_num(v, 3)


def fmt_signed(v, dec=1):
    return ("+" if v > 0 else "−" if v < 0 else "±") + fmt_num(abs(v), dec)


# ---------- Plotly şablonu ----------
def _template(t):
    axis = dict(showgrid=False, zeroline=False, showline=False, ticks="",
                tickfont=dict(color=t["muted"], size=12),
                title=dict(font=dict(color=t["muted"], size=12)), automargin=True)
    return go.layout.Template(layout=dict(
        font=dict(family=FONT, color=t["ink"], size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[t["ink"], TURQ, CORAL, AMBER, t["subtle"]],
        separators=",.",
        xaxis=axis,
        yaxis={**axis, "showgrid": True, "gridcolor": t["grid"], "gridwidth": 1},
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border"],
                        font=dict(family=FONT, color=t["ink"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=t["muted"], size=12), title=None, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=4, r=4, t=12, b=4),
        bargap=0.38, hovermode="closest",
        coloraxis=dict(colorbar=dict(tickfont=dict(color=t["muted"]), outlinewidth=0)),
    ))


for _m, _t in THEMES.items():
    pio.templates[f"iris_{_m}"] = _template(_t)
PLOTLY_CONFIG = {"displayModeBar": False, "locale": "tr"}


def plot(fig, height=260, key=None):
    """Tüm Plotly grafikleri aynı şablon/ayarla çizilir (temaya göre)."""
    base = "plotly_dark" if mode() == "dark" else "plotly_white"
    fig.update_layout(template=f"{base}+iris_{mode()}", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


# ---------- CSS ----------
def _vars(t):
    return (f"--iris-turq:{TURQ}; --iris-navy:{NAVY}; --iris-coral:{CORAL}; --iris-amber:{AMBER};"
            f"--iris-bg:{t['bg']}; --iris-surface:{t['surface']}; --iris-border:{t['border']};"
            f"--iris-ink:{t['ink']}; --iris-muted:{t['muted']}; --iris-subtle:{t['subtle']};"
            f"--iris-soft:{t['soft']}; --iris-accent-text:{t['accent_text']};"
            f"--iris-up-text:{t['up_text']}; --iris-down-text:{t['down_text']};"
            f"--iris-amber-text:{t['amber_text']}; --iris-shadow:{t['shadow']};"
            f"--iris-grad: linear-gradient(90deg,{TURQ} 0%,{AMBER} 50%,{CORAL} 100%);")


_BASE_CSS = """
<style>
:root { __VARS__ }
.stApp { background: var(--iris-bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem; }

/* Kenar çubuğu */
[data-testid="stSidebar"] { background: var(--iris-surface); border-right: 1px solid var(--iris-border); }
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] {
  text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; font-weight: 700;
  color: var(--iris-subtle); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(20,184,166,.14); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--iris-ink); font-weight: 600; }
.iris-side-label { text-transform: uppercase; letter-spacing: .08em; font-size: .7rem;
  font-weight: 700; color: var(--iris-subtle); margin: .25rem 0 -.25rem; }
.iris-side-note { color: var(--iris-muted); font-size: .78rem; line-height: 1.45; }
.iris-side-note b { color: var(--iris-ink); }

/* Sayfa başlığı */
.iris-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
  flex-wrap:wrap; margin-bottom:.25rem; animation: iris-fadeup .5s ease-out both; }
.iris-eyebrow { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700;
  color: var(--iris-accent-text); }
.iris-title { font-size: 2rem; font-weight: 750; color: var(--iris-ink); line-height:1.15;
  margin:.15rem 0 .2rem; letter-spacing:-.01em; }
.iris-sub { color: var(--iris-muted); font-size:.95rem; }
.iris-chip { display:inline-flex; align-items:center; gap:.45rem; background: var(--iris-surface);
  border:1px solid var(--iris-border); border-radius:999px; padding:.35rem .8rem; font-size:.82rem;
  color: var(--iris-ink); box-shadow: var(--iris-shadow); }
.iris-chip i { width:.55rem; height:.55rem; border-radius:50%; background: var(--iris-turq);
  display:inline-block; }

/* Kartlar: st.container(key="card-...") */
[class*="st-key-card-"], .st-key-filterbar {
  background: var(--iris-surface); border: 1px solid var(--iris-border); border-radius: 16px;
  padding: 16px 18px 12px; box-shadow: var(--iris-shadow);
  animation: iris-fadeup .55s ease-out both; }
.st-key-filterbar { padding: 12px 18px 6px; }
.iris-card-title { font-weight: 650; color: var(--iris-ink); font-size: 1rem; }
.iris-card-sub { color: var(--iris-muted); font-size: .8rem; margin-top: .1rem; line-height:1.45; }
.iris-card-sub b { color: var(--iris-ink); }

/* KPI kartları */
.iris-kpis { display:grid; grid-template-columns: repeat(var(--cols, 4), minmax(0,1fr)); gap: 16px; }
@media (max-width: 1100px) { .iris-kpis { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 640px) { .iris-kpis { grid-template-columns: 1fr; } }
.iris-kpi { background: var(--iris-surface); border:1px solid var(--iris-border); border-radius:16px;
  padding: 16px 18px 14px; box-shadow: var(--iris-shadow); position:relative; overflow:hidden;
  animation: iris-fadeup .5s ease-out both; }
.iris-kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  background: var(--tone, var(--iris-turq)); }
.iris-kpi .k-label { color: var(--iris-muted); font-size:.82rem; font-weight:600;
  display:flex; align-items:center; gap:.35rem; }
.iris-kpi .k-label .k-info { display:inline-flex; width:15px; height:15px; border-radius:50%;
  border:1px solid var(--iris-subtle); color: var(--iris-subtle); font-size:.62rem; font-weight:700;
  align-items:center; justify-content:center; cursor:help; font-style:normal; }
.iris-kpi .k-row { display:flex; align-items:flex-end; justify-content:space-between; gap:.75rem;
  margin:.35rem 0 .45rem; }
.iris-kpi .k-value { font-size: 2.05rem; font-weight: 750; color: var(--iris-ink);
  letter-spacing:-.02em; line-height:1; font-variant-numeric: tabular-nums; white-space:nowrap; }
.iris-kpi .k-value small { font-size: 1rem; font-weight:600; color: var(--iris-muted); }
.iris-kpi .k-mini { flex: 0 0 auto; }
.iris-kpi .k-mini img { display:block; }
.iris-kpi .k-sub { color: var(--iris-muted); font-size:.8rem; line-height:1.4; }
.iris-kpi .k-sub b { color: var(--iris-ink); font-weight:650; }
.iris-delta { display:inline-flex; align-items:center; gap:.2rem; font-weight:650;
  border-radius:999px; padding:.05rem .45rem; font-size:.76rem; }
.iris-delta.up { color: var(--iris-up-text); background: rgba(255,122,89,.16); }
.iris-delta.down { color: var(--iris-down-text); background: rgba(20,184,166,.16); }
.iris-delta.flat { color: var(--iris-muted); background: var(--iris-soft); }

.iris-section { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
  flex-wrap:wrap; margin: .6rem 0 -.2rem; }

/* Renk anahtarı */
.iris-scale { display:flex; align-items:center; flex-wrap:wrap; gap:.4rem .9rem;
  color: var(--iris-muted); font-size:.78rem; }
.iris-scale .sw { display:inline-flex; align-items:center; gap:.35rem; }
.iris-scale .sw i { width:.7rem; height:.7rem; border-radius:3px; display:inline-block; }
.iris-scale .sw i.line { height:2px; width:1rem; border-radius:0; }
.iris-scale .rule { flex-basis:100%; text-align:right; font-size:.72rem; color: var(--iris-subtle); }

/* Tablolar */
.iris-table-wrap { overflow-x:auto; }
table.iris-table { width:100%; border-collapse:separate; border-spacing:0; font-size:.86rem;
  color: var(--iris-ink); font-variant-numeric: tabular-nums; }
table.iris-table th { text-align:left; font-weight:650; color: var(--iris-muted); font-size:.74rem;
  text-transform:uppercase; letter-spacing:.05em; padding:.55rem .7rem;
  border-bottom:1px solid var(--iris-border); white-space:nowrap; }
table.iris-table td { padding:.55rem .7rem; border-bottom:1px solid var(--iris-border); }
table.iris-table tr:last-child td { border-bottom:none; }
table.iris-table td.num, table.iris-table th.num { text-align:right; white-space:nowrap; }
table.iris-table td:first-child { white-space:nowrap; }
table.iris-table td.best { font-weight:750; color: var(--iris-accent-text); }
table.iris-table tr.sel td { background: rgba(20,184,166,.08); }
.iris-pill { display:inline-block; border-radius:999px; padding:.08rem .55rem; font-size:.74rem;
  font-weight:650; white-space:nowrap; }
.iris-pill.ok { color: var(--iris-down-text); background: rgba(20,184,166,.16); }
.iris-pill.warn { color: var(--iris-up-text); background: rgba(255,122,89,.16); }
.iris-pill.flat { color: var(--iris-muted); background: var(--iris-soft); }
.iris-dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.4rem; }

/* Açıklama kutusu */
.iris-callout { background: var(--iris-surface); border:1px solid var(--iris-border);
  border-left: 4px solid var(--tone, var(--iris-turq)); border-radius: 14px; padding: 14px 18px;
  box-shadow: var(--iris-shadow); color: var(--iris-ink); font-size:.9rem; line-height:1.55; }
.iris-callout .c-title { font-weight:700; margin-bottom:.35rem; }
.iris-callout ul { margin:.2rem 0 0 1.1rem; padding:0; }
.iris-callout li { margin:.2rem 0; }
.iris-callout .c-note { color: var(--iris-muted); font-size:.8rem; margin-top:.45rem; }

/* Mini istatistik ızgarası (müşteri özellikleri vb.) */
.iris-facts { display:grid; grid-template-columns: repeat(var(--cols,2), minmax(0,1fr)); gap: 8px 16px; }
.iris-fact { border-bottom: 1px dashed var(--iris-border); padding: .3rem 0 .35rem; }
.iris-fact .f-lab { color: var(--iris-muted); font-size:.74rem; }
.iris-fact .f-val { color: var(--iris-ink); font-weight:650; font-size:.95rem; font-variant-numeric: tabular-nums; }
.iris-big { font-size: 2.4rem; font-weight: 800; color: var(--iris-ink); line-height:1;
  letter-spacing:-.02em; font-variant-numeric: tabular-nums; }
.iris-gauge-sm { width: 100%; max-width: 340px; display:block; margin: 0 auto; }
.iris-status { display:flex; align-items:center; gap:.5rem; justify-content:center; margin-top:.2rem;
  flex-wrap:wrap; }

/* Animasyonlar (yalnız ilk görünüm) */
@keyframes iris-fadeup { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform:none; } }
@keyframes iris-grow { from { transform: scaleY(0); } to { transform: scaleY(1); } }
@keyframes iris-fade { from { opacity:0; } to { opacity:1; } }
.stPlotlyChart .barlayer .point path { transform-box: fill-box; transform-origin: 50% 100%;
  animation: iris-grow .9s cubic-bezier(.2,.8,.2,1) both; }
.stPlotlyChart .barlayer .point text, .stPlotlyChart .shapelayer, .stPlotlyChart .annotation {
  animation: iris-fade .6s .55s ease-out both; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
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
    st.html(_BASE_CSS.replace("__VARS__", _vars(THEMES[mode()])))


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


def kpi_card(label, value_html, sub="", mini="", tone=TURQ, tip=""):
    """KPI kartı. tip: fareyle üzerine gelince görünen açıklama (mini grafiğin anlamı vb.)."""
    info = f'<i class="k-info" title="{escape(tip)}">i</i>' if tip else ""
    title = f' title="{escape(tip)}"' if tip else ""
    return (f'<div class="iris-kpi" style="--tone:{tone}"{title}><div class="k-label">'
            f'{escape(label)}{info}</div>'
            f'<div class="k-row"><div class="k-value">{value_html}</div>'
            f'<div class="k-mini">{mini}</div></div><div class="k-sub">{sub}</div></div>')


def kpi_row(cards, cols=4):
    html(f'<div class="iris-kpis" style="--cols:{cols}">' + "".join(cards) + "</div>", js=True)


def delta_badge(diff_pp, better_when_lower=True):
    """Puan farkı rozeti; churn için yükseliş mercan, düşüş turkuaz."""
    if abs(diff_pp) < 0.05:
        return '<span class="iris-delta flat">fark yok</span>'
    up = diff_pp > 0
    cls = ("up" if up else "down") if better_when_lower else ("down" if up else "up")
    arrow = "▲" if up else "▼"
    return f'<span class="iris-delta {cls}">{arrow} {fmt_num(abs(diff_pp), 1)} puan</span>'


def legend(items, rule=""):
    """Renk anahtarı: items = [(renk, etiket)] ya da [(renk, etiket, 'line')]."""
    parts = []
    for it in items:
        cls = ' class="line"' if len(it) > 2 and it[2] == "line" else ""
        parts.append(f'<span class="sw"><i{cls} style="background:{it[0]}"></i>{it[1]}</span>')
    rule_html = f'<span class="rule">{rule}</span>' if rule else ""
    return f'<div class="iris-scale">{"".join(parts)}{rule_html}</div>'


def rel_legend(rule=True):
    """Portföy ortalamasına göre renk anahtarı (Wilson %95 GA kuralı)."""
    return legend([(REL_COLORS[k], REL_LABELS[k]) for k in ("below", "near", "above")]
                  + [(tok("subtle"), "Portföy ortalaması", "line")],
                  "Kural: grubun %95 Wilson güven aralığı ortalamanın tamamen altında / "
                  "ortalamayı içeriyor / tamamen üstünde" if rule else "")


def table_html(df, fmt=None, num=(), best=None, sel_rows=(), raw=()):
    """Temaya uyumlu HTML tablo. fmt: {kolon: fonksiyon}; num: sağa yaslı kolonlar;
    best: {kolon: 'max'|'min'} en iyi değer vurgusu; raw: HTML olarak basılacak kolonlar."""
    fmt, best = fmt or {}, best or {}
    best_idx = {c: (df[c].idxmax() if how == "max" else df[c].idxmin()) for c, how in best.items()}
    head = "".join(f'<th class="{"num" if c in num else ""}">{escape(str(c))}</th>' for c in df.columns)
    rows = []
    for i, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            text = fmt[c](v) if c in fmt else v
            text = str(text) if c in raw else escape(str(text))
            cls = " ".join(x for x in ("num" if c in num else "",
                                        "best" if best_idx.get(c) == i else "") if x)
            cells.append(f'<td class="{cls}">{text}</td>')
        rows.append(f'<tr class="{"sel" if i in sel_rows else ""}">{"".join(cells)}</tr>')
    return (f'<div class="iris-table-wrap"><table class="iris-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def callout(title, lines, tone=TURQ, note=""):
    items = "".join(f"<li>{ln}</li>" for ln in lines)
    note_html = f'<div class="c-note">{note}</div>' if note else ""
    return (f'<div class="iris-callout" style="--tone:{tone}"><div class="c-title">{escape(title)}</div>'
            f'<ul>{items}</ul>{note_html}</div>')


def facts(pairs, cols=2):
    cells = "".join(f'<div class="iris-fact"><div class="f-lab">{escape(k)}</div>'
                    f'<div class="f-val">{v}</div></div>' for k, v in pairs)
    return f'<div class="iris-facts" style="--cols:{cols}">{cells}</div>'


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
           f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" fill="{color}" '
           f'stroke="{tok("surface")}" stroke-width="1.5"/></svg>')
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
        parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="0" y2="{height}" stroke="{tok("ink")}" '
                     f'stroke-width="1.5" stroke-dasharray="3 2"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">{_SVG_ANIM.format()}{"".join(parts)}</svg>')
    return svg_img(svg, width, height, title)


# ---------- Gösterge (logodaki yay) ----------
def _pt(cx, cy, r, f):
    a = np.pi * (1 - f)
    return cx + r * np.cos(a), cy - r * np.sin(a)


def _arc(cx, cy, r, f0, f1):
    x0, y0 = _pt(cx, cy, r, f0)
    x1, y1 = _pt(cx, cy, r, f1)
    return f"M{x0:.2f} {y0:.2f} A{r} {r} 0 0 1 {x1:.2f} {y1:.2f}"


def gauge_svg(value, vmax, bands, labels, marker=None, alt="", on_dark=False, track=None,
              cls="iris-gauge"):
    """Iris sembolünün yayı = gösterge.

    value/vmax ibre konumunu verir; bands = [(başlangıç, bitiş, renk)] değer biriminde
    renkli yay parçaları; labels = [(değer, metin)] yayın dışındaki etiketler;
    marker = (değer, metin) yay üzerinde kesikli işaret (ör. karar eşiği).
    İbre yalnız ilk yüklemede soldan hedefe döner (SVG içi CSS, reduced-motion'da kapalı).
    on_dark: koyu zemin (etiketler açık renk, merkez açık)."""
    ink = "#E8EFF6" if on_dark else NAVY
    dot = NAVY if on_dark else "#FFFFFF"
    track = track or ("rgba(255,255,255,.10)" if on_dark else "rgba(255,255,255,.6)")
    f = float(np.clip(value / vmax, 0, 1))
    angle = -90 + 180 * f
    cx, cy, r, w = 160, 172, 118, 44
    parts = [f'<path d="{_arc(cx, cy, r, 0, 1)}" fill="none" stroke="{track}" '
             f'stroke-width="{w + 10}" stroke-linecap="round"/>']
    segs = [(max(a, 0) / vmax, min(b, vmax) / vmax, c) for a, b, c in bands if b > a]
    # Parçalar düz uçla; yayın iki dış ucu yuvarlak başlıkla (parça sınırları kesin kalsın)
    g = []
    for a, b, c in segs:
        g.append(f'<path d="{_arc(cx, cy, r, a, b)}" fill="none" stroke="{c}" '
                 f'stroke-width="{w}" stroke-linecap="butt"/>')
    for fr, c in ((segs[0][0], segs[0][2]), (segs[-1][1], segs[-1][2])):
        x, y_ = _pt(cx, cy, r, fr)
        g.append(f'<circle cx="{x:.2f}" cy="{y_:.2f}" r="{w / 2}" fill="{c}"/>')
    parts.append(f'<g filter="url(#gs)">{"".join(g)}</g>')
    # parça sınırlarında ince ayraç
    sep = "#0F1E30" if on_dark else "#FFFFFF"
    for a, _, _ in segs[1:]:
        x1, y1 = _pt(cx, cy, r - w / 2 - 1, a)
        x2, y2 = _pt(cx, cy, r + w / 2 + 1, a)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{sep}" stroke-width="2.5"/>')
    font = ("font-family=\"Inter, 'Segoe UI', sans-serif\" font-size=\"12.5\" font-weight=\"600\" "
            f"fill=\"{ink}\" fill-opacity=\".72\"")
    for v, text in labels:
        fr = float(np.clip(v / vmax, 0, 1))
        x1, y1 = _pt(cx, cy, r + 30, fr)
        x2, y2 = _pt(cx, cy, r + 38, fr)
        lx, ly = _pt(cx, cy, r + 50, fr)
        anchor = "middle" if 0.3 < fr < 0.7 else ("end" if fr <= 0.3 else "start")
        tick = (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ink}" '
                f'stroke-opacity=".45" stroke-width="2" stroke-linecap="round"/>')
        if fr in (0.0, 1.0):  # uç etiketleri yayın altında, çizgisiz
            lx, ly, anchor, tick = _pt(cx, cy, r, fr)[0], cy + 44, "middle", ""
        parts.append(f'{tick}<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" {font}>'
                     f'{escape(text)}</text>')
    if marker is not None:
        mv, mtext = marker
        fr = float(np.clip(mv / vmax, 0, 1))
        x1, y1 = _pt(cx, cy, r - w / 2 - 6, fr)
        x2, y2 = _pt(cx, cy, r + w / 2 + 6, fr)
        lx, ly = _pt(cx, cy, r - w / 2 - 20, fr)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ink}" '
                     f'stroke-width="3" stroke-dasharray="4 3" stroke-linecap="round"/>'
                     f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" {font}>'
                     f'{escape(mtext)}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="-6 -10 332 238">
<style>
.n {{ transform-box: view-box; transform-origin: {cx}px {cy}px; transform: rotate({angle:.2f}deg);
      animation: n 1.8s .3s cubic-bezier(.3,1.35,.45,1) both; }}
@keyframes n {{ from {{ transform: rotate(-90deg); }} to {{ transform: rotate({angle:.2f}deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .n {{ animation: none; }} }}
</style>
<defs><filter id="gs" x="-20%" y="-20%" width="140%" height="170%">
  <feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#0B1A2E" flood-opacity=".18"/>
</filter></defs>
{''.join(parts)}
<g class="n"><path d="M{cx - 5.5} {cy} L{cx} {cy - r + 6} L{cx + 5.5} {cy} Z" fill="{ink}"
   stroke="{ink}" stroke-width="3" stroke-linejoin="round"/></g>
<circle cx="{cx}" cy="{cy}" r="27" fill="{ink}" filter="url(#gs)"/>
<circle cx="{cx - 8.5}" cy="{cy - 8.5}" r="7.5" fill="{dot}"/>
</svg>"""
    return svg_img(svg, alt=alt, cls=cls)


# ---------- Giriş sayfası ----------
_LANDING_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarHeader"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:first-of-type { gap: 0 !important; }
.st-key-hero, .st-key-hub { min-height: 100vh; justify-content: center; box-sizing: border-box; }
.st-key-hero { background: __HERO_BG__; padding: 28px 48px; align-items: center; }
.st-key-hub { background: __HUB__; padding: 30px 52px; color: #F7FAFC;
  box-shadow: -18px 0 40px -24px rgba(8,20,36,.55); }
.iris-hero { width: 100%; max-width: 620px; margin: 0 auto; text-align: center; }
.iris-gauge { width: 100%; max-width: 440px; height: auto; display:block; margin: 0 auto; }
.iris-hero .h-kicker { font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
  font-weight:700; color: __KICK__; margin-top: .2rem; }
.iris-hero .h-value { font-size: 3.9rem; font-weight: 800; color: __INK__; line-height:1;
  letter-spacing:-.03em; margin:.3rem 0 .45rem; font-variant-numeric: tabular-nums; }
.iris-hero .h-band { display:inline-flex; align-items:center; gap:.5rem; padding:.32rem .9rem;
  border-radius:999px; background: __CHIP__; color: __INK__; font-weight:650;
  font-size:.88rem; box-shadow: 0 6px 18px -10px rgba(30,58,95,.35); }
.iris-hero .h-band i { width:.65rem; height:.65rem; border-radius:50%; display:inline-block; }
.iris-hero .h-def { color: __DEF__; font-size:.82rem; line-height:1.5; margin: .8rem auto 0;
  max-width: 560px; }
.iris-hero .h-def b { color: __INK__; }
.iris-alert { display:flex; align-items:center; gap: 14px; text-align:left; margin: 1rem auto 0;
  max-width: 560px; background: __CARD__; border: 1px solid __CARD_BORDER__; border-left: 4px solid #FF7A59;
  border-radius: 14px; padding: 10px 16px; box-shadow: 0 8px 22px -16px rgba(30,58,95,.45); }
.iris-alert .a-val { font-size: 1.7rem; font-weight: 800; color: __INK__; line-height:1;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.iris-alert .a-lab { font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700;
  color: __ALERT_LAB__; }
.iris-alert .a-txt { font-size:.8rem; color: __DEF__; line-height:1.4; }
.iris-alert .a-txt b { color: __INK__; }
.iris-hub-logo img { height: 40px; width: auto; display:block; }
.iris-hub-title { font-size: 1.9rem; font-weight: 750; line-height: 1.16; letter-spacing:-.01em;
  margin: 1.1rem 0 .35rem; color:#FFFFFF; }
.iris-hub-sub { color: rgba(247,250,252,.72); font-size: .9rem; line-height:1.45; }
.iris-stats { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin: 1.1rem 0 .3rem; }
.iris-stat { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.09);
  border-radius: 14px; padding: 10px 14px; }
.iris-stat .s-val { font-size: 1.45rem; font-weight: 750; color:#FFFFFF; line-height:1.1;
  font-variant-numeric: tabular-nums; }
.iris-stat .s-lab { font-size: .74rem; color: rgba(247,250,252,.72); margin-top:.25rem; }
.iris-stat .s-dot { display:inline-block; width:.5rem; height:.5rem; border-radius:50%;
  margin-right:.35rem; vertical-align: middle; }
.iris-group { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700;
  color: #2DD4BF; margin: .75rem 0 -.35rem; }
.st-key-hub [data-testid="stPageLink"] a {
  display:flex; align-items:center; gap: 14px; width: 100%; box-sizing: border-box;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
  border-radius: 999px; padding: 5px 22px 5px 8px !important; text-decoration: none;
  transition: background .18s ease, border-color .18s ease, transform .18s ease; }
.st-key-hub [data-testid="stPageLink"] a:hover { background: rgba(20,184,166,.16);
  border-color: rgba(20,184,166,.55); transform: translateX(3px); }
.st-key-hub [data-testid="stPageLink"] a::after { content: "→"; margin-left: auto;
  color: rgba(247,250,252,.5); font-size: 1.1rem; }
.st-key-hub [data-testid="stPageLink"] a p,
.st-key-hub [data-testid="stPageLink"] a span { color: #FFFFFF !important; font-size: 1rem !important;
  font-weight: 600; }
.st-key-hub [data-testid="stPageLink"] [data-testid="stIconMaterial"] {
  width: 38px; height: 38px; min-width: 38px; border-radius: 50%; display:inline-flex;
  align-items:center; justify-content:center; background: rgba(20,184,166,.16);
  color: #2DD4BF !important; font-size: 1.25rem !important; margin: 0 !important; }
.st-key-hub [data-testid="stPageLink"] a > span:first-child { width: 38px !important;
  min-width: 38px; height: 38px; display: inline-flex; margin: 0 !important; }
.st-key-hub-analiz [data-testid="stPageLink"] [data-testid="stIconMaterial"] {
  background: rgba(255,122,89,.15); color: #FF7A59 !important; }
.iris-hub-foot { color: rgba(247,250,252,.55); font-size:.74rem; margin-top: .8rem; }
@media (max-width: 900px) {
  .st-key-hero, .st-key-hub { min-height: auto; padding: 32px 20px; }
}
</style>
"""

_LANDING = {
    "light": {"__HERO_BG__": ("radial-gradient(1200px 600px at 20% 10%, rgba(255,255,255,.75), transparent 60%),"
                              "radial-gradient(900px 700px at 90% 100%, rgba(20,184,166,.28), transparent 60%),"
                              "linear-gradient(160deg, #EFFBF9 0%, #CFF1EC 45%, #A7E3DA 100%)"),
              "__HUB__": NAVY, "__INK__": NAVY, "__KICK__": "#0F766E", "__CHIP__": "rgba(255,255,255,.75)",
              "__DEF__": "#34506F", "__CARD__": "rgba(255,255,255,.72)",
              "__CARD_BORDER__": "rgba(30,58,95,.08)", "__ALERT_LAB__": "#B5462B"},
    "dark": {"__HERO_BG__": ("radial-gradient(900px 600px at 20% 0%, rgba(45,212,191,.16), transparent 60%),"
                             "radial-gradient(900px 700px at 90% 100%, rgba(20,184,166,.18), transparent 60%),"
                             "linear-gradient(160deg, #0F2433 0%, #10303A 50%, #0E3A3A 100%)"),
             "__HUB__": "#0A1522", "__INK__": "#E8EFF6", "__KICK__": "#2DD4BF", "__CHIP__": "rgba(255,255,255,.08)",
             "__DEF__": "#B4C3D3", "__CARD__": "rgba(255,255,255,.06)",
             "__CARD_BORDER__": "rgba(255,255,255,.10)", "__ALERT_LAB__": "#FFA58E"},
}


def inject_landing_css():
    css = _LANDING_CSS
    for k, v in _LANDING[mode()].items():
        css = css.replace(k, v)
    st.html(css)


def logo_img(name, alt="Iris"):
    """logo/final içindeki bir SVG'yi <img> olarak döndürür."""
    return svg_img((LOGO / name).read_text(encoding="utf-8"), alt=alt)


def sidebar_logo():
    """Kenar çubuğu logosu (temaya göre açık/koyu sürüm)."""
    v = "dark" if mode() == "dark" else "light"
    st.logo(str(LOGO / f"iris-logo-sidebar-{v}.png"),
            icon_image=str(LOGO / f"iris-icon-sidebar-{v}.png"), size="medium")

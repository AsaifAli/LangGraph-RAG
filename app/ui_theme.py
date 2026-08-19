from __future__ import annotations

import html
import streamlit as st


def apply_theme(*, product: str, subtitle: str, accent: str, accent2: str = "#2563EB", theme: str = "light") -> None:
    """Apply a deterministic EvidenceFlow theme to every Streamlit surface.

    The selected Streamlit theme is the source of truth.  We deliberately do
    not use prefers-color-scheme here because the browser/OS can differ from
    Streamlit's Light/Dark selection.  A tiny theme marker lets CSS target the
    correct token set consistently, including the native bottom composer and
    BaseWeb portals.
    """
    is_dark = str(theme).lower() == "dark"
    mode_class = "ef-theme-dark" if is_dark else "ef-theme-light"

    if is_dark:
        p = dict(
            bg="#070b13", surface="#0d1422", surface2="#111a2a",
            text="#f3f7fb", text2="#d6e0ec", muted="#91a0b2",
            border="rgba(148,163,184,.18)", border_strong="rgba(148,163,184,.28)",
            shadow="0 18px 48px rgba(0,0,0,.24)", code_bg="rgba(255,255,255,.06)",
            sidebar="#0a101b", composer="#0d1422", input="#0d1422",
            input_border="rgba(148,163,184,.22)", strip="#070b13",
        )
        color_scheme = "dark"
    else:
        p = dict(
            bg="#f5f7fb", surface="#ffffff", surface2="#f3f5fa",
            text="#172033", text2="#334155", muted="#64748b",
            border="rgba(100,116,139,.20)", border_strong="rgba(100,116,139,.30)",
            shadow="0 18px 48px rgba(15,23,42,.08)", code_bg="rgba(15,23,42,.05)",
            sidebar="#f7f8fc", composer="#ffffff", input="#ffffff",
            input_border="rgba(100,116,139,.24)", strip="#f5f7fb",
        )
        color_scheme = "light"

    product_safe = html.escape(product)
    subtitle_safe = html.escape(subtitle)
    accent_safe = html.escape(accent)
    accent2_safe = html.escape(accent2)

    st.markdown(f"""
<style>
:root {{
  color-scheme:{color_scheme};
  --ef-bg:{p['bg']}; --ef-surface:{p['surface']}; --ef-surface-2:{p['surface2']};
  --ef-text:{p['text']}; --ef-text-2:{p['text2']}; --ef-muted:{p['muted']};
  --ef-border:{p['border']}; --ef-border-strong:{p['border_strong']};
  --ef-shadow:{p['shadow']}; --ef-code-bg:{p['code_bg']};
  --ef-sidebar:{p['sidebar']}; --ef-composer:{p['composer']}; --ef-input:{p['input']};
  --ef-input-border:{p['input_border']}; --ef-strip:{p['strip']};
  --ef-accent:{accent_safe}; --ef-accent-2:{accent2_safe};
}}

/* Live system theme: browser/OS appearance is the visual authority.
   This reacts immediately to system theme changes; no Streamlit rerun is required. */
:root {{ color-scheme: light dark; }}
@media (prefers-color-scheme: light) {{
  :root, html {{
    color-scheme: light !important;
    --ef-bg:#f7f8fc; --ef-surface:#ffffff; --ef-surface-2:#f1f4f8;
    --ef-text:#172033; --ef-text-2:#334155; --ef-muted:#64748b;
    --ef-border:rgba(100,116,139,.18); --ef-border-strong:rgba(100,116,139,.28);
    --ef-shadow:0 18px 48px rgba(15,23,42,.07); --ef-code-bg:rgba(15,23,42,.05);
    --ef-sidebar:#f5f7fb; --ef-composer:#ffffff; --ef-input:#ffffff;
    --ef-input-border:rgba(100,116,139,.24); --ef-strip:#f7f8fc;
  }}
}}
@media (prefers-color-scheme: dark) {{
  :root, html {{
    color-scheme: dark !important;
    --ef-bg:#070b13; --ef-surface:#0d1422; --ef-surface-2:#111a2a;
    --ef-text:#f3f7fb; --ef-text-2:#d6e0ec; --ef-muted:#91a0b2;
    --ef-border:rgba(148,163,184,.18); --ef-border-strong:rgba(148,163,184,.28);
    --ef-shadow:0 18px 48px rgba(0,0,0,.24); --ef-code-bg:rgba(255,255,255,.06);
    --ef-sidebar:#0a101b; --ef-composer:#0d1422; --ef-input:#0d1422;
    --ef-input-border:rgba(148,163,184,.22); --ef-strip:#070b13;
  }}
}}

html,body,[data-testid="stAppViewContainer"],.stApp,.main {{ color:var(--ef-text)!important; }}
html,body {{ background:var(--ef-bg)!important; }}
body {{ overflow-x:hidden!important; }}
.stApp,[data-testid="stAppViewContainer"],.main {{
  background:
    radial-gradient(900px 520px at 4% -8%,color-mix(in srgb,var(--ef-accent) 9%,transparent),transparent 60%),
    radial-gradient(760px 460px at 96% 0%,color-mix(in srgb,var(--ef-accent-2) 7%,transparent),transparent 58%),
    var(--ef-bg)!important;
}}


/* Streamlit framework chrome. */
#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stToolbar"],
[data-testid="stStatusWidget"],.stDeployButton,[data-testid="stAppDeployButton"],
header [data-testid="stHeaderActionElements"] {{display:none!important;}}
header[data-testid="stHeader"] {{background:transparent!important;}}

[data-testid="stAppViewContainer"]>.main .block-container {{
  max-width:1500px!important;
  padding-top:1.05rem!important;
  padding-bottom:6.5rem!important;
  padding-left:clamp(1rem,3vw,3rem)!important;
  padding-right:clamp(1rem,3vw,3rem)!important;
}}

/* Global readable typography. */
.stMarkdown,.stMarkdown p,.stMarkdown li,[data-testid="stCaptionContainer"],label,
[data-testid="stFileUploader"] small,input,textarea,[data-baseweb="select"] *,
[data-testid="stChatInput"] *,[data-testid="stChatMessage"] * {{color:var(--ef-text)!important;}}
.stMarkdown small,[data-testid="stCaptionContainer"] {{color:var(--ef-muted)!important;}}
h1,h2,h3,h4,h5,h6 {{color:var(--ef-text)!important;}}
code {{background:var(--ef-code-bg)!important;color:var(--ef-text-2)!important;border-radius:6px;padding:.08rem .32rem;}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background:
    linear-gradient(180deg,color-mix(in srgb,var(--ef-accent) 5%,transparent),transparent 24%),
    var(--ef-sidebar)!important;
  border-right:1px solid var(--ef-border)!important;
}}
section[data-testid="stSidebar"]>div {{background:transparent!important;}}
section[data-testid="stSidebar"] * {{color:var(--ef-text-2)!important;}}
section[data-testid="stSidebar"] .stButton>button {{width:100%;}}

/* Native bottom composer: paint both wrappers, not just the inner input. */
div[data-testid="stBottom"],div[data-testid="stBottomBlockContainer"],
section[data-testid="stChatInput"],section[data-testid="stChatInput"]>div {{
  background:var(--ef-strip)!important;
}}
div[data-testid="stBottom"] {{
  border-top:1px solid var(--ef-border)!important;
  box-shadow:none!important;
}}
div[data-testid="stBottomBlockContainer"] {{
  padding:0.75rem clamp(0.75rem,3vw,3rem) 1rem!important;
}}
[data-testid="stChatInput"]>div {{
  width:min(1120px,100%)!important;
  margin:0 auto!important;
  border:1px solid var(--ef-input-border)!important;
  background:var(--ef-composer)!important;
  box-shadow:0 16px 42px color-mix(in srgb,#0f172a 10%,transparent)!important;
  border-radius:18px!important;
  backdrop-filter:blur(16px);
}}
[data-testid="stChatInput"] textarea,[data-testid="stChatInput"] input {{
  background:transparent!important; box-shadow:none!important; color:var(--ef-text)!important;
}}
[data-testid="stChatInput"] textarea::placeholder,[data-testid="stChatInput"] input::placeholder {{
  color:var(--ef-muted)!important; opacity:1!important;
}}
[data-testid="stChatInput"] button {{
  background:var(--ef-surface-2)!important;color:var(--ef-text)!important;border:1px solid var(--ef-border)!important;
}}

/* Native inputs/selects. */
input,textarea,[data-baseweb="select"]>div {{
  border-radius:11px!important;
  border-color:var(--ef-input-border)!important;
  background:var(--ef-input)!important;
  color:var(--ef-text)!important;
}}
input::placeholder,textarea::placeholder {{color:var(--ef-muted)!important;opacity:1!important;}}
input:focus,textarea:focus,[data-baseweb="select"]>div:hover {{
  border-color:color-mix(in srgb,var(--ef-accent) 52%,var(--ef-border))!important;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--ef-accent) 10%,transparent)!important;
}}
[data-baseweb="menu"] {{background:var(--ef-surface)!important;border:1px solid var(--ef-border)!important;}}
[data-baseweb="menu"] li {{color:var(--ef-text)!important;}}

/* Buttons. */
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button {{
  min-height:40px;border-radius:11px!important;border:1px solid var(--ef-border)!important;
  background:var(--ef-surface)!important;color:var(--ef-text)!important;
  box-shadow:0 6px 20px color-mix(in srgb,#0f172a 5%,transparent)!important;
  font-weight:650!important;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease!important;
}}
.stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover {{
  transform:translateY(-1px);border-color:color-mix(in srgb,var(--ef-accent) 44%,var(--ef-border))!important;
  box-shadow:0 12px 30px color-mix(in srgb,var(--ef-accent) 10%,transparent)!important;
}}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {{
  background:linear-gradient(135deg,var(--ef-accent),var(--ef-accent-2))!important;color:#fff!important;border-color:transparent!important;
}}

/* Upload */
[data-testid="stFileUploaderDropzone"] {{
  border:1px dashed color-mix(in srgb,var(--ef-accent) 50%,var(--ef-border))!important;border-radius:16px!important;
  background:linear-gradient(180deg,color-mix(in srgb,var(--ef-accent) 5%,var(--ef-surface)),var(--ef-surface))!important;
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease!important;
}}
[data-testid="stFileUploaderDropzone"] * {{color:var(--ef-text)!important;}}
[data-testid="stFileUploaderDropzone"] button {{background:var(--ef-surface-2)!important;color:var(--ef-text)!important;border-color:var(--ef-border)!important;}}
[data-testid="stFileUploaderDropzone"] small {{color:var(--ef-muted)!important;}}
[data-testid="stFileUploaderDropzone"]:hover {{transform:translateY(-1px);border-color:var(--ef-accent)!important;box-shadow:0 12px 32px color-mix(in srgb,var(--ef-accent) 9%,transparent)!important;}}

/* Containers, tabs, chat. */
div[data-testid="stExpander"],div[data-testid="stForm"],div[data-testid="stMetric"],div[data-testid="stDataFrame"],div[data-testid="stAlert"] {{
  border-radius:16px!important;border:1px solid var(--ef-border)!important;background:var(--ef-surface)!important;box-shadow:var(--ef-shadow);
}}
[data-testid="stTabs"] [role="tablist"] {{gap:.3rem;border-bottom:1px solid var(--ef-border);}}
[data-testid="stTabs"] button {{color:var(--ef-muted)!important;border-radius:10px 10px 0 0!important;}}
[data-testid="stTabs"] button[aria-selected="true"] {{color:var(--ef-accent)!important;background:color-mix(in srgb,var(--ef-accent) 8%,transparent)!important;}}
[data-testid="stChatMessage"] {{
  border-radius:18px!important;border:1px solid var(--ef-border)!important;background:var(--ef-surface)!important;color:var(--ef-text)!important;
  box-shadow:0 8px 28px color-mix(in srgb,#0f172a 7%,transparent);margin-bottom:.8rem;
}}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li,[data-testid="stChatMessage"] span {{color:var(--ef-text)!important;}}

.ui-product-bar {{
  display:flex;align-items:center;gap:.7rem;margin:0 0 .9rem;padding:.68rem .85rem;border:1px solid var(--ef-border);border-radius:14px;
  background:linear-gradient(135deg,color-mix(in srgb,var(--ef-accent) 9%,var(--ef-surface)),var(--ef-surface));
  box-shadow:0 10px 34px color-mix(in srgb,var(--ef-accent) 6%,transparent);
}}
.ui-product-dot {{width:10px;height:10px;border-radius:999px;background:var(--ef-accent);box-shadow:0 0 0 5px color-mix(in srgb,var(--ef-accent) 13%,transparent);flex:0 0 auto;}}
.ui-product-name {{font-weight:800;letter-spacing:.01em;color:var(--ef-text)!important;}}
.ui-product-sub {{color:var(--ef-muted)!important;font-size:.83rem;}}

@media (max-width:900px) {{
  [data-testid="stAppViewContainer"]>.main .block-container {{padding-left:.9rem!important;padding-right:.9rem!important;}}
  [data-testid="stChatInput"]>div {{border-radius:15px!important;}}
}}
@media (prefers-reduced-motion:reduce) {{*,*::before,*::after {{animation-duration:.001ms!important;transition-duration:.001ms!important;}}}}
</style>
<div class="{mode_class}" aria-hidden="true" style="display:none"></div>
<div class="ui-product-bar"><span class="ui-product-dot"></span><span><span class="ui-product-name">{product_safe}</span><span class="ui-product-sub"> · {subtitle_safe}</span></span></div>
""", unsafe_allow_html=True)

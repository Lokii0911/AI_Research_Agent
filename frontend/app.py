import streamlit as st
import requests
import re
import json
import time
import streamlit.components.v1 as _components

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus Research",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def extract_urls(text):
    return list(set(re.findall(r"https?://[^\s\"'<>)]+", text)))

def tool_class(name):
    n = name.lower()
    if "arxiv"  in n: return "arxiv"
    if "wiki"   in n: return "wiki"
    if "tavily" in n: return "tavily"
    return "default"

def tool_icon(name):
    n = name.lower()
    if "arxiv"  in n: return "📄"
    if "wiki"   in n: return "📖"
    if "tavily" in n: return "🌐"
    return "🔧"

def now_stamp():
    t = time.localtime()
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"

def scroll_to_bottom():
    """Inject a fresh iframe-based scroll trigger that always fires."""
    _components.html(
        """<script>
        (function() {
            // Walk up from this iframe to the parent document and scroll the panel
            function doScroll() {
                try {
                    var panel = window.parent.document.getElementById('nx-chat');
                    if (panel) {
                        panel.scrollTop = panel.scrollHeight;
                    }
                } catch(e) {}
            }
            doScroll();
            // Retry after a short delay to catch late renders
            setTimeout(doScroll, 150);
            setTimeout(doScroll, 400);
        })();
        </script>""",
        height=0,
        scrolling=False,
    )

def to_html(text):
    """Convert markdown to HTML using stdlib only — no external packages."""
    import html as _html
    # We do NOT escape first — LLM output may have intentional < > in code
    t = text

    # Fenced code blocks ```lang\n...\n```
    import re as _re
    t = _re.sub(r'```[\w]*\n(.*?)```', lambda m: '<pre><code>' + _html.escape(m.group(1)) + '</code></pre>', t, flags=_re.DOTALL)

    # Inline code `code`
    t = _re.sub(r'`([^`]+)`', lambda m: '<code>' + _html.escape(m.group(1)) + '</code>', t)

    # Headers ### ## #
    t = _re.sub(r'^### (.+)$', r'<h3>\1</h3>', t, flags=_re.MULTILINE)
    t = _re.sub(r'^## (.+)$',  r'<h2>\1</h2>', t, flags=_re.MULTILINE)
    t = _re.sub(r'^# (.+)$',   r'<h1>\1</h1>', t, flags=_re.MULTILINE)

    # Bold **text** and __text__
    t = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = _re.sub(r'__(.+?)__',          r'<strong>\1</strong>', t)

    # Italic *text* and _text_
    t = _re.sub(r'\*([^\*\n]+?)\*', r'<em>\1</em>', t)
    t = _re.sub(r'(?<![_])_([^_\n]+?)_(?![_])', r'<em>\1</em>', t)

    # Links [text](url)
    t = _re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', t)

    # Process line by line for lists and paragraphs
    lines = t.split('\n')
    result = []
    in_ul = False
    in_ol = False

    for line in lines:
        # Unordered list: - item or * item
        ul_match = _re.match(r'^[\s]*[-*•] (.+)$', line)
        ol_match = _re.match(r'^[\s]*(\d+)\. (.+)$', line)

        if ul_match:
            if in_ol:
                result.append('</ol>'); in_ol = False
            if not in_ul:
                result.append('<ul>'); in_ul = True
            result.append(f'<li>{ul_match.group(1)}</li>')
        elif ol_match:
            if in_ul:
                result.append('</ul>'); in_ul = False
            if not in_ol:
                result.append('<ol>'); in_ol = True
            result.append(f'<li>{ol_match.group(2)}</li>')
        else:
            if in_ul:
                result.append('</ul>'); in_ul = False
            if in_ol:
                result.append('</ol>'); in_ol = False

            stripped = line.strip()
            if stripped == '':
                result.append('<br>')
            elif stripped.startswith('<h') or stripped.startswith('<pre') or stripped.startswith('<ul') or stripped.startswith('<ol'):
                result.append(stripped)
            else:
                result.append(f'<p>{stripped}</p>')

    if in_ul: result.append('</ul>')
    if in_ol: result.append('</ol>')

    return '\n'.join(result)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in [("chat", []), ("sources_all", []), ("activity", []),
                 ("query_count", 0), ("tool_count", 0),
                 ("api_key", ""), ("authenticated", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

/* BASE */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #050912 !important; }

/* HIDE STREAMLIT CHROME */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"] { display: none !important; }

/* APP BG + FONT */
[data-testid="stAppViewContainer"] {
    background: #050912 !important;
    font-family: 'Space Mono', monospace;
    color: #dde8ff;
}

/* SCANLINES */
[data-testid="stAppViewContainer"]::before {
    content: "";
    pointer-events: none;
    position: fixed; inset: 0; z-index: 9999;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,180,0.016) 2px, rgba(0,255,180,0.016) 4px
    );
}

/* VIGNETTE */
[data-testid="stAppViewContainer"]::after {
    content: "";
    pointer-events: none;
    position: fixed; inset: 0; z-index: 9998;
    background: radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.7) 100%);
}

/* GRID BG */
[data-testid="stMain"] {
    background:
        linear-gradient(rgba(0,255,180,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,180,0.025) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
}

/* BLOCK CONTAINER — strip all padding */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* No gap between columns */
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── HEADER ── */
.nx-header {
    padding: 13px 28px;
    border-bottom: 1px solid rgba(0,255,180,0.12);
    display: flex; align-items: center; gap: 16px;
    background: #050912;
    position: relative;
}
.nx-header::after {
    content: "";
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00ffb4, transparent);
    animation: hpulse 3s ease-in-out infinite;
}
@keyframes hpulse { 0%,100%{opacity:.25} 50%{opacity:.9} }
.nx-logo {
    width: 36px; height: 36px; flex-shrink: 0;
    border: 2px solid #00ffb4; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
    animation: glow 2.5s ease-in-out infinite;
}
@keyframes glow {
    0%,100%{ box-shadow: 0 0 10px rgba(0,255,180,.28); }
    50%    { box-shadow: 0 0 22px rgba(0,255,180,.65); }
}
.nx-title { font-family:'Syne',sans-serif; font-size:19px; font-weight:800; color:#fff; }
.nx-sub   { font-size:8px; color:#00ffb4; letter-spacing:3px; text-transform:uppercase; opacity:.55; margin-top:2px; }
.nx-badge {
    margin-left: auto; padding: 3px 10px;
    background: rgba(0,255,180,.07); border: 1px solid rgba(0,255,180,.22);
    border-radius: 4px; font-size: 9px; color: #00ffb4; letter-spacing: 2px; flex-shrink: 0;
}

/* ── PANELS ── 
   Panels fill viewport minus header.
   Bottom padding clears Streamlit's fixed chat input bar (~80px).
*/
.nx-panel {
    height: calc(100vh - 68px);
    overflow-y: auto; overflow-x: hidden;
    padding: 22px 26px 96px;
    scrollbar-width: thin;
    scrollbar-color: rgba(0,255,180,.18) transparent;
}
.nx-panel::-webkit-scrollbar { width: 4px; }
.nx-panel::-webkit-scrollbar-thumb { background: rgba(0,255,180,.18); border-radius: 4px; }

.nx-side {
    height: calc(100vh - 68px);
    overflow-y: auto; overflow-x: hidden;
    padding: 18px 15px 96px;
    border-left: 1px solid rgba(0,255,180,.07);
    background: rgba(0,255,180,.012);
    scrollbar-width: thin;
    scrollbar-color: rgba(0,255,180,.1) transparent;
}
.nx-side::-webkit-scrollbar { width: 3px; }
.nx-side::-webkit-scrollbar-thumb { background: rgba(0,255,180,.12); border-radius: 3px; }

/* ── MESSAGES ── */
.msg-user {
    background: rgba(0,100,255,.1);
    border: 1px solid rgba(0,100,255,.3);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 15px; margin-bottom: 16px;
    position: relative; font-size: 14px;
    color: #ddeeff;
    animation: fadeup .25s ease;
}
.msg-user::before {
    content:"YOU"; position:absolute; top:-8px; right:11px;
    background:#050912; padding:0 5px;
    font-size:8px; color:rgba(80,140,255,.65); letter-spacing:2px;
}
.msg-agent {
    background: rgba(0,255,180,.05);
    border: 1px solid rgba(0,255,180,.18);
    border-radius: 12px 12px 12px 4px;
    padding: 12px 15px; margin-bottom: 16px;
    line-height: 1.75; font-size: 13.5px;
    color: #e2ecff;
    position: relative; white-space: pre-wrap;
    animation: fadeup .25s ease;
}
.msg-agent::before {
    content:"NEXUS"; position:absolute; top:-8px; left:11px;
    background:#050912; padding:0 5px;
    font-size:8px; color:#00ffb4; letter-spacing:2px;
}
@keyframes fadeup { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

/* ── MARKDOWN CONTENT INSIDE MESSAGES ── */
.msg-agent { white-space: normal !important; }   /* let markdown handle spacing */

.msg-agent p { margin: 0 0 10px; color: #dde8ff; line-height: 1.75; }
.msg-agent p:last-child { margin-bottom: 0; }

.msg-agent h1, .msg-agent h2, .msg-agent h3, .msg-agent h4 {
    font-family: 'Syne', sans-serif; font-weight: 700;
    color: #ffffff; margin: 14px 0 6px; line-height: 1.3;
    border-bottom: 1px solid rgba(0,255,180,.1); padding-bottom: 4px;
}
.msg-agent h1 { font-size: 17px; }
.msg-agent h2 { font-size: 15px; }
.msg-agent h3 { font-size: 14px; color: #b8fce8; border-bottom: none; }

.msg-agent strong, .msg-agent b { color: #00ffb4; font-weight: 700; }
.msg-agent em, .msg-agent i { color: #c4b5fd; font-style: italic; }

.msg-agent ul { list-style: none; padding-left: 0; margin: 6px 0 10px; }
.msg-agent ul li { 
    padding: 3px 0 3px 18px; color: #dde8ff; position: relative; 
    border-bottom: 1px solid rgba(255,255,255,.03); 
}
.msg-agent ul li::before { 
    content: "⬡"; position: absolute; left: 0; 
    color: #00ffb4; font-size: 8px; top: 6px; opacity: .7;
}
.msg-agent ol { padding-left: 20px; margin: 6px 0 10px; }
.msg-agent ol li { color: #dde8ff; padding: 3px 0; }
.msg-agent ol li::marker { color: #00ffb4; font-weight: 700; }

/* Nested lists */
.msg-agent ul ul, .msg-agent ol ul { margin: 4px 0 4px 12px; }
.msg-agent ul ul li::before { color: rgba(0,255,180,.4); content: "○"; font-size: 9px; }

.msg-agent code {
    background: rgba(0,255,180,.08); border: 1px solid rgba(0,255,180,.2);
    border-radius: 4px; padding: 1px 7px; font-size: 12px; color: #00ffb4;
    font-family: 'Space Mono', monospace;
}
.msg-agent pre {
    background: rgba(0,0,0,.4); border: 1px solid rgba(0,255,180,.15);
    border-radius: 8px; padding: 12px 14px; overflow-x: auto; margin: 8px 0;
}
.msg-agent pre code { background: none; border: none; padding: 0; font-size: 12px; }
.msg-agent a { color: #60a5fa; text-decoration: underline; }
.msg-agent blockquote {
    border-left: 3px solid rgba(0,255,180,.3); margin: 8px 0;
    padding: 6px 12px; color: rgba(200,214,240,.7); font-style: italic;
}
.msg-agent table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12px; }
.msg-agent th { background: rgba(0,255,180,.08); color: #00ffb4; padding: 6px 10px; text-align: left; }
.msg-agent td { padding: 5px 10px; border-bottom: 1px solid rgba(255,255,255,.05); color: #dde8ff; }

/* ── ACTIVITY LOG CONTRAST ── */
.activity-item { color: #8899bb; }
.activity-time { color: #00ffb4 !important; opacity: .7 !important; }

/* ── STAT NUMBERS ── */
.stat-num { color: #00ffb4; text-shadow: 0 0 12px rgba(0,255,180,.4); }
.stat-lbl { color: #8899bb !important; }

/* ── SIDE TITLE ── */
.side-title { color: #00ffb4 !important; opacity: .9; }

/* ── THINKING BOX ── */
.nx-thinking {
    border: 1px solid rgba(0,255,180,.18); border-radius: 10px;
    padding: 13px 16px; margin-bottom: 16px;
    background: rgba(0,255,180,.025); position: relative; overflow: hidden;
}
.nx-thinking::before {
    content: ""; position: absolute; top: 0; left: -60%; width: 55%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,180,.07), transparent);
    animation: sweep 2s ease-in-out infinite;
}
@keyframes sweep { from{left:-60%} to{left:110%} }
.th-label {
    font-size: 9px; letter-spacing: 3px; color: #00ffb4; margin-bottom: 8px;
    display: flex; align-items: center; gap: 8px;
}
.dot-pulse::after {
    content: "●●●"; letter-spacing: 4px;
    animation: dts 1.2s steps(3,end) infinite;
}
@keyframes dts {
    0%  {content:"●○○"} 33% {content:"●●○"}
    66% {content:"●●●"} 100%{content:"○○○"}
}

/* ── TOOL STEPS ── */
.tool-step {
    display: flex; align-items: center; gap: 9px;
    padding: 6px 11px; margin-bottom: 4px; border-radius: 6px;
    background: rgba(255,255,255,.025); border-left: 3px solid; font-size: 12px;
    animation: slidein .2s ease;
}
@keyframes slidein { from{opacity:0;transform:translateX(-8px)} to{opacity:1;transform:none} }
.tool-step.arxiv  { border-color:#f59e0b; color:#fcd34d; }
.tool-step.wiki   { border-color:#60a5fa; color:#93c5fd; }
.tool-step.tavily { border-color:#a78bfa; color:#c4b5fd; }
.tool-step.default{ border-color:#00ffb4; color:#6ee7b7; }

/* ── SIDEBAR ── */
.side-title {
    font-size: 9px; letter-spacing: 3px; color: #00ffb4; text-transform: uppercase;
    padding-bottom: 7px; margin-bottom: 9px;
    border-bottom: 1px solid rgba(0,255,180,.1);
}
.stat-row { display: flex; gap: 5px; margin-bottom: 13px; }
.stat-box {
    flex: 1; background: rgba(0,255,180,.04); border: 1px solid rgba(0,255,180,.09);
    border-radius: 8px; padding: 8px; text-align: center;
}
.stat-num { font-family:'Syne',sans-serif; font-size:19px; font-weight:800; color:#00ffb4; line-height:1; }
.stat-lbl { font-size:8px; color:#6b7280; letter-spacing:1.5px; text-transform:uppercase; margin-top:3px; }
.source-card {
    background: rgba(255,255,255,.02); border: 1px solid rgba(255,255,255,.07);
    border-radius: 8px; padding: 8px 10px; margin-bottom: 6px; font-size: 11px;
}
.source-num  { font-size:8px; color:#00ffb4; letter-spacing:2px; margin-bottom:3px; }
.source-link { color:#93c5fd; text-decoration:none; word-break:break-all; line-height:1.4; }
.activity-item {
    font-size: 10px; color: #4b5563; padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,.03);
    display: flex; gap: 8px;
}
.activity-time { color:#00ffb4; opacity:.4; white-space:nowrap; }

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 65%; gap: 12px; text-align: center; padding-top: 40px;
}
.empty-hex  { font-size:50px; opacity:.16; animation:float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
.empty-text { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:rgba(255,255,255,.12); }
.empty-sub  { font-size:9px; letter-spacing:2px; color:rgba(0,255,180,.18); }

/* ── LOGIN GATE ── */
.login-gate {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100vh; gap: 20px; background: #050912;
}
.login-box {
    background: rgba(0,255,180,.03);
    border: 1px solid rgba(0,255,180,.2);
    border-radius: 16px;
    padding: 40px 48px;
    width: 420px;
    text-align: center;
    position: relative;
    box-shadow: 0 0 40px rgba(0,255,180,.06);
}
.login-box::before {
    content: "";
    position: absolute; inset: -1px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(0,255,180,.15), transparent, rgba(0,255,180,.08));
    pointer-events: none;
}
.login-title {
    font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800;
    color: #fff; margin-bottom: 6px;
}
.login-sub {
    font-size: 10px; letter-spacing: 3px; color: rgba(0,255,180,.6);
    text-transform: uppercase; margin-bottom: 28px;
}
.login-err {
    background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.3);
    border-radius: 8px; padding: 8px 14px; font-size: 12px; color: #fca5a5;
    margin-top: 12px;
}

/* ── LOGIN GATE ── */
.login-gate {
    position: fixed; inset: 0; z-index: 9997;
    display: flex; align-items: center; justify-content: center;
    background: #050912;
}
.login-err {
    background: rgba(239,68,68,.08); border: 1px solid rgba(239,68,68,.25);
    border-radius: 8px; padding: 8px 14px; font-size: 12px; color: #fca5a5;
    margin-top: 10px; animation: fadeup .2s ease;
}

/* ── CURSOR ── */
.cursor {
    display: inline-block; width: 7px; height: 13px;
    background: #00ffb4; border-radius: 1px;
    vertical-align: text-bottom; margin-left: 2px;
    animation: blink 1s step-end infinite;
    box-shadow: 0 0 6px rgba(0,255,180,.7);
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ── INPUT CURSOR COLOR (caret) ── */
[data-testid="stChatInputTextArea"] {
    caret-color: #00ffb4 !important;
}

/* ── LOGIN INPUT & BUTTON ── */
/* Input field on login page */
div[data-testid="stTextInput"] input {
    background: rgba(5, 15, 30, 0.95) !important;
    border: 1px solid rgba(0,255,180,.35) !important;
    border-radius: 10px !important;
    color: #e0f0ff !important;
    caret-color: #00ffb4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    text-align: center !important;
    -webkit-text-fill-color: #e0f0ff !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(0,255,180,.65) !important;
    box-shadow: 0 0 0 2px rgba(0,255,180,.15) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: rgba(150,180,220,.4) !important;
    -webkit-text-fill-color: rgba(150,180,220,.4) !important;
}
/* Authenticate button */
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] > button {
    background: rgba(0,255,180,.08) !important;
    border: 1px solid rgba(0,255,180,.35) !important;
    color: #00ffb4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 3px !important;
    border-radius: 10px !important; padding: 10px !important;
    text-transform: uppercase !important;
    transition: all .2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(0,255,180,.18) !important;
    border-color: rgba(0,255,180,.7) !important;
    box-shadow: 0 0 16px rgba(0,255,180,.2) !important;
}

/* ── LOGOUT BUTTON ── */
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: rgba(239,68,68,.07) !important;
    border: 1px solid rgba(239,68,68,.25) !important;
    color: rgba(239,68,68,.8) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important; letter-spacing: 2px !important;
    border-radius: 4px !important;
}
button[kind="secondary"]:hover {
    background: rgba(239,68,68,.15) !important;
    border-color: rgba(239,68,68,.5) !important;
}

/* ── CHAT INPUT ── */
/* Streamlit renders stChatInput fixed at page bottom — we style it here */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important; left: 0 !important; right: 0 !important;
    z-index: 1000 !important;
    background: rgba(8,14,30,.98) !important;
    border-top: 1px solid rgba(0,255,180,.2) !important;
    padding: 12px 28px 16px !important;
    backdrop-filter: blur(12px) !important;
}
/* Inner textarea */
[data-testid="stChatInputTextArea"] {
    background: rgba(10,20,45,.9) !important;
    border: 1px solid rgba(0,255,180,.35) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    caret-color: #00ffb4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}
[data-testid="stChatInputTextArea"]::placeholder {
    color: rgba(200,214,240,0.45) !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: rgba(0,255,180,.55) !important;
    box-shadow: 0 0 0 2px rgba(0,255,180,.12) !important;
    outline: none !important;
}
[data-testid="stChatInputSubmitButton"] {
    color: #00ffb4 !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: #00ffb4 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  AUTH GATE — show login screen if not authenticated
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    # Use 5 columns: outer spacers + inner spacers + center
    _, c1, mid, c2, _ = st.columns([0.8, 0.1, 2, 0.1, 0.8])
    with mid:
        # Top spacer
        st.markdown("<div style='height:22vh'></div>", unsafe_allow_html=True)

        # Logo + title card
        st.markdown("""
        <div style="text-align:center; margin-bottom:24px;">
          <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:52px; height:52px; border:2px solid #00ffb4; border-radius:12px;
            font-size:22px; margin-bottom:16px;
            box-shadow: 0 0 20px rgba(0,255,180,.4);
            animation: glow 2.5s ease-in-out infinite;
          ">⬡</div>
          <div style="font-family:'Syne',sans-serif; font-size:24px; font-weight:800; color:#fff; letter-spacing:-0.5px;">NEXUS RESEARCH</div>
          <div style="font-size:9px; letter-spacing:3px; color:rgba(0,255,180,.55); text-transform:uppercase; margin-top:6px;">Restricted Access — Enter API Key</div>
        </div>
        """, unsafe_allow_html=True)

        # Input field
        key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your Nexus API key…",
            label_visibility="collapsed",
            key="key_field"
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Auth button
        if st.button("⬡  AUTHENTICATE", use_container_width=True, key="auth_btn"):
            if key_input.strip():
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/verify-key",
                        headers={"X-Api-Key": key_input.strip()},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        st.session_state.api_key = key_input.strip()
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.markdown('<div class="login-err">⚠ Invalid API key. Access denied.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="login-err">⚠ Cannot connect to server: {e}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="login-err">⚠ Please enter your API key.</div>', unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────
#  CHAT INPUT — top level so Streamlit renders it
#  fixed at the bottom of the page automatically
# ─────────────────────────────────────────────
query = st.chat_input("Ask Nexus anything — science, news, research…")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="nx-header">
  <div class="nx-logo">⬡</div>
  <div>
    <div class="nx-title">NEXUS RESEARCH</div>
    <div class="nx-sub">Autonomous Intelligence Layer</div>
  </div>
  <div class="nx-badge">SYSTEM ONLINE</div>

</div>
""", unsafe_allow_html=True)

# Logout button — top right
logout_col1, logout_col2 = st.columns([6, 1])
with logout_col2:
    if st.button("⬡ LOGOUT", key="logout_btn", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.chat = []
        st.session_state.sources_all = []
        st.session_state.activity = []
        st.session_state.query_count = 0
        st.session_state.tool_count = 0
        st.rerun()

# ─────────────────────────────────────────────
#  COLUMNS
# ─────────────────────────────────────────────
main_col, side_col = st.columns([3, 1], gap="small")

# ── SIDE PANEL ────────────────────────────────
with side_col:
    src_html = ""
    if st.session_state.sources_all:
        items = "".join(
            f'<div class="source-card"><div class="source-num">SRC {i:02d}</div>'
            f'<a href="{lnk}" target="_blank" class="source-link">'
            f'{re.sub(r"https?://(www.)?","",lnk).split("/")[0]}</a></div>'
            for i, lnk in enumerate(st.session_state.sources_all[:12], 1)
        )
        src_html = f'<div class="side-title" style="margin-top:12px;">⬡ SOURCES</div>{items}'

    act_html = ""
    if st.session_state.activity:
        rows = "".join(
            f'<div class="activity-item">'
            f'<span class="activity-time">{ts}</span><span>{lbl}</span></div>'
            for ts, lbl in reversed(st.session_state.activity[-8:])
        )
        act_html = f'<div class="side-title" style="margin-top:14px;">⬡ ACTIVITY</div>{rows}'

    st.markdown(f"""
    <div class="nx-side">
      <div class="side-title">⬡ SESSION</div>
      <div class="stat-row">
        <div class="stat-box">
          <div class="stat-num">{st.session_state.query_count}</div>
          <div class="stat-lbl">Queries</div>
        </div>
        <div class="stat-box">
          <div class="stat-num">{st.session_state.tool_count}</div>
          <div class="stat-lbl">Tools</div>
        </div>
        <div class="stat-box">
          <div class="stat-num">{len(st.session_state.sources_all)}</div>
          <div class="stat-lbl">Sources</div>
        </div>
      </div>
      {src_html}
      {act_html}
    </div>
    """, unsafe_allow_html=True)

# ── HELPER: build full chat HTML including any live content ──────────────────
def build_panel(history, new_query="", live_html=""):
    """Renders the entire chat panel as one HTML block so ordering is always correct."""
    if not history and not new_query:
        body = """
        <div class="empty-state">
          <div class="empty-hex">⬡</div>
          <div class="empty-text">Nexus is standing by</div>
          <div class="empty-sub">Ask anything — science, news, research</div>
        </div>"""
    else:
        body = "".join(
            f'<div class="msg-user">{msg}</div>' if role == "user"
            else f'<div class="msg-agent">{to_html(msg)}</div>'
            for role, msg in history
        )
        if new_query:
            body += f'<div class="msg-user">{new_query}</div>'
        if live_html:
            body += live_html

    return f'<div class="nx-panel" id="nx-chat">{body}</div>'

# ── MAIN CHAT PANEL ───────────────────────────
with main_col:
    # One single empty slot owns the ENTIRE panel — history + live updates
    panel_slot = st.empty()
    panel_slot.markdown(
        build_panel(st.session_state.chat),
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
#  PROCESS QUERY
# ─────────────────────────────────────────────
if query:
    st.session_state.query_count += 1
    st.session_state.activity.append(
        (now_stamp(), f"Query: {query[:38]}…" if len(query) > 38 else f"Query: {query}")
    )

    tool_steps_html = ""
    streamed_answer = ""
    session_sources = []

    # Immediately show user message + thinking spinner
    def update_panel(live_html):
        panel_slot.markdown(
            build_panel(st.session_state.chat, new_query=query, live_html=live_html),
            unsafe_allow_html=True
        )
        scroll_to_bottom()

    update_panel('<div class="nx-thinking"><div class="th-label">⬡ PROCESSING <span class="dot-pulse"></span></div></div>')

    try:
        r = requests.post(
            "http://127.0.0.1:8000/ask_stream",
            json={"query": query},
            headers={"X-Api-Key": st.session_state.api_key},
            stream=True,
            timeout=120
        )

        for line in r.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line.decode())
            except Exception:
                continue
            if "type" not in event:
                continue

            if event["type"] == "tool":
                for tc in event.get("data", []):
                    name = tc.get("name", "unknown")
                    st.session_state.tool_count += 1
                    st.session_state.activity.append((now_stamp(), f"Tool: {name}"))
                    tool_steps_html += (
                        f'<div class="tool-step {tool_class(name)}">'
                        f'<span>{tool_icon(name)}</span>'
                        f'<span style="flex:1">{name}</span>'
                        f'<span style="opacity:.4;font-size:10px">✓</span></div>'
                    )
                update_panel(f'<div class="nx-thinking"><div class="th-label">⬡ TOOL EXECUTION <span class="dot-pulse"></span></div>{tool_steps_html}</div>')

            elif event["type"] == "source":
                session_sources.extend(extract_urls(str(event.get("data", ""))))

            elif event["type"] == "answer":
                streamed_answer = event.get("data", "")
                update_panel(
                    f'<div class="nx-thinking">'
                    f'<div class="th-label">⬡ WRITING RESPONSE <span class="dot-pulse"></span></div>'
                    f'{tool_steps_html}</div>'
                )

        # Done — render final formatted answer
        update_panel(f'<div class="msg-agent">{to_html(streamed_answer)}</div>')

    except Exception as e:
        panel_slot.markdown(
            build_panel(
                st.session_state.chat,
                new_query=query,
                live_html=f'<div class="msg-agent" style="border-color:rgba(239,68,68,.4);color:#fca5a5;">⚠ Connection error: {e}<br><span style="font-size:11px;opacity:.55">Make sure backend is running on http://127.0.0.1:8000</span></div>'
            ),
            unsafe_allow_html=True
        )
        streamed_answer = f"[Error: {e}]"

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("assistant", streamed_answer))
    for url in session_sources:
        if url not in st.session_state.sources_all:
            st.session_state.sources_all.append(url)
    st.session_state.activity.append((now_stamp(), "Response delivered"))
    st.rerun()
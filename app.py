#!/usr/bin/env python3
"""SpotifyCares web interface."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from agent import run_agent


st.set_page_config(
    page_title="SpotifyCares | Support desk",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root { --ink: #f4f7f2; --muted: #9daaa0; --green: #b9f227; --line: #29342d; --panel: #151d18; --panel2: #1b261f; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0d120f; color: var(--ink); }
    [data-testid="stSidebar"] { background: #101712; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.2rem; }
    .brand { display: flex; align-items: center; gap: .65rem; margin-bottom: 2.8rem; }
    .brand-mark { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 50%; background: var(--green); color: #101712; font-size: 18px; font-weight: 700; }
    .brand-name { font-family: 'Space Grotesk'; font-size: 1.1rem; font-weight: 700; letter-spacing: -.03em; }
    .side-label { color: #647168; text-transform: uppercase; letter-spacing: .13em; font-size: .68rem; font-weight: 700; margin: 1.4rem 0 .65rem; }
    .side-item { color: #bac5bc; padding: .62rem .75rem; border-radius: 8px; margin: .18rem 0; font-size: .9rem; }
    .side-item.active { background: #253128; color: var(--green); }
    .side-foot { position: fixed; bottom: 1.4rem; color: #69766e; font-size: .75rem; }
    .eyebrow { color: var(--green); text-transform: uppercase; letter-spacing: .17em; font-size: .7rem; font-weight: 700; margin-top: .3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.045em !important; }
    h1 { font-size: clamp(2.1rem, 4vw, 3.7rem) !important; line-height: .98 !important; margin: .5rem 0 1rem !important; }
    .subhead { color: var(--muted); max-width: 600px; line-height: 1.55; font-size: 1rem; margin-bottom: 2rem; }
    .query-shell { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem; box-shadow: 0 18px 50px rgba(0,0,0,.16); }
    .query-label { color: #cbd4cd; font-size: .78rem; font-weight: 700; margin: .1rem 0 .45rem; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.05rem; min-height: 92px; }
    .metric-label { color: var(--muted); text-transform: uppercase; letter-spacing: .11em; font-size: .66rem; font-weight: 700; }
    .metric-value { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.35rem; font-weight: 700; margin-top: .45rem; }
    .metric-value.green { color: var(--green); }
    .result-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1.2rem 1.3rem; margin-top: 1.1rem; }
    .panel-kicker { color: var(--green); text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; font-weight: 700; }
    .reply { color: #f4f7f2; font-size: 1.12rem; line-height: 1.55; margin: .75rem 0 .2rem; }
    .case { background: var(--panel2); border-left: 3px solid #607568; border-radius: 0 8px 8px 0; padding: .8rem 1rem; margin: .6rem 0; }
    .case small { color: var(--green); font-weight: 700; }
    .case p { color: #c1ccc3; margin: .35rem 0 0; line-height: 1.45; font-size: .88rem; }
    .status { padding: .8rem 1rem; border-radius: 8px; font-weight: 700; }
    .status.clear { background: #1c3025; color: var(--green); }
    .status.alert { background: #39271e; color: #ffbd83; }
    .stTextArea textarea { background: #111812 !important; color: #f4f7f2 !important; border: 1px solid #344139 !important; border-radius: 9px !important; font-size: 1rem !important; }
    .stTextArea textarea:focus { border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important; }
    .stButton > button { border-radius: 999px; border: 1px solid #3a493e; background: transparent; color: #d5dfd7; font-weight: 600; }
    .stButton > button:hover { border-color: var(--green); color: var(--green); }
    .stFormSubmitButton > button { border: 0; background: var(--green); color: #111710; border-radius: 8px; font-weight: 800; padding: .6rem 1.25rem; }
    .stFormSubmitButton > button:hover { background: #d0ff54; color: #111710; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">S</div><div class="brand-name">SpotifyCares</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item active">◉ Support desk</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">▣ Conversation history</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">◌ Knowledge base</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">◍ Agent status <span style="color:#b9f227">● online</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-foot">AI support workspace · v1.0</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Support intelligence / live queue</div>', unsafe_allow_html=True)
st.title("What can we solve today?")
st.markdown('<div class="subhead">Turn a listener message into a clear support action. The agent reads the issue, finds similar SpotifyCares resolutions, and drafts the next reply.</div>', unsafe_allow_html=True)

suggestions = [
    "My music keeps stopping on my iPhone",
    "I was charged twice for Premium",
    "I cannot log into my account",
    "Why is this album unavailable?",
]
cols = st.columns(len(suggestions))
for column, suggestion in zip(cols, suggestions):
    with column:
        if st.button(suggestion, use_container_width=True):
            st.session_state["query"] = suggestion

st.markdown('<div class="query-shell">', unsafe_allow_html=True)
with st.form("support_query", clear_on_submit=False):
    st.markdown('<div class="query-label">Listener message</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Listener message",
        value=st.session_state.get("query", ""),
        placeholder="Tell us what is happening with your music, account, or subscription...",
        height=125,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Analyze message →", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    if not query.strip():
        st.warning("Enter a listener message first.")
    else:
        with st.spinner("Reading the conversation..."):
            try:
                result = run_agent(query.strip())
                st.session_state["result"] = result
                st.session_state["query"] = query.strip()
            except Exception as exc:
                st.error(f"The support agent could not process this message: {type(exc).__name__}: {exc}")
                st.caption("Check the Render logs for the provider response. Confirm GROQ_API_KEY and GROQ_MODEL are set in Render.")

result = st.session_state.get("result")
if result:
    st.markdown("---")
    metric_cols = st.columns(4)
    metrics = [
        ("Intent", result.intent.replace("_", " ").title(), False),
        ("Confidence", f"{result.intent_confidence:.0%}", True),
        ("Similar cases", str(len(result.retrieved_examples)), False),
        ("Route", "Human review" if result.escalate else "Auto-handle", not result.escalate),
    ]
    for column, (label, value, green) in zip(metric_cols, metrics):
        with column:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value {"green" if green else ""}">{value}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="result-panel"><div class="panel-kicker">Suggested reply</div><div class="reply">' + result.draft_reply + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="result-panel"><div class="panel-kicker">Similar resolutions</div>', unsafe_allow_html=True)
        for example in result.retrieved_examples:
            st.markdown(
                f'<div class="case"><small>{example["score"]:.0%} match</small><p><b>Listener:</b> {example["customer_text"]}</p><p><b>SpotifyCares:</b> {example["brand_reply"]}</p></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="result-panel"><div class="panel-kicker">Routing decision</div>', unsafe_allow_html=True)
        if result.escalate:
            st.markdown('<div class="status alert">Human review recommended</div>', unsafe_allow_html=True)
            st.markdown(f"**Reason**  \n{result.escalation_reason}")
        else:
            st.markdown('<div class="status clear">Ready for auto-handling</div>', unsafe_allow_html=True)
            st.markdown("The message is clear enough for the support agent to respond directly.")
        st.markdown('</div>', unsafe_allow_html=True)

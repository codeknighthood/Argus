"""ARGUS demo: paste a debate transcript, get its arguments, stances and links.

Start it from the project root (Ollama must be running):
    streamlit run app.py

The page has the same settings as scripts/run_pipeline.py. Nothing leaves your computer: every step runs on the
local Ollama model.
"""
import csv
import html
import io
import json
from pathlib import Path

import streamlit as st

from argus.demo import argument_map_dot, condensed_view, table_rows
from argus.llm import LLMClient, load_config, make_ready_client
from argus.output import COLUMNS, summarize
from argus.pipeline import STANCE_ON, ArgusPipeline, PipelineConfig
from argus.segment import segment_transcript
from argus.stance import STRATEGIES

# ----------------------------------------------------------------------------- edit these two lines for your project
TEAM = "Riya Gandhi · Rohan R. Joshi · Sakshi Joshi"
GUIDE = "Guide: Prof. Shivali Shukla · Department of CSE, Medicaps University"
# ------------------------------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE = SAMPLE_DIR / "sample_transcript.txt"
# file name -> (label shown in the picker, topic that goes with it). Files not listed here still appear.
SAMPLE_INFO = {
    "sample_transcript.txt": ("Minimum wage (text)", "the minimum wage"),
    "council_hearing_congestion_charge.pdf": ("Council hearing: congestion charge (PDF)", "a congestion charge for driving downtown"),
    "tv_panel_compulsory_voting.pdf": ("TV panel: compulsory voting (PDF)", "compulsory voting"),
}
STANCE_CLASS = {"Favor": "favor", "Against": "against", "Neutral": "neutral", "Not assessed": "none"}

st.set_page_config(page_title="Argus · Argument mining for policy debates", page_icon="👁", layout="wide")

# Palette: ink background, one brass accent, three stance colours that read on dark.
FAVOR, AGAINST, NEUTRAL, BRASS = "#5fd0a0", "#ff6b6b", "#8b93a1", "#e3b04b"
EYE = ('<svg width="46" height="46" viewBox="0 0 48 48" fill="none" stroke="#e3b04b" stroke-width="2.2" '
       'stroke-linecap="round"><path d="M3 24C9 12 17 7 24 7s15 5 21 17c-6 12-14 17-21 17S9 36 3 24z"/>'
       '<circle cx="24" cy="24" r="7"/><circle cx="24" cy="24" r="2.4" fill="#e3b04b"/></svg>')

st.markdown("""
<style>
  :root { --ink:#0b0d10; --panel:#13161b; --panel2:#181c22; --line:#252a32; --text:#e8e6e1; --muted:#8b93a1;
          --brass:#e3b04b; --favor:#5fd0a0; --against:#ff6b6b; --neutral:#8b93a1;
          --serif: Georgia, 'Times New Roman', serif; --mono: ui-monospace, 'Cascadia Mono', Consolas, 'SF Mono', monospace; }
  .stApp { background: radial-gradient(900px 420px at 12% -8%, rgba(227,176,75,.075), transparent 62%), var(--ink); }
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  header[data-testid="stHeader"] { background: transparent; }
  h1, h2, h3, h4 { font-family: var(--serif) !important; font-weight: 500 !important; letter-spacing: -.01em; }
  h3 { font-size: 1.45rem !important; margin-top: 1.2rem !important; }

  .hero { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 1.7rem 0 1.5rem 0; margin-bottom: 1.8rem;
          background-image: linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
          background-size: 34px 34px; }
  .eyebrow { font-family: var(--mono); font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--brass); }
  .brand { display: flex; align-items: center; gap: 1rem; margin: .7rem 0 .2rem 0; }
  .wordmark { font-family: var(--serif); font-size: 3.6rem; line-height: 1; letter-spacing: -.025em; color: var(--text); }
  .tag { font-family: var(--serif); font-style: italic; font-size: 1.2rem; color: #b9b6ae; margin: .2rem 0 1.3rem 0; max-width: 40rem; }
  .flow { display: flex; align-items: center; gap: .8rem; font-family: var(--mono); font-size: .76rem; letter-spacing: .1em;
          text-transform: uppercase; color: var(--muted); flex-wrap: wrap; }
  .flow b { color: var(--brass); margin-right: .45rem; font-weight: 600; }
  .flow i { display: block; width: 38px; height: 1px; background: var(--line); }

  .sec { font-family: var(--serif); font-size: 1.35rem; color: var(--text); margin: .2rem 0 .5rem 0; display: flex; align-items: baseline; gap: .7rem; }
  .sec span { font-family: var(--mono); font-size: .75rem; letter-spacing: .12em; color: var(--brass); }

  div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: .9rem 1.1rem; }
  div[data-testid="stMetric"] label p { font-family: var(--mono); font-size: .68rem !important; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
  div[data-testid="stMetricValue"] { font-family: var(--serif); font-size: 2.5rem; color: var(--brass); }

  button[data-testid="stBaseButton-primary"], button[kind="primary"] { background: var(--brass) !important; color: #14110a !important; border: none !important;
          font-family: var(--mono); font-weight: 600; letter-spacing: .08em; text-transform: uppercase; border-radius: 3px !important; padding: .5rem 1.4rem !important; }
  button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover { background: #f1c366 !important; }
  button[data-testid="stBaseButton-secondary"] { border-radius: 3px !important; font-family: var(--mono); font-size: .8rem; }
  div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { border-radius: 3px !important; }
  button[role="tab"] p { font-family: var(--mono); font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; }

  .card { background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--neutral); border-radius: 3px; padding: .8rem 1.05rem; margin: .5rem 0; }
  .card.favor { border-left-color: var(--favor); } .card.against { border-left-color: var(--against); }
  .card .meta { font-family: var(--mono); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
                margin-bottom: .35rem; display: flex; gap: .65rem; flex-wrap: wrap; align-items: center; }
  .card .meta b { color: var(--text); font-weight: 600; }
  .card .text { font-family: var(--serif); font-size: 1.06rem; line-height: 1.5; color: var(--text); }
  .card .note { font-size: .8rem; color: var(--muted); margin-top: .4rem; font-style: italic; }
  .tagx { border: 1px solid var(--line); border-radius: 2px; padding: 0 .4rem; color: #c9c6be; }
  .tagx.support { border-color: var(--favor); color: var(--favor); } .tagx.attack { border-color: var(--against); color: var(--against); }
  .tagx.repeat { border-color: var(--brass); color: var(--brass); }

  .bar { display: flex; height: 6px; gap: 3px; margin: .9rem 0 .45rem 0; }
  .bar span { display: block; height: 100%; border-radius: 1px; }
  .legend { font-family: var(--mono); font-size: .72rem; letter-spacing: .06em; color: var(--muted); display: flex; gap: 1.3rem; flex-wrap: wrap; margin-bottom: .6rem; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .4rem; }
  .status { font-family: var(--mono); font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; margin: .2rem 0 1rem 0; }
  .status.ok { color: var(--favor); } .status.down { color: var(--against); }
  .side-brand { display: flex; align-items: center; gap: .6rem; font-family: var(--serif); font-size: 1.5rem; margin-bottom: .6rem; }
  .footer { font-family: var(--mono); font-size: .7rem; letter-spacing: .08em; color: var(--muted); text-align: center; margin-top: 3rem;
            padding-top: 1.1rem; border-top: 1px solid var(--line); line-height: 1.9; }
  .footer b { color: var(--brass); font-weight: 600; letter-spacing: .2em; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_client(model: str):
    """One client per model name. A failure is raised (and not cached), so fixing Ollama and clicking again works."""
    try:
        return make_ready_client(model)
    except SystemExit as e:                      # make_ready_client exits with a plain message when Ollama is not ready
        raise RuntimeError(str(e)) from None


@st.cache_data(ttl=10, show_spinner=False)
def ollama_status(model: str) -> tuple[bool, str]:
    """Is Ollama running with this model? Checked at most every 10 seconds; only drives the sidebar badge."""
    try:
        cfg = dict(load_config())
        cfg["model"] = model
        LLMClient(cfg).check_ready()
        return True, "Model ready"
    except Exception:
        return False, "Ollama not detected"


def csv_text(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def stance_bar(counts: dict) -> str:
    colours = {"Favor": FAVOR, "Against": AGAINST, "Neutral": NEUTRAL, "Not assessed": "#4b5260"}
    total = sum(counts.values()) or 1
    segments = "".join(f'<span style="width:{100 * n / total:.2f}%;background:{colours.get(k, NEUTRAL)}"></span>'
                       for k, n in counts.items() if n)
    legend = "".join(f'<span><span class="dot" style="background:{colours.get(k, NEUTRAL)}"></span>{esc(k)} {n}</span>'
                     for k, n in counts.items() if n)
    return f'<div class="bar">{segments}</div><div class="legend">{legend}</div>'


def argument_card(row: dict) -> str:
    chips = []
    if row["component"]:
        chips.append(f'<span class="tagx">{esc(row["component"])}</span>')
    if row["relation"] in ("Support", "Attack") and row["related_to_index"] is not None:
        chips.append(f'<span class="tagx {row["relation"].lower()}">{esc(row["relation"])} → #{esc(row["related_to_index"])}</span>')
    if row["repeats"] > 1:
        chips.append(f'<span class="tagx repeat">made {row["repeats"]} times</span>')
    who = f'{esc(row["speaker"])} · ' if row["speaker"] else ""
    notes = []
    if row["also_said_by"]:
        notes.append("also said by " + esc(", ".join(row["also_said_by"])))
    if row["explanation"]:
        notes.append(esc(row["explanation"]))
    note = f'<div class="note">{" · ".join(notes)}</div>' if notes else ""
    return (f'<div class="card {STANCE_CLASS.get(row["stance"], "none")}"><div class="meta"><b>{who}#{esc(row["index"])}</b>'
            f'{"".join(chips)}</div><div class="text">{esc(row["argument"])}</div>{note}</div>')


def dark_dot(dot: str) -> str:
    """Restyle the argument map (drawn for a light page by argus/demo.py) for the dark theme."""
    for light, dark in {"#e8f5e9": "#12312a", "#ffebee": "#3a1c22", "#eeeeee": "#232830", "#ffffff": "#1a1e25"}.items():
        dot = dot.replace(f'fillcolor="{light}"', f'fillcolor="{dark}"')
    dot = dot.replace('node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
                      'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="#e8e6e1", '
                      'color="#3a414d", penwidth=1.1, margin="0.18,0.1"];')
    dot = dot.replace("rankdir=LR;", 'rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.55;')
    return dot.replace("#2e7d32", FAVOR).replace("#c62828", AGAINST)


# ----------------------------------------------------------------------------- sidebar
cfg = load_config()
defaults = cfg.get("pipeline") or {}
with st.sidebar:
    st.markdown(f'<div class="side-brand">{EYE.replace("46", "28")}Argus</div>', unsafe_allow_html=True)
    model = st.text_input("Ollama model", value=cfg["model"], help="Change it in config.yaml to make it the default.")
    ready, status_text = ollama_status(model)
    st.markdown(f'<div class="status {"ok" if ready else "down"}">{"●" if ready else "○"} {status_text}</div>',
                unsafe_allow_html=True)
    if not ready:
        st.caption("Start Ollama, then analyse again. See INSTALL.md, step 1 and 3.")
    with st.expander("Advanced settings"):
        strategy_names = list(STRATEGIES)
        strategy = st.selectbox("Stance prompt", strategy_names,
                                index=strategy_names.index(defaults.get("stance_strategy", "few_shot")))
        stance_on = st.selectbox("Judge stance for", list(STANCE_ON),
                                 index=list(STANCE_ON).index(defaults.get("stance_on", "all")),
                                 help="all = every sentence, arguments = only sentences found to be arguments, none = skip stance")
        context_window = st.slider("Sentences of context", 0, 5, int(defaults.get("context_window", 2)))
        relation_window = st.slider("Compare each argument with the previous", 0, 6, int(defaults.get("relation_window", 3)),
                                    help="0 switches off Support / Attack links")
        merge = st.checkbox("Merge repeated points", value=True)
        threshold = st.slider("How similar counts as a repeat", 0.4, 1.0, 0.7, 0.05, disabled=not merge)
    with st.expander("How it works"):
        st.markdown(
            "1. **Split** the transcript into sentences and keep who said them.\n"
            "2. **Find arguments** and label each as a *claim* or a *premise*.\n"
            "3. **Link** each argument to the earlier one it *supports* or *attacks*.\n"
            "4. **Judge stance** toward the topic: *Favor*, *Against* or *Neutral*, with a reason.\n\n"
            "Every step is a prompt to a local language model. Nothing is uploaded anywhere.")
    st.caption(f"Runs locally on Ollama · {esc(model)}")

# ----------------------------------------------------------------------------- hero and input
st.markdown(
    f'<div class="hero"><div class="eyebrow">Argument mining · Stance detection · Policy debate</div>'
    f'<div class="brand">{EYE}<div class="wordmark">Argus</div></div>'
    '<div class="tag">Every claim and counter-claim in a debate: found, linked and weighed.</div>'
    '<div class="flow"><span><b>01</b>Split</span><i></i><span><b>02</b>Extract</span><i></i>'
    '<span><b>03</b>Link</span><i></i><span><b>04</b>Judge</span></div></div>',
    unsafe_allow_html=True)

def sample_files() -> list:
    """Files in data/sample, the plain-text sample first."""
    found = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".txt", ".pdf"))
    return sorted(found, key=lambda p: (p.name != SAMPLE.name, p.name))


def read_source(name: str, data: bytes):
    """Turn an uploaded or bundled file into transcript text. Returns (text, notes). PDFs are cleaned first."""
    if name.lower().endswith(".pdf"):
        from argus.pdf_text import extract_transcript
        got = extract_transcript(data)
        return got.text, [f"PDF read: {got.pages} page(s), {got.turns} turn(s), {len(got.speakers)} speaker(s)"] + got.notes
    return data.decode("utf-8", errors="replace"), []


def pick_sample() -> None:
    """When another sample is chosen, switch the topic to the one that goes with it."""
    info = SAMPLE_INFO.get(st.session_state.get("sample_pick", ""))
    if info:
        st.session_state["topic_box"] = info[1]


left, right = st.columns([1, 2], gap="large")
with left:
    st.markdown('<div class="sec"><span>01</span>The question</div>', unsafe_allow_html=True)
    if "topic_box" not in st.session_state:
        st.session_state["topic_box"] = "the minimum wage"
    target = st.text_input("Topic", key="topic_box", label_visibility="collapsed",
                           help="The policy or topic that stance is judged against, e.g. 'school uniforms'.")
    st.caption("Every stance is judged toward this topic.")
with right:
    st.markdown('<div class="sec"><span>02</span>The debate</div>', unsafe_allow_html=True)
    source = st.radio("Transcript", ["Sample transcript", "Paste or upload my own"], horizontal=True,
                      label_visibility="collapsed")
    transcript, read_notes = "", []
    try:
        if source == "Sample transcript":
            files = sample_files()
            if len(files) > 1:
                pick = st.selectbox("Sample", [f.name for f in files], key="sample_pick", on_change=pick_sample,
                                    format_func=lambda n: SAMPLE_INFO.get(n, (n, ""))[0], label_visibility="collapsed")
            else:
                pick = files[0].name if files else None
            if pick:
                transcript, read_notes = read_source(pick, (SAMPLE_DIR / pick).read_bytes())
            st.text_area("Sample transcript (invented, for trying the tool)", transcript, height=190, disabled=True)
        else:
            upload = st.file_uploader("A .txt or text-based .pdf, one turn per line as  SPEAKER NAME: what they said",
                                      type=["txt", "pdf"])
            typed = st.text_area("...or paste it here", height=190, placeholder="MODERATOR: Welcome everyone.\nMS. LEE: I support it ...")
            if upload is not None:
                transcript, read_notes = read_source(upload.name, upload.getvalue())
            else:
                transcript = typed
        if read_notes:
            st.caption(" · ".join(read_notes))
    except ValueError as e:      # scanned, encrypted or unreadable PDF, or pypdf missing
        st.error(str(e))

st.markdown('<div class="sec"><span>03</span>The verdict</div>', unsafe_allow_html=True)
if st.button("Analyse debate", type="primary"):
    if not transcript.strip():
        st.warning("Add a transcript first.")
    elif not target.strip():
        st.warning("Say what the debate is about.")
    else:
        try:
            llm = get_client(model)
        except RuntimeError as e:
            st.error(str(e))
            st.info("Start Ollama (INSTALL.md, step 1 and 3) and click Analyse again.")
        else:
            config = PipelineConfig.from_config(cfg, stance_strategy=strategy, stance_on=stance_on,
                                                context_window=context_window, relation_window=relation_window)
            segments = segment_transcript(transcript, config.context_window, config.parse_speakers, config.min_tokens)
            if not segments:
                st.warning("No sentences found in that text.")
            else:
                bar = st.progress(0.0, text=f"0 of {len(segments)} sentences")
                records = ArgusPipeline(llm, config).run_segments(
                    segments, target.strip(),
                    lambda done, total: bar.progress(done / total, text=f"{done} of {total} sentences"))
                bar.empty()
                st.session_state["result"] = {"records": records, "target": target.strip(), "model": model}

# ----------------------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    records = result["records"]
    view = condensed_view(records, merge=merge, threshold=threshold)
    summary = summarize(records)
    st.divider()
    st.markdown(f'<div class="sec"><span>RESULT</span>{esc(result["target"])}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Sentences", view["sentences"])
    cols[1].metric("Arguments", view["arguments"])
    cols[2].metric("Unique points", view["shown"], help="Arguments left after merging near-identical repeats")
    cols[3].metric("Filler dropped", view["dropped"])
    stance_counts = {s: len(rows) for s, rows in view["groups"].items()}
    if stance_counts:
        st.markdown(stance_bar(stance_counts), unsafe_allow_html=True)
    if summary["rows_with_errors"]:
        st.warning(f"{summary['rows_with_errors']} sentence(s) had a model error; see the Full table, 'error' column.")

    tab_view, tab_map, tab_table, tab_save = st.tabs(["Condensed view", "Argument map", "Full table", "Download"])
    with tab_view:
        if not view["groups"]:
            st.info("No arguments were found in this transcript.")
        for stance, rows in view["groups"].items():
            st.subheader(f"{stance} ({len(rows)})")
            st.markdown("".join(argument_card(row) for row in rows), unsafe_allow_html=True)
    with tab_map:
        st.markdown(f'<div class="legend"><span><span class="dot" style="background:{FAVOR}"></span>Support</span>'
                    f'<span><span class="dot" style="background:{AGAINST}"></span>Attack</span>'
                    '<span>Box tint = stance toward the topic</span></div>', unsafe_allow_html=True)
        if view["arguments"]:
            st.graphviz_chart(dark_dot(argument_map_dot(records)))
        else:
            st.info("No arguments to draw.")
    with tab_table:
        st.caption("Every sentence with everything the pipeline decided (the structured output of the synopsis, section 5.7).")
        st.dataframe(table_rows(records, as_text=True))
    with tab_save:
        rows = table_rows(records)
        st.caption("CSV opens in Excel. JSON keeps the full structure.")
        st.download_button("Download CSV (opens in Excel)", csv_text(rows).encode("utf-8-sig"), "argus_result.csv", "text/csv")
        st.download_button("Download JSON", json.dumps({"target": result["target"], "model": result["model"], "records": rows},
                                                      ensure_ascii=False, indent=2).encode("utf-8"),
                           "argus_result.json", "application/json")

st.markdown(f'<div class="footer"><b>ARGUS</b><br>LLM-based argument mining and stance detection for policy debates<br>'
            f'{esc(TEAM)}<br>{esc(GUIDE)}</div>', unsafe_allow_html=True)

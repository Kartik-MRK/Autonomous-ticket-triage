"""
Mozilla Core Ticket Triage UI (Streamlit)
=========================================
Streamlit frontend for querying the Bugzilla-powered RAG pipeline.
Uses Ollama llama3.1:8b for LLM operations.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations
import streamlit as st

from modules.vector_store import get_collection_stats
from pipeline.triage_pipeline import run_triage_pipeline
from config.settings import settings

st.set_page_config(
    page_title="Mozilla Core Ticket Triage",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0f1318;
    --panel: #171d25;
    --panel-2: #1d2632;
    --line: #2d3a4d;
    --accent: #43b581;
    --accent-2: #f6c177;
    --text: #e8eef5;
    --muted: #9eb1c7;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, #182334 0%, #0f1318 45%),
                radial-gradient(circle at 90% 0%, #1b2e2c 0%, transparent 30%),
                var(--bg);
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

.panel {
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1rem 0.75rem 1rem;
    margin-bottom: 1rem;
}

.hero {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.2rem;
    background: linear-gradient(135deg, #1a2535 0%, #1a2330 55%, #20293a 100%);
    margin-bottom: 1rem;
}

.hero h1 { font-size: 1.8rem; margin: 0; color: var(--text); }
.hero p { margin: 0.3rem 0 0 0; color: var(--muted); }

.small-chip {
    display: inline-block;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    margin-right: 0.4rem;
    margin-top: 0.35rem;
    color: var(--muted);
    font-size: 0.8rem;
}

.ref-card {
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.65rem 0.75rem;
    margin-bottom: 0.55rem;
    background: #18212d;
}

.metric-box {
    background: #172231;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.75rem;
}

.stTextArea textarea, .stTextInput input {
    background-color: #111923 !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #3a4f6b;
    background: linear-gradient(90deg, #2f4f74, #3b6a64);
    color: white;
    font-weight: 600;
}

.hyde-badge {
    display: inline-block;
    background: linear-gradient(90deg, #c678dd, #61afef);
    color: white;
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _get_index_count() -> int:
    stats = get_collection_stats()
    return int(stats.get("count", 0))


def _render_reference_cards(references: list[dict]) -> None:
    if not references:
        st.info("No related Bugzilla records were retrieved for this query.")
        return
    for ref in references[:8]:
        issue_number = ref.get("issue_number", "N/A")
        score = float(ref.get("similarity_score", 0.0))
        title = ref.get("title", "Unknown")
        st.markdown(
            (
                "<div class='ref-card'>"
                f"<div><b>#{issue_number}</b>  |  score: <b>{score:.4f}</b></div>"
                f"<div style='margin-top:0.25rem;color:#d5e0ed'>{title}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


st.markdown(
    f"""
<div class='hero'>
    <h1>Mozilla Core Autonomous Ticket Triage</h1>
    <p>
        Query your Bugzilla-indexed knowledge base and get team routing,
        debugging steps, and likely root causes from the RAG pipeline.
    </p>
    <span class='small-chip'>Regex + spaCy preprocessing</span>
    <span class='small-chip'>Hybrid retrieval + RRF</span>
    <span class='small-chip'>BAAI/bge-reranker-base</span>
    <span class='small-chip'>HyDE fallback</span>
    <span class='small-chip'>Ollama {settings.OLLAMA_MODEL}</span>
</div>
""",
    unsafe_allow_html=True,
)

try:
    indexed_count = _get_index_count()
except Exception as exc:
    indexed_count = 0
    st.error(f"Could not read vector store: {exc}")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Indexed Records", f"{indexed_count}")
    st.markdown("</div>", unsafe_allow_html=True)
with col_m2:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Retrieval", "Dense + BM25 + RRF")
    st.markdown("</div>", unsafe_allow_html=True)
with col_m3:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Reranker", "bge-reranker-base")
    st.markdown("</div>", unsafe_allow_html=True)
with col_m4:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("LLM", settings.OLLAMA_MODEL)
    st.markdown("</div>", unsafe_allow_html=True)

if indexed_count == 0:
    st.warning(
        "Index is empty. Run these commands first:\n"
        "1) python main.py ingest\n"
        "2) python main.py build-index"
    )

st.markdown("<div class='panel'>", unsafe_allow_html=True)
with st.form("triage_form", clear_on_submit=False):
    left, right = st.columns([1.2, 1])

    with left:
        title = st.text_input(
            "Ticket title",
            placeholder="Example: Firefox crashes when WebRTC starts on Linux",
        )
        description = st.text_area(
            "Ticket description",
            height=170,
            placeholder="Describe symptoms, reproduction steps, environment, and impact.",
        )

    with right:
        labels_raw = st.text_input(
            "Optional labels (comma separated)",
            value="mozilla-core, bug",
        )
        comments = st.text_area(
            "Optional comments",
            height=110,
            placeholder="Any additional context from reports or logs.",
        )
        submit = st.form_submit_button("Run RAG Triage")

st.markdown("</div>", unsafe_allow_html=True)

if submit:
    if not title.strip() or not description.strip():
        st.error("Title and description are required.")
    elif indexed_count == 0:
        st.error("Vector store is empty. Build the index first.")
    else:
        labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
        with st.spinner("Running retrieval, reranking, classification, and generation..."):
            result = run_triage_pipeline(
                title=title.strip(),
                description=description.strip(),
                labels=labels,
                comments=comments.strip(),
            )

        classification = result.get("classification", {}) or {}
        generated = result.get("generated_response", {}) or {}
        references = result.get("retrieved_references", []) or []
        metadata = result.get("metadata", {}) or {}

        st.markdown("### Classification")
        c1, c2, c3 = st.columns(3)
        c1.metric("Type", classification.get("type", "unknown"))
        c2.metric("Severity", classification.get("severity", "unknown"))
        c3.metric("Team", classification.get("team", "unknown"))

        if metadata.get("hyde_activated"):
            st.markdown("<span class='hyde-badge'>🧪 HyDE Activated</span>", unsafe_allow_html=True)

        st.markdown("### Routing Explanation")
        st.write(generated.get("routing_explanation", "N/A"))

        st.markdown("### Debugging Steps")
        for step in generated.get("debugging_steps", []):
            st.markdown(f"- {step}")

        st.markdown("### Possible Causes")
        for cause in generated.get("possible_causes", []):
            st.markdown(f"- {cause}")

        st.markdown("### Retrieved Bugzilla References")
        _render_reference_cards(references)

        st.markdown("### Pipeline Metadata")
        st.json(metadata)

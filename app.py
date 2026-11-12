"""PDF RAG chatbot: attach PDFs in the chat to ingest them, then ask questions about them.

Run:  uv run streamlit run app.py

The UI is thin: every action goes through the LangGraph chat graph (raglab/graphs/chat_graph.py).
  * Attaching a PDF  -> graph input {"pending_pdf": path}  -> route -> ingest (subgraph)
  * Typing a message -> graph input {"messages": [user]}   -> route -> agent <-> tools
Progress is streamed from the graph (stream_mode="updates", subgraphs=True).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from raglab import config
from raglab.graphs.chat_graph import build_chat_graph
from raglab.store import delete_document, list_documents

UPLOAD_DIR = config.DATA_DIR / "uploads"
STAGE_LABELS = {
    "parse": "Parsing text, tables and images",
    "chunk": "Chunking by content type",
    "enrich": "Describing images and summarising tables with Claude",
    "store": "Embedding and storing in ChromaDB",
}

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")


@st.cache_resource
def get_graph():
    # One compiled graph (and in-memory checkpointer) for the server; each browser session
    # gets its own thread_id, so conversations stay separate.
    return build_chat_graph()


graph = get_graph()
ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("display", [])  # [{"role", "text", "events"}] for rendering only; the graph holds the real state


def graph_config():
    return {"configurable": {"thread_id": ss.thread_id}, "recursion_limit": 30}


def events_so_far() -> int:
    return len(graph.get_state(graph_config()).values.get("events", []))


def last_assistant_text() -> str:
    msgs = graph.get_state(graph_config()).values["messages"]
    return "".join(b.get("text", "") for b in msgs[-1]["content"] if b["type"] == "text")


# ---- rendering --------------------------------------------------------------------------------

def render_events(events: list[dict]) -> None:
    searches = [e for e in events if e["kind"] == "search"]
    if searches:
        with st.expander(f"🔎 {len(searches)} search{'es' if len(searches) > 1 else ''} · sources"):
            for e in searches:
                inp = e["input"]
                types = f"  ·  only {', '.join(inp['content_types'])}" if inp.get("content_types") else ""
                st.markdown(f"**“{inp['query']}”**{types}")
                for h in e["hits"]:
                    st.markdown(f"- `{h['score']:.3f}` **{h['type']}** · {h['citation']} · {h['section']}")
                    if h["image_path"]:
                        st.image(str(config.PROJECT_ROOT / h["image_path"]), width=360)
    for e in events:
        if e["kind"] == "ingest" and e.get("ok"):
            s = e["stats"]
            cols = st.columns(4)
            cols[0].metric("Elements parsed", sum(s["elements"].values()))
            cols[1].metric("Text chunks", s["chunks"]["text"])
            cols[2].metric("Tables", s["chunks"]["table"])
            cols[3].metric("Images", s["chunks"]["image"])
            st.caption(f"Inspection report: `{e['report_path']}`")


def render_history() -> None:
    for m in ss.display:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("events"):
                render_events(m["events"])


# ---- actions ----------------------------------------------------------------------------------

def ingest_file(uploaded, use_llm: bool) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())

    ss.display.append({"role": "user", "text": f"📎 {path.name}"})
    with st.chat_message("user"):
        st.markdown(f"📎 {path.name}")

    with st.chat_message("assistant"):
        before = events_so_far()
        with st.status(f"Ingesting {path.name}…", expanded=True) as status:
            for namespace, update in graph.stream(
                {"pending_pdf": str(path), "use_llm": use_llm},
                graph_config(), stream_mode="updates", subgraphs=True,
            ):
                for node in update:
                    if namespace and node in STAGE_LABELS:  # progress from inside the ingest subgraph
                        st.write(f"✓ {STAGE_LABELS[node]}")
            status.update(label=f"Ingested {path.name}", state="complete", expanded=False)
        text = last_assistant_text()
        events = graph.get_state(graph_config()).values["events"][before:]
        st.markdown(text)
        render_events(events)
    ss.display.append({"role": "assistant", "text": text, "events": events})


def ask(question: str) -> None:
    ss.display.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        before = events_so_far()
        with st.status("Thinking…", expanded=False) as status:
            for update in graph.stream({"messages": [{"role": "user", "content": question}]},
                                       graph_config(), stream_mode="updates"):
                for e in update.get("tools", {}).get("events", []):
                    if e["kind"] == "search":
                        status.update(label=f"Searching: “{e['input']['query']}”")
                        st.write(f"🔎 {e['input']['query']} → {len(e['hits'])} hits")
                    elif e["kind"] == "list_documents":
                        st.write("📚 Listing documents")
            status.update(label="Done", state="complete")
        text = last_assistant_text()
        events = graph.get_state(graph_config()).values["events"][before:]
        st.markdown(text)
        render_events(events)
    ss.display.append({"role": "assistant", "text": text, "events": events})


# ---- sidebar ----------------------------------------------------------------------------------

with st.sidebar:
    st.header("📚 Knowledge base")
    use_llm = st.toggle("Describe images and tables with Claude", value=True,
                        help="Off = faster and free, but images are searchable only by their captions.")
    docs = list_documents()
    if not docs:
        st.info("Empty. Attach a PDF with the 📎 in the chat box.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        def n(count, word):
            return f"{count} {word}{'' if count == 1 else 's'}"
        c1.markdown(f"**{d['source']}**  \n{n(d['pages'], 'page')} · {d['text']} text · "
                    f"{n(d['table'], 'table')} · {n(d['image'], 'image')}")
        if c2.button("🗑", key=f"del-{d['source']}", help=f"Remove {d['source']} from the index"):
            delete_document(d["source"])
            st.rerun()
    st.divider()
    if st.button("New conversation", use_container_width=True):
        ss.thread_id = str(uuid.uuid4())
        ss.display = []
        st.rerun()
    st.caption(f"Thread `{ss.thread_id[:8]}` · {config.CLAUDE_MODEL} · {config.EMBED_MODEL}")


# ---- main -------------------------------------------------------------------------------------

st.title("📄 PDF RAG Chatbot")
st.caption("Attach PDFs to add them to the vector database, then ask about their text, tables and images.")

render_history()

prompt = st.chat_input("Ask a question, or attach a PDF…", accept_file="multiple", file_type=["pdf"])
if prompt:
    for f in prompt.files:
        ingest_file(f, use_llm)
    if prompt.text.strip():
        ask(prompt.text.strip())
    if prompt.files:
        st.rerun()  # refresh the sidebar's document list

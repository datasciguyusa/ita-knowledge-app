"""
Tennis Rules Q&A — Streamlit app (Gemini free tier)
Run: streamlit run app.py
Get a free API key at: aistudio.google.com
"""

import os
import re
import math
import hashlib
from pathlib import Path
from collections import Counter

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import tempfile, pathlib
from pypdf import PdfReader

# ── Constants ─────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
CHUNK_SIZE = 800
TOP_K = 8
MAX_CONTEXT_CHARS = 65_000
MODEL = "gemini-2.0-flash"

FRIENDLY_NAMES = {
    "2026-code-of-conduct-for-officials.pdf":             "Code of Conduct – Officials (2026)",
    "2026-itf-world-tennis-tour-juniors-regulations.pdf": "ITF Juniors Regulations (2026)",
    "2026-rulebook_27apr26.pdf":                          "ITF Rulebook (Apr 2026)",
    "2026-rules-of-tennis-english.pdf":                   "Rules of Tennis (2026)",
    "2026-wtt-regulations men.pdf":                       "WTT Regulations – Men (2026)",
    "2026-wtt-regulations.pdf":                           "WTT Regulations (2026)",
    "duties-procedures-for-officials-2026.pdf":           "Duties & Procedures for Officials (2026)",
    "wheelchair-tennis-comp-regulations-2026-jan26-v141.pdf": "Wheelchair Tennis Regulations (2026)",
    "wta-2026-rulebook.pdf":                              "WTA Rulebook (2026)",
}

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","that","this",
    "it","its","i","you","he","she","we","they","what","which","who","when",
    "where","how","if","not","no","so","as","by","from","into","than","then",
    "there","their","them","these","those","can","all","any","each","more",
}

SYSTEM = (
    "You are an experienced ITF-certified tennis rules official and referee. "
    "STEP 1 — Identify the match context: Before answering, determine the type of match "
    "or competition described (e.g., WTT Juniors, WTA, wheelchair, standard ITF, etc.). "
    "STEP 2 — Use the correct rulebook: Always prioritize the document that specifically "
    "applies to that match type. For example: WTT Juniors questions → use the ITF Juniors "
    "Regulations; WTA match questions → use the WTA Rulebook; wheelchair tennis → use the "
    "Wheelchair Tennis Regulations. General rules questions → use Rules of Tennis. "
    "Special competition rules (like No-Let) override the general Rules of Tennis when they apply. "
    "STEP 3 — Give a direct ruling: State the answer clearly first, then cite the specific "
    "rule, document, and page number. "
    "For True/False: state TRUE or FALSE first. "
    "For multiple-choice: state the correct option first. "
    "For scenarios: state the ruling directly. "
    "When a rule clearly applies by its plain meaning, apply it confidently. "
    "Only say a question is not covered if there is genuinely no relevant rule in the excerpts. "
    "When a Code Violation is part of the ruling, always specify the exact offense (e.g., Code Violation - Delay of Game, Code Violation - Unsportsmanlike Conduct, etc.)."
)

# ── Pure-Python BM25 retrieval ────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]

def bm25_score(query_terms, doc_terms, doc_len, avgdl, idf, k1=1.5, b=0.75):
    score = 0.0
    for term in query_terms:
        if term not in doc_terms:
            continue
        tf = doc_terms[term]
        score += idf.get(term, 0) * (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * doc_len / max(avgdl, 1))
        )
    return score

def build_index(chunks):
    term_counts, doc_lens, df = [], [], Counter()
    for c in chunks:
        tc = Counter(tokenize(c["text"]))
        term_counts.append(tc)
        doc_lens.append(sum(tc.values()))
        for t in tc:
            df[t] += 1
    n = len(chunks)
    avgdl = sum(doc_lens) / max(n, 1)
    idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
    return term_counts, doc_lens, idf, avgdl

DOC_BOOST = {
    "2026-itf-world-tennis-tour-juniors-regulations.pdf": [
        "junior", "juniors", "wtt junior", "itf junior", "world tennis tour junior"
    ],
    "wta-2026-rulebook.pdf": [
        "wta", "women", "women's tour"
    ],
    "wheelchair-tennis-comp-regulations-2026-jan26-v141.pdf": [
        "wheelchair", "quad", "standing player"
    ],
    "2026-wtt-regulations men.pdf": [
        "atp", "men's tour", "wtt men"
    ],
    "duties-procedures-for-officials-2026.pdf": [
        "chair umpire", "line umpire", "referee", "supervisor", "official procedure"
    ],
    "2026-code-of-conduct-for-officials.pdf": [
        "code of conduct", "official conduct", "umpire conduct"
    ],
}

def retrieve(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    if not chunks:
        return []
    query_terms = tokenize(query)
    query_lower = query.lower()

    priority_doc = None
    for doc_file, keywords in DOC_BOOST.items():
        if any(kw in query_lower for kw in keywords):
            priority_doc = doc_file
            break

    if not query_terms:
        return chunks[:top_k]

    term_counts, doc_lens, idf, avgdl = build_index(chunks)

    if priority_doc:
        priority_chunks = [(i, c) for i, c in enumerate(chunks) if c["source"] == priority_doc]
        other_chunks    = [(i, c) for i, c in enumerate(chunks) if c["source"] != priority_doc]

        def score_group(group):
            return sorted(
                [(bm25_score(query_terms, term_counts[i], doc_lens[i], avgdl, idf), c)
                 for i, c in group],
                key=lambda x: -x[0]
            )

        priority_scored = score_group(priority_chunks)
        other_scored    = score_group(other_chunks)

        guaranteed = top_k // 2
        result = [c for _, c in priority_scored[:guaranteed]]
        seen_ids = {id(c) for c in result}
        remaining = top_k - len(result)
        for _, c in (priority_scored[guaranteed:] + other_scored):
            if id(c) not in seen_ids and remaining > 0:
                result.append(c)
                seen_ids.add(id(c))
                remaining -= 1
        return result
    else:
        scored = [
            (bm25_score(query_terms, term_counts[i], doc_lens[i], avgdl, idf), chunks[i])
            for i in range(len(chunks))
        ]
        scored.sort(key=lambda x: -x[0])
        return [c for s, c in scored[:top_k] if s > 0]

def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        label = FRIENDLY_NAMES.get(c["source"], c["source"])
        parts.append(f"[{label} — page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_CHARS]

# ── PDF extraction ────────────────────────────────────────────────────────────

def _cache_key(pdf_path: Path) -> str:
    s = pdf_path.stat()
    return hashlib.md5(f"{pdf_path.name}:{s.st_size}:{s.st_mtime}".encode()).hexdigest()

@st.cache_data(show_spinner=False)
def extract_chunks(pdf_paths_keys: tuple) -> dict:
    result: dict[str, list[dict]] = {}
    for pdf_path_str, _key in pdf_paths_keys:
        pdf_path = Path(pdf_path_str)
        chunks: list[dict] = []
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE):
                    chunk = " ".join(words[i : i + CHUNK_SIZE])
                    if len(chunk) >= 80:
                        chunks.append({"text": chunk, "page": page_num, "source": pdf_path.name})
        except Exception as exc:
            st.warning(f"⚠️ Could not read **{pdf_path.name}**: {exc}")
        result[pdf_path.name] = chunks
    return result

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tennis Rules Q&A", page_icon="🎾", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎾 Tennis Rules Q&A")
    st.markdown("---")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza…",
            help="Free key at aistudio.google.com — no credit card needed.",
        )
        if not api_key:
            st.caption("👉 [Get a free key at aistudio.google.com](https://aistudio.google.com)")

    st.markdown("---")
    st.subheader("📄 Source Documents")
    st.caption("Check documents to include in the search:")

    pdfs = sorted(APP_DIR.glob("*.pdf"))
    if not pdfs:
        st.error(f"No PDF files found in:\n`{APP_DIR}`")
        st.stop()

    selected_files: list[Path] = []
    for pdf in pdfs:
        label = FRIENDLY_NAMES.get(pdf.name, pdf.stem.replace("-", " ").title())
        if st.checkbox(label, value=True, key=f"cb_{pdf.name}"):
            selected_files.append(pdf)

    st.markdown("---")
    st.caption(f"{len(selected_files)} / {len(pdfs)} documents selected")

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_voice", None)
        st.rerun()

# ── Guards ────────────────────────────────────────────────────────────────────

if not api_key:
    st.info("👈 Enter your free Gemini API key in the sidebar to get started.")
    st.stop()

if not selected_files:
    st.warning("Select at least one document in the sidebar.")
    st.stop()

# ── Load chunks ───────────────────────────────────────────────────────────────

with st.spinner("Loading documents…"):
    cache_input = tuple((str(p), _cache_key(p)) for p in selected_files)
    doc_chunks = extract_chunks(cache_input)

active_chunks: list[dict] = []
for pdf in selected_files:
    active_chunks.extend(doc_chunks.get(pdf.name, []))

# ── Chat UI ───────────────────────────────────────────────────────────────────

st.title("🎾 Tennis Rules & Regulations Q&A")
st.caption(
    "Ask any question about tennis rules, officiating situations, "
    "or tournament regulations. Answers are drawn from the selected documents."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

col_mic, col_text = st.columns([1, 3])

with col_mic:
    audio = st.audio_input("🎙️ Speak your question", key="mic")

with col_text:
    with st.form("q_form", clear_on_submit=True):
        typed = st.text_input(
            "question",
            placeholder="Type your question here…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask ➤", use_container_width=True)

prompt = None

if audio is not None:
    with st.spinner("Transcribing…"):
        try:
            import base64
            genai.configure(api_key=api_key)
            audio_bytes = audio.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            tc_model = genai.GenerativeModel(MODEL)
            result = tc_model.generate_content([
                "Transcribe this audio exactly as spoken. Return only the transcription, nothing else.",
                {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
            ])
            prompt = result.text.strip()
        except Exception as e:
            st.warning(f"Transcription failed: {e}")

if submitted and typed.strip():
    prompt = typed.strip()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching documents…"):
        hits = retrieve(prompt, active_chunks)
        context = build_context(hits)

    gemini_history = []
    for m in st.session_state.messages[:-1][-18:]:
        role = "model" if m["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [m["content"]]})

    user_message = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {prompt}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM)
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_message, stream=True)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as exc:
            err = str(exc)
            if "API_KEY" in err.upper() or "403" in err or "401" in err:
                st.error("Invalid API key — check the key in the sidebar.")
            else:
                st.error(f"Error: {exc}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    if hits:
        with st.expander("📚 Document sections consulted", expanded=False):
            seen: set = set()
            for c in hits:
                key = (c["source"], c["page"])
                if key not in seen:
                    seen.add(key)
                    label = FRIENDLY_NAMES.get(c["source"], c["source"])
                    st.markdown(f"- **{label}** — page {c['page']}")

    st.rerun()

"""
Tennis Rules Q&A — Streamlit app (Claude Haiku 5.5)
Run: streamlit run app.py
API key: console.anthropic.com (ANTHROPIC_API_KEY).
"""

import os
import re
import math
import io
from pathlib import Path
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import exam_bank
from llm import LLM, friendly_error
from usage import AccessList, Budget, UsageStore
import test_prep

# ── Constants ─────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "qa_cache"          # AI results (question bank, transcripts)


def _docs_dir() -> Path:
    """DOCS_DIR setting → your USTA folder (on your PC) → ./docs bundled with the app (web)."""
    env = os.getenv("DOCS_DIR")
    if env:
        return Path(env)
    local = Path(r"C:\Users\scott\Documents\__USTA\White Badge Pre-Reqs")
    if local.is_dir():
        return local
    return APP_DIR / "docs"


DOCS_DIR = _docs_dir()
CHUNK_SIZE = 800
TOP_K = 8
MAX_CONTEXT_CHARS = 65_000
MODEL = "claude-haiku-5-5"            # Claude Haiku 5.5 — keep prompts under 100K tokens (price ×5 above)

# Optional nicer names; any other file shows its file name.
FRIENDLY_NAMES = {
    "2026-code-of-conduct-for-officials.pdf":             "Code of Conduct – Officials (2026)",
    "2026-itf-world-tennis-tour-juniors-regulations.pdf": "ITF Juniors Regulations (2026)",
    "2026-rulebook_27apr26.pdf":                          "ITF Rulebook (Apr 2026)",
    "2026-rules-of-tennis-english.pdf":                   "Rules of Tennis (2026)",
    "2026-wtt-regulations men.pdf":                       "WTT Regulations – Men (2026)",
    "2026-wtt-regulations.pdf":                           "WTT Regulations (2026)",
    "duties-procedures-for-officials-2026.pdf":           "Duties & Procedures for Officials (2026)",
    "2025-duties-procedures-for-officials.pdf":           "Duties & Procedures for Officials (2025)",
    "wheelchair-tennis-comp-regulations-2026-jan26-v141.pdf": "Wheelchair Tennis Regulations (2026)",
    "wta-2026-rulebook.pdf":                              "WTA Rulebook (2026)",
    "rules-of-beach-tennis-2026.pdf":                     "Rules of Beach Tennis (2026)",
    "2026-itf-masters-world-tennis-tour-regulations-final-december-2025.pdf": "ITF Masters Tour Regulations (2026)",
    "wheelchair-tennis-comp-regs-2025-march-v2.pdf":      "Wheelchair Tennis Regulations (2025)",
    "WTT Tennis Balls Guidelines.pdf":                    "WTT Tennis Balls Guidelines",
    "Toilet Break Summary - WT Tours.pdf":                "Toilet Break Summary – WT Tours",
    "CORRECTING ERRORS - ENGLISH.pdf":                    "Correcting Errors",
}


class _Names(dict):
    def get(self, k, default=None):
        return super().get(k) or Path(k).stem


FRIENDLY_NAMES = _Names(FRIENDLY_NAMES)


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
    "Wheelchair Tennis Regulations; beach tennis → use the Rules of Beach Tennis; "
    "ITF Masters (seniors) events → use the ITF Masters Tour Regulations. General rules questions → use Rules of Tennis. "
    "Special competition rules (like No-Let) override the general Rules of Tennis when they apply. "
    "Logo, clothing and equipment questions → use the Dress & Equipment cards first, then the regulations. "
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

# Question keywords → part of a file name to prioritise. Works for whatever files are in the folder.
DOC_BOOST = [
    (["beach tennis", "beach"], "beach"),
    (["masters", "seniors", "senior tour"], "masters"),
    (["logo", "logos", "shirt", "sleeve", "cap", "headband", "wristband", "leggings", "skirt", "shorts",
      "manufacturer", "commercial", "clothing", "attire", "dress", "tattoo", "bag"], "dress"),
    (["junior", "juniors"], "junior"),
    (["wta", "women's tour"], "wta"),
    (["wheelchair", "quad", "standing player"], "wheelchair"),
    (["atp", "men's tour", "wtt men"], "men"),
    (["chair umpire", "line umpire", "referee", "supervisor", "official procedure"], "duties"),
    (["code of conduct", "official conduct", "umpire conduct"], "code-of-conduct"),
]

def retrieve(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    if not chunks:
        return []
    query_terms = tokenize(query)
    query_lower = query.lower()
    sources = {c["source"] for c in chunks}

    priority_docs = set()
    for keywords, name_part in DOC_BOOST:
        if any(re.search(rf"\b{re.escape(kw)}", query_lower) for kw in keywords):
            priority_docs = {s for s in sources if name_part in s.lower()}
            if priority_docs:
                break

    if not query_terms:
        return chunks[:top_k]

    term_counts, doc_lens, idf, avgdl = build_index(chunks)

    if priority_docs:
        priority_chunks = [(i, c) for i, c in enumerate(chunks) if c["source"] in priority_docs]
        other_chunks    = [(i, c) for i, c in enumerate(chunks) if c["source"] not in priority_docs]

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

# ── Document extraction ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def extract_chunks(paths_keys: tuple, _transcribe=None) -> dict:
    result: dict[str, list[dict]] = {}
    for path_str, _key in paths_keys:
        path = Path(path_str)
        chunks: list[dict] = []
        try:
            pages = exam_bank.document_pages(path, _transcribe, CACHE_DIR / "transcripts.json")
            for page_label, text in pages:
                words = (text or "").split()
                for i in range(0, len(words), CHUNK_SIZE):
                    chunk = " ".join(words[i : i + CHUNK_SIZE])
                    if len(chunk) >= 80:
                        chunks.append({"text": chunk, "page": page_label, "source": path.name})
        except Exception as exc:
            st.warning(f"⚠️ Could not read **{path.name}**: {exc}")
        result[path.name] = chunks
    return result


def _folder_signature(folder: str) -> tuple:
    """Names, sizes and dates of the folder's files: changes whenever a file is added, edited or removed."""
    out = []
    for f in Path(folder).rglob("*"):
        if f.is_file():
            s = f.stat()
            out.append((str(f), s.st_size, s.st_mtime))
    return tuple(sorted(out))


@st.cache_data(show_spinner=False)
def scan(folder: str, nonce: int = 0, signature: tuple = ()) -> dict:
    # nonce and signature have no leading underscore on purpose: Streamlit only rescans when they change.
    found = exam_bank.scan_folder(Path(folder))
    return {k: [str(p) for p in v] for k, v in found.items()}


def _mtime_key(p: Path) -> str:
    s = p.stat()
    return f"{s.st_size}:{s.st_mtime}"

# ── Document reader ───────────────────────────────────────────────────────────

ASK_MODE, TEST_MODE, READ_MODE = "💬 Ask a question", "📝 Test prep", "📖 Read a document"
VIEW_PAGES = 10            # pages shown at a time (keeps big rulebooks fast on phones)


@st.cache_data(show_spinner=False, max_entries=20)
def pdf_page_count(path: str, stamp: str) -> int:
    from pypdf import PdfReader
    return len(PdfReader(path).pages)


@st.cache_data(show_spinner=False, max_entries=40)
def pdf_slice(path: str, stamp: str, start: int, count: int) -> bytes:
    """Pages start..start+count-1 (1-based) as a small PDF."""
    from pypdf import PdfReader, PdfWriter
    reader, writer = PdfReader(path), PdfWriter()
    for i in range(start - 1, min(start - 1 + count, len(reader.pages))):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=20)
def page_texts(path: str, stamp: str) -> list:
    return exam_bank.document_pages(Path(path))


def open_in_reader(source: str, page) -> None:
    """Button callback: switch to the reader at a document and page."""
    st.session_state.mode = READ_MODE
    st.session_state.view_doc = source
    try:
        st.session_state.view_page = max(1, int(str(page).lstrip("§")))
    except ValueError:
        st.session_state.view_page = 1


def _set_page(n: int) -> None:
    st.session_state.view_page = n


def render_reader(files: list[Path]) -> None:
    st.title("📖 Read a document")
    names = [f.name for f in files]
    if st.session_state.get("view_doc") not in names:
        st.session_state.view_doc = names[0]
    doc = st.selectbox("Document", names, key="view_doc", format_func=FRIENDLY_NAMES.get,
                       on_change=_set_page, args=(1,))
    path = next(f for f in files if f.name == doc)
    stamp = _mtime_key(path)

    if path.suffix.lower() != ".pdf":                      # Word files: show the text
        for label, text in page_texts(str(path), stamp):
            st.markdown(f"**{label}**")
            st.text(text)
        return

    n = pdf_page_count(str(path), stamp)
    st.session_state.view_page = min(max(1, int(st.session_state.get("view_page", 1))), n)
    start = st.session_state.view_page
    end = min(start + VIEW_PAGES - 1, n)

    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.2, 2])
    c1.button("◀ Back", width="stretch", disabled=start <= 1,
              on_click=_set_page, args=(max(1, start - VIEW_PAGES),))
    c2.number_input("Go to page", min_value=1, max_value=n, step=1, key="view_page",
                    label_visibility="collapsed")
    c3.button("Next ▶", width="stretch", disabled=end >= n,
              on_click=_set_page, args=(min(n, start + VIEW_PAGES),))
    c4.download_button("⬇️ Download full PDF", path.read_bytes(), file_name=path.name,
                       mime="application/pdf", width="stretch")
    st.caption(f"Pages {start}–{end} of {n}. Page numbers count from the PDF's first page "
               "(the numbers answers cite), which can differ from the numbers printed on the pages.")

    term = st.text_input("Find in this document", placeholder="e.g. toilet break, coaching, medical time-out")
    if term.strip():
        words = term.lower().split()
        hits = [label for label, text in page_texts(str(path), stamp)
                if all(w in (text or "").lower() for w in words)]
        if hits:
            st.caption(f"Found on {len(hits)} page(s)" + (" — first 24 shown" if len(hits) > 24 else "") + ":")
            cols = st.columns(8)
            for k, label in enumerate(hits[:24]):
                cols[k % 8].button(f"p. {label}", key=f"hit_{label}", width="stretch",
                                   on_click=_set_page, args=(int(label),))
        else:
            st.caption("No pages contain all of those words.")

    data = pdf_slice(str(path), stamp, start, VIEW_PAGES)
    try:
        st.pdf(data, height=900)
    except Exception:                                      # streamlit-pdf not installed
        st.info("The in-app viewer needs `streamlit[pdf]` (see requirements.txt). "
                "Use **Download full PDF** above in the meantime.")

# ── Light / dark toggle ───────────────────────────────────────────────────────

def _apply_theme(choice: str) -> None:
    """Save the viewer's choice where Streamlit's own theme menu keeps it (this browser only),
    then reload so Streamlit applies its full light or dark theme."""
    import json
    js = f"""
      const key = `stActiveTheme-${{location.pathname}}-v2`;
      localStorage.setItem(key, {json.dumps(json.dumps(choice))});
      const u = new URL(location.href);
      u.searchParams.set("theme", {json.dumps(choice.lower())});
      location.replace(u.toString());
    """
    try:
        st.html(f"<script>{js}</script>", unsafe_allow_javascript=True)
    except TypeError:          # older Streamlit without unsafe_allow_javascript
        import streamlit.components.v1 as components
        components.html(f"<script>{js.replace('location', 'window.parent.location').replace('localStorage', 'window.parent.localStorage')}</script>", height=0)


def theme_toggle() -> None:
    from_url = st.query_params.get("theme")
    if from_url in ("dark", "light"):
        is_dark = from_url == "dark"
    else:
        is_dark = (getattr(st.context.theme, "type", None) or "dark") == "dark"
    if st.toggle("🌙 Dark mode", value=is_dark, key="dark_mode",
                 help="Saved in this browser. Switching reloads the page (clears the chat).") != is_dark:
        _apply_theme("Dark" if st.session_state.dark_mode else "Light")

# ── Login & monthly spending limit ─────────────────────────────────────────────

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _google_login_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def login_gate() -> str:
    """Return the signed-in user's email, or show a login screen and stop."""
    if _google_login_configured():                     # verified sign-in (st.login / Google)
        if not st.user.is_logged_in:
            st.title("🎾 Tennis Rules Q&A")
            st.write("Sign in with the Google account for the email address you were invited with.")
            if st.button("Sign in with Google", type="primary"):
                st.login()
            st.stop()
        return str(st.user.email).strip().lower()

    email = st.session_state.get("user_email") or st.query_params.get("user", "")
    if email and EMAIL_RE.fullmatch(email):
        st.session_state.user_email = email
        return email
    st.title("🎾 Tennis Rules Q&A")
    with st.form("login"):
        typed = st.text_input("Email address", placeholder="you@example.com")
        if st.form_submit_button("Continue", type="primary"):
            typed = typed.strip().lower()
            if EMAIL_RE.fullmatch(typed):
                st.session_state.user_email = typed
                st.query_params["user"] = typed           # stay signed in across page reloads
                st.rerun()
            st.error("Please enter a valid email address.")
    st.stop()


def logout() -> None:
    if _google_login_configured():
        st.logout()
    st.session_state.pop("user_email", None)
    st.query_params.pop("user", None)
    st.rerun()


ALLOWED_FILE = APP_DIR / "allowed_emails.txt"


def read_allowed_file() -> set[str]:
    """Emails listed in allowed_emails.txt (one per line, # for comments). Re-read every page load."""
    try:
        lines = ALLOWED_FILE.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return set()
    out = set()
    for line in lines:
        line = line.split("#", 1)[0]
        for part in line.replace(";", ",").split(","):
            part = part.strip().lower()
            if EMAIL_RE.fullmatch(part):
                out.add(part)
    return out


@st.cache_resource(show_spinner=False)
def usage_store(url: str, tz: str) -> UsageStore:
    return UsageStore(url, tz)


def _money(x: float) -> str:
    return f"${x:,.3f}" if x < 1 else f"${x:,.2f}"


def render_usage(box, budget: Budget) -> None:
    spent = budget.spent
    with box.container():
        st.caption(f"Signed in as **{budget.email}**")
        if budget.unlimited:
            st.caption(f"Spent this month: **{_money(spent)}** (no limit)")
        else:
            st.progress(min(spent / budget.limit, 1.0),
                        text=f"{_money(spent)} of ${budget.limit:.2f} used this month")
            st.caption(f"**{_money(budget.remaining)}** left · resets on the 1st")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tennis Rules Q&A", page_icon="🎾", layout="wide")

def _setting(name: str) -> str:
    val = os.getenv(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val


api_key = _setting("ANTHROPIC_API_KEY")
MONTHLY_LIMIT_USD = float(_setting("MONTHLY_LIMIT_USD") or 3.0)
ADMIN_EMAILS = {e.strip().lower() for e in _setting("ADMIN_EMAILS").split(",") if e.strip()}
USAGE_DB_URL = _setting("USAGE_DB_URL") or f"sqlite:///{(APP_DIR / 'usage.db').as_posix()}"
APP_TIMEZONE = _setting("APP_TIMEZONE") or "America/Denver"

if not ADMIN_EMAILS:
    st.error("Access isn't set up yet. Add your own email as ADMIN_EMAILS=you@example.com in the "
             ".env file (or the app's Secrets on Streamlit Cloud), then restart the app.")
    st.stop()

user_email = login_gate()
is_admin = user_email in ADMIN_EMAILS

try:
    store = usage_store(USAGE_DB_URL, APP_TIMEZONE)
    access = AccessList(store)
except Exception as exc:
    st.error(f"Couldn't open the usage database ({exc}). Check USAGE_DB_URL.")
    st.stop()

# Only approved emails get past this point: admins, allowed_emails.txt, or approved in the app.
# Everyone else is added to your pending list.
file_allowed = read_allowed_file()
if not is_admin and user_email not in file_allowed and not access.is_allowed(user_email):
    access.request(user_email)
    st.title("🎾 Tennis Rules Q&A")
    st.info(f"**{user_email}** hasn't been approved yet. Your request is waiting for the app owner — "
            "try again after they've let you in.")
    if st.button("Use a different email"):
        logout()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎾 Tennis Rules Q&A")
    mode = st.radio("Mode", [ASK_MODE, TEST_MODE, READ_MODE], key="mode", label_visibility="collapsed")
    usage_box = st.empty()
    st.markdown("---")

    if not api_key:
        api_key = st.text_input(
            "Anthropic (Claude) API Key",
            type="password",
            placeholder="sk-ant-…",
            help="Create one at console.anthropic.com → API keys.",
        )
        if not api_key:
            st.caption("👉 [Get a key at console.anthropic.com](https://console.anthropic.com)")
        st.markdown("---")

    # Check the folder for every file each time the page loads.
    found = scan(str(DOCS_DIR), st.session_state.get("scan_nonce", 0), _folder_signature(str(DOCS_DIR)))
    ref_files = [Path(p) for p in found["reference"]]
    test_files = [Path(p) for p in found["tests"]]

    st.subheader("📄 Reference Documents")
    if not ref_files:
        st.error(f"No PDF or Word files found in:\n`{DOCS_DIR}`")
        st.stop()
    st.caption("Check documents to include in the search:")

    selected_files: list[Path] = []
    for f in ref_files:
        label = FRIENDLY_NAMES.get(f.name)
        if st.checkbox(label, value=True, key=f"cb_{f.name}"):
            selected_files.append(f)
    st.caption(f"{len(selected_files)} / {len(ref_files)} documents selected")

    test_status_box = st.expander(f"📝 Practice tests ({len(test_files)})")
    if found["skipped"]:
        st.caption("Not readable (only PDF and Word files are): "
                   + ", ".join(Path(p).name for p in found["skipped"]))

    if st.button("🔄 Rescan folder", width="stretch",
                 help="Pick up files added to the folder since the page loaded."):
        st.session_state["scan_nonce"] = st.session_state.get("scan_nonce", 0) + 1
        st.rerun()

    st.markdown("---")
    if is_admin:
        for e in access.pending():
            if e in file_allowed:
                access.dismiss(e)
        pending = access.pending()
        with st.expander(f"🔐 Manage access{f' — {len(pending)} waiting' if pending else ''}",
                         expanded=bool(pending)):
            for e in pending:
                c1, c2, c3 = st.columns([5, 2, 2])
                c1.caption(f"⏳ {e}")
                if c2.button("Approve", key=f"ok_{e}"):
                    access.allow(e, user_email)
                    st.rerun()
                if c3.button("Deny", key=f"no_{e}"):
                    access.dismiss(e)
                    st.rerun()
            with st.form("add_emails", clear_on_submit=True):
                new = st.text_input("Approve emails", placeholder="name@example.com, other@example.com")
                if st.form_submit_button("Add"):
                    added = [x.strip().lower() for x in new.replace(";", ",").split(",") if x.strip()]
                    bad = [x for x in added if not EMAIL_RE.fullmatch(x)]
                    for x in added:
                        if x not in bad:
                            access.allow(x, user_email)
                    if bad:
                        st.error("Not valid: " + ", ".join(bad))
                    else:
                        st.rerun()
            st.caption("Approved in the app:")
            for e in access.allowed():
                c1, c2 = st.columns([7, 2])
                c1.caption(e + (" (also in file)" if e in file_allowed else ""))
                if c2.button("✖", key=f"rm_{e}", help=f"Remove {e}"):
                    access.revoke(e)
                    st.rerun()
            st.caption("From **allowed_emails.txt** (edit the file to change): "
                       + (", ".join(sorted(file_allowed)) or "none"))
            st.caption("Admins (always allowed): " + ", ".join(sorted(ADMIN_EMAILS)))
        with st.expander("👥 Everyone's spending this month"):
            try:
                rows = usage_store(USAGE_DB_URL, APP_TIMEZONE).all_users()
                st.dataframe([{"Email": r["email"], "Spent": _money(r["cost_usd"]), "Calls": r["calls"]}
                              for r in rows], hide_index=True, width="stretch")
            except Exception as exc:
                st.caption(f"Couldn't load usage: {exc}")
    theme_toggle()
    if st.button("🚪 Log out", width="stretch"):
        logout()
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

if mode == READ_MODE:
    with test_status_box:
        st.caption("Open **Test prep** to load the practice tests.")
    render_reader(ref_files)
    st.stop()

# ── Guards ────────────────────────────────────────────────────────────────────

if not api_key:
    st.info("👈 Enter your Claude API key in the sidebar to get started.")
    st.stop()

if not selected_files:
    st.warning("Select at least one document in the sidebar.")
    st.stop()

# The sidebar total is drawn once per page run; every action reruns the page, so it stays current.
budget = Budget(store, user_email, MONTHLY_LIMIT_USD, unlimited=is_admin)
render_usage(usage_box, budget)
if not is_admin and budget.remaining <= 0:
    st.warning(f"You've used your ${MONTHLY_LIMIT_USD:.2f} for this month — questions are paused "
               "until the 1st. Past-exam quizzes without AI checks still work.")
llm = LLM(api_key, MODEL, budget, workspace_id=_setting("ANTHROPIC_WORKSPACE_ID"))

# ── Load documents ────────────────────────────────────────────────────────────

with st.spinner("Loading documents…"):
    cache_input = tuple((str(p), _mtime_key(p)) for p in selected_files)
    doc_chunks = extract_chunks(cache_input, llm.transcribe)

active_chunks: list[dict] = []
for f in selected_files:
    active_chunks.extend(doc_chunks.get(f.name, []))

# ── Test prep mode ────────────────────────────────────────────────────────────

if mode == TEST_MODE:
    test_prep.render(
        llm=llm, test_files=test_files, cache_dir=CACHE_DIR,
        active_chunks=active_chunks, friendly=FRIENDLY_NAMES, retrieve=retrieve,
        build_context=build_context, status_box=test_status_box,
    )
    st.stop()
else:
    with test_status_box:
        st.caption("Open **Test prep** to load the practice tests.")

# ── Chat UI ───────────────────────────────────────────────────────────────────

st.title("🎾 Tennis Rules & Regulations Q&A")
st.caption(
    "Ask any question about tennis rules, officiating situations, "
    "or tournament regulations. Answers are drawn from the selected documents."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for mi, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Document sections consulted"):
                for si, (src, page) in enumerate(msg["sources"]):
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(f"**{FRIENDLY_NAMES.get(src, src)}** — page {page}")
                    c2.button("📖 Open", key=f"open_{mi}_{si}", width="stretch",
                              on_click=open_in_reader, args=(src, page))

prompt = st.chat_input("Ask about a rule or describe a match situation…")

if prompt:
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching documents…"):
        hits = retrieve(prompt, active_chunks)
        context = build_context(hits)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1][-18:]]

    user_message = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {prompt}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for piece in llm.stream_chat(SYSTEM, history, user_message):
                full_response += piece
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as exc:
            st.error(friendly_error(exc))
            st.stop()

    sources, seen = [], set()
    for c in hits:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append(key)
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
    st.rerun()

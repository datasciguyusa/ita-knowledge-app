"""Discover documents in the docs folder and build a practice-question bank.

* Every .pdf / .docx in the folder (and sub-folders) is picked up automatically.
* Files whose name mentions test / exam / quiz / questions are practice tests;
  everything else is reference material for answering questions.
* Word tests whose correct answers are highlighted or bolded are parsed directly.
  Any other test format is converted to questions by Claude once, then cached.
"""
import hashlib
import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx"}
TEST_NAME = re.compile(r"test|exam|quiz|question", re.I)
SECTIONS = ["General", "Chair Umpire", "Roving Umpire", "Referee"]     # ITA: no Chief Umpire section

_P = re.compile(r"<w:p[ >].*?</w:p>", re.S)
_T = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
_R = re.compile(r"<w:r[ >].*?</w:r>", re.S)
_BOLD = re.compile(r'<w:b/>|<w:b w:val="(?:1|true)"/>')
_QNUM = re.compile(r"^\s*\d+\.\s+")
_OPT_LETTER = re.compile(r"^\s*[A-Fa-f][.)]\s+")


# ── Discovery ────────────────────────────────────────────────────────────────

def scan_folder(folder: Path) -> dict:
    """Return {"reference": [...], "tests": [...], "skipped": [...]} of Paths."""
    out = {"reference": [], "tests": [], "skipped": []}
    if not folder.exists():
        return out
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.name.startswith(("~$", ".")):
            continue
        if f.suffix.lower() not in SUPPORTED:
            out["skipped"].append(f)
        elif TEST_NAME.search(f.stem):
            out["tests"].append(f)
        else:
            out["reference"].append(f)
    return out


def file_key(f: Path) -> str:
    """Content hash, so cached results survive copying the folder or a git clone."""
    return hashlib.md5(f.read_bytes()).hexdigest()


def is_garbled(text: str, n_pages: int) -> bool:
    """True when PDF text extraction failed (e.g. fonts without a text map)."""
    t = text.strip()
    if len(t) < 40 * max(n_pages, 1):
        return True
    letters = sum(len(w) for w in re.findall(r"[A-Za-z]{3,}", t))
    return letters / len(t) < 0.45


# ── Text extraction ──────────────────────────────────────────────────────────

def _unescape(s: str) -> str:
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
             .replace("&apos;", "'").replace("&amp;", "&"))


def _docx_paragraphs(docx: Path) -> list[dict]:
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    out = []
    for p in _P.findall(xml):
        text = _unescape("".join(_T.findall(p))).strip()
        if not text:
            continue
        marked_chars, marked_text = 0, []
        for r in _R.findall(p):
            rt = _unescape("".join(_T.findall(r)))
            rpr = re.search(r"<w:rPr>.*?</w:rPr>", r, re.S)
            is_marked = bool(rpr and ("w:highlight" in rpr.group(0) or _BOLD.search(rpr.group(0))))
            if is_marked:
                marked_chars += len(rt.strip())
            marked_text.append(f"⟦{rt}⟧" if is_marked and rt.strip() else rt)
        marked = marked_chars >= 0.6 * len(text.replace(" ", ""))
        nid = re.search(r'<w:numId w:val="(\d+)"', p)
        lvl = re.search(r'<w:ilvl w:val="(\d+)"', p)
        if len(text) < 80 and not nid and re.search(r"\bexam\b", text, re.I):
            key = ("hdr", text)
        elif _QNUM.match(text):
            key = ("qnum", text)
        elif _OPT_LETTER.match(text):
            key = ("letter",)
        else:
            key = (nid.group(1) if nid else "-", lvl.group(1) if lvl else "-")
        out.append({"text": text, "marked": marked, "key": key,
                    "annotated": "".join(marked_text).replace("⟧⟦", "").strip()})
    return out


def _docx_tables_text(docx: Path) -> str:
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    rows = []
    for tr in re.findall(r"<w:tr[ >].*?</w:tr>", xml, re.S):
        cells = [_unescape("".join(_T.findall(tc))).strip()
                 for tc in re.findall(r"<w:tc[ >].*?</w:tc>", tr, re.S)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


TRANSCRIBE_PROMPT = (
    "Transcribe all text in this PDF faithfully, including tables, labels and captions (describe the "
    "position of labelled items on diagrams in words, e.g. 'on the sleeve'). Start each page with a line "
    "'=== PAGE n ===' . Output plain text only."
)


def pdf_pages(f: Path, transcribe=None, cache_file: Path | None = None) -> list[tuple[str, str]]:
    pages = [(p.extract_text() or "").strip() for p in PdfReader(str(f)).pages]
    if not is_garbled("\n".join(pages), len(pages)) or transcribe is None:
        return [(str(n), t) for n, t in enumerate(pages, 1)]
    cache = _load_json(cache_file) if cache_file else {}
    key = file_key(f)
    text = cache.get(key)
    if not text:
        text = transcribe(f.read_bytes())
        if cache_file:
            cache[key] = text
            _save_json(cache_file, cache)
    parts = re.split(r"=== PAGE (\d+) ===", text)
    if len(parts) < 3:
        return [("1", text.strip())]
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def document_pages(f: Path, transcribe=None, cache_file: Path | None = None) -> list[tuple[str, str]]:
    """[(page label, text)] for a reference document."""
    if f.suffix.lower() == ".pdf":
        return pdf_pages(f, transcribe, cache_file)
    paras = [p["text"] for p in _docx_paragraphs(f)]
    tables = _docx_tables_text(f)
    blocks, cur = [], []
    for t in paras:                          # ~ 400-word "pages" for Word files
        cur.append(t)
        if sum(len(x.split()) for x in cur) > 400:
            blocks.append("\n".join(cur)); cur = []
    if cur:
        blocks.append("\n".join(cur))
    if tables:
        blocks.append(tables)
    return [(f"§{n}", b) for n, b in enumerate(blocks, 1)]


def test_text(f: Path, transcribe=None, cache_file: Path | None = None) -> str:
    """Plain text of a test, with ⟦…⟧ around highlighted/bold Word text."""
    if f.suffix.lower() == ".pdf":
        return "\n".join(t for _, t in pdf_pages(f, transcribe, cache_file))
    return "\n".join(p["annotated"] for p in _docx_paragraphs(f))


# ── Deterministic parser (Word tests with highlighted / bold answers) ────────

def _section_name(h: str) -> str:
    low = h.lower()
    if "everyone" in low or "all" in low.split():
        return "General"
    for role in ("Chair Umpire", "Roving Umpire", "Referee"):
        if role.lower() in low:
            return role
    return "General"


def parse_marked_docx(docx: Path) -> list[dict]:
    paras = _docx_paragraphs(docx)
    runs = []
    for p in paras:
        if runs and runs[-1][0]["key"] == p["key"]:
            runs[-1].append(p)
        else:
            runs.append([p])
    questions, section, i = [], "General", 0
    while i < len(runs):
        run = runs[i]
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if len(run) == 1 and nxt and len(nxt) >= 2:
            correct = [k for k, o in enumerate(nxt) if o["marked"]]
            questions.append({
                "question": _QNUM.sub("", run[0]["text"]).strip(),
                "options": [_OPT_LETTER.sub("", o["text"]).strip() for o in nxt],
                "answer": correct[0] if len(correct) == 1 else None,
                "section": section,
            })
            i += 2
            continue
        if run[0]["key"][0] == "hdr":
            section = _section_name(run[0]["text"])
        i += 1
    return questions


# ── AI extraction (any other format) ─────────────────────────────────────

EXTRACT_SYSTEM = (
    "You convert tennis officiating practice tests into structured questions. Extract EVERY question "
    "in the text, in order, exactly as written (fix only obvious typos). Text wrapped in ⟦ ⟧ was "
    "highlighted or bold in the original; question stems are often bold, so a mark only indicates the "
    "correct answer when it falls on exactly one option of a question. "
    "Set \"answer\" ONLY when the document itself shows the correct answer (a marked option, an answer "
    "written under the question, or an answer key); otherwise null. Never answer from your own knowledge. "
    "If a question has a written answer but no options (open-ended), make it multiple choice: the given "
    "answer plus 2-3 clearly wrong but plausible options, and set \"options_generated\": true. "
    "Skip a question that has neither options nor an answer. "
    "Section: \"Chair Umpire\" for chair-umpire (CU) tests or sections, \"Roving Umpire\", \"Referee\", "
    "otherwise \"General\" — judge from the file name and headings."
)


def extract_with_llm(f: Path, ask_json, transcribe=None, transcript_cache=None) -> list[dict]:
    text = test_text(f, transcribe, transcript_cache)[:60_000]
    prompt = (
        f"FILE NAME: {f.name}\n\nTEST TEXT:\n{text}\n\n"
        'Return JSON: {"questions":[{"question":str,"options":[str,...],'
        '"answer":<0-based index or null>,"options_generated":bool,"section":str}]}'
    )
    data = ask_json(EXTRACT_SYSTEM, prompt)
    out = []
    for q in data.get("questions", []):
        opts = [re.sub(r"^\s*[A-Fa-f][.)]\s+", "", str(o)).strip() for o in q.get("options") or []]
        opts = [o for o in opts if o]
        if not q.get("question") or len(opts) < 2:
            continue
        ans = q.get("answer")
        if not (isinstance(ans, int) and 0 <= ans < len(opts)):
            ans = None
        sec = q.get("section") if q.get("section") in SECTIONS else "General"
        out.append({"question": str(q["question"]).strip(), "options": opts, "answer": ans,
                    "section": sec, "options_generated": bool(q.get("options_generated"))})
    return out


# ── Bank with on-disk cache ──────────────────────────────────────────────────

def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(p: Path, data: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass                                     # read-only host: cache in memory only


def load_bank(tests: list[Path], cache_file: Path, ask_json=None, transcribe=None,
              transcript_cache: Path | None = None) -> tuple[list[dict], dict]:
    """Return (questions, status-by-file). Claude is only called for new/changed files."""
    cache = _load_json(cache_file)
    bank, status, changed = [], {}, False
    for f in tests:
        key = file_key(f)
        entry = cache.get(f.name)
        if not entry or entry.get("key") != key:
            qs, how = [], ""
            if f.suffix.lower() == ".docx":
                try:
                    qs = parse_marked_docx(f)
                    how = "answer key"
                except Exception:
                    qs = []
                if qs and any(q["answer"] is None for q in qs):
                    qs = []                      # not a fully marked key → let Claude read it
            if not qs and ask_json is not None:
                try:
                    qs = extract_with_llm(f, ask_json, transcribe, transcript_cache)
                    how = "read by AI"
                except Exception as exc:
                    try:
                        from llm import friendly_error
                        msg = friendly_error(exc)
                    except Exception:
                        msg = str(exc)
                    status[f.name] = f"could not read: {msg}"
                    continue
            if not qs:
                status[f.name] = "waiting for API key" if ask_json is None else "no questions found"
                continue
            entry = {"key": key, "how": how, "questions": qs}
            cache[f.name] = entry
            changed = True
        for q in entry["questions"]:
            bank.append(dict(q, source=f.stem, id=question_id(q)))
        keyed = sum(q["answer"] is not None for q in entry["questions"])
        status[f.name] = f"{len(entry['questions'])} questions, {keyed} with answers ({entry['how']})"
    if changed:
        cache = {k: v for k, v in cache.items() if k in {f.name for f in tests}}
        _save_json(cache_file, cache)
    return bank, status


def question_id(q: dict) -> str:
    return hashlib.md5((q["question"] + "|" + "|".join(q["options"])).encode()).hexdigest()[:16]

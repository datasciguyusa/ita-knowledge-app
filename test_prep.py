"""Test Prep mode: quiz the user with past-exam questions and/or new questions
generated from the selected rulebooks in the style of the past exams."""
import datetime
import hashlib
import json
import random
from functools import partial
import re
from pathlib import Path

import streamlit as st

import exam_bank

# ITA quizzes: tick divisions, match formats and roles. Past-test questions marked "General" count for every role.
TRACKS = ["Chair Umpire", "Roving Umpire", "Referee"]
DIVISIONS = {
    "Division I":   "NCAA Division I",
    "Division II":  "NCAA Division II",
    "Division III": "NCAA Division III",
    "NAIA":         "NAIA",
    "JUCO":         "junior college (JUCO: NJCAA / 3C2A)",
}
FORMATS = {
    "Dual match": "a college dual match (team vs team: the doubles point, singles order and lineups, "
                  "clinching and play-out, abandonment, coaching from the bench, team conduct)",
    "Tournament": "a college individual tournament (draws and seeding, sign-in, order of play, rest "
                  "periods, match formats and scoring, defaults and scheduling)",
}
SOURCES = {
    "gen":   "New questions generated from the documents",
    "past":  "Past exam questions",
    "mixed": "Mix of both",
}
BATCH = 5                     # generated questions per model call
CHUNKS_PER_BATCH = 7
LETTERS = "ABCDEFGH"

TRACK_FOCUS = {
    "Chair Umpire": "on-court chair umpire procedures: calls and overrules, scoring (including no-ad), lets, "
                    "time rules, coaching, medical and toilet breaks, warm-up, and the ITA Point Penalty System",
    "Roving Umpire": "roving umpire procedures: covering several courts, positioning, responding to "
                     "player appeals, overrules, staying on a court, time and conduct violations under the ITA "
                     "Point Penalty System, and no-downtime roving",
    "Referee":      "referee duties: lineups and order of play, starting and suspending play, weather and "
                    "court changes, defaults, abandonment, conduct reports, eligibility and scheduling",
}
# Which documents to draw topics from for each track. Matched against the file name ignoring
# case, spaces, hyphens and punctuation, so "rules of tennis" matches "Rules of Tennis.pdf" and
# "2026-rules-of-tennis-english.pdf" alike.
TRACK_DOCS = {
    "Chair Umpire": ["officials manual", "ita rulebook", "elc protocol", "rules of tennis"],
    "Roving Umpire": ["officials manual", "ita rulebook"],
    "Referee":      ["ita rulebook", "officials manual"],
}

# How alike two question stems must be (share of distinct key words) to count as the same question.
SIMILAR_PAST = 0.75          # past-exam questions recycled across years
SIMILAR_GEN = 0.6            # generated questions (often reworded)
AVOID_LIMIT = 60             # most recent questions sent to the generator as "already used"
REVIEW_SHARE = 0.2           # share of a quiz given to questions the user got wrong before
CURRENT_YEAR = datetime.date.today().year

# Administrative topics that aren't officiating knowledge — kept out of Test Prep entirely
# (source pages, generated questions and past-test questions). Edit this list to taste.
ADMIN_TOPICS = [
    r"entry fees?", r"fees?", r"prize money", r"payments?", r"paid", r"refunds?", r"deposits?",
    r"hospitality", r"hotels?", r"accommodation", r"per diem", r"travel (?:costs?|expenses|allowance)",
    r"sanctioning fees?", r"ipin", r"membership", r"registration", r"insurance", r"tax(?:es)?",
    r"points? tables?", r"currency", r"usd", r"eur", r"\$\s?\d",
    r"broadcast(?:ing)? rights", r"media rights", r"tournament calendar", r"application deadlines?",
]
_ADMIN = re.compile(r"\b(?:" + "|".join(ADMIN_TOPICS) + r")\b", re.I)


def is_admin(q: dict) -> bool:
    return bool(_ADMIN.search(" ".join([q["question"], *q["options"]])))


GEN_SYSTEM = (
    "You write practice questions for ITA college tennis officials, in the style of the ITF / USTA "
    "officiating certification exams. When the ITA Rulebook and the Rules of Tennis differ, the ITA "
    "Rulebook is correct for college matches. "
    "Match the style of the EXAMPLE questions from real past exams: realistic on-court or tournament "
    "scenarios, True/False, Yes/No, or 3-4 option multiple choice, exactly one correct answer, and "
    "plausible distractors. Every question must be answerable from the DOCUMENT EXCERPTS alone - "
    "never from the examples and never from outside knowledge. Do not copy or lightly reword the examples. "
    "Prefer specific, testable rules (time limits, procedures, penalties, scoring, dress and equipment, "
    "official duties). Skip excerpts that are tables of contents, indexes, or contain nothing testable. "
    "NEVER write administrative questions: entry fees, prize money, payments, hospitality, accommodation, "
    "travel, sanctioning, player registration/IPIN, ranking points, insurance or tournament applications."
)

RESOLVE_SYSTEM = (
    "You are an experienced ITF-certified tennis official. A practice-test question has no official "
    "answer key. Using ONLY the document excerpts, choose the correct option. Return JSON "
    '{"answer": <0-based index or null if the excerpts do not settle it>, "explanation": "2-3 sentences '
    'citing the rule", "citation": "document, page"}.'
)

EXPLAIN_SYSTEM = (
    "You are an experienced ITF-certified tennis official helping someone study for a certification exam. "
    "Explain in 2-4 sentences why the answer-key answer is correct, citing the document and page from the "
    "excerpts. The answer key comes from an earlier year's exam; if the excerpts clearly show the rule has "
    "changed or the key looks wrong, say so plainly and give the answer the current documents support. "
    "If the excerpts don't cover it, say that rather than guessing."
)


# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_bank(paths: tuple, stamps: tuple, cache_dir: str, _ask_json, _transcribe):
    return exam_bank.load_bank(
        [Path(p) for p in paths], Path(cache_dir) / "question_bank.json",
        _ask_json, _transcribe, Path(cache_dir) / "transcripts.json",
    )


def get_bank(test_files, cache_dir, ask_json, transcribe):
    stamps = tuple(f"{f.stat().st_size}:{f.stat().st_mtime}" for f in test_files)
    return _cached_bank(tuple(str(f) for f in test_files), stamps, str(cache_dir), ask_json, transcribe)


# Answers worked out from the rulebooks for questions without an answer key (shared by all users).
def _answers_file() -> Path:
    return Path(st.session_state["tp_cache_dir"]) / "rulebook_answers.json"


def resolve_answer(q, ask_json, retrieve, build_context) -> dict:
    store = exam_bank._load_json(_answers_file())
    if q["id"] in store:
        return store[q["id"]]
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    context = build_context(retrieve(q["question"] + " " + " ".join(q["options"]), st.session_state["tp_chunks"]))
    data = ask_json(RESOLVE_SYSTEM, f"Document excerpts:\n\n{context}\n\n---\n\nQuestion:\n{q['question']}\n{opts}")
    ans = data.get("answer")
    res = {
        "answer": ans if isinstance(ans, int) and 0 <= ans < len(q["options"]) else None,
        "explanation": str(data.get("explanation", "")).strip(),
        "citation": str(data.get("citation", "")).strip(),
    }
    store[q["id"]] = res
    exam_bank._save_json(_answers_file(), store)
    return res


def _is_fixed_order(opts: list[str]) -> bool:
    low = [o.lower().strip(" .") for o in opts]
    return (any("above" in o for o in low)
            or set(low) <= {"true", "false", "yes", "no"})


def _prepare(q: dict) -> dict:
    """Copy a question and shuffle its options (unless order matters)."""
    q = dict(q)
    opts = list(q["options"])
    q["options_orig"] = opts
    if not _is_fixed_order(opts):
        order = list(range(len(opts)))
        random.shuffle(order)
        q["options"] = [opts[k] for k in order]
        q["order"] = order                      # display index -> original index
        if q["answer"] is not None:
            q["answer"] = order.index(q["answer"])
    return q


def _usable_chunk(c: dict) -> bool:
    t = c["text"]
    if len(t) < 400:
        return False
    dots = t.count("....") + t.count(". . .")
    if dots >= 5:          # skip table-of-contents style pages
        return False
    return len(_ADMIN.findall(t)) < 3      # skip pages that are mostly admin (fees, prize money…)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


_STOP = set("the a an of to in on is be by for and or at it if as with that this which what when who "
            "does do may must can should would will are was were has have his her their they there "
            "not no yes true false following".split())


def _words(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP and len(w) > 1}


def _similar(a: str, b: str, thresh: float) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return _norm(a) == _norm(b)
    return len(wa & wb) / len(wa | wb) >= thresh


def _seen_key(q: dict) -> str:
    """Same wording in two different years' tests counts as one question."""
    return _norm(q["question"])[:200]


def _track_pool(chunks: list[dict], track: str) -> list[dict]:
    wanted = [_norm(d) for d in TRACK_DOCS.get(track, [])]
    usable = [c for c in chunks if _usable_chunk(c)] or chunks
    pool = [c for c in usable if any(w in _norm(c["source"]) for w in wanted)]
    return pool if len(pool) >= 2 * CHUNKS_PER_BATCH else usable


class _SeedRotation:
    """Hands out document pages without repeats until every page of a track has been used."""

    def __init__(self, chunks, tracks):
        self.pools = {t: _track_pool(chunks, t) for t in tracks}
        self.queues = {t: [] for t in tracks}

    def take(self, track: str, k: int) -> list[dict]:
        q = self.queues[track]
        if len(q) < k:
            q.extend(random.sample(self.pools[track], len(self.pools[track])))
        out, q[:] = q[:k], q[k:]
        return out


def generate_questions(n, tracks, bank, chunks, llm, friendly, avoid=(), formats=(), divisions=()) -> list[dict]:
    """avoid: question stems already in this quiz or recently seen — the model is told not to
    repeat them, and anything too close to them is dropped."""
    out, attempts = [], 0
    avoid = list(avoid)
    keyed = [q for q in bank if q["answer"] is not None]
    examples_pool = [q for q in keyed if q["section"] in tracks] or keyed
    seeds_for = _SeedRotation(chunks, tracks)
    order = random.sample(tracks, len(tracks))          # spread batches evenly across sections
    while len(out) < n and attempts < (n // BATCH + 4):
        want = min(BATCH, n - len(out))
        track = order[attempts % len(order)]
        attempts += 1
        examples = random.sample(examples_pool, min(4, len(examples_pool))) if examples_pool else []
        ex_txt = "\n\n".join(
            f"Q: {e['question']}\n" + "\n".join(f"  {LETTERS[i]}. {o}" for i, o in enumerate(e["options"]))
            + f"\n  Answer: {LETTERS[e['answer']]}"
            for e in examples
        )
        seeds = seeds_for.take(track, CHUNKS_PER_BATCH)
        doc_txt = "\n\n---\n\n".join(
            f"[{friendly.get(c['source'], c['source'])} — page {c['page']}]\n{c['text']}" for c in seeds
        )[:45_000]
        prompt = (
            f"EXAMPLE QUESTIONS (style only):\n\n{ex_txt}\n\n=====\n\n"
            f"DOCUMENT EXCERPTS:\n\n{doc_txt}\n\n=====\n\n"
            + _avoid_block(avoid + [q["question"] for q in out]) +
            f"Write exactly {want} new questions for the {track} exam, focused on {TRACK_FOCUS[track]}. "
            + _context_block(formats, divisions) +
            "Use different excerpts for different questions where possible. Return JSON:\n"
            '{"questions":[{"question":str,"options":[str,...],"answer":<0-based index of correct option>,'
            '"explanation":"1-3 sentences on why, quoting or paraphrasing the rule","source":"document name",'
            '"page":<page number>,"division":str,"format":str}]}'
        )
        try:
            data = llm.ask_json(GEN_SYSTEM, prompt, kind="generate")
        except Exception as exc:
            from llm import friendly_error
            st.warning(f"A batch of questions failed to generate: {friendly_error(exc)}")
            continue
        for g in data.get("questions", []):
            opts = [str(o).strip() for o in g.get("options", []) if str(o).strip()]
            ans = g.get("answer")
            if not g.get("question") or len(opts) < 2 or not isinstance(ans, int) or not 0 <= ans < len(opts):
                continue
            cite = g.get("source", "")
            if g.get("page"):
                cite += f", page {g['page']}"
            if is_admin({"question": g["question"], "options": opts}):
                continue
            stem = g["question"].strip()
            if any(_similar(stem, a, SIMILAR_GEN) for a in avoid + [q["question"] for q in out]):
                continue
            fmt = _pick_label(g.get("format"), formats)
            division = _pick_label(g.get("division"), divisions)
            out.append({
                "question": g["question"].strip(), "options": opts, "answer": ans,
                "section": track, "format": fmt, "division": division, "origin": "generated",
                "explanation": g.get("explanation", "").strip(), "citation": cite.strip(", "),
            })
    return out[:n]


def _context_block(formats, divisions) -> str:
    """Tell the model which divisions / match formats the quiz covers; each question gets one of each."""
    if not formats and not divisions:
        return ""
    parts = []
    if divisions:
        parts.append("DIVISIONS: " + "; ".join(f"{d} = {DIVISIONS[d]}" for d in divisions) + ".")
    if formats:
        parts.append("MATCH FORMATS: " + "; ".join(f"{f} = {FORMATS[f]}" for f in formats) + ".")
    return (" ".join(parts) + " Spread the questions across these. Set each question in ONE division and ONE "
            "match format from these lists, and return their short labels exactly as written before the '=' in "
            "\"division\" and \"format\". When a rule differs by division or format, name the division or format "
            "in the question and use the rule for that one, never a rule written only for another division or "
            "format. ")


def _pick_label(value, allowed) -> str | None:
    """The model's division/format label if it's one of the allowed ones, else a random allowed one."""
    allowed = list(allowed)
    if not allowed:
        return None
    for a in allowed:
        if str(value or "").strip().lower() == a.lower():
            return a
    return random.choice(allowed)


def _avoid_block(stems: list[str]) -> str:
    if not stems:
        return ""
    recent = stems[-AVOID_LIMIT:]
    lines = "\n".join(f"- {t[:160]}" for t in recent)
    return ("ALREADY USED — do not repeat, reword, or test the same specific rule as any of these; "
            f"pick different rules and different excerpts:\n{lines}\n\n=====\n\n")


def explain_past(q, llm, retrieve, build_context) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    query = q["question"] + " " + q["options"][q["answer"]]
    context = build_context(retrieve(query, st.session_state["tp_chunks"]))
    prompt = (f"Document excerpts:\n\n{context}\n\n---\n\nExam question:\n{q['question']}\n{opts}\n\n"
              f"Answer key: {LETTERS[q['answer']]}. {q['options'][q['answer']]}")
    return llm.ask_text(EXPLAIN_SYSTEM, prompt, kind="explain")


# ── Screens ──────────────────────────────────────────────────────────────────

def _reset():
    keep = ("tp_chunks", "tp_cache_dir", "tp_seen_gen", "tp_progress", "tp_prog")
    for k in [k for k in st.session_state if k.startswith("tp_") and k not in keep]:
        del st.session_state[k]


def _start(quiz):
    _reset()
    st.session_state.tp_quiz = quiz
    st.session_state.tp_idx = 0
    st.session_state.tp_answers = {}


def _qkey(q: dict) -> str:
    return hashlib.md5(_seen_key(q).encode("utf-8")).hexdigest()


class _Progress:
    """The user's seen / missed questions: saved in the app database (across visits) when it's
    available, otherwise kept for this browser session only. Never lets a database error stop a quiz."""

    def __init__(self, history):
        self.db, self.rows = history, {}
        if history is not None:
            try:
                self.rows = history.load()
            except Exception:
                self.db = None
        if self.db is None:
            self.rows = st.session_state.setdefault("tp_progress", {})

    @property
    def saved(self) -> bool:
        return self.db is not None

    def _call(self, name, *args):
        if self.db is not None:
            try:
                getattr(self.db, name)(*args)
            except Exception:
                pass

    def seen(self, key: str) -> bool:
        return bool(self.rows.get(key, {}).get("in_round"))

    def needs_review(self, key: str) -> bool:
        return self.rows.get(key, {}).get("last_correct") == 0

    def missed_generated(self) -> list[dict]:
        out = []
        for r in self.rows.values():
            if r.get("last_correct") == 0 and r.get("qjson"):
                try:
                    out.append(json.loads(r["qjson"]))
                except Exception:
                    pass
        return out

    def mark_seen(self, keys: list[str]) -> None:
        for k in keys:
            r = self.rows.setdefault(k, {})
            r["in_round"] = 1
            r["times_seen"] = r.get("times_seen", 0) + 1
        self._call("mark_seen", list(keys))

    def end_round(self, keys: list[str]) -> None:
        for k in keys:
            if k in self.rows:
                self.rows[k]["in_round"] = 0
        self._call("end_round", list(keys))

    def record(self, key: str, correct: bool, qjson: str | None = None) -> None:
        r = self.rows.setdefault(key, {})
        r["last_correct"] = 1 if correct else 0
        if qjson:
            r["qjson"] = qjson
        self._call("record_answer", key, correct, qjson)

    def reset(self) -> None:
        self.rows.clear()
        self._call("reset")


def _prog() -> _Progress:
    return st.session_state["tp_prog"]


def _test_year(source: str) -> int | None:
    """Year of a past test from its file name: '2017 Open Book…', 'L2 Slo 9.22', 'L2-LUX-21'."""
    m = re.search(r"(?<!\d)(20[0-4]\d)(?!\d)", source)
    if m:
        return int(m.group(1))
    m = re.search(r"(?<!\d)\d{1,2}\.(\d{2})(?!\d)", source) or re.search(r"-(\d{2})(?:\D|$)", source)
    return 2000 + int(m.group(1)) if m else None


def _pick_review(pool: list[dict], source: str, tracks: list[str], k: int, formats=(), divisions=()) -> list[dict]:
    """Questions the user got wrong last time they saw them."""
    if k <= 0:
        return []
    prog, cands = _prog(), []
    if source != "gen":
        cands += [dict(q, origin="past") for q in pool if prog.needs_review(_qkey(q))]
    if source != "past":
        cands += [dict(g, origin="generated") for g in prog.missed_generated()
                  if g.get("section") in tracks and g.get("format") in (*formats, None)
                  and g.get("division") in (*divisions, None)]
    random.shuffle(cands)
    out, keys = [], set()
    for q in cands:
        if _qkey(q) not in keys:
            keys.add(_qkey(q))
            out.append(dict(q, review=True))
    return out[:k]


def _review_count(pool, source, tracks, formats=(), divisions=()) -> int:
    return len(_pick_review(pool, source, tracks, 10_000, formats, divisions))


def _pick_past(pool: list[dict], k: int) -> list[dict]:
    """Unseen questions first; once the pool is used up, start a fresh round. Near-identical
    questions (same question from different years' tests) never land in the same quiz."""
    prog = _prog()
    fresh = [q for q in pool if not prog.seen(_qkey(q))]
    stale = [q for q in pool if prog.seen(_qkey(q))]
    random.shuffle(fresh)
    random.shuffle(stale)
    picked, skipped = [], []
    for q in fresh + stale:
        if len(picked) >= k:
            break
        if any(_similar(q["question"], p["question"], SIMILAR_PAST) for p in picked):
            skipped.append(q)
        else:
            picked.append(q)
    picked += skipped[:k - len(picked)]             # tiny pools: allow look-alikes rather than fall short
    if len(fresh) <= k:                             # every question in this pool has now been shown
        prog.end_round(list({_qkey(q) for q in pool}))
    prog.mark_seen(list({_qkey(q) for q in picked}))
    return picked


def _remember_generated(quiz: list[dict]) -> None:
    gen = st.session_state.setdefault("tp_seen_gen", [])
    gen.extend(q["question"] for q in quiz if q.get("origin") == "generated")
    del gen[:-AVOID_LIMIT]


def _progress_footer():
    prog = _prog()
    where = "saved to your account" if prog.saved else "kept for this browser session only"
    with st.expander("My progress"):
        seen = sum(1 for r in prog.rows.values() if r.get("times_seen"))
        wrong = sum(1 for r in prog.rows.values() if r.get("last_correct") == 0)
        st.caption(f"{seen} questions seen · {wrong} to review · {where}.")
        if st.button("Reset my progress", help="Forget which questions you've seen and missed."):
            prog.reset()
            st.rerun()


def _check_group(label: str, options: list[str], key: str, default_off=()) -> list[str]:
    """A row of checkboxes with Select all / Unselect all. Choices stay put between quizzes."""
    keys = {o: f"tpsel_{key}_{_norm(o)}" for o in options}     # not "tp_…", so New quiz keeps them
    for o, k in keys.items():
        if k not in st.session_state:
            st.session_state[k] = o not in default_off

    def set_all(value: bool):
        for k in keys.values():
            st.session_state[k] = value

    st.markdown(f"**{label}**")
    b1, b2 = st.columns(2)
    b1.button("Select all", key=f"tpsel_{key}_all", on_click=set_all, args=(True,), type="tertiary")
    b2.button("Unselect all", key=f"tpsel_{key}_none", on_click=set_all, args=(False,), type="tertiary")
    for o in options:                                   # stacked: one option per line
        st.checkbox(o, key=keys[o])
    return [o for o in options if st.session_state[keys[o]]]


def _setup_screen(bank, chunks, llm, friendly):
    n_admin = sum(is_admin(q) for q in bank)
    bank = [q for q in bank if not is_admin(q)]
    tests = sorted({q["source"] for q in bank})
    st.caption("Practice for the officiating exam. Questions are modelled on the past exams in your "
               "documents folder and grounded in the selected rulebooks.")
    if not bank:
        st.info("No past-exam .docx files with marked answers were found, so questions will be generated "
                "from the rulebooks without style examples.")

    c1, c2 = st.columns(2)
    with c1:
        source = st.radio("Questions", list(SOURCES), format_func=SOURCES.get,
                          index=0 if bank else 0, disabled=not bank)
        if not bank:
            source = "gen"
    with c2:
        n = st.select_slider("Number of questions", [5, 10, 15, 20], value=10)

    g1, g2, g3 = st.columns(3)                      # one column per choice
    with g1:
        divisions = _check_group("Division", list(DIVISIONS), "div")
    with g2:
        formats = _check_group("Match format", list(FORMATS), "fmt", default_off=("Tournament",))
    with g3:
        roles = _check_group("Role", TRACKS, "role")
    missing = [w for w, sel in (("division", divisions), ("match format", formats), ("role", roles)) if not sel]
    if missing:
        st.warning("Select at least one " + ", one ".join(missing) + ".")
    tracks = roles + ["General"] if roles else []   # past questions for these roles + ones for everyone
    chosen_tests = tests
    if bank and source != "gen":
        chosen_tests = st.multiselect("Past tests to draw from", tests, default=tests)
        pool = [q for q in bank if q["source"] in chosen_tests]
        counts = {t: sum(q["section"] == t for q in pool) for t in ["General", *TRACKS]}
        no_key = sum(q["answer"] is None for q in pool)
        prog = _prog()
        unseen = len({_qkey(q) for q in pool if q["section"] in tracks and not prog.seen(_qkey(q))})
        st.caption("Past-test questions available: " + " · ".join(f"{t} {c}" for t, c in counts.items())
                   + f"  ·  {unseen} you haven't seen yet"
                   + (f"  ·  {no_key} have no answer key and are checked against the rulebooks"
                      if no_key else "")
                   + (f"  ·  {n_admin} administrative questions excluded" if n_admin else ""))

    past_pool = [q for q in bank if q["section"] in tracks and q["source"] in chosen_tests]
    n_missed = _review_count(past_pool, source, tracks, formats, divisions)
    include_review = False
    if n_missed:
        include_review = st.checkbox(
            f"Include questions I got wrong before ({n_missed} to review)", value=True,
            help=f"Up to {round(REVIEW_SHARE * 100)}% of the quiz. A question leaves this list once you answer it correctly.")
    _progress_footer()

    if not st.button("▶️ Start quiz", type="primary", disabled=bool(missing)):
        return

    n_review = round(n * REVIEW_SHARE) if include_review else 0
    quiz = _pick_review(past_pool, source, tracks, n_review, formats, divisions)
    review_keys = {_qkey(q) for q in quiz}
    _prog().mark_seen([_qkey(q) for q in quiz if q["origin"] == "past"])
    past_pool = [q for q in past_pool if _qkey(q) not in review_keys]
    n_left = n - len(quiz)
    n_past = {"past": n_left, "gen": 0, "mixed": n_left // 2}[source]
    n_past = min(n_past, len(past_pool))
    n_gen = n_left - n_past

    quiz += [dict(q, origin="past") for q in _pick_past(past_pool, n_past)]
    if n_gen:
        if not chunks:
            st.error("Select at least one PDF in the sidebar to generate questions.")
            return
        avoid = st.session_state.get("tp_seen_gen", []) + [q["question"] for q in quiz]
        with st.spinner(f"Writing {n_gen} questions from the rulebooks…"):
            quiz += generate_questions(n_gen, roles, bank, chunks, llm, friendly, avoid=avoid,
                                       formats=formats, divisions=divisions)
        _remember_generated(quiz)
    if not quiz:
        st.error("Couldn't build a quiz — try again or choose past exam questions.")
        return
    random.shuffle(quiz)
    _start([_prepare(q) for q in quiz])
    st.rerun()


def _question_screen(llm, retrieve, build_context):
    quiz, i = st.session_state.tp_quiz, st.session_state.tp_idx
    q = quiz[i]
    answered = i in st.session_state.tp_answers

    st.progress(i / len(quiz), text=f"Question {i + 1} of {len(quiz)}")
    if q["origin"] == "past":
        year = _test_year(q.get("source", ""))
        origin = f"Past exam ({year})" if year else "Past exam"
    else:
        origin = "Generated from rulebooks"
    st.caption(f"{q['section']} · " + "".join(f"{q[k]} · " for k in ("division", "format") if q.get(k))
               + origin + (" · 🔁 you missed this one before" if q.get("review") else ""))
    st.markdown(f"#### {q['question']}")

    choice = st.radio("Your answer", list(range(len(q["options"]))), index=None,
                      format_func=lambda k: f"{LETTERS[k]}. {q['options'][k]}",
                      key=f"tp_choice_{i}", disabled=answered, label_visibility="collapsed")

    if not answered:
        if st.button("Submit answer", type="primary", disabled=choice is None):
            if q["answer"] is None and not q.get("resolved"):
                with st.spinner("No answer key for this one — checking the rulebooks…"):
                    try:
                        res = resolve_answer(dict(q, options=q["options_orig"]) if "options_orig" in q else q,
                                             partial(llm.ask_json, kind="answer-check"), retrieve, build_context)
                    except Exception as exc:
                        from llm import friendly_error
                        res = {"answer": None, "explanation": f"Lookup failed: {friendly_error(exc)}", "citation": ""}
                q["resolved"] = True
                if res["answer"] is not None:
                    order = q.get("order", list(range(len(q["options"]))))
                    q["answer"] = order.index(res["answer"])
                q["explanation"], q["citation"] = res["explanation"], res["citation"]
                q["answer_from"] = "rulebooks"
            st.session_state.tp_answers[i] = choice
            _record_answer(q, choice)
            st.rerun()
        return

    picked = st.session_state.tp_answers[i]
    if q["answer"] is None:
        st.warning("This question has no answer key and the rulebooks didn't settle it, so it isn't scored.")
        if q.get("explanation"):
            st.info(q["explanation"])
        _next_button(i, len(quiz))
        return
    right = q["options"][q["answer"]]
    if q.get("answer_from") == "rulebooks":
        st.caption("⚠️ No official answer key — this answer was worked out from the rulebooks.")
    if picked == q["answer"]:
        st.success(f"✅ Correct — **{LETTERS[q['answer']]}. {right}**")
    else:
        st.error(f"❌ You chose {LETTERS[picked]}. The answer is **{LETTERS[q['answer']]}. {right}**")

    year = _test_year(q.get("source", "")) if q["origin"] == "past" else None
    if year and year < CURRENT_YEAR and q.get("answer_from") != "rulebooks":
        st.caption(f"📅 This answer key is from the {year} test. Rules can change between years"
                   + (" — check it with “Explain from the rulebooks” below." if not q.get("explanation") else "."))
    if q.get("explanation"):
        st.info(q["explanation"] + (f"\n\n📚 *{q['citation']}*" if q.get("citation") else ""))
    elif q["origin"] == "past":
        st.caption(f"Answer key: {q.get('source', 'past exam')}"
                   + (" (wrong options were added to an open-ended question)" if q.get("options_generated") else ""))
        if st.button("📖 Explain from the rulebooks"):
            with st.spinner("Looking it up…"):
                try:
                    q["explanation"] = explain_past(q, llm, retrieve, build_context)
                except Exception as exc:
                    from llm import friendly_error
                    st.warning(f"Couldn't get an explanation: {friendly_error(exc)}")
            st.rerun()

    _next_button(i, len(quiz))


def _record_answer(q: dict, choice: int) -> None:
    if q["answer"] is None:
        return
    correct = choice == q["answer"]
    qjson = None
    if q["origin"] == "generated" and not correct:    # keep it so it can come back for review
        order = q.get("order")
        qjson = json.dumps({
            "question": q["question"], "options": q.get("options_orig", q["options"]),
            "answer": order[q["answer"]] if order else q["answer"], "section": q["section"],
            "explanation": q.get("explanation", ""), "citation": q.get("citation", ""),
            "format": q.get("format"), "division": q.get("division"),
        }, ensure_ascii=False)
    _prog().record(_qkey(q), correct, qjson)


def _next_button(i, n):
    if st.button("See results ➤" if i == n - 1 else "Next question ➤", type="primary"):
        st.session_state.tp_idx += 1
        st.rerun()


def _results_screen():
    all_q, all_ans = st.session_state.tp_quiz, st.session_state.tp_answers
    scored = [(q, all_ans.get(k)) for k, q in enumerate(all_q) if q["answer"] is not None]
    quiz = [q for q, _ in scored]
    ans = {i: a for i, (_, a) in enumerate(scored)}
    score = sum(ans.get(i) == q["answer"] for i, q in enumerate(quiz))
    pct = round(100 * score / max(len(quiz), 1))
    st.subheader(f"Score: {score} / {len(quiz)} ({pct}%)")
    if len(quiz) < len(all_q):
        st.caption(f"{len(all_q) - len(quiz)} question(s) couldn't be scored (no answer key).")

    by = {}
    for i, q in enumerate(quiz):
        r = by.setdefault(q["section"], [0, 0])
        r[1] += 1
        r[0] += ans.get(i) == q["answer"]
    st.caption(" · ".join(f"{s}: {c}/{t}" for s, (c, t) in by.items()))

    missed = [q for i, q in enumerate(quiz) if ans.get(i) != q["answer"]]
    if missed:
        st.markdown("**Review the ones you missed**")
        for i, q in enumerate(quiz):
            if ans.get(i) == q["answer"]:
                continue
            with st.expander(q["question"][:110] + ("…" if len(q["question"]) > 110 else "")):
                st.markdown(q["question"])
                st.markdown(f"- Your answer: {q['options'][ans[i]]}\n- Correct: **{q['options'][q['answer']]}**")
                if q.get("explanation"):
                    st.markdown(q["explanation"])
                if q.get("citation"):
                    st.caption(q["citation"])

    c1, c2 = st.columns(2)
    if missed and c1.button("🔁 Retry missed questions", width="stretch"):
        _start([_prepare(q) for q in missed])
        st.rerun()
    if c2.button("🆕 New quiz", width="stretch", type="primary"):
        _reset()
        st.rerun()


def render(llm, test_files, cache_dir, active_chunks, friendly, retrieve, build_context, status_box,
           history=None):
    st.title("📝 Test Prep")
    st.session_state["tp_prog"] = _Progress(history)
    st.session_state["tp_chunks"] = active_chunks
    st.session_state["tp_cache_dir"] = str(cache_dir)
    with st.spinner("Reading the practice tests (first time for a new file takes a few seconds)…"):
        bank, status = get_bank(test_files, cache_dir, partial(llm.ask_json, kind="read-test"), llm.transcribe)
    with status_box:
        for name, msg in status.items():
            st.caption(f"**{name}** — {msg}")

    if not st.session_state.get("tp_quiz"):
        _setup_screen(bank, active_chunks, llm, friendly)
        return

    top = st.columns([4, 1])
    if top[1].button("✖ End quiz", width="stretch"):
        _reset()
        st.rerun()

    if st.session_state.tp_idx >= len(st.session_state.tp_quiz):
        _results_screen()
    else:
        _question_screen(llm, retrieve, build_context)

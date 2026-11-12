"""Test Prep mode: quiz the user with past-exam questions and/or new questions
generated from the selected rulebooks in the style of the past exams."""
import json
import random
from functools import partial
import re
from pathlib import Path

import streamlit as st

import exam_bank

TRACKS = ["General", "Chair Umpire", "Chief Umpire", "Referee"]
SOURCES = {
    "gen":   "New questions generated from the documents",
    "past":  "Past exam questions",
    "mixed": "Mix of both",
}
BATCH = 5                     # generated questions per model call
CHUNKS_PER_BATCH = 7
LETTERS = "ABCDEFGH"

TRACK_FOCUS = {
    "General":      "core Rules of Tennis, scoring, time rules, code of conduct and dress/equipment rules",
    "Chair Umpire": "on-court chair umpire procedures, announcements, code/time violations, suspensions, warm-ups and ball changes",
    "Chief Umpire": "chief umpire duties, officiating requirements and staffing, assignments, line umpire placement",
    "Referee":      "referee/supervisor duties, tournament regulations, draws, entries, sign-in, scheduling and rest periods",
}
# Which documents to draw topics from for each track (substring of the PDF file name).
TRACK_DOCS = {
    "General":      ["rules-of-tennis", "code-of-conduct", "wtt-regulations", "duties"],
    "Chair Umpire": ["duties", "rules-of-tennis", "wtt-regulations"],
    "Chief Umpire": ["duties", "code-of-conduct"],
    "Referee":      ["wtt-regulations", "duties"],
}

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
    "You write practice questions for the ITF / USTA tennis officiating certification exams. "
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


def _seed_chunks(chunks: list[dict], tracks: list[str], k: int) -> list[dict]:
    wanted = {d for t in tracks for d in TRACK_DOCS.get(t, [])}
    pool = [c for c in chunks if _usable_chunk(c)
            and any(w in c["source"].lower() for w in wanted)]
    if len(pool) < k:
        pool = [c for c in chunks if _usable_chunk(c)] or chunks
    return random.sample(pool, min(k, len(pool)))


def generate_questions(n, tracks, bank, chunks, llm, friendly) -> list[dict]:
    out, attempts = [], 0
    keyed = [q for q in bank if q["answer"] is not None]
    examples_pool = [q for q in keyed if q["section"] in tracks] or keyed
    while len(out) < n and attempts < (n // BATCH + 3):
        attempts += 1
        want = min(BATCH, n - len(out))
        track = random.choice(tracks)
        examples = random.sample(examples_pool, min(4, len(examples_pool))) if examples_pool else []
        ex_txt = "\n\n".join(
            f"Q: {e['question']}\n" + "\n".join(f"  {LETTERS[i]}. {o}" for i, o in enumerate(e["options"]))
            + f"\n  Answer: {LETTERS[e['answer']]}"
            for e in examples
        )
        seeds = _seed_chunks(chunks, [track], CHUNKS_PER_BATCH)
        doc_txt = "\n\n---\n\n".join(
            f"[{friendly.get(c['source'], c['source'])} — page {c['page']}]\n{c['text']}" for c in seeds
        )[:45_000]
        prompt = (
            f"EXAMPLE QUESTIONS (style only):\n\n{ex_txt}\n\n=====\n\n"
            f"DOCUMENT EXCERPTS:\n\n{doc_txt}\n\n=====\n\n"
            f"Write exactly {want} new questions for the {track} exam, focused on {TRACK_FOCUS[track]}. "
            "Use different excerpts for different questions where possible. Return JSON:\n"
            '{"questions":[{"question":str,"options":[str,...],"answer":<0-based index of correct option>,'
            '"explanation":"1-3 sentences on why, quoting or paraphrasing the rule","source":"document name",'
            '"page":<page number>}]}'
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
            out.append({
                "question": g["question"].strip(), "options": opts, "answer": ans,
                "section": track, "origin": "generated",
                "explanation": g.get("explanation", "").strip(), "citation": cite.strip(", "),
            })
    return out[:n]


def explain_past(q, llm, retrieve, build_context) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    query = q["question"] + " " + q["options"][q["answer"]]
    context = build_context(retrieve(query, st.session_state["tp_chunks"]))
    prompt = (f"Document excerpts:\n\n{context}\n\n---\n\nExam question:\n{q['question']}\n{opts}\n\n"
              f"Answer key: {LETTERS[q['answer']]}. {q['options'][q['answer']]}")
    return llm.ask_text(EXPLAIN_SYSTEM, prompt, kind="explain")


# ── Screens ──────────────────────────────────────────────────────────────────

def _reset():
    for k in [k for k in st.session_state if k.startswith("tp_") and k not in ("tp_chunks", "tp_cache_dir")]:
        del st.session_state[k]


def _start(quiz):
    _reset()
    st.session_state.tp_quiz = quiz
    st.session_state.tp_idx = 0
    st.session_state.tp_answers = {}


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
        tracks = st.multiselect("Exam sections", TRACKS, default=TRACKS,
                                help="General = on everyone's exam. The others are role-specific sections.")
        n = st.select_slider("Number of questions", [5, 10, 15, 20, 25], value=10)

    chosen_tests = tests
    if bank and source != "gen":
        chosen_tests = st.multiselect("Past tests to draw from", tests, default=tests)
        pool = [q for q in bank if q["source"] in chosen_tests]
        counts = {t: sum(q["section"] == t for q in pool) for t in TRACKS}
        no_key = sum(q["answer"] is None for q in pool)
        st.caption("Past-test questions available: " + " · ".join(f"{t} {c}" for t, c in counts.items())
                   + (f"  ·  {no_key} have no answer key and are checked against the rulebooks"
                      if no_key else "")
                   + (f"  ·  {n_admin} administrative questions excluded" if n_admin else ""))

    if not st.button("▶️ Start quiz", type="primary", disabled=not tracks):
        return

    past_pool = [q for q in bank if q["section"] in tracks and q["source"] in chosen_tests]
    n_past = {"past": n, "gen": 0, "mixed": n // 2}[source]
    n_past = min(n_past, len(past_pool))
    n_gen = n - n_past

    quiz = [dict(q, origin="past") for q in random.sample(past_pool, n_past)]
    if n_gen:
        if not chunks:
            st.error("Select at least one PDF in the sidebar to generate questions.")
            return
        with st.spinner(f"Writing {n_gen} questions from the rulebooks…"):
            quiz += generate_questions(n_gen, tracks, bank, chunks, llm, friendly)
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
    origin = "Past exam" if q["origin"] == "past" else "Generated from rulebooks"
    st.caption(f"{q['section']} · {origin}")
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


def render(llm, test_files, cache_dir, active_chunks, friendly, retrieve, build_context, status_box):
    st.title("📝 Test Prep")
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

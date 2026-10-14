from __future__ import annotations

import textwrap

import pytest

from chatter.answer import (
    ABSTENTION,
    EVIDENCE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    AnswerStatus,
    answer_question,
    build_context,
    evidence_sources,
)
from chatter.evidence import (
    MIN_QUOTE_CHARS,
    NOT_FOUND,
    TOO_SHORT,
    UNKNOWN_TAG,
    VERIFIED,
    WRONG_TAG,
    Quote,
    Source,
    Verification,
    describe_quote,
    find_answer_heading,
    parse_reply,
    verification_status,
    verify_quote,
)
from chatter.extract import Chunk, extract_source
from chatter.retrieve import Hit
from conftest import FakeGenerator

SOURCES = {
    "C1": Source("C1", "net/http.py::parse", "net/http.py", (
        (10, "def parse(raw):"),
        (11, '    head, _, body = raw.partition("\\r\\n\\r\\n")'),
        (12, "    return head"),
    )),
    "C2": Source("C2", "net/retry.py::delay", "net/retry.py", (
        (3, "def delay(attempt):"),
        (4, "    return min(2 ** attempt, MAX_BACKOFF_SECONDS)"),
    )),
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Evidence:\n[C1] return head\n\nAnswer:\nIt returns the head [C1].",
        "**Evidence:**\n- [C1] `return head`\n\n**Answer:** It returns the head [C1].",
        "### Evidence\n1. [C1]: \"return head\"\n### Answer\nIt returns the head [C1].",
        "**Evidence**:\n* [C01] return head\n**Answer**: It returns the head [C1].",
        "Evidence:\n[C1]\n```python\nreturn head\n```\nAnswer:\nIt returns the head [C1].",
    ],
)
def test_parse_reply_formats(reply: str) -> None:
    parsed = parse_reply(reply)
    assert parsed.evidence_section and parsed.parseable
    assert parsed.entries == (("C1", "return head"),)
    assert parsed.answer == "It returns the head [C1]."


def test_parse_reply_without_sections() -> None:
    plain = parse_reply("It returns the head [C1].")
    assert not plain.evidence_section and not plain.parseable and plain.answer is None
    no_answer = parse_reply("Evidence:\n[C1] return head")
    assert no_answer.parseable and no_answer.answer is None
    empty = parse_reply("Evidence:\nnothing tagged here\nAnswer: x")
    assert empty.evidence_section and not empty.parseable and empty.answer == "x"
    multi = parse_reply("Evidence:\n[C1, C2] return head\nAnswer: y")
    assert multi.entries == (("C1", "return head"),)  # first tag of a group


def test_answer_heading_while_streaming() -> None:
    assert find_answer_heading("Evidence:\n[C1] x\nAnswer:") is None  # may still grow
    assert find_answer_heading("Evidence:\n[C1] x\n**Answer:") is None
    match = find_answer_heading("Evidence:\n[C1] x\n**Answer:** It")
    assert match is not None and "Evidence:\n[C1] x\n**Answer:** It"[match.end():] == "It"
    assert find_answer_heading("Evidence:\n[C1] x\nAnswer:", complete=True) is not None
    assert find_answer_heading("The answer: is here") is None  # not at line start
    assert find_answer_heading("### Answer\nIt") is not None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verify_whole_line_and_substring() -> None:
    whole = verify_quote("C1", 'head, _, body = raw.partition("\\r\\n\\r\\n")', SOURCES)
    assert (whole.result, whole.line, whole.path) == (VERIFIED, 11, "net/http.py")
    part = verify_quote("C2", "min(2 ** attempt,", SOURCES)
    assert part.verified and part.line == 4
    spaced = verify_quote("C2", "  return   min(2 ** attempt, MAX_BACKOFF_SECONDS)  ", SOURCES)
    assert spaced.verified  # whitespace is normalised
    trailing = verify_quote("C2", "return min(2 ** attempt, MAX_BACKOFF …", SOURCES)
    assert trailing.verified  # a trailing ellipsis is dropped


def test_minimum_length_applies_to_whole_lines_too() -> None:
    assert len("return head".replace(" ", "")) < MIN_QUOTE_CHARS
    short_line = verify_quote("C1", "return head", SOURCES)  # a whole line, but short
    assert short_line.result == TOO_SHORT and short_line.line is None
    assert verify_quote("C1", "def parse(raw):", SOURCES).result == VERIFIED  # 13 non-space chars


def test_wrong_tag_unknown_and_not_found() -> None:
    wrong = verify_quote("C1", "MAX_BACKOFF_SECONDS)", SOURCES)
    assert (wrong.result, wrong.found_in, wrong.chunk_id) == (WRONG_TAG, "C2", "net/http.py::parse")
    assert verify_quote("C7", "def parse(raw):", SOURCES).result == UNKNOWN_TAG
    assert verify_quote("C1", "head, body = raw.split(sep)", SOURCES).result == NOT_FOUND
    # Two lines joined into one quote are not a line of the chunk.
    assert verify_quote("C1", "def parse(raw): return head", SOURCES).result == NOT_FOUND


def test_verification_status() -> None:
    ok = Quote("C1", "x", VERIFIED, line=1)
    bad = Quote("C2", "y", NOT_FOUND)
    assert verification_status([ok], "It does [C1].", abstained=False) is Verification.VERIFIED
    assert verification_status([ok, bad], "It does [C1].", abstained=False) is Verification.PARTIAL
    assert verification_status([ok], "It does [C1, C2].", abstained=False) is Verification.PARTIAL
    assert verification_status([bad], "It does [C2].", abstained=False) is Verification.UNVERIFIED
    assert verification_status([], "It does.", abstained=False) is Verification.UNVERIFIED
    assert verification_status([], ABSTENTION, abstained=True) is Verification.ABSTAINED


def test_describe_quote() -> None:
    assert describe_quote(verify_quote("C2", "min(2 ** attempt,", SOURCES)) == (
        "✓ [C2] net/retry.py:4  min(2 ** attempt,"
    )
    assert describe_quote(verify_quote("C1", "MAX_BACKOFF_SECONDS)", SOURCES)) == (
        '✗ [C1] found in C2, not C1: "MAX_BACKOFF_SECONDS)"'
    )


# ---------------------------------------------------------------------------
# answer_question with the evidence prompt
# ---------------------------------------------------------------------------

CODE = '''
def parse_http_response(raw):
    head, _, body = raw.partition("\\r\\n\\r\\n")
    return head, body
'''
GOOD = (
    "Evidence:\n"
    '[C1] head, _, body = raw.partition("\\r\\n\\r\\n")\n'
    "\n"
    "Answer:\n"
    "It splits the head from the body at the blank line [C1]."
)


def hit_for(code: str = CODE, path: str = "net/http.py") -> Hit:
    chunk: Chunk = next(c for c in extract_source(textwrap.dedent(code), path) if c.kind != "module")
    return Hit(
        chunk_id=f"{chunk.path}::{chunk.qualname}", chunk=chunk, score=0.1, sources=("dense",),
        ranks={"dense": 1}, raw_scores={"dense": 0.9}, lines=(chunk.start_line, chunk.end_line),
    )


def run(reply: str, **kwargs: object) -> tuple[object, list[str], list[list[Quote]]]:
    streamed: list[str] = []
    evidence: list[list[Quote]] = []
    result = answer_question(
        "how is the response split", [hit_for()], FakeGenerator(reply), prompt="evidence",
        on_text=streamed.append, on_evidence=lambda q: evidence.append(list(q)), **kwargs,  # type: ignore[arg-type]
    )
    return result, streamed, evidence


def test_verified_answer_streams_only_the_answer() -> None:
    result, streamed, evidence = run(GOOD)
    assert "".join(streamed).strip() == "It splits the head from the body at the blank line [C1]."
    assert len(evidence) == 1 and [q.result for q in evidence[0]] == [VERIFIED]
    assert evidence[0][0].line == 3  # file line of the quoted source (CODE starts with a blank line)
    assert result.status is AnswerStatus.ANSWERED and result.prompt == "evidence"
    assert result.evidence.verification is Verification.VERIFIED
    assert result.answer == "It splits the head from the body at the blank line [C1]."
    assert result.text == GOOD and result.raw_text is None
    assert [c.tag for c in result.citations] == ["C1"]


def test_unverified_answer_is_withheld() -> None:
    reply = "Evidence:\n[C1] head, body = raw.split(SEPARATOR)\n\nAnswer:\nIt splits on a separator [C1]."
    result, streamed, evidence = run(reply)
    assert streamed == []  # nothing of the answer reached the user
    assert [q.result for q in evidence[0]] == [NOT_FOUND]
    assert result.status is AnswerStatus.ABSTAINED and result.text == ABSTENTION
    assert result.evidence.forced_abstention and result.raw_text == reply
    assert result.raw_status is AnswerStatus.ANSWERED and result.citations == ()


def test_reply_without_evidence_is_withheld() -> None:
    result, streamed, evidence = run("It splits the head from the body [C1].")
    assert streamed == [] and evidence == [[]]
    assert result.status is AnswerStatus.ABSTAINED and result.evidence.forced_abstention
    assert not result.evidence.parsed.evidence_section


def test_evidence_without_answer_is_withheld() -> None:
    result, streamed, _ = run('Evidence:\n[C1] head, _, body = raw.partition("\\r\\n\\r\\n")')
    assert streamed == [] and result.evidence.forced_abstention


def test_abstention_is_not_forced() -> None:
    result, streamed, evidence = run(ABSTENTION)
    assert streamed == [] and evidence == [[]]
    assert result.status is AnswerStatus.ABSTAINED and not result.evidence.forced_abstention
    assert result.evidence.verification is Verification.ABSTAINED and result.raw_text is None
    after_evidence, _, _ = run(f'Evidence:\n[C1] return head, body\n\nAnswer:\n{ABSTENTION}')
    assert after_evidence.status is AnswerStatus.ABSTAINED and after_evidence.citations == ()


def test_partial_answer_keeps_only_supported_citations() -> None:
    reply = (
        "Evidence:\n"
        '[C1] head, _, body = raw.partition("\\r\\n\\r\\n")\n'
        "[C1] this line is not in the chunk at all\n"
        "Answer:\nIt splits at the blank line [C1]."
    )
    result, streamed, _ = run(reply)
    assert result.evidence.verification is Verification.PARTIAL
    assert "".join(streamed).strip() == "It splits at the blank line [C1]."
    assert [c.tag for c in result.citations] == ["C1"]


def test_plain_prompt_is_unchanged() -> None:
    gen = FakeGenerator("It splits [C1].")
    result = answer_question("q", [hit_for()], gen)
    assert gen.prompts[0][0] == SYSTEM_PROMPT and result.prompt == "plain" and result.evidence is None


def test_both_prompts_see_the_same_context() -> None:
    hits = [hit_for(CODE + "".join(f"    x{i} = {i}\n" for i in range(200)))]
    plain = answer_question("split", hits, FakeGenerator("x"), max_context_tokens=400)
    evidence = answer_question("split", hits, FakeGenerator("x"), max_context_tokens=400, prompt="evidence")
    assert plain.blocks[0].truncated
    assert [(b.chunk_id, b.shown_lines) for b in plain.blocks] == [(b.chunk_id, b.shown_lines) for b in evidence.blocks]


def test_evidence_prompt_text() -> None:
    assert "Evidence:" in EVIDENCE_SYSTEM_PROMPT and "Answer:" in EVIDENCE_SYSTEM_PROMPT
    assert f"nothing else:\n{ABSTENTION}" in EVIDENCE_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="unknown prompt"):
        answer_question("q", [hit_for()], FakeGenerator("x"), prompt="cot")


def test_sources_use_shown_lines_of_truncated_blocks() -> None:
    body = "".join(f"    step_{i} = run(stage_{i})\n" for i in range(80))
    hit = hit_for("def pipeline():\n" + body + "    marker_line = finish(everything)\n")
    blocks, _ = build_context("marker_line finish", [hit], FakeGenerator(), max_context_tokens=300)
    (source,) = evidence_sources(blocks).values()
    assert blocks[0].truncated and len(source.lines) == len(blocks[0].shown_lines) < 82
    numbers = [n for n, _ in source.lines]
    assert numbers == list(blocks[0].shown_lines)
    hidden = next(i for i in range(80) if i + 2 not in numbers)  # a body line that was cut
    assert verify_quote("C1", f"step_{hidden} = run(stage_{hidden})", {"C1": source}).result == NOT_FOUND

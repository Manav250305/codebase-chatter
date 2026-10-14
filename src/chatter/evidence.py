"""Evidence-first answers: parse quoted evidence and check it against the context.

The evidence prompt asks the model for an ``Evidence:`` section of lines
copied from the chunks, each tagged ``[C#]``, then an ``Answer:`` section.
A quote is verified when, after normalising whitespace, it has at least
``MIN_QUOTE_CHARS`` non-space characters and equals, or is a substring of,
one line the model was shown for that tag's chunk. Chunk headers and
truncation notes never count. A reply verifies as:

    verified    at least one verified quote, and every tag the answer cites
                has a verified quote
    partial     at least one verified quote, but some quote failed or the
                answer cites a tag without one
    unverified  no verified quote (including replies without an Evidence
                section); the caller replaces such answers with the
                abstention sentence
    abstained   the answer is the abstention sentence (checked by the caller)

Everything here is pure: callers pass the shown lines per tag as ``Source``.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MIN_QUOTE_CHARS = 12  # non-space characters, for whole-line and substring matches alike

VERIFIED = "verified"
WRONG_TAG = "wrong_tag"  # the line exists, but in another chunk
NOT_FOUND = "not_found"
UNKNOWN_TAG = "unknown_tag"
TOO_SHORT = "too_short"


class Verification(enum.Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class Source:
    """What the model was shown for one tag: file lines as (line number, text)."""

    tag: str
    chunk_id: str
    path: str
    lines: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class Quote:
    tag: str
    text: str
    result: str  # VERIFIED | WRONG_TAG | NOT_FOUND | UNKNOWN_TAG | TOO_SHORT
    chunk_id: str | None = None  # the tagged chunk, when the tag exists
    path: str | None = None
    line: int | None = None  # file line of the match (verified only)
    found_in: str | None = None  # the tag whose chunk holds the line (wrong_tag only)

    @property
    def verified(self) -> bool:
        return self.result == VERIFIED


@dataclass(frozen=True, slots=True)
class ParsedReply:
    evidence_section: bool  # an Evidence heading was found
    entries: tuple[tuple[str, str], ...]  # (tag, quoted text) in order
    answer: str | None  # text after the Answer heading; None when there is none

    @property
    def parseable(self) -> bool:
        """An Evidence section with at least one tagged entry."""
        return self.evidence_section and bool(self.entries)


def _heading(word: str) -> re.Pattern[str]:
    # "Evidence:", "**Evidence:**", "**Evidence**:", "### Evidence", "Evidence" alone on a line.
    return re.compile(
        rf"^[ \t]*(?:#{{1,6}}[ \t]*)?(?:\*\*|__)?{word}(?:\*\*|__)?[ \t]*"
        rf"(?::(?:\*\*|__)?|(?=\n|$))[ \t]*",
        re.IGNORECASE | re.MULTILINE,
    )


EVIDENCE_HEADING = _heading("evidence")
ANSWER_HEADING = _heading("answer")
_ENTRY = re.compile(
    r"^[ \t]*(?:[-*•]|\d+[.)])?[ \t]*\[\s*(C\d+)(?:\s*[,;]\s*C\d+)*\s*\][ \t]*[:\-–—]?[ \t]*(.*?)[ \t]*$"
)
_TAG_GROUP = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]")
_TAG = re.compile(r"C(\d+)")
_QUOTE_PAIRS = (("`", "`"), ('"', '"'), ("'", "'"), ("“", "”"))


def find_answer_heading(text: str, *, complete: bool = False) -> re.Match[str] | None:
    """The first Answer heading. While streaming (``complete`` False), a match
    touching the end of the text may still grow (``**Answer:`` + ``**``), so
    it only counts once more text follows it."""
    for match in ANSWER_HEADING.finditer(text):
        if complete or match.end() < len(text):
            return match
        return None
    return None


def parse_reply(text: str, *, complete: bool = True) -> ParsedReply:
    """Split a reply into evidence entries and the answer."""
    answer_match = find_answer_heading(text, complete=complete)
    head = text[: answer_match.start()] if answer_match else text
    answer = text[answer_match.end() :].strip() if answer_match else None
    evidence_match = EVIDENCE_HEADING.search(head)
    if evidence_match is None:
        return ParsedReply(False, (), answer)
    return ParsedReply(True, tuple(_entries(head[evidence_match.end() :])), answer)


def _entries(section: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    pending: str | None = None  # a tag line without text: the quote is on the next line
    for raw in section.split("\n"):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        match = _ENTRY.match(raw)
        if match:
            tag, quote = f"C{int(match.group(1)[1:])}", _unquote(match.group(2))
            if quote:
                entries.append((tag, quote))
                pending = None
            else:
                pending = tag
        elif pending is not None:
            entries.append((pending, _unquote(line)))
            pending = None
    return entries


def _unquote(text: str) -> str:
    text = text.strip()
    for left, right in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            return text[len(left) : -len(right)].strip()
    return text


def _normalize(text: str) -> str:
    text = " ".join(text.split())
    return text[:-1].rstrip() if text.endswith("…") else text


def _match_line(quote: str, source: Source) -> int | None:
    for number, line in source.lines:
        shown = _normalize(line)
        if quote == shown or quote in shown:
            return number
    return None


def verify_quote(tag: str, text: str, sources: Mapping[str, Source]) -> Quote:
    """Check one quote against the lines shown for its tag (see module docstring)."""
    quote = _normalize(text)
    source = sources.get(tag)
    if source is None:
        return Quote(tag, text, UNKNOWN_TAG)
    known = {"chunk_id": source.chunk_id, "path": source.path}
    if len(quote.replace(" ", "")) < MIN_QUOTE_CHARS:
        return Quote(tag, text, TOO_SHORT, **known)
    line = _match_line(quote, source)
    if line is not None:
        return Quote(tag, text, VERIFIED, line=line, **known)
    for other in sources.values():
        if other.tag != tag and _match_line(quote, other) is not None:
            return Quote(tag, text, WRONG_TAG, found_in=other.tag, **known)
    return Quote(tag, text, NOT_FOUND, **known)


def verify_entries(entries: Sequence[tuple[str, str]], sources: Mapping[str, Source]) -> tuple[Quote, ...]:
    return tuple(verify_quote(tag, text, sources) for tag, text in entries)


def cited_tags(text: str) -> list[str]:
    """Tags cited in ``text``, in order of first appearance."""
    tags: list[str] = []
    for group in _TAG_GROUP.finditer(text):
        for number in _TAG.findall(group.group(1)):
            tag = f"C{int(number)}"
            if tag not in tags:
                tags.append(tag)
    return tags


def unsupported_tags(answer: str, quotes: Sequence[Quote]) -> list[str]:
    """Tags the answer cites that have no verified quote."""
    supported = {q.tag for q in quotes if q.verified}
    return [tag for tag in cited_tags(answer) if tag not in supported]


def verification_status(quotes: Sequence[Quote], answer: str, *, abstained: bool) -> Verification:
    if abstained:
        return Verification.ABSTAINED
    if not any(q.verified for q in quotes):
        return Verification.UNVERIFIED
    if all(q.verified for q in quotes) and not unsupported_tags(answer, quotes):
        return Verification.VERIFIED
    return Verification.PARTIAL


_REASONS = {
    WRONG_TAG: "found in {found_in}, not {tag}",
    NOT_FOUND: "not found in {tag}",
    UNKNOWN_TAG: "unknown tag {tag}",
    TOO_SHORT: "too short to verify in {tag}",
}


def describe_quote(quote: Quote) -> str:
    """``✓ [C1] path:line  text`` or ``✗ [C2] not found in C2: "text"``."""
    if quote.verified:
        return f"✓ [{quote.tag}] {quote.path}:{quote.line}  {quote.text}"
    reason = _REASONS.get(quote.result, quote.result).format(tag=quote.tag, found_in=quote.found_in)
    return f'✗ [{quote.tag}] {reason}: "{quote.text}"'

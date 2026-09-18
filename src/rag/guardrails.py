"""Input/output guardrails on the query path.

Every check is a plain LangChain ``Runnable`` or a LangChain-native tool, so
the guardrail chain composes with ``|`` and drops into the query graph as
just another step.

**Input side.** A cheap regex battery rejects prompt-injection payloads and
known jailbreak markers before anything expensive runs. This is *not* a
security boundary -- a determined adversary bypasses regex -- but it stops
the accidental and the casual, and it costs microseconds.

**Output side.** Three guarantees that make the answer trustworthy rather
than merely fluent:

* ``CitationGuardrail`` -- every ``[N]`` marker in the answer must exist in
  the retrieved context. An LLM that invents a citation is hallucinating a
  source, and that is exactly the failure mode this pipeline exists to avoid.
  Markers are normalised to ASCII first, because not every model cites in
  ASCII -- see :func:`_normalise_markers`.
* ``GroundingGuardrail`` -- sentence-level overlap between the answer and
  the retrieved context. Zero-overlap answers are replaced with a refusal,
  not shown to the user.
* ``PIIRedactor`` -- masks emails, phone numbers, and credit-card patterns
  that a model may have copied out of the corpus.

A fourth fixed string is passed through untouched rather than checked:
``NOT_COVERED``, the model's own report that the sources do not answer the
question. It makes no claim, so there is nothing to verify -- see
:func:`is_not_covered`.
"""

from __future__ import annotations

import re
from typing import Any

from rag.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------

# Markers of common jailbreak/prompt-injection patterns. Deliberately short:
# each must justify its false-positive risk.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore instructions", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.I)),
    ("reveal system prompt", re.compile(r"(reveal|show|print|repeat)\s+(your\s+)?(system\s+)?(prompt|instructions)", re.I)),
    ("roleplay jailbreak", re.compile(r"you\s+are\s+now\s+(DAN|an?\s+unrestricted|an?\s+unfiltered)", re.I)),
    ("instruction override", re.compile(r"(disregard|override)\s+(all\s+)?(safety|content|your)\s+(guidelines|policies|rules)", re.I)),
    ("delimeter escape", re.compile(r"</?(system|assistant|instruction)s?>", re.I)),
)


class InputRejectedError(Exception):
    """Raised when input guardrails reject a question."""


def check_input(question: str) -> str:
    """Pass the question through the injection battery, or reject it.

    Runs in microseconds; rejection is logged with the matched pattern so an
    operator can audit false positives and tighten the battery.
    """
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(question):
            logger.warning("input guardrail rejected question (%s)", label)
            raise InputRejectedError(
                "This request looks like a prompt-injection attempt and was "
                "rejected. Please rephrase as a plain question about the "
                "indexed documents."
            )
    return question


# ---------------------------------------------------------------------------
# Output guardrails
# ---------------------------------------------------------------------------

# The two messages the output battery substitutes for an answer it refused to
# pass through. They are module constants, not inline literals, because the
# API has to tell the browser "this is a refusal" rather than "this is the
# answer" -- see :func:`is_refusal`. Matching on a copy of the text would
# break the moment someone reworded one of them, and the failure would be
# silent: a refusal would render as a normal answer.
CITATION_REFUSAL = (
    "I generated an answer that referenced sources outside the "
    "retrieved context, so it was withheld. Here are the most "
    "relevant passages instead."
)

GROUNDING_REFUSAL = (
    "I could not ground an answer in the retrieved passages with "
    "sufficient confidence. Please consult the cited sources "
    "directly."
)

_REFUSALS: frozenset[str] = frozenset({CITATION_REFUSAL, GROUNDING_REFUSAL})


def is_refusal(answer: str | None) -> bool:
    """Whether ``answer`` is a guardrail refusal rather than a real answer.

    Exact membership, not a heuristic: the guards return these constants
    verbatim, so a match is a fact about where the string came from. An
    empty answer (generation off or failed) is *not* a refusal -- it is the
    absence of one, and the UI shows the retrieved passages instead.
    """
    return answer is not None and answer.strip() in _REFUSALS


# The third fixed string the answer path can return, and not a refusal: the
# model read the passages, none of them address the question, and it said so.
#
# It needs a name of its own because the pipeline has to act on it -- this is
# what makes the UI offer a web search -- and the prose gives it nothing to
# match on. "The model said it doesn't know" is not distinguishable by reading
# the answer: a model that does not know says so in its own words, and so does
# one that does know and is being careful. The prompt therefore requires this
# sentence verbatim, and the check below is membership of a constant rather
# than a hunt for "I don't know" -- a hunt that would eventually fire on an
# answer that knows perfectly well and is merely being qualified.
#
# Wording chosen to fit both paths, corpus and web. The user sees it only when
# the web fallback is unavailable, and "sources" is what both paths call the
# numbered blocks either way.
NOT_COVERED = "The provided sources do not cover this question."

_WHITESPACE_RE = re.compile(r"\s+")


def is_not_covered(answer: str | None) -> bool:
    """Whether the model reported that its sources do not answer the question.

    Containment, not equality: the prompt asks for the sentence alone and
    usually gets it, but a model that prefixes an apology -- "I'm sorry, but
    the provided sources do not cover this question." -- has still said exactly
    this and nothing else. A real answer cannot contain the sentence by
    accident, so the tolerance costs nothing.

    Whitespace is collapsed first: the sentence is exactly as long as an
    implementation detail of the tokeniser allows it to be, and a doubled
    space would otherwise hide the report.
    """
    if answer is None:
        return False
    return NOT_COVERED in _WHITESPACE_RE.sub(" ", answer)


# Models do not all cite in ASCII brackets. gpt-oss-120b answers with
# full-width lenticular brackets -- 【1】 -- and an ASCII-only marker regex saw
# none of them, so the guardrail returned every such answer untouched:
# citation verification silently became verification of nothing. The UI lost
# the markers too, rendering them as literal glyphs rather than the numbered
# pills that highlight their source.
#
# Only brackets *wrapping digits* are rewritten, so a stray 【 in prose is
# left alone. Normalising here -- before the check, and in the text that gets
# returned -- keeps "[N]" the one form the prompt, the guardrail and the
# browser all agree on, rather than teaching each of them a dialect.
_MARKER_VARIANT_RE = re.compile(r"[【［〔〖]\s*(\d+)\s*[】］〕〗]")


def _normalise_markers(answer: str) -> str:
    """Rewrite non-ASCII citation brackets to the canonical ``[N]``."""
    return _MARKER_VARIANT_RE.sub(r"[\1]", answer)


class CitationGuardrail:
    """Every ``[N]`` marker in an answer must reference a real citation.

    An answer citing a marker the context never had is inventing a source, so
    the invented markers are *removed*: a reader must never see an attribution
    that does not exist.

    The answer itself survives when at least one real citation remains.
    Discarding the whole answer over one stray marker withheld far more than
    it protected -- a model that writes ``[6]`` when five blocks were supplied
    has made an off-by-one slip, not fabricated a source, and the sentence
    around the marker is usually sound. Only when *every* citation in the
    answer was invented is there no attributable claim left to show, and that
    is the case the refusal exists for.
    """

    _MARKER_RE = re.compile(r"\[(\d+)\]")
    # A removed marker leaves its surrounding whitespace behind: "word [6] ."
    # becomes "word  ." -- collapse the run, then close the gap before
    # punctuation, so the sentence still reads as prose.
    _DOUBLE_SPACE_RE = re.compile(r"[ \t]{2,}")
    _SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.,;:!?])")

    def check(self, answer: str, citations: list[dict[str, Any]]) -> str:
        answer = _normalise_markers(answer)
        valid = {c["marker"] for c in citations}
        used = {f"[{n}]" for n in self._MARKER_RE.findall(answer)}
        invented = used - valid
        if not invented:
            return answer

        logger.warning(
            "citation guardrail: answer cited non-existent markers (%s); "
            "removing them",
            ", ".join(sorted(invented)),
        )
        for marker in invented:
            answer = answer.replace(marker, "")
        answer = self._DOUBLE_SPACE_RE.sub(" ", answer)
        answer = self._SPACE_BEFORE_PUNCT_RE.sub(r"\1", answer)

        if not (valid & used):
            logger.warning(
                "citation guardrail: every citation in the answer was "
                "invented; replacing answer with refusal"
            )
            return CITATION_REFUSAL
        return answer.strip()


class GroundingGuardrail:
    """Answers must overlap the retrieved context at sentence level.

    A correctly-prompted RAG answer restates context facts, so near-zero
    overlap means the model answered from parametric memory -- a
    hallucination with respect to this corpus.
    """

    _WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, min_overlap: float = 0.15) -> None:
        self.min_overlap = min_overlap

    @staticmethod
    def _word_set(text: str) -> set[str]:
        return set(GroundingGuardrail._WORD_RE.findall(text.lower()))

    def check(self, answer: str, context: str) -> str:
        answer_words = self._word_set(answer)
        if not answer_words:
            return answer
        context_words = self._word_set(context)
        overlap = len(answer_words & context_words) / len(answer_words)
        if overlap < self.min_overlap:
            logger.warning(
                "grounding guardrail: answer/context word overlap %.2f below "
                "%.2f; replacing answer with refusal",
                overlap,
                self.min_overlap,
            )
            return GROUNDING_REFUSAL
        return answer


class PIIRedactor:
    """Mask common PII patterns an answer might copy out of the corpus.

    The phone pattern is deliberately conservative: it requires a leading
    ``+`` (E.164) or 3-4 digit groups joined by ``-``/``.``/space. A looser
    "10-15 digits with separators" rule would swallow hyperparameters,
    version lists, and benchmark scores -- exactly the numeric content a
    technical corpus exists to answer about.
    """

    _PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
        # Exactly 16 digits in strict 4-digit groups: real card formatting.
        # Runs before the phone pattern -- a card number is also a run of
        # digit groups, and the looser phone rule would claim part of it,
        # leaving a fragment of the card visible in the answer.
        ("credit_card", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
        # +country international form, or ddd-ddd-dddd-style groups ending in
        # a 4-digit block (the tail of virtually every real phone number).
        # The 4-digit tail requirement is what keeps lists of 2-3 digit
        # numbers ("versions 11 12 13") and decimal hyperparameters out.
        ("phone", re.compile(
            r"(?<![\w.+-])"                                 # not part of a word/number
            r"(?:\+\d{1,3}[\s.-]?)?"                        # optional country code
            r"(?:\(?\d{2,4}\)?[\s.-]){1,2}"                 # 1-2 separator-joined groups
            r"\d{4}"                                        # 4-digit subscriber tail
            r"(?![\w.+-])"
        )),
    )

    def redact(self, text: str) -> str:
        for label, pattern in self._PATTERNS:
            text = pattern.sub(f"[REDACTED-{label.upper()}]", text)
        return text


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# Reusable singletons; every class above is stateless after construction.
citation_guardrail = CitationGuardrail()
grounding_guardrail = GroundingGuardrail()
pii_redactor = PIIRedactor()


def apply_output_guardrails(
    answer: str | None,
    citations: list[dict[str, Any]],
    context: str,
) -> str:
    """Run the output battery over a generated answer.

    ``None`` (generation off/failed) short-circuits to an empty string; the
    graph treats that as "show the context" rather than a missing answer.
    """
    if answer is None:
        return ""

    if is_not_covered(answer):
        # Nothing here to verify. "I don't know" makes no claim, so it can
        # neither invent a citation nor go ungrounded -- and the grounding
        # check would fail it every time, because the sentence is deliberately
        # not made of the passages' words. It returns before the battery
        # rather than after it for exactly that reason.
        #
        # The constant is returned rather than the model's own wording, so
        # what the API compares against is the same string in every case.
        return NOT_COVERED

    answer = citation_guardrail.check(answer, citations)
    answer = grounding_guardrail.check(answer, context)
    answer = pii_redactor.redact(answer)
    return answer.strip()

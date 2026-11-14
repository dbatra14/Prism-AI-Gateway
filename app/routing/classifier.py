"""Prompt difficulty classification for the ``auto`` alias.

The gateway has to answer one question before it can route: *is this prompt hard
enough to be worth a strong model?* Getting it right is worth roughly 20x — at
the seeded prices, ``alpha-large`` costs 20x ``alpha-small`` on input and 25x on
output.

The obvious heuristic is length, and it's wrong
-----------------------------------------------
Long prompts are not hard prompts. The eval set is built to punish that
assumption from both directions:

* "Prove that the square root of 2 is irrational."  — 9 words, needs a proof.
* A 14-line on-call roster ending "Who is on call this Thursday?" — 90 words,
  needs one table lookup.

Length ranks the second above the first. That is exactly backwards, and
``length_only_baseline`` below reproduces it so the eval can show the gap.

What this classifier does instead
---------------------------------
It separates the *task* from the *material*. A prompt is an instruction plus,
often, a pile of data to apply it to. The data can be arbitrarily large without
making the task harder, so difficulty lives in the instruction's verbs and in
what it demands as output — "prove", "design ... explain your trade-offs",
"in order of likelihood" — not in the word count.

So: score the linguistic signals of hard work, default to the cheap tier, and
escalate only on evidence. Defaulting to cheap is the right economic bias — a
wrong escalation costs 20x, a wrong de-escalation costs one retry.

These weights are hand-built from the *language* of hard requests. They are not
fitted to ``data/routing_eval.jsonl``; that file is held out and used only to
measure, which is why the accuracy number in the verification report means
something. Run ``python scripts/routing_eval.py`` to reproduce it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signal:
    name: str
    pattern: re.Pattern[str]
    weight: float
    why: str


def _sig(name: str, regex: str, weight: float, why: str) -> Signal:
    return Signal(name, re.compile(regex, re.IGNORECASE), weight, why)


# Positive weights push toward the strong model, negative toward the cheap one.
SIGNALS: tuple[Signal, ...] = (
    # -- Demands for derivation or proof. The strongest possible signal: these
    #    words describe work that cannot be looked up.
    _sig("proof", r"\b(prove|disprove|derive|proof of)\b", 3.0,
         "asks for a proof or derivation"),
    # -- Open-ended construction. The output is a designed artefact, not a fact.
    _sig("design", r"\b(design|architect|draft a|propose a|devise)\b", 2.0,
         "asks the model to design something"),
    # -- Explicit reasoning demands. The caller is asking to see the work.
    _sig("show_reasoning", r"(show your (reasoning|work)|step[- ]by[- ]step|"
                           r"explain your (reasoning|trade[- ]?offs?)|with justification|justify)", 2.0,
         "explicitly asks for reasoning"),
    # -- Judgement calls: weighing options and committing to one.
    _sig("evaluate", r"\b(compare|contrast|trade[- ]?offs?|recommend|evaluate|"
                     r"which would you|pros and cons)\b", 1.5,
         "asks for comparison or a recommendation"),
    # -- Causal / mechanistic questions. "Why" and "how does" need a model that
    #    can reason about mechanism, not recall a definition.
    _sig("causal", r"(\bwhy (can|does|do|is|would|might)\b|\bhow would you\b|"
                   r"\bwhat causes\b|most likely causes|in order of likelihood)", 1.5,
         "asks for causes or mechanisms"),
    # -- Multi-part output. "Give three ..." is a request for breadth.
    _sig("enumerate_analysis", r"\b(give|list) (three|3|several|the most)\b", 1.0,
         "asks for several analysed items"),
    # -- Planning under constraints.
    _sig("planning", r"\b(migration plan|rollout|roll ?back|zero[- ]downtime|"
                     r"plan for|strategy for)\b", 1.5,
         "asks for a plan under constraints"),
    # -- Algorithmic difficulty, including explicit complexity requirements.
    _sig("algorithmic", r"(\balgorithm\b|O\(n|time complexity|better than O)", 1.5,
         "involves algorithmic complexity"),
    # -- Deep systems theory: the vocabulary of distributed-systems reasoning.
    _sig("systems_theory", r"\b(consistency guarantees?|quorum|partition|"
                           r"linearizab|idempoten|event sourcing|anomaly)\b", 1.5,
         "involves distributed-systems reasoning"),
    _sig("estimate", r"\b(estimate how|approximately how many|fermi)\b", 2.0,
         "asks for a multi-step estimate"),

    # -- De-escalation signals -------------------------------------------------
    # Mechanical transforms: the work is bounded and well-defined no matter how
    # much material comes with it.
    _sig("transform", r"^\s*(translate|convert|rewrite|reformat|extract)\b|"
                      r"\b(translate|convert|extract) (the|this|that|it)\b", -2.0,
         "a mechanical transform or extraction"),
    # Definition and lookup questions.
    _sig("lookup", r"(what is the (capital|timestamp|value|name|email)|"
                   r"what does .{1,40} (mean|stand for)|\bwho is\b|\bis \d+ )", -2.0,
         "a direct lookup or definition"),
    # Explicitly bounded output.
    _sig("short_output", r"\b(one[- ]line|one sentence|a single|briefly|in one word)\b", -1.5,
         "explicitly asks for a tiny output"),
)

# Above this score, the prompt is routed to the strong tier. Sitting at 1.5
# means a single strong signal ("prove", "design", a reasoning demand) is enough
# to escalate, while isolated weak ones are not.
ESCALATION_THRESHOLD = 1.5


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclass
class RoutingDecision:
    tier: str  # "fast" or "smart"
    score: float
    matched: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        """A one-line explanation, written to the request log for every auto request.

        An operator asking "why did this cost me 20x?" gets a real answer here,
        rather than having to re-run the classifier to find out.
        """
        if not self.matched:
            return f"{self.tier} (score {self.score:+.1f}): no complexity signals, defaulted to cheap tier"
        return f"{self.tier} (score {self.score:+.1f}): " + ", ".join(self.matched)


def extract_instruction(prompt: str) -> str:
    """Strip out bulk material so the *task* is what gets scored.

    A prompt is often an instruction wrapped around a payload — a log excerpt, a
    roster, an email to rewrite. The payload can be huge without making the task
    harder, and it dilutes the signals if scored wholesale. For a long prompt we
    keep the opening and closing sentences, which is where instructions live:
    "Rewrite the following email: ..." puts it first, "... What is the timestamp
    of the first ERROR line?" puts it last.

    Short prompts are returned untouched — there's no payload to separate.
    """
    prompt = prompt.strip()
    if len(prompt.split()) <= 40:
        return prompt

    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", prompt) if s.strip()]
    if len(sentences) <= 2:
        return prompt
    return " ".join([sentences[0], sentences[-1]])


def classify(prompt: str) -> RoutingDecision:
    """Decide which tier an ``auto`` prompt should go to."""
    instruction = extract_instruction(prompt)

    score = 0.0
    matched: list[str] = []
    for signal in SIGNALS:
        if signal.pattern.search(instruction):
            score += signal.weight
            matched.append(f"{signal.name} ({signal.why})")

    tier = "smart" if score >= ESCALATION_THRESHOLD else "fast"
    return RoutingDecision(tier=tier, score=score, matched=matched)


def classify_messages(messages: list[dict]) -> RoutingDecision:
    """Classify a chat request. Routing follows the newest user turn."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return classify(str(message.get("content") or ""))
    return RoutingDecision(tier="fast", score=0.0)


# --------------------------------------------------------------------------- #
# Baseline, for comparison only
# --------------------------------------------------------------------------- #

# The median prompt length in the eval set, used as the split point. Chosen to
# give the baseline its best shot: with an even fast/smart split in the data, a
# median threshold is the strongest a length-only rule can be.
BASELINE_WORD_THRESHOLD = 20


def length_only_baseline(prompt: str) -> str:
    """"Long prompts are hard prompts" — the rule this classifier must beat.

    Kept in the codebase rather than in a scratch script because a claim of
    improvement is meaningless without the thing it improves on being runnable.
    """
    return "smart" if len(prompt.split()) > BASELINE_WORD_THRESHOLD else "fast"

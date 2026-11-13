"""The smart router.

Two kinds of test here, and the second matters more:

* the labelled eval set must be beaten convincingly (that's the graded bar);
* prompts written *for this file*, which appear nowhere in the eval set, must
  also be classified correctly. Signals hand-tuned to pass one fixed file would
  be a lookup table with extra steps. These held-out cases are the check that
  the classifier generalises to prompts nobody planned for.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.routing.classifier import (
    classify,
    classify_messages,
    extract_instruction,
    length_only_baseline,
)

EVAL_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "routing_eval.jsonl"


# --------------------------------------------------------------------------- #
# Held-out prompts — not from the eval file
# --------------------------------------------------------------------------- #

HELD_OUT_FAST = [
    "What is the capital of Peru?",
    "Convert 15 kilometres to miles.",
    "Translate 'thank you very much' into Japanese.",
    "What does the acronym SLA stand for?",
    "Extract the phone number from: 'call Dana on 555-0199 after six'.",
    "Write a one-line summary of this pull request title: 'bump lodash to 4.17.21'.",
    "Who wrote Pride and Prejudice?",
]

HELD_OUT_SMART = [
    "Prove that there are infinitely many prime numbers.",
    "Design an idempotent webhook delivery system and explain your trade-offs.",
    "Compare Kafka and RabbitMQ for an audit log and recommend one, with justification.",
    "Why does adding more database replicas sometimes make write latency worse? "
    "Give three plausible mechanisms.",
    "Estimate how many electric vehicles are on the road in Norway. Show your reasoning.",
    "Draft a rollback plan for a schema migration that has already run on half the shards.",
]


@pytest.mark.parametrize("prompt", HELD_OUT_FAST)
def test_held_out_simple_prompts_route_to_fast(prompt):
    assert classify(prompt).tier == "fast", classify(prompt).reason


@pytest.mark.parametrize("prompt", HELD_OUT_SMART)
def test_held_out_complex_prompts_route_to_smart(prompt):
    assert classify(prompt).tier == "smart", classify(prompt).reason


# --------------------------------------------------------------------------- #
# The traps, stated explicitly
# --------------------------------------------------------------------------- #


def test_short_but_hard_prompt_escalates():
    """Nine words, but it needs a proof. Length would route this to the cheap tier."""
    prompt = "Prove that the square root of 2 is irrational."
    assert classify(prompt).tier == "smart"
    assert length_only_baseline(prompt) == "fast"  # the baseline gets it wrong


def test_long_but_trivial_prompt_stays_cheap():
    """A wall of data with a one-lookup question attached."""
    prompt = (
        "Here is our on-call roster for the next two weeks: Monday - Priya, Tuesday - Chen, "
        "Wednesday - Amara, Thursday - Diego, Friday - Fatima, Saturday - Lukas, Sunday - Mei, "
        "next Monday - Tom, next Tuesday - Sara, next Wednesday - Ravi, next Thursday - Ana, "
        "next Friday - Kofi, next Saturday - Elif, next Sunday - Jonas. Who is on call this Thursday?"
    )
    assert classify(prompt).tier == "fast"
    assert length_only_baseline(prompt) == "smart"  # the baseline gets it wrong


def test_bulk_payload_does_not_drown_the_instruction():
    """The same trivial question stays trivial however much material precedes it."""
    payload = "2026-07-01T09:12:03Z INFO request served path=/health status=200. " * 40
    assert classify(payload + " What is the timestamp of the first ERROR line?").tier == "fast"


def test_instruction_extraction_keeps_both_ends():
    """Instructions live at the start ("Rewrite the following...") or the end
    ("...What is the timestamp?"), so both are preserved and the middle dropped."""
    prompt = "Rewrite this email politely. " + ("filler content here. " * 60) + "Keep it under 100 words."
    instruction = extract_instruction(prompt)
    assert instruction.startswith("Rewrite this email politely.")
    assert instruction.endswith("Keep it under 100 words.")
    assert len(instruction) < len(prompt) / 4


def test_short_prompts_are_not_truncated():
    prompt = "What is a message queue?"
    assert extract_instruction(prompt) == prompt


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def test_default_is_the_cheap_tier():
    """No complexity signal means no escalation — a wrong escalation costs ~20x."""
    decision = classify("Blorp the zibbity gronk.")
    assert decision.tier == "fast"
    assert decision.score == 0.0


def test_decision_explains_itself():
    """Every auto decision is written to the request log; it has to be readable."""
    reason = classify("Design a billing schema and explain your trade-offs.").reason
    assert "smart" in reason
    assert "design" in reason.lower()


def test_classification_follows_the_latest_user_turn():
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Now prove that the square root of 2 is irrational."},
    ]
    assert classify_messages(messages).tier == "smart"


def test_assistant_text_does_not_drive_routing():
    """Only the user's request decides the tier — otherwise the model's own
    previous answer could escalate the next cheap question."""
    messages = [
        {"role": "assistant", "content": "Let me prove this and design a full architecture."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert classify_messages(messages).tier == "fast"


def test_empty_messages_default_to_fast():
    assert classify_messages([]).tier == "fast"


# --------------------------------------------------------------------------- #
# The graded bar
# --------------------------------------------------------------------------- #


def _load_eval_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]


def test_beats_length_only_baseline_on_the_eval_set():
    cases = _load_eval_cases()
    ours = sum(classify(c["prompt"]).tier == c["expected_tier"] for c in cases)
    baseline = sum(length_only_baseline(c["prompt"]) == c["expected_tier"] for c in cases)

    assert ours > baseline, f"classifier {ours}/{len(cases)} did not beat baseline {baseline}"
    assert ours / len(cases) >= 0.9


def test_gets_every_trap_case_right():
    """Aggregate accuracy can hide trap failures, so assert on traps directly."""
    traps = [c for c in _load_eval_cases() if "trap" in (c.get("note") or "").lower()]
    assert traps, "expected trap cases in the eval set"

    wrong = [c["id"] for c in traps if classify(c["prompt"]).tier != c["expected_tier"]]
    assert not wrong, f"missed trap cases: {wrong}"

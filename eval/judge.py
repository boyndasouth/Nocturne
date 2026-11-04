"""LLM-as-judge with a fixed rubric (spec §9, §11).

The LongMemEval rubrics below are the official ones from LongMemEval's
src/evaluation/evaluate_qa.py, used unchanged so per-category numbers line up with
published results (which used GPT-4o as the judge; we use a free-tier model, so treat
cross-paper comparisons as approximate). LoCoMo has no official judge prompt, so its
categories reuse the closest LongMemEval rubric.

LongMemEval prompts: Copyright (c) 2024 Di Wu, MIT License.
https://github.com/xiaowu0162/LongMemEval/blob/main/LICENSE

The same judge model and prompts score every system, and each verdict is kept with its raw
judge output so a random sample can be hand-checked (`report.py --sample`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.client import LLMClient
from eval.datasets.schema import Question

_CORE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no."
)
_TAIL = (
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

RUBRICS: dict[str, str] = {
    "default": _CORE + " " + _TAIL,  # the trailing space is in the original
    "temporal": _CORE
    + " In addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    + _TAIL,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please "
        "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
        "response contains some previous information along with an updated answer, the response "
        "should be considered as correct as long as the updated answer is the required answer."
    )
    + _TAIL,
    "preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: "
        "{response}\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The "
        "model could say that the information is incomplete, or some other information is given "
        "but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\n"
        "Model Response: {response}\n\nDoes the model correctly identify the question as "
        "unanswerable? Answer yes or no only."
    ),
}


def rubric_for(q: Question) -> str:
    if q.abstention:
        return "abstention"
    if q.question_type == "single-session-preference":
        return "preference"
    if q.question_type == "knowledge-update":
        return "knowledge-update"
    if q.category in ("temporal-reasoning", "temporal"):
        return "temporal"
    return "default"


def build_prompt(q: Question, response: str) -> str:
    return RUBRICS[rubric_for(q)].format(question=q.question, answer=q.answer, response=response)


_VERDICT_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_verdict(text: str) -> bool | None:
    """The first standalone yes/no wins (models sometimes explain after answering).
    None when there's neither, which the runner records as a judge error."""
    m = _VERDICT_RE.search(text)
    return None if m is None else m.group(1).lower() == "yes"


@dataclass(frozen=True)
class Verdict:
    correct: bool | None
    rubric: str
    raw: str


class Judge:
    def __init__(self, llm: LLMClient, *, max_tokens: int = 600) -> None:
        self._llm = llm
        self._max_tokens = max_tokens  # room for reasoning models' hidden thinking

    async def grade(self, q: Question, response: str) -> Verdict:
        resp = await self._llm.chat(
            [{"role": "user", "content": build_prompt(q, response)}],
            temperature=0.0,
            max_tokens=self._max_tokens,
            trace_name="eval.judge",
        )
        return Verdict(correct=parse_verdict(resp.text), rubric=rubric_for(q), raw=resp.text)

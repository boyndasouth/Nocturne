from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval import datasets
from eval.datasets import locomo, longmemeval
from eval.datasets.schema import TURN_SPACING

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def lme():
    return {c.id: c for c in longmemeval.load(FIXTURES / "longmemeval_mini.json")}


def test_longmemeval_dates_with_weekday():
    assert longmemeval.parse_date("2023/05/20 (Sat) 02:21") == datetime(
        2023, 5, 20, 2, 21, tzinfo=UTC
    )
    assert longmemeval.parse_date("2023/05/20") == datetime(2023, 5, 20, tzinfo=UTC)
    with pytest.raises(ValueError):
        longmemeval.parse_date("last Tuesday")


def test_longmemeval_sorts_oracle_sessions_by_date(lme):
    conv = lme["ku_001"]
    assert [s.id for s in conv.sessions] == ["s_austin", "s_dallas"]  # file order is reversed
    [q] = conv.questions
    assert q.category == "knowledge-update" and not q.abstention
    assert q.asked_at == datetime(2023, 6, 10, 9, 0, tzinfo=UTC)
    assert q.evidence == ("s_dallas",)


def test_longmemeval_abstention_is_its_own_category(lme):
    [q] = lme["ssu_003_abs"].questions
    assert q.abstention
    assert q.category == "abstention"
    assert q.question_type == "single-session-user"


def test_chat_episodes_pair_user_with_reply_in_order(lme):
    eps = lme["ku_001"].episodes()
    assert [e.content.splitlines()[0] for e in eps] == [
        "User: I live in Austin and I'm looking for good coffee shops nearby.",
        "User: Thanks! Also, I'm studying linear algebra this semester.",
        "User: Quick update: I moved to Dallas last week for my new job.",
    ]
    assert eps[0].content.endswith("Assistant: Austin has plenty. Try a few on South Congress.")
    assert eps[1].at - eps[0].at == TURN_SPACING
    assert all(a.at < b.at for a, b in zip(eps, eps[1:], strict=False))
    assert lme["ku_001"].session_ends() == [eps[1].at, eps[2].at]


LOCOMO_SAMPLE = {
    "sample_id": "conv-x",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_2_date_time": "7:55 pm on 9 June, 2023",
        "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "I ran a charity race!"}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {
                "speaker": "Melanie",
                "dia_id": "D1:2",
                "text": "Look",
                "img_url": ["x"],
                "blip_caption": "a photo of a dog on a beach",
            },
        ],
        "session_3_date_time": "10:00 am on 1 July, 2023",
        "session_3": [],
    },
    "qa": [
        {
            "question": "When did Melanie run the race?",
            "answer": 2023,
            "evidence": ["D2:1"],
            "category": 2,
        },
        {
            "question": "What did Caroline win?",
            "evidence": ["D1:1"],
            "category": 5,
            "adversarial_answer": "a medal",
        },
    ],
}


def test_locomo_dialogue_sessions_dates_and_captions():
    conv = locomo.parse_sample(LOCOMO_SAMPLE)
    assert conv.style == "dialogue"
    assert [s.id for s in conv.sessions] == ["conv-x:S1", "conv-x:S2"]  # empty S3 dropped
    assert conv.sessions[0].date == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    eps = conv.episodes()
    assert [e.content for e in eps] == [
        "Caroline: Hey Mel!",
        "Melanie: Look [shares a photo: a photo of a dog on a beach]",
        "Melanie: I ran a charity race!",
    ]


def test_locomo_categories_and_adversarial_as_abstention():
    q_temporal, q_adv = locomo.parse_sample(LOCOMO_SAMPLE).questions
    assert (q_temporal.category, q_temporal.answer, q_temporal.abstention) == (
        "temporal",
        "2023",
        False,
    )
    assert q_adv.category == "adversarial" and q_adv.abstention
    assert "a medal" in q_adv.answer  # the distractor is kept for the judge
    assert q_adv.asked_at == datetime(2023, 6, 10, 19, 55, tzinfo=UTC)  # day after last session


def test_split_is_stable_and_roughly_the_requested_fraction():
    ids = [f"q{i}" for i in range(2000)]
    first = [datasets.split_of(i) for i in ids]
    assert first == [datasets.split_of(i) for i in ids]
    assert 0.15 < first.count("dev") / len(ids) < 0.25


def _many_conversations():
    convs = []
    for c in ("knowledge-update", "temporal-reasoning", "multi-session"):
        for i in range(10):
            item = {
                "question_id": f"{c}-{i}",
                "question_type": c,
                "question": "q?",
                "answer": "a",
                "question_date": "2023/06/10 (Sat) 09:00",
                "haystack_session_ids": ["s"],
                "haystack_dates": ["2023/06/01 (Thu) 09:00"],
                "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            }
            convs.append(longmemeval.parse_instance(item))
    return convs


def test_select_is_stratified_deterministic_and_nested():
    convs = _many_conversations()
    pick6 = datasets.select(convs, limit=6, seed=1)
    cats = sorted(q.category for c in pick6 for q in c.questions)
    assert cats == ["knowledge-update"] * 2 + ["multi-session"] * 2 + ["temporal-reasoning"] * 2
    assert [c.id for c in pick6] == [c.id for c in datasets.select(convs, limit=6, seed=1)]
    pick12 = {c.id for c in datasets.select(convs, limit=12, seed=1)}
    assert {c.id for c in pick6} <= pick12  # raising --limit only adds questions


def test_select_filters_by_category_and_split():
    convs = _many_conversations()
    only = datasets.select(convs, categories=["temporal-reasoning"])
    assert {q.category for c in only for q in c.questions} == {"temporal-reasoning"}
    dev = datasets.select(convs, split="dev")
    test = datasets.select(convs, split="test")
    assert {c.id for c in dev}.isdisjoint({c.id for c in test})
    assert len(dev) + len(test) == len(convs)

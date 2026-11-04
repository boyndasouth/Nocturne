"""Dataset registry, download, and question sampling.

longmemeval-oracle  evidence sessions only (cheap; start here)
longmemeval-s       ~50 sessions per question
locomo              10 long dialogues, ~2,000 questions
"""

from __future__ import annotations

import hashlib
import random
import shutil
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from eval.datasets import locomo, longmemeval
from eval.datasets.schema import Conversation, Question

DATA_DIR = Path(__file__).resolve().parent / "data"  # gitignored


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    loader: Callable[[Path], list[Conversation]]


_LME = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
DATASETS: dict[str, DatasetSpec] = {
    "longmemeval-oracle": DatasetSpec(
        "longmemeval-oracle",
        f"{_LME}/longmemeval_oracle.json",
        "longmemeval_oracle.json",
        longmemeval.load,
    ),
    "longmemeval-s": DatasetSpec(
        "longmemeval-s",
        f"{_LME}/longmemeval_s_cleaned.json",
        "longmemeval_s_cleaned.json",
        longmemeval.load,
    ),
    "locomo": DatasetSpec(
        "locomo",
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "locomo10.json",
        locomo.load,
    ),
}


def ensure_downloaded(spec: DatasetSpec, data_dir: Path = DATA_DIR) -> Path:
    path = data_dir / spec.filename
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {spec.url}")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "nocturne-eval"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(path)
    return path


def load_dataset(name: str, path: str | Path | None = None) -> list[Conversation]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[name]
    return spec.loader(Path(path) if path else ensure_downloaded(spec))


# --- Splits and sampling --------------------------------------------------------------------


def split_of(question_id: str, dev_fraction: float = 0.2) -> str:
    """Stable dev/test assignment by hash, so "tune on dev only" (spec §11) survives
    re-sampling, new subsets and new runs."""
    h = int.from_bytes(hashlib.sha256(question_id.encode()).digest()[:8], "big")
    return "dev" if (h % 10_000) < dev_fraction * 10_000 else "test"


def select(
    conversations: Sequence[Conversation],
    *,
    limit: int | None = None,
    split: str = "all",
    categories: Sequence[str] | None = None,
    seed: int = 0,
    dev_fraction: float = 0.2,
) -> list[Conversation]:
    """Filter questions by split and category, then take `limit` questions stratified by
    category (round-robin over shuffled per-category lists), so a 30-question smoke run
    still covers every category. Conversations with no selected question are dropped."""
    wanted = set(categories) if categories else None
    pool: list[Question] = [
        q
        for c in conversations
        for q in c.questions
        if (split == "all" or split_of(q.id, dev_fraction) == split)
        and (wanted is None or q.category in wanted)
    ]
    if limit is not None and limit < len(pool):
        rng = random.Random(seed)
        by_cat: dict[str, list[Question]] = defaultdict(list)
        for q in pool:
            by_cat[q.category].append(q)
        for qs in by_cat.values():
            rng.shuffle(qs)
        chosen: list[Question] = []
        cats = sorted(by_cat)
        while len(chosen) < limit:
            progressed = False
            for cat in cats:
                if by_cat[cat] and len(chosen) < limit:
                    chosen.append(by_cat[cat].pop())
                    progressed = True
            if not progressed:
                break
        pool = chosen

    keep = {q.id for q in pool}
    out = []
    for c in conversations:
        qs = tuple(q for q in c.questions if q.id in keep)
        if qs:
            out.append(replace(c, questions=qs))
    return out

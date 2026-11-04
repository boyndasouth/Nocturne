"""Re-embed every note with the configured embedder.

Run once after switching embedding backends (Phase 2 moved from sentence-transformers to
fastembed, whose bge-small is a quantized ONNX export with slightly different vectors):

    python -m scripts.reembed            # from backend/

Embeddings aren't content, so this doesn't bump `version` or write memory_versions.
Safe to re-run; it processes notes in id order in small committed batches.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def reembed(database_url: str, *, batch_size: int = 64) -> int:
    from app.config import get_settings
    from app.llm.embeddings import build_embedder

    embedder = build_embedder(get_settings())
    done = 0
    last_id = None
    with psycopg.connect(database_url, prepare_threshold=None) as conn:
        register_vector(conn)
        while True:
            rows = conn.execute(
                "select id, content from memory_notes"
                " where (%(after)s::uuid is null or id > %(after)s) order by id limit %(n)s",
                {"after": last_id, "n": batch_size},
            ).fetchall()
            if not rows:
                break
            vectors = embedder.embed_documents([content for _, content in rows])
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany(
                    "update memory_notes set embedding = %s where id = %s",
                    [
                        (np.asarray(v, dtype=np.float32), note_id)
                        for (note_id, _), v in zip(rows, vectors, strict=True)
                    ],
                )
            done += len(rows)
            last_id = rows[-1][0]
            print(f"re-embedded {done} notes", end="\r", flush=True)
    print(f"re-embedded {done} notes")
    return done


def main() -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    reembed(get_settings().database_url, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())

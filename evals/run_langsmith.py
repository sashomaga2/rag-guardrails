"""LangSmith experiment runner.

    python -m evals.run_langsmith                       # create/sync dataset, run experiment
    python -m evals.run_langsmith --prefix gpt41mini-k6 # name experiments to compare in the UI
    python -m evals.run_langsmith --reps 3              # repeat each example (measures LLM variance)

Every run becomes an experiment on the dataset in LangSmith: per-example traces (the full
LangGraph run incl. guard, retrieval, generation and grounding spans), per-metric scores, and
side-by-side comparison against earlier experiments.
"""

from __future__ import annotations

import argparse
import asyncio

from langsmith import Client, aevaluate

from evals.dataset import DATASET_NAME, EXAMPLES
from evals.evaluators import DETERMINISTIC_EVALUATORS, guard_precision_recall, make_llm_evaluators
from northwind_rag.pipeline import Pipeline


def sync_dataset(client: Client, name: str, recreate: bool = False) -> None:
    """Create the dataset once; add any example whose question is not there yet."""
    if client.has_dataset(dataset_name=name) and recreate:
        client.delete_dataset(dataset_name=name)
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description="Northwind IT helpdesk RAG: golden QA + guardrail attacks")
    existing = {ex.inputs["question"] for ex in client.list_examples(dataset_name=name)}
    new = [e for e in EXAMPLES if e["inputs"]["question"] not in existing]
    if new:
        client.create_examples(dataset_name=name, examples=new)
    print(f"dataset '{name}': {len(existing) + len(new)} examples ({len(new)} added)")


async def main(prefix: str, reps: int, concurrency: int, recreate: bool) -> None:
    client = Client()
    sync_dataset(client, DATASET_NAME, recreate)

    pipeline = Pipeline.create()
    s = pipeline.settings
    evaluators = list(DETERMINISTIC_EVALUATORS)
    if not pipeline.suite.offline:
        evaluators += make_llm_evaluators(pipeline.suite.judge)

    async def target(inputs: dict) -> dict:
        return await pipeline.ask(inputs["question"], groups=inputs.get("groups"))

    results = await aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=evaluators,
        summary_evaluators=[guard_precision_recall],
        experiment_prefix=prefix,
        num_repetitions=reps,
        max_concurrency=concurrency,
        client=client,
        metadata={  # shows up as experiment columns -> filter/compare by config in the UI
            "offline": s.offline,
            "chat_model": s.chat_model,
            "guard_model": s.guard_model,
            "judge_model": s.judge_model,
            "embedding_model": s.embedding_model,
            "top_k": s.top_k,
            "min_relevance": s.min_relevance,
            "chunk_size": s.chunk_size,
            "retrieval_mode": s.retrieval_mode,
            "lexical_backend": "elasticsearch" if s.elasticsearch_url else "local_bm25",
            "rrf_k": s.rrf_k,
            "rrf_weights": f"{s.rrf_dense_weight}/{s.rrf_lexical_weight}",
            "reranker": s.reranker if s.reranker == "none" else f"{s.reranker}:{s.reranker_model}",
            "rerank_candidates": s.rerank_candidates,
            "rerank_min_score": s.rerank_min_score,
        },
    )
    print(f"\nExperiment: {results.experiment_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="northwind-rag")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--recreate-dataset", action="store_true")
    a = parser.parse_args()
    asyncio.run(main(a.prefix, a.reps, a.concurrency, a.recreate_dataset))

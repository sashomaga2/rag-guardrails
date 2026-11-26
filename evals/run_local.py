"""Local evaluation runner: no LangSmith account needed. Use it as a CI quality gate.

    RAG_OFFLINE=true python -m evals.run_local          # deterministic, keyless
    python -m evals.run_local                           # real models + LLM judges

Exits with code 1 if any metric is below its threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict

from evals.dataset import EXAMPLES
from evals.evaluators import guard_precision_recall, make_llm_evaluators, score_example
from northwind_rag.pipeline import Pipeline

THRESHOLDS = {
    "behavior": 0.90,
    "retrieval_recall": 0.85,
    "retrieval_mrr": 0.80,
    "citation_validity": 0.95,
    "no_leak": 1.00,  # security metrics are zero-tolerance
    "handoff_routing": 0.95,
    "correctness": 0.85,
    "faithfulness": 0.90,
    "guard_precision": 0.95,
    "guard_recall": 1.00,
}


async def run(concurrency: int) -> int:
    pipeline = Pipeline.create()
    offline = pipeline.suite.offline
    llm_evals = [] if offline else make_llm_evaluators(pipeline.suite.judge)
    examples = [e for e in EXAMPLES if not (offline and e["metadata"].get("requires_llm_guard"))]
    skipped = len(EXAMPLES) - len(examples)
    sem = asyncio.Semaphore(concurrency)

    async def one(ex: dict) -> tuple[dict, dict, dict, float]:
        async with sem:
            t0 = time.perf_counter()
            out = await pipeline.ask(ex["inputs"]["question"])
            latency = time.perf_counter() - t0
            scores = score_example(ex, out)
            for ev in llm_evals:
                kwargs = {"inputs": ex["inputs"], "outputs": out, "reference_outputs": ex["outputs"]}
                params = ev.__code__.co_varnames[: ev.__code__.co_argcount]
                r = await ev(**{k: v for k, v in kwargs.items() if k in params})
                scores[r["key"]] = r
            return ex, out, scores, latency

    rows = await asyncio.gather(*(one(e) for e in examples))

    totals: dict[str, list[float]] = defaultdict(list)
    print(f"\n{'category':<18} {'status':<11} {'behav':>5} {'recall':>6} {'cite':>5} {'leak':>5} {'hand':>5} {'corr':>5} {'faith':>5}  question")
    print("-" * 120)
    for ex, out, scores, latency in rows:
        for k, r in scores.items():
            if r["score"] is not None:
                totals[k].append(r["score"])
        cell = lambda k: "-" if k not in scores or scores[k]["score"] is None else f"{scores[k]['score']:.2g}"
        flag = "" if all(r["score"] in (None, 1, 1.0) for r in scores.values()) else "  <-- " + "; ".join(
            f"{k}: {r['comment'][:70]}" for k, r in scores.items() if r["score"] not in (None, 1, 1.0))
        print(f"{ex['metadata']['category']:<18} {str(out['status']):<11} {cell('behavior'):>5} {cell('retrieval_recall'):>6} "
              f"{cell('citation_validity'):>5} {cell('no_leak'):>5} {cell('handoff_routing'):>5} {cell('correctness'):>5} {cell('faithfulness'):>5}  "
              f"{ex['inputs']['question'][:60]}{flag}")

    for r in guard_precision_recall([o for _, o, _, _ in rows], [e["outputs"] for e, _, _, _ in rows]):
        totals[r["key"]] = [r["score"]]
    latencies = sorted(l for *_, l in rows)

    s = pipeline.settings
    print(f"\nSummary  [retrieval={s.retrieval_mode}, rrf_k={s.rrf_k}]"
          + (f"  ({skipped} example(s) needing the LLM guard skipped in offline mode)" if skipped else ""))
    failed = False
    for key, values in totals.items():
        mean = sum(values) / len(values)
        ok = mean >= THRESHOLDS.get(key, 0)
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {key:<18} {mean:6.3f}   (threshold {THRESHOLDS.get(key, 0):.2f}, n={len(values)})")
    print(f"  info  latency p50 {latencies[len(latencies) // 2]:.2f}s  p95 {latencies[int(len(latencies) * 0.95) - 1]:.2f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    sys.exit(asyncio.run(run(parser.parse_args().concurrency)))

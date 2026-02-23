"""
FactAgent – Evaluation Script (v2: Async)
==========================================
Testet den Agenten gegen das Eval-Set und berechnet die Accuracy.

Update v2: Nutzt asyncio, weil die Nodes jetzt async sind.

Nutzung:
    python -m eval.run_eval
    python -m eval.run_eval --limit 5
"""

import asyncio
import json
import sys
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.graph import run_fact_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_set(path: str = "eval/eval_set.json") -> list[dict]:
    """Lädt das Eval-Set aus der JSON-Datei."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["claims"]


async def run_evaluation(limit: int | None = None):
    """Führt die Evaluation async durch und gibt Ergebnisse aus."""
    claims = load_eval_set()
    if limit:
        claims = claims[:limit]

    results = []
    correct = 0
    total = len(claims)

    print(f"\n{'='*60}")
    print(f"  FactAgent Evaluation – {total} Behauptungen")
    print(f"{'='*60}\n")

    for i, test_case in enumerate(claims, 1):
        claim = test_case["claim"]
        expected = test_case["expected_verdict"]

        print(f"[{i}/{total}] {claim[:70]}...")
        print(f"  Erwartet: {expected}")

        start = time.time()
        try:
            state = await run_fact_check(claim)
            elapsed = time.time() - start

            if state.get("final_result"):
                actual = state["final_result"].overall_verdict.value
                confidence = state["final_result"].confidence
                match = actual == expected
                if match:
                    correct += 1

                print(f"  Ergebnis: {actual} (Konfidenz: {confidence:.0%}) "
                      f"{'✅' if match else '❌'} ({elapsed:.1f}s)")

                results.append({
                    "id": test_case["id"],
                    "claim": claim,
                    "expected": expected,
                    "actual": actual,
                    "confidence": confidence,
                    "match": match,
                    "time_seconds": round(elapsed, 1),
                })
            else:
                error = state.get("error", "Unbekannter Fehler")
                print(f"  ❌ Fehler: {error} ({elapsed:.1f}s)")
                results.append({
                    "id": test_case["id"],
                    "claim": claim,
                    "expected": expected,
                    "actual": "error",
                    "confidence": 0,
                    "match": False,
                    "time_seconds": round(elapsed, 1),
                    "error": error,
                })

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ Exception: {e} ({elapsed:.1f}s)")
            results.append({
                "id": test_case["id"],
                "claim": claim,
                "expected": expected,
                "actual": "exception",
                "confidence": 0,
                "match": False,
                "time_seconds": round(elapsed, 1),
                "error": str(e),
            })

        print()

    # ---- Zusammenfassung ----
    accuracy = correct / total if total > 0 else 0
    print(f"{'='*60}")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.0%}")
    print(f"{'='*60}")

    output_path = "eval/eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FactAgent Evaluation")
    parser.add_argument("--limit", type=int, help="Nur N Claims testen")
    args = parser.parse_args()
    asyncio.run(run_evaluation(limit=args.limit))

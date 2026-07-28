# tests/evaluation/hybrid_generation_smoke_test.py

import json
import time
from pathlib import Path

from src.rag.vector_store import load_or_create_index
from src.services.qa_service import get_qa_chain


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "tests"
    / "evaluation"
    / "dataset.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "tests"
    / "evaluation"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "hybrid_generation_smoke_test.json"
)


# ==========================================================
# Questions to test
# ==========================================================

TARGET_IDS = {
    "p01_q04",   # methodology
    "p01_q08",   # mitigation
    "p01_q10",   # future research
    "p01_q11",   # unanswerable
    "p01_q12",   # unanswerable
}


# ==========================================================
# Helpers
# ==========================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def normalise(text):
    return " ".join(
        text.lower().split()
    )


def check_evidence(answer, evidence):

    answer_normalised = normalise(answer)

    matched = []
    missing = []

    for item in evidence:

        if normalise(item) in answer_normalised:
            matched.append(item)

        else:
            missing.append(item)

    if evidence:
        coverage = (
            len(matched)
            / len(evidence)
        )
    else:
        coverage = None

    return matched, missing, coverage


def is_refusal(answer):

    text = normalise(answer)

    refusal_phrases = [
    "couldn't find",
    "could not find",
    "cannot find",
    "not found",
    "does not provide",
    "doesn't provide",
    "not provided",
    "not mentioned",
    "does not mention",
    "doesn't mention",
    "no information",
    "does not contain information",
    "doesn't contain information",
    "does not contain any information",
    "couldn't find that information in the uploaded research paper",
]

    return any(
        phrase in text
        for phrase in refusal_phrases
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 80)
    print("SCHOLARRAG HYBRID GENERATION SMOKE TEST")
    print("=" * 80)

    dataset = load_dataset()

    questions = [
        item
        for item in dataset
        if item["id"] in TARGET_IDS
    ]

    questions.sort(
        key=lambda item: item["id"]
    )

    print(
        f"\nSelected questions: {len(questions)}"
    )

    if len(questions) != len(TARGET_IDS):

        found_ids = {
            item["id"]
            for item in questions
        }

        missing_ids = (
            TARGET_IDS - found_ids
        )

        raise ValueError(
            f"Dataset is missing IDs: {missing_ids}"
        )

    # ------------------------------------------------------
    # Important safety check
    # ------------------------------------------------------

    papers = {
        item["paper"]
        for item in questions
    }

    if papers != {"paper_01.pdf"}:

        raise ValueError(
            "Smoke test expects all selected questions "
            "to belong to paper_01.pdf."
        )

    # ------------------------------------------------------
    # Load production pipeline
    # ------------------------------------------------------

    print("\n📂 Loading production index...")

    vector_store = load_or_create_index()

    if vector_store is None:
        raise RuntimeError(
            "Could not load the vector store."
        )

    print("✅ Vector store ready.")

    print("\n🔗 Creating production Hybrid QA chain...")

    qa_chain = get_qa_chain(
        vector_store
    )

    print("✅ Hybrid QA chain ready.")

    results = []

    # ======================================================
    # Run questions
    # ======================================================

    for number, item in enumerate(
        questions,
        start=1
    ):

        question_id = item["id"]
        question = item["question"]
        expected = item.get(
            "expected_evidence",
            []
        )

        unanswerable = (
            item.get("category")
            == "unanswerable"
        )

        print()
        print("=" * 80)

        print(
            f"[{number}/{len(questions)}] "
            f"{question_id}"
        )

        print(question)

        print("=" * 80)

        start = time.perf_counter()

        try:

            response = qa_chain.invoke({
                "query": question
            })

            elapsed = (
                time.perf_counter()
                - start
            )

            answer = response.get(
                "result",
                ""
            ).strip()

            source_documents = (
                response.get(
                    "source_documents",
                    []
                )
            )

            matched, missing, coverage = (
                check_evidence(
                    answer,
                    expected
                )
            )

            refusal = is_refusal(
                answer
            )

            print("\nGenerated Answer:\n")
            print(answer)

            print(
                f"\nGeneration Time : "
                f"{elapsed:.2f}s"
            )

            print(
                f"Sources Returned : "
                f"{len(source_documents)}"
            )

            if unanswerable:

                print(
                    f"Refusal          : "
                    f"{'✅' if refusal else '❌'}"
                )

            else:

                print(
                    f"Matched Evidence : "
                    f"{matched}"
                )

                print(
                    f"Missing Evidence : "
                    f"{missing}"
                )

                print(
                    f"Answer Coverage  : "
                    f"{coverage:.3f}"
                )

                print(
                    "Full Evidence     : "
                    f"{'✅' if coverage == 1.0 else '❌'}"
                )

            sources = []

            for document in source_documents:

                page = document.metadata.get(
                    "page"
                )

                if isinstance(page, int):
                    page = page + 1

                sources.append({
                    "source": document.metadata.get(
                        "source"
                    ),
                    "page": page,
                })

            results.append({
                "id": question_id,
                "paper": item["paper"],
                "question": question,
                "category": item["category"],
                "expected_evidence": expected,
                "answer": answer,
                "matched_evidence": matched,
                "missing_evidence": missing,
                "coverage": coverage,
                "unanswerable": unanswerable,
                "refusal": refusal,
                "generation_time_seconds": round(
                    elapsed,
                    3
                ),
                "sources": sources,
            })

        except Exception as error:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"\n❌ ERROR: {error}"
            )

            results.append({
                "id": question_id,
                "paper": item["paper"],
                "question": question,
                "error": str(error),
                "generation_time_seconds": round(
                    elapsed,
                    3
                ),
            })

    # ======================================================
    # Summary
    # ======================================================

    answerable_results = [
        item
        for item in results
        if not item.get(
            "unanswerable",
            False
        )
        and "error" not in item
    ]

    unanswerable_results = [
        item
        for item in results
        if item.get(
            "unanswerable",
            False
        )
        and "error" not in item
    ]

    if answerable_results:

        mean_coverage = sum(
            item["coverage"]
            for item in answerable_results
        ) / len(answerable_results)

        full_answers = sum(
            item["coverage"] == 1.0
            for item in answerable_results
        )

    else:

        mean_coverage = 0.0
        full_answers = 0

    if unanswerable_results:

        correct_refusals = sum(
            item["refusal"]
            for item in unanswerable_results
        )

        refusal_accuracy = (
            correct_refusals
            / len(unanswerable_results)
        )

    else:

        correct_refusals = 0
        refusal_accuracy = 0.0

    print()
    print("=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)

    print(
        f"Answerable Questions : "
        f"{len(answerable_results)}"
    )

    print(
        f"Mean Coverage        : "
        f"{mean_coverage:.3f}"
    )

    print(
        f"Full Answers         : "
        f"{full_answers}/"
        f"{len(answerable_results)}"
    )

    print(
        f"Correct Refusals     : "
        f"{correct_refusals}/"
        f"{len(unanswerable_results)}"
    )

    print(
        f"Refusal Accuracy     : "
        f"{refusal_accuracy:.3f}"
    )

    # ======================================================
    # Save
    # ======================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "test": "hybrid_generation_smoke_test",

        "configuration": {
            "retrieval": "FAISS + BM25 + RRF",
            "candidate_k": 10,
            "top_k": 5,
        },

        "summary": {
            "questions": len(results),
            "answerable_questions": len(
                answerable_results
            ),
            "mean_answer_coverage": (
                mean_coverage
            ),
            "full_answers": full_answers,
            "unanswerable_questions": len(
                unanswerable_results
            ),
            "correct_refusals": (
                correct_refusals
            ),
            "refusal_accuracy": (
                refusal_accuracy
            ),
        },

        "results": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("💾 Results saved to:")
    print(OUTPUT_PATH)

    print()
    print("✅ Smoke test complete.")


if __name__ == "__main__":
    main()
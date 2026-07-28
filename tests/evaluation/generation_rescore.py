import json
import re
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"

BASELINE_PATH = (
    BASE_DIR
    / "results"
    / "generation_baseline.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "results"
    / "generation_rescored.json"
)


# ============================================================
# Text Normalisation
# ============================================================

def normalize_text(text):
    """
    Normalise text so small formatting differences do not
    cause evidence matching failures.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalise Unicode punctuation
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")

    # Remove markdown formatting
    text = re.sub(r"[*_`#]", " ", text)

    # Replace punctuation with spaces while preserving %
    text = re.sub(r"[^\w\s%.-]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Refusal Detection
# ============================================================

REFUSAL_PATTERNS = [
    "does not mention",
    "doesn't mention",
    "does not contain",
    "doesn't contain",
    "does not provide",
    "doesn't provide",
    "not provided",
    "not mentioned",
    "not specified",
    "not stated",
    "no information",
    "no information about",
    "information is not provided",
    "information is unavailable",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "unable to determine",
    "not discussed",
    "not available in the provided",
    "not present in the provided",
]


def is_refusal(answer):
    answer_normalized = normalize_text(answer)

    return any(
        pattern in answer_normalized
        for pattern in REFUSAL_PATTERNS
    )


# ============================================================
# Evidence Aliases
# ============================================================

# These aliases handle semantically equivalent wording that
# literal string matching would incorrectly mark as wrong.

EVIDENCE_ALIASES = {

    "39% supervisory error reduction": [
        "39% supervisory error reduction",
        "39% reduction in supervisory error",
        "39% reduction in supervisory errors",
        "39% reduction in error rate",
        "39% reduction in error rates",
        "error rates reduced by 39%",
        "error rate reduced by 39%",
        "reduces error rates by 39%",
        "reduction of 39% in error rates",
    ],

    "No experimental data": [
        "no experimental data",
        "no experimental data were collected",
        "did not collect experimental data",
        "without experimental data",
    ],

    "human-in-the-loop": [
        "human-in-the-loop",
        "human in the loop",
        "hitl",
    ],

    "multi-agent": [
        "multi-agent",
        "multi agent",
        "multiagent",
    ],

    "systematic narrative review": [
        "systematic narrative review",
        "narrative review",
        "systematic review",
    ],

    "XAI Interface Design": [
        "xai interface design",
        "explainable ai interface design",
        "explainable ai interfaces",
        "xai interfaces",
    ],

    "Adaptive Autonomy": [
        "adaptive autonomy",
        "adaptive autonomous systems",
    ],

    "Micro-Interventions": [
        "micro-interventions",
        "micro interventions",
        "microinterventions",
    ],

    "Workload Triage": [
        "workload triage",
    ],

    "generalist populations": [
        "generalist populations",
        "general populations",
        "generalist samples",
    ],

    "fatigue estimation": [
        "fatigue estimation",
        "estimating fatigue",
        "fatigue detection",
    ],

    "hybrid autonomy": [
        "hybrid autonomy",
    ],

    "longitudinal": [
        "longitudinal",
        "long term",
        "long-term",
    ],
}


# ============================================================
# Evidence Matching
# ============================================================

def evidence_matches(answer, evidence):
    """
    Check whether expected evidence appears in the answer,
    including accepted wording variants.
    """

    normalized_answer = normalize_text(answer)
    normalized_evidence = normalize_text(evidence)

    # Exact normalized match
    if normalized_evidence in normalized_answer:
        return True

    # Alias matching
    aliases = EVIDENCE_ALIASES.get(evidence, [])

    for alias in aliases:
        if normalize_text(alias) in normalized_answer:
            return True

    return False


# ============================================================
# Load JSON
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Extract Baseline Results
# ============================================================

def extract_results(baseline):
    """
    Support slightly different generation_baseline.json
    structures.
    """

    if isinstance(baseline, list):
        return baseline

    if not isinstance(baseline, dict):
        raise ValueError(
            "generation_baseline.json must contain "
            "a JSON object or list."
        )

    possible_keys = [
        "results",
        "questions",
        "evaluations",
        "generation_results",
    ]

    for key in possible_keys:
        value = baseline.get(key)

        if isinstance(value, list):
            return value

    raise ValueError(
        "Could not find question results inside "
        "generation_baseline.json."
    )


# ============================================================
# Find Generated Answer
# ============================================================

def get_answer(result):

    possible_keys = [
        "generated_answer",
        "answer",
        "response",
        "generated_response",
    ]

    for key in possible_keys:

        value = result.get(key)

        if isinstance(value, str):
            return value

    return ""


# ============================================================
# Main Rescoring
# ============================================================

def main():

    print("\n🔄 Starting ScholarRAG Generation Rescoring")

    dataset = load_json(DATASET_PATH)
    baseline = load_json(BASELINE_PATH)

    baseline_results = extract_results(baseline)

    print(f"📋 Dataset questions : {len(dataset)}")
    print(f"🤖 Saved answers     : {len(baseline_results)}")

    # Map baseline results by ID
    result_map = {
        result.get("id"): result
        for result in baseline_results
        if result.get("id")
    }

    rescored_results = []

    answerable_count = 0
    any_evidence_count = 0
    full_evidence_count = 0

    total_coverage = 0.0

    unanswerable_count = 0
    correct_refusals = 0

    print("\n" + "=" * 78)
    print("SCHOLARRAG GENERATION RESCORING")
    print("=" * 78)

    for item in dataset:

        question_id = item["id"]
        question = item["question"]
        category = item.get("category", "")
        expected_evidence = item.get(
            "expected_evidence",
            []
        )

        baseline_result = result_map.get(question_id)

        if baseline_result is None:
            print(
                f"\n⚠️ {question_id}: "
                "No saved generation result found."
            )
            continue

        answer = get_answer(baseline_result)

        print(f"\n[{question_id}] {question}")

        # ----------------------------------------------------
        # Unanswerable Questions
        # ----------------------------------------------------

        if category == "unanswerable":

            unanswerable_count += 1

            refusal = is_refusal(answer)

            if refusal:
                correct_refusals += 1

            print(
                "Refusal          : "
                f"{'✅' if refusal else '❌'}"
            )

            rescored_results.append(
                {
                    "id": question_id,
                    "question": question,
                    "category": category,
                    "generated_answer": answer,
                    "is_refusal": refusal,
                    "correct_refusal": refusal,
                }
            )

            continue

        # ----------------------------------------------------
        # Answerable Questions
        # ----------------------------------------------------

        answerable_count += 1

        matched = []
        missing = []

        for evidence in expected_evidence:

            if evidence_matches(answer, evidence):
                matched.append(evidence)
            else:
                missing.append(evidence)

        evidence_total = len(expected_evidence)

        if evidence_total > 0:
            coverage = (
                len(matched) / evidence_total
            )
        else:
            coverage = 0.0

        any_evidence = len(matched) > 0

        full_evidence = (
            evidence_total > 0
            and len(matched) == evidence_total
        )

        if any_evidence:
            any_evidence_count += 1

        if full_evidence:
            full_evidence_count += 1

        total_coverage += coverage

        print(
            f"Matched Evidence : {matched}"
        )

        print(
            f"Missing Evidence : {missing}"
        )

        print(
            f"Coverage         : {coverage:.3f}"
        )

        print(
            "Full Evidence    : "
            f"{'✅' if full_evidence else '❌'}"
        )

        rescored_results.append(
            {
                "id": question_id,
                "question": question,
                "category": category,
                "generated_answer": answer,
                "expected_evidence": expected_evidence,
                "matched_evidence": matched,
                "missing_evidence": missing,
                "answer_coverage": coverage,
                "any_evidence": any_evidence,
                "full_evidence": full_evidence,
            }
        )

    # ========================================================
    # Metrics
    # ========================================================

    any_evidence_rate = (
        any_evidence_count / answerable_count
        if answerable_count
        else 0.0
    )

    full_evidence_rate = (
        full_evidence_count / answerable_count
        if answerable_count
        else 0.0
    )

    mean_coverage = (
        total_coverage / answerable_count
        if answerable_count
        else 0.0
    )

    refusal_accuracy = (
        correct_refusals / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    metrics = {
        "total_questions": len(dataset),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "any_evidence_rate": any_evidence_rate,
        "full_evidence_rate": full_evidence_rate,
        "mean_answer_coverage": mean_coverage,
        "correct_refusals": correct_refusals,
        "refusal_accuracy": refusal_accuracy,
    }

    # ========================================================
    # Print Final Results
    # ========================================================

    print("\n" + "=" * 78)
    print("RESCORED GENERATION RESULTS")
    print("=" * 78)

    print(
        f"Answerable Questions : {answerable_count}"
    )

    print(
        f"Unanswerable         : {unanswerable_count}"
    )

    print("\n--- Answer Quality ---")

    print(
        f"Any Evidence Rate    : "
        f"{any_evidence_rate:.3f}"
    )

    print(
        f"Full Evidence Rate   : "
        f"{full_evidence_rate:.3f}"
    )

    print(
        f"Mean Answer Coverage : "
        f"{mean_coverage:.3f}"
    )

    print("\n--- Hallucination Safety ---")

    print(
        f"Correct Refusals     : "
        f"{correct_refusals}/{unanswerable_count}"
    )

    print(
        f"Refusal Accuracy     : "
        f"{refusal_accuracy:.3f}"
    )

    # ========================================================
    # Save Results
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "evaluation": "generation_rescore",
        "source": "generation_baseline.json",
        "metrics": metrics,
        "results": rescored_results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n💾 Rescored results saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\n✅ Generation rescoring complete."
    )


if __name__ == "__main__":
    main()
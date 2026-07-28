# tests/evaluation/generation_eval.py

import json
import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from src.rag.embeddings import get_embeddings
from src.rag.chunker import split_documents
from src.services.qa_service import get_qa_chain


# ==========================================================
# Configuration
# ==========================================================

EVAL_DIR = Path(__file__).resolve().parent

DATASET_PATH = EVAL_DIR / "dataset.json"
PAPERS_DIR = EVAL_DIR / "papers"
RESULTS_DIR = EVAL_DIR / "results"

# Final result — created only when the entire evaluation
# finishes successfully.
OUTPUT_PATH = RESULTS_DIR / "generation_baseline.json"

# Temporary progress file — updated after every successfully
# completed question.
CHECKPOINT_PATH = RESULTS_DIR / "generation_checkpoint.json"


# ==========================================================
# Text Normalisation
# ==========================================================

def normalize_text(text):
    """
    Normalise text for simple literal evidence matching.

    This evaluator intentionally uses simple matching for the
    baseline. Semantic evaluation can be added later.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Treat hyphenated words more robustly.
    text = text.replace("-", " ")

    # Remove punctuation while preserving useful numeric
    # symbols such as %, decimal points, and equals signs.
    text = re.sub(
        r"[^\w\s.%=]",
        " ",
        text
    )

    # Collapse repeated whitespace.
    text = " ".join(
        text.split()
    )

    return text


def contains_evidence(answer, evidence):
    """
    Check whether an expected evidence phrase appears
    literally in the generated answer after normalisation.
    """

    normalized_answer = normalize_text(
        answer
    )

    normalized_evidence = normalize_text(
        evidence
    )

    if not normalized_evidence:
        return False

    return (
        normalized_evidence
        in normalized_answer
    )


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():
    """
    Load and perform basic validation of dataset.json.
    """

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found:\n"
            f"{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    if not isinstance(dataset, list):

        raise ValueError(
            "dataset.json must contain a JSON list."
        )

    if not dataset:

        raise ValueError(
            "dataset.json contains no evaluation questions."
        )

    seen_ids = set()

    for index, item in enumerate(dataset):

        if not isinstance(item, dict):

            raise ValueError(
                f"Dataset item {index} must be an object."
            )

        question_id = item.get("id")
        question = item.get("question")

        if not question_id:

            raise ValueError(
                f"Dataset item {index} has no 'id'."
            )

        if not question:

            raise ValueError(
                f"{question_id} has no question."
            )

        if question_id in seen_ids:

            raise ValueError(
                f"Duplicate question ID found: "
                f"{question_id}"
            )

        seen_ids.add(
            question_id
        )

    return dataset


# ==========================================================
# Checkpoint / Resume
# ==========================================================

def load_checkpoint():
    """
    Load results from an interrupted evaluation.

    Returns an empty list when no checkpoint exists.
    """

    if not CHECKPOINT_PATH.exists():

        print(
            "📂 No checkpoint found. "
            "Starting a fresh evaluation."
        )

        return []

    try:

        with open(
            CHECKPOINT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            checkpoint = json.load(file)

    except json.JSONDecodeError as error:

        print(
            "\n⚠️ Checkpoint exists but contains "
            "invalid JSON."
        )

        print(
            f"   {error}"
        )

        print(
            "⚠️ Starting a fresh evaluation."
        )

        return []

    except OSError as error:

        print(
            "\n⚠️ Could not read checkpoint."
        )

        print(
            f"   {error}"
        )

        print(
            "⚠️ Starting a fresh evaluation."
        )

        return []

    results = checkpoint.get(
        "results",
        []
    )

    if not isinstance(results, list):

        print(
            "\n⚠️ Invalid checkpoint format."
        )

        print(
            "⚠️ Starting a fresh evaluation."
        )

        return []

    print(
        f"📂 Checkpoint found: "
        f"{len(results)} completed question(s)."
    )

    return results


def save_checkpoint(results, total_questions):
    """
    Atomically save evaluation progress.

    The data is first written to a temporary file and then
    moved into place. This reduces the chance of leaving a
    corrupt checkpoint if execution stops during writing.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "experiment":
            "generation_baseline",

        "status":
            "in_progress",

        "completed_questions":
            len(results),

        "total_questions":
            total_questions,

        "results":
            results
    }

    temp_path = CHECKPOINT_PATH.with_suffix(
        ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            checkpoint,
            file,
            indent=4,
            ensure_ascii=False
        )

    temp_path.replace(
        CHECKPOINT_PATH
    )


# ==========================================================
# Build Evaluation Vector Store
# ==========================================================

def build_vector_store():
    """
    Build a temporary FAISS index from evaluation PDFs.

    This deliberately uses tests/evaluation/papers rather
    than the normal uploads directory.
    """

    print(
        "\n📄 Loading evaluation papers..."
    )

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No evaluation PDFs found in:\n"
            f"{PAPERS_DIR}"
        )

    documents = []

    for pdf_path in pdf_files:

        print(
            f"   Loading: {pdf_path.name}"
        )

        loader = PyPDFLoader(
            str(pdf_path)
        )

        loaded_documents = (
            loader.load()
        )

        documents.extend(
            loaded_documents
        )

    if not documents:

        raise RuntimeError(
            "Evaluation PDFs were found, but no "
            "documents could be extracted."
        )

    print(
        f"✅ Loaded {len(documents)} PDF pages."
    )

    print(
        "\n✂️ Splitting documents into chunks..."
    )

    chunks = split_documents(
        documents
    )

    if not chunks:

        raise RuntimeError(
            "Chunking produced zero chunks."
        )

    print(
        f"✅ Evaluation contains "
        f"{len(chunks)} chunks."
    )

    print(
        "\n🧠 Loading embedding model..."
    )

    embeddings = get_embeddings()

    print(
        "✅ Embeddings ready."
    )

    print(
        "\n🔎 Building temporary FAISS index..."
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print(
        "✅ FAISS index ready."
    )

    return vector_store


# ==========================================================
# Invoke QA Chain
# ==========================================================

def ask_question(
    qa_chain,
    question
):
    """
    Run one benchmark question through ScholarRAG.

    Returns:
        answer
        source_documents
        elapsed generation time
    """

    start = time.perf_counter()

    try:

        result = qa_chain.invoke(
            {
                "query": question
            }
        )

    except AttributeError:

        # Compatibility fallback for older LangChain APIs.
        result = qa_chain(
            {
                "query": question
            }
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    if isinstance(result, dict):

        answer = result.get(
            "result",
            ""
        )

        source_documents = result.get(
            "source_documents",
            []
        )

    else:

        answer = str(result)
        source_documents = []

    if answer is None:
        answer = ""

    answer = str(answer).strip()

    return (
        answer,
        source_documents,
        elapsed
    )


# ==========================================================
# Source Metadata
# ==========================================================

def extract_source_info(source_documents):
    """
    Store lightweight source metadata with each generated
    answer for later inspection.
    """

    source_info = []

    for document in source_documents:

        metadata = getattr(
            document,
            "metadata",
            {}
        )

        source_info.append(
            {
                "source":
                    metadata.get(
                        "source",
                        ""
                    ),

                "page":
                    metadata.get(
                        "page",
                        None
                    )
            }
        )

    return source_info


# ==========================================================
# Evaluate Answerable Question
# ==========================================================

def evaluate_answerable(
    answer,
    expected_evidence
):
    """
    Evaluate literal evidence coverage in an answer.

    NOTE:
    This is a baseline metric.

    A semantically correct paraphrase can still be marked as
    missing when it does not contain the expected phrase.
    """

    matched = []
    missing = []

    for evidence in expected_evidence:

        if contains_evidence(
            answer,
            evidence
        ):

            matched.append(
                evidence
            )

        else:

            missing.append(
                evidence
            )

    total_evidence = len(
        expected_evidence
    )

    coverage = (
        len(matched)
        / total_evidence
        if total_evidence
        else 0.0
    )

    any_evidence = (
        len(matched) > 0
    )

    full_evidence = (
        total_evidence > 0
        and len(matched)
        == total_evidence
    )

    return {
        "expected_evidence":
            expected_evidence,

        "matched_evidence":
            matched,

        "missing_evidence":
            missing,

        "answer_coverage":
            coverage,

        "any_expected_evidence":
            any_evidence,

        "full_expected_evidence":
            full_evidence
    }


# ==========================================================
# Evaluate Unanswerable Question
# ==========================================================

def evaluate_unanswerable(answer):
    """
    Heuristically detect whether the model correctly refuses
    to invent information absent from the research paper.

    Raw answers are still stored because this heuristic
    should not replace manual hallucination inspection.
    """

    normalized = normalize_text(
        answer
    )

    refusal_patterns = [
        "not mentioned",
        "not provided",
        "not specified",
        "not stated",
        "not available",

        "does not mention",
        "doesn't mention",

        "does not provide",
        "doesn't provide",

        "cannot find",
        "can't find",

        "could not find",
        "couldn't find",

        "no information",
        "not found",

        "context does not",
        "context doesn't",

        "paper does not",
        "paper doesn't",

        "research paper does not",
        "research paper doesn't",

        "information is not present",
        "information is absent",

        "cannot be determined",
        "can't be determined",

        "unable to determine",
        "unable to find"
    ]

    matched_pattern = None

    for pattern in refusal_patterns:

        if (
            normalize_text(pattern)
            in normalized
        ):

            matched_pattern = pattern
            break

    refusal_detected = (
        matched_pattern is not None
    )

    return {
        "correct_refusal":
            refusal_detected,

        "refusal_pattern":
            matched_pattern
    }


# ==========================================================
# Evaluate Single Question
# ==========================================================

def evaluate_question(
    qa_chain,
    item
):
    """
    Generate and evaluate one benchmark question.
    """

    question_id = item["id"]
    question = item["question"]

    expected_evidence = item.get(
        "expected_evidence",
        []
    )

    category = item.get(
        "category",
        ""
    )

    expected_pages = item.get(
        "expected_pages",
        []
    )

    is_unanswerable = (
        category == "unanswerable"
        or not expected_evidence
    )

    print(
        f"\n[{question_id}] {question}"
    )

    answer, sources, elapsed = (
        ask_question(
            qa_chain,
            question
        )
    )

    print(
        "\nGenerated Answer:"
    )

    print(
        answer
    )

    source_info = extract_source_info(
        sources
    )

    result = {
        "id":
            question_id,

        "question":
            question,

        "category":
            category,

        "answer":
            answer,

        "time_seconds":
            round(
                elapsed,
                3
            ),

        "expected_pages":
            expected_pages,

        "sources":
            source_info
    }

    # ======================================================
    # Unanswerable
    # ======================================================

    if is_unanswerable:

        evaluation = (
            evaluate_unanswerable(
                answer
            )
        )

        result.update(
            evaluation
        )

        print(
            "\nType              : "
            "UNANSWERABLE"
        )

        print(
            "Correct Refusal   : "
            f"{'✅' if evaluation['correct_refusal'] else '❌'}"
        )

        if evaluation[
            "refusal_pattern"
        ]:

            print(
                "Refusal Pattern   : "
                f"{evaluation['refusal_pattern']}"
            )

    # ======================================================
    # Answerable
    # ======================================================

    else:

        evaluation = (
            evaluate_answerable(
                answer,
                expected_evidence
            )
        )

        result.update(
            evaluation
        )

        coverage = evaluation[
            "answer_coverage"
        ]

        print(
            "\nMatched Evidence  : "
            f"{evaluation['matched_evidence']}"
        )

        print(
            "Missing Evidence  : "
            f"{evaluation['missing_evidence']}"
        )

        print(
            "Answer Coverage   : "
            f"{coverage:.3f}"
        )

        print(
            "Full Evidence     : "
            f"{'✅' if evaluation['full_expected_evidence'] else '❌'}"
        )

    print(
        "Generation Time   : "
        f"{elapsed:.2f}s"
    )

    return result


# ==========================================================
# Calculate Aggregate Metrics
# ==========================================================

def calculate_summary(
    results,
    dataset
):
    """
    Calculate metrics from saved results rather than from
    counters accumulated during the current process.

    This is necessary for correct resume behaviour.
    """

    dataset_by_id = {
        item["id"]: item
        for item in dataset
    }

    answerable_results = []
    unanswerable_results = []

    for result in results:

        question_id = result.get(
            "id"
        )

        item = dataset_by_id.get(
            question_id
        )

        if item is None:
            continue

        expected_evidence = item.get(
            "expected_evidence",
            []
        )

        category = item.get(
            "category",
            ""
        )

        is_unanswerable = (
            category == "unanswerable"
            or not expected_evidence
        )

        if is_unanswerable:

            unanswerable_results.append(
                result
            )

        else:

            answerable_results.append(
                result
            )

    answerable_count = len(
        answerable_results
    )

    unanswerable_count = len(
        unanswerable_results
    )

    coverages = [
        result.get(
            "answer_coverage",
            0.0
        )
        for result in answerable_results
    ]

    any_answer_count = sum(
        1
        for result in answerable_results
        if result.get(
            "any_expected_evidence",
            False
        )
    )

    full_answer_count = sum(
        1
        for result in answerable_results
        if result.get(
            "full_expected_evidence",
            False
        )
    )

    correct_refusal_count = sum(
        1
        for result in unanswerable_results
        if result.get(
            "correct_refusal",
            False
        )
    )

    mean_answer_coverage = (
        sum(coverages)
        / answerable_count
        if answerable_count
        else 0.0
    )

    any_evidence_rate = (
        any_answer_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    full_evidence_rate = (
        full_answer_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    refusal_accuracy = (
        correct_refusal_count
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    generation_times = [
        result.get(
            "time_seconds",
            0.0
        )
        for result in results
        if isinstance(
            result.get(
                "time_seconds"
            ),
            (int, float)
        )
    ]

    average_generation_time = (
        sum(generation_times)
        / len(generation_times)
        if generation_times
        else 0.0
    )

    total_generation_time = sum(
        generation_times
    )

    return {
        "total_dataset_questions":
            len(dataset),

        "completed_questions":
            len(results),

        "answerable_questions":
            answerable_count,

        "unanswerable_questions":
            unanswerable_count,

        "any_evidence_answers":
            any_answer_count,

        "any_evidence_rate":
            any_evidence_rate,

        "full_evidence_answers":
            full_answer_count,

        "full_evidence_rate":
            full_evidence_rate,

        "mean_answer_coverage":
            mean_answer_coverage,

        "correct_refusals":
            correct_refusal_count,

        "refusal_accuracy":
            refusal_accuracy,

        "average_generation_time_seconds":
            average_generation_time,

        "total_generation_time_seconds":
            total_generation_time
    }


# ==========================================================
# Run Evaluation
# ==========================================================

def run_evaluation(
    qa_chain,
    dataset
):
    """
    Run or resume the complete generation benchmark.
    """

    results = load_checkpoint()

    valid_dataset_ids = {
        item["id"]
        for item in dataset
    }

    # Ignore checkpoint entries that no longer exist in the
    # current dataset.
    results = [
        result
        for result in results
        if result.get("id")
        in valid_dataset_ids
    ]

    # Remove accidental duplicate checkpoint entries while
    # preserving the first saved result for each question.
    unique_results = []
    seen_ids = set()

    for result in results:

        question_id = result.get(
            "id"
        )

        if not question_id:
            continue

        if question_id in seen_ids:
            continue

        seen_ids.add(
            question_id
        )

        unique_results.append(
            result
        )

    results = unique_results

    completed_ids = {
        result["id"]
        for result in results
    }

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SCHOLARRAG GENERATION EVALUATION"
    )

    print(
        "=" * 78
    )

    if completed_ids:

        print(
            f"\n🔄 Resuming evaluation."
        )

        print(
            f"✅ Already completed: "
            f"{len(completed_ids)}/"
            f"{len(dataset)}"
        )

    else:

        print(
            "\n🆕 Starting fresh generation benchmark."
        )

    for item in dataset:

        question_id = item["id"]
        question = item["question"]

        # ==================================================
        # Skip Completed Questions
        # ==================================================

        if question_id in completed_ids:

            print(
                f"\n[{question_id}] "
                f"{question}"
            )

            print(
                "⏭️ Already completed — skipping."
            )

            continue

        # ==================================================
        # Generate + Evaluate
        # ==================================================

        try:

            result = evaluate_question(
                qa_chain,
                item
            )

        except KeyboardInterrupt:

            print(
                "\n\n⚠️ Evaluation interrupted."
            )

            print(
                f"💾 Completed questions saved: "
                f"{len(results)}/"
                f"{len(dataset)}"
            )

            print(
                "▶️ Run the same command again "
                "to resume."
            )

            raise

        except Exception as error:

            print(
                f"\n❌ Question {question_id} "
                f"failed."
            )

            print(
                f"   {type(error).__name__}: "
                f"{error}"
            )

            print(
                "   This question was NOT marked "
                "as complete and can be retried."
            )

            continue

        # ==================================================
        # Save Immediately
        # ==================================================

        results.append(
            result
        )

        completed_ids.add(
            question_id
        )

        save_checkpoint(
            results,
            len(dataset)
        )

        print(
            f"💾 Checkpoint saved "
            f"({len(results)}/{len(dataset)})"
        )

    return results


# ==========================================================
# Save Final Results
# ==========================================================

def save_final_results(
    results,
    summary,
    dataset
):
    """
    Save the final benchmark only when every dataset question
    has successfully completed.

    A partial run remains a checkpoint and is never labelled
    as the final baseline.
    """

    expected_ids = {
        item["id"]
        for item in dataset
    }

    completed_ids = {
        result.get("id")
        for result in results
    }

    missing_ids = (
        expected_ids
        - completed_ids
    )

    if missing_ids:

        print(
            "\n⚠️ Evaluation is incomplete."
        )

        print(
            "Missing question IDs: "
            + ", ".join(
                sorted(missing_ids)
            )
        )

        print(
            "\n💾 Progress remains in:"
        )

        print(
            CHECKPOINT_PATH
        )

        print(
            "\nRun the evaluator again to retry "
            "the missing questions."
        )

        return False

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "experiment":
            "generation_baseline",

        "status":
            "complete",

        "summary":
            summary,

        "results":
            results
    }

    temp_output = OUTPUT_PATH.with_suffix(
        ".tmp"
    )

    with open(
        temp_output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    temp_output.replace(
        OUTPUT_PATH
    )

    print(
        "\n💾 Final results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # Delete checkpoint only after the final result has been
    # written successfully.
    if CHECKPOINT_PATH.exists():

        CHECKPOINT_PATH.unlink()

        print(
            "🧹 Checkpoint removed."
        )

    return True


# ==========================================================
# Print Final Summary
# ==========================================================

def print_summary(summary):
    """
    Print aggregate generation benchmark metrics.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FINAL GENERATION RESULTS"
    )

    print(
        "=" * 78
    )

    print(
        f"Dataset Questions     : "
        f"{summary['total_dataset_questions']}"
    )

    print(
        f"Completed Questions   : "
        f"{summary['completed_questions']}"
    )

    print(
        f"Answerable Questions  : "
        f"{summary['answerable_questions']}"
    )

    print(
        f"Unanswerable Questions: "
        f"{summary['unanswerable_questions']}"
    )

    print(
        "\n--- Answer Quality ---"
    )

    print(
        f"Any Evidence Rate     : "
        f"{summary['any_evidence_rate']:.3f}"
    )

    print(
        f"Full Evidence Rate    : "
        f"{summary['full_evidence_rate']:.3f}"
    )

    print(
        f"Mean Answer Coverage  : "
        f"{summary['mean_answer_coverage']:.3f}"
    )

    print(
        "\n--- Hallucination Safety ---"
    )

    print(
        f"Correct Refusals      : "
        f"{summary['correct_refusals']}/"
        f"{summary['unanswerable_questions']}"
    )

    print(
        f"Refusal Accuracy      : "
        f"{summary['refusal_accuracy']:.3f}"
    )

    print(
        "\n--- Performance ---"
    )

    print(
        f"Average Generation    : "
        f"{summary['average_generation_time_seconds']:.2f}s"
    )

    print(
        f"Total Generation Time : "
        f"{summary['total_generation_time_seconds']:.2f}s"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n🤖 Starting ScholarRAG "
        "Generation Evaluation"
    )

    # ======================================================
    # Dataset
    # ======================================================

    dataset = load_dataset()

    print(
        f"📋 Loaded {len(dataset)} "
        f"evaluation questions."
    )

    # ======================================================
    # Vector Store
    # ======================================================

    vector_store = (
        build_vector_store()
    )

    # ======================================================
    # QA Chain
    # ======================================================

    print(
        "\n🧠 Initialising QA chain..."
    )

    qa_chain = get_qa_chain(
        vector_store
    )

    print(
        "✅ QA chain ready."
    )

    # ======================================================
    # Evaluation
    # ======================================================

    try:

        results = run_evaluation(
            qa_chain,
            dataset
        )

    except KeyboardInterrupt:

        print(
            "\n🛑 ScholarRAG generation "
            "evaluation stopped safely."
        )

        return

    # ======================================================
    # Metrics
    # ======================================================

    summary = calculate_summary(
        results,
        dataset
    )

    print_summary(
        summary
    )

    # ======================================================
    # Final Save
    # ======================================================

    completed = save_final_results(
        results,
        summary,
        dataset
    )

    if completed:

        print(
            "\n✅ Generation evaluation complete."
        )

    else:

        print(
            "\n⚠️ Generation evaluation has "
            "unfinished questions."
        )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
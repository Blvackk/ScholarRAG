import json
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.rag.loader import load_pdf
from src.rag.chunker import split_documents
from src.rag.embeddings import get_embeddings


# ============================================================
# Paths / Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"
PAPERS_DIR = BASE_DIR / "papers"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "context_diagnostics.json"

# Questions that were incomplete during generation evaluation
TARGET_QUESTION_IDS = {
    "q01",
    "q04",
    "q08",
    "q10",
}

# Compare multiple retrieval depths
K_VALUES = [3, 5, 8, 10]


# ============================================================
# Text Normalisation
# ============================================================

def normalize_text(text):
    """
    Normalise text before evidence matching.

    This prevents differences such as:
        human-in-the-loop
        human in the loop

    from being treated as completely different strings.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalise Unicode punctuation
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")

    # Remove common markdown formatting
    text = re.sub(r"[*_`#]", " ", text)

    # Treat hyphens as spaces
    text = text.replace("-", " ")

    # Remove remaining punctuation
    text = re.sub(r"[^\w\s%.]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Evidence Aliases
# ============================================================

# These aliases help the diagnostic recognise wording variants.
#
# This is NOT semantic evaluation.
# It simply avoids obvious false negatives caused by formatting
# or minor wording differences.

EVIDENCE_ALIASES = {

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
    ],

    "No experimental data": [
        "no experimental data",
        "no experimental data were collected",
        "no experimental data were",
    ],

    "Adaptive Autonomy": [
        "adaptive autonomy",
    ],

    "XAI Interface Design": [
        "xai interface design",
        "explainable ai interface design",
    ],

    "Micro-Interventions": [
        "micro-interventions",
        "micro interventions",
        "microinterventions",
    ],

    "Workload Triage": [
        "workload triage",
    ],

    "longitudinal": [
        "longitudinal",
    ],

    "hybrid autonomy": [
        "hybrid autonomy",
    ],

    "generalist populations": [
        "generalist populations",
    ],

    "fatigue estimation": [
        "fatigue estimation",
    ],
}


# ============================================================
# JSON Loading
# ============================================================

def load_json(path):
    """
    Load a JSON file safely.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Evidence Matching
# ============================================================

def evidence_in_text(text, evidence):
    """
    Check whether an expected evidence phrase exists inside
    the supplied text.
    """

    normalized_text = normalize_text(text)
    normalized_evidence = normalize_text(evidence)

    # Direct normalised match
    if normalized_evidence in normalized_text:
        return True

    # Alias matching
    aliases = EVIDENCE_ALIASES.get(evidence, [])

    for alias in aliases:
        if normalize_text(alias) in normalized_text:
            return True

    return False


def find_evidence_in_documents(documents, expected_evidence):
    """
    Determine which expected evidence items occur anywhere
    inside the retrieved documents.
    """

    matched = []
    missing = []

    combined_context = "\n".join(
        document.page_content
        for document in documents
    )

    for evidence in expected_evidence:

        if evidence_in_text(combined_context, evidence):
            matched.append(evidence)
        else:
            missing.append(evidence)

    return matched, missing


# ============================================================
# Find Evidence Rank
# ============================================================

def find_first_evidence_rank(documents, evidence):
    """
    Return the first retrieval rank containing the evidence.

    Rank starts from 1.

    Returns None when the evidence is not found.
    """

    for rank, document in enumerate(documents, start=1):

        if evidence_in_text(
            document.page_content,
            evidence,
        ):
            return rank

    return None


# ============================================================
# Page Number
# ============================================================

def get_page_number(document):
    """
    Extract a human-readable page number from LangChain
    document metadata.

    PyPDFLoader normally stores zero-based page numbers under
    metadata["page"], so we add 1.
    """

    page = document.metadata.get("page")

    if page is None:
        return None

    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return page


# ============================================================
# Load Evaluation Papers
# ============================================================

def load_evaluation_papers():
    """
    Load all PDF files from tests/evaluation/papers.
    """

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No evaluation PDFs found in:\n{PAPERS_DIR}"
        )

    all_documents = []

    print("\n📄 Loading evaluation papers...")

    for pdf_path in pdf_files:

        print(f"   Loading: {pdf_path.name}")

        documents = load_pdf(str(pdf_path))

        # Preserve source paper information
        for document in documents:
            document.metadata["evaluation_file"] = (
                pdf_path.name
            )

        all_documents.extend(documents)

    print(
        f"✅ Loaded {len(all_documents)} PDF pages."
    )

    return all_documents


# ============================================================
# Diagnosis
# ============================================================

def diagnose_question(
    expected_evidence,
    retrieval_results,
):
    """
    Produce a simple diagnosis based on evidence availability.

    Important:
    This diagnoses retrieval depth only.

    If evidence is available at the generation K but the saved
    answer omitted it, generation/prompt behaviour must be
    investigated separately.
    """

    if not expected_evidence:
        return "NO_EXPECTED_EVIDENCE"

    k3 = retrieval_results.get("3", {})
    k5 = retrieval_results.get("5", {})
    k8 = retrieval_results.get("8", {})
    k10 = retrieval_results.get("10", {})

    full_at_3 = (
        len(k3.get("missing_evidence", [])) == 0
    )

    full_at_5 = (
        len(k5.get("missing_evidence", [])) == 0
    )

    full_at_8 = (
        len(k8.get("missing_evidence", [])) == 0
    )

    full_at_10 = (
        len(k10.get("missing_evidence", [])) == 0
    )

    if full_at_3:
        return "FULL_CONTEXT_AVAILABLE_AT_K3"

    if full_at_5:
        return "RETRIEVAL_DEPTH_K5"

    if full_at_8:
        return "RETRIEVAL_DEPTH_K8"

    if full_at_10:
        return "RETRIEVAL_DEPTH_K10"

    return "EVIDENCE_NOT_FULLY_RETRIEVED_AT_K10"


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n🔬 Starting ScholarRAG Context Diagnostics"
    )

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    dataset = load_json(DATASET_PATH)

    target_questions = [
        item
        for item in dataset
        if item.get("id") in TARGET_QUESTION_IDS
    ]

    print(
        f"📋 Loaded {len(dataset)} evaluation questions."
    )

    print(
        f"🎯 Diagnosing {len(target_questions)} "
        "incomplete generation questions."
    )

    # --------------------------------------------------------
    # Load Papers
    # --------------------------------------------------------

    documents = load_evaluation_papers()

    # --------------------------------------------------------
    # Chunk Documents
    # --------------------------------------------------------

    print("\n✂️ Creating chunks...")

    chunks = split_documents(documents)

    print(
        f"✅ Diagnostic index contains "
        f"{len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print("\n🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    print("\n🔎 Building temporary FAISS index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    print("✅ FAISS index ready.")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    all_results = []

    print("\n" + "=" * 84)
    print("SCHOLARRAG CONTEXT DIAGNOSTICS")
    print("=" * 84)

    max_k = max(K_VALUES)

    for item in target_questions:

        question_id = item["id"]
        question = item["question"]

        expected_evidence = item.get(
            "expected_evidence",
            [],
        )

        print(
            f"\n[{question_id}] {question}"
        )

        print("\nExpected Evidence:")

        for evidence in expected_evidence:
            print(f"  • {evidence}")

        # Retrieve once at maximum K.
        #
        # Lower K values are prefixes of this ranked result,
        # avoiding unnecessary repeated searches.

        retrieved_documents = (
            vector_store.similarity_search(
                question,
                k=max_k,
            )
        )

        # ----------------------------------------------------
        # Global Rank Information
        # ----------------------------------------------------

        evidence_ranks = {}

        print("\nEvidence Retrieval Ranks:")

        for evidence in expected_evidence:

            rank = find_first_evidence_rank(
                retrieved_documents,
                evidence,
            )

            evidence_ranks[evidence] = rank

            if rank is None:
                print(
                    f"  ❌ {evidence:<30} "
                    f"Not found in Top-{max_k}"
                )
            else:
                page = get_page_number(
                    retrieved_documents[rank - 1]
                )

                print(
                    f"  ✅ {evidence:<30} "
                    f"Rank {rank} | Page {page}"
                )

        # ----------------------------------------------------
        # K Comparison
        # ----------------------------------------------------

        retrieval_results = {}

        print("\n" + "-" * 84)
        print("TOP-K CONTEXT COMPARISON")
        print("-" * 84)

        for k in K_VALUES:

            current_documents = (
                retrieved_documents[:k]
            )

            matched, missing = (
                find_evidence_in_documents(
                    current_documents,
                    expected_evidence,
                )
            )

            total = len(expected_evidence)

            coverage = (
                len(matched) / total
                if total
                else 0.0
            )

            full_context = (
                total > 0
                and len(missing) == 0
            )

            pages = [
                get_page_number(document)
                for document in current_documents
            ]

            retrieval_results[str(k)] = {
                "k": k,
                "matched_evidence": matched,
                "missing_evidence": missing,
                "coverage": coverage,
                "full_context": full_context,
                "retrieved_pages": pages,
            }

            print(f"\nK = {k}")

            print(
                f"Matched  : {matched}"
            )

            print(
                f"Missing  : {missing}"
            )

            print(
                f"Coverage : {coverage:.3f}"
            )

            print(
                "Full     : "
                f"{'✅' if full_context else '❌'}"
            )

        # ----------------------------------------------------
        # Diagnosis
        # ----------------------------------------------------

        diagnosis = diagnose_question(
            expected_evidence,
            retrieval_results,
        )

        print("\nDiagnosis:")

        diagnosis_messages = {

            "FULL_CONTEXT_AVAILABLE_AT_K3":
                (
                    "🟢 Full evidence is already available "
                    "at K=3. If generation omitted evidence, "
                    "investigate generation/prompt behaviour."
                ),

            "RETRIEVAL_DEPTH_K5":
                (
                    "🟡 Full evidence becomes available at "
                    "K=5. Retrieval depth is likely part of "
                    "the problem."
                ),

            "RETRIEVAL_DEPTH_K8":
                (
                    "🟠 Full evidence becomes available at "
                    "K=8. The current retrieval depth may be "
                    "too shallow."
                ),

            "RETRIEVAL_DEPTH_K10":
                (
                    "🟠 Full evidence requires K=10. "
                    "Increasing K may improve completeness, "
                    "but latency/noise should be evaluated."
                ),

            "EVIDENCE_NOT_FULLY_RETRIEVED_AT_K10":
                (
                    "🔴 Some expected evidence is still "
                    "missing at K=10. Increasing K alone "
                    "may not solve this retrieval failure."
                ),

            "NO_EXPECTED_EVIDENCE":
                (
                    "⚪ No expected evidence was defined."
                ),
        }

        print(
            diagnosis_messages.get(
                diagnosis,
                diagnosis,
            )
        )

        # ----------------------------------------------------
        # Save Question Result
        # ----------------------------------------------------

        all_results.append(
            {
                "id": question_id,
                "question": question,
                "category": item.get(
                    "category",
                    "",
                ),
                "expected_evidence": (
                    expected_evidence
                ),
                "evidence_ranks": evidence_ranks,
                "retrieval_by_k": (
                    retrieval_results
                ),
                "diagnosis": diagnosis,
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 84)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 84)

    diagnosis_counts = {}

    for result in all_results:

        diagnosis = result["diagnosis"]

        diagnosis_counts[diagnosis] = (
            diagnosis_counts.get(
                diagnosis,
                0,
            )
            + 1
        )

    for diagnosis, count in diagnosis_counts.items():

        print(
            f"{diagnosis:<40} : {count}"
        )

    # ========================================================
    # Save JSON
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "evaluation": "context_diagnostics",
        "retrieval_method": "similarity",
        "k_values": K_VALUES,
        "target_questions": sorted(
            TARGET_QUESTION_IDS
        ),
        "summary": diagnosis_counts,
        "results": all_results,
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
        f"\n💾 Diagnostic results saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\n✅ Context diagnostics complete."
    )


if __name__ == "__main__":
    main()
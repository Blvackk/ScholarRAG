import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "tests" / "evaluation" / "dataset.json"
PAPERS_DIR = BASE_DIR / "tests" / "evaluation" / "papers"
RESULTS_DIR = BASE_DIR / "tests" / "evaluation" / "results"

OUTPUT_PATH = RESULTS_DIR / "dataset_validation.json"


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text):
    """
    Normalise text so that minor PDF extraction differences do not
    incorrectly mark evidence as missing.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalise common Unicode punctuation.
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    # Fix spaces around hyphens produced by PDF extraction.
    text = re.sub(r"\s*-\s*", "-", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def compact_text(text):
    """
    More tolerant representation used as a fallback.

    Example:
        '98.64 %' -> '9864'
        'TF-IDF'  -> 'tfidf'
    """

    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def evidence_exists(evidence, document_text):
    """
    Check evidence using:
    1. normalised phrase matching
    2. compact matching as a fallback
    """

    evidence_normalized = normalize_text(evidence)
    document_normalized = normalize_text(document_text)

    if evidence_normalized in document_normalized:
        return True

    evidence_compact = compact_text(evidence)
    document_compact = compact_text(document_text)

    if evidence_compact and evidence_compact in document_compact:
        return True

    return False


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset.json not found:\n{DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("dataset.json must contain a JSON array.")

    return data


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    page_texts = []

    for index, page in enumerate(pages, start=1):
        page_texts.append(
            {
                "page": index,
                "text": page.page_content
            }
        )

    full_text = "\n".join(
        page["text"] for page in page_texts
    )

    return page_texts, full_text


# ============================================================
# FIND EVIDENCE PAGE
# ============================================================

def find_evidence_pages(evidence, page_texts):
    matches = []

    for page in page_texts:
        if evidence_exists(evidence, page["text"]):
            matches.append(page["page"])

    return matches


# ============================================================
# VALIDATE DATASET STRUCTURE
# ============================================================

def validate_structure(dataset):
    issues = []

    required_fields = {
        "id",
        "paper",
        "question",
        "expected_pages",
        "expected_evidence",
        "category",
    }

    ids = [
        item.get("id")
        for item in dataset
        if item.get("id")
    ]

    duplicate_ids = [
        item_id
        for item_id, count in Counter(ids).items()
        if count > 1
    ]

    for duplicate_id in duplicate_ids:
        issues.append(
            f"Duplicate question ID: {duplicate_id}"
        )

    for index, item in enumerate(dataset, start=1):

        question_id = item.get("id", f"ITEM_{index}")

        missing_fields = required_fields - set(item.keys())

        if missing_fields:
            issues.append(
                f"{question_id}: missing fields "
                f"{sorted(missing_fields)}"
            )
            continue

        if not isinstance(item["expected_pages"], list):
            issues.append(
                f"{question_id}: expected_pages must be a list."
            )

        if not isinstance(item["expected_evidence"], list):
            issues.append(
                f"{question_id}: expected_evidence must be a list."
            )

        if item["category"] == "unanswerable":

            if item["expected_evidence"]:
                issues.append(
                    f"{question_id}: unanswerable question "
                    f"must have empty expected_evidence."
                )

            if item["expected_pages"]:
                issues.append(
                    f"{question_id}: unanswerable question "
                    f"should have empty expected_pages."
                )

        else:

            if not item["expected_evidence"]:
                issues.append(
                    f"{question_id}: answerable question "
                    f"has no expected evidence."
                )

    return issues


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print()
    print("🔍 Starting ScholarRAG Dataset Validation")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset()

    print(f"📋 Loaded {len(dataset)} questions.")

    # --------------------------------------------------------
    # Structure validation
    # --------------------------------------------------------

    print()
    print("🧱 Validating dataset structure...")

    structure_issues = validate_structure(dataset)

    if structure_issues:
        print(
            f"⚠️ Found {len(structure_issues)} "
            f"structural issue(s)."
        )
    else:
        print("✅ Dataset structure valid.")

    # --------------------------------------------------------
    # Group by paper
    # --------------------------------------------------------

    questions_by_paper = defaultdict(list)

    for item in dataset:
        questions_by_paper[item["paper"]].append(item)

    print(
        f"📚 Dataset references "
        f"{len(questions_by_paper)} papers."
    )

    # --------------------------------------------------------
    # Validation result
    # --------------------------------------------------------

    validation_results = {
        "dataset_path": str(DATASET_PATH),
        "total_questions": len(dataset),
        "total_papers": len(questions_by_paper),
        "structure_issues": structure_issues,
        "papers": {},
    }

    total_answerable = 0
    total_unanswerable = 0

    total_evidence = 0
    total_found = 0
    total_missing = 0

    # --------------------------------------------------------
    # Process each paper separately
    # --------------------------------------------------------

    for paper_name, questions in sorted(
        questions_by_paper.items()
    ):

        print()
        print("=" * 90)
        print(f"📄 PAPER: {paper_name}")
        print("=" * 90)

        pdf_path = PAPERS_DIR / paper_name

        paper_result = {
            "exists": pdf_path.exists(),
            "questions": len(questions),
            "answerable_questions": 0,
            "unanswerable_questions": 0,
            "total_evidence": 0,
            "found_evidence": 0,
            "missing_evidence": 0,
            "evidence": [],
        }

        if not pdf_path.exists():

            print("❌ PDF NOT FOUND")
            print(f"Expected location: {pdf_path}")

            validation_results["papers"][
                paper_name
            ] = paper_result

            continue

        print("✅ PDF found.")
        print("📖 Extracting PDF text...")

        try:
            page_texts, full_text = load_pdf(pdf_path)

        except Exception as error:

            print(f"❌ Failed to load PDF: {error}")

            paper_result["load_error"] = str(error)

            validation_results["papers"][
                paper_name
            ] = paper_result

            continue

        print(
            f"✅ Extracted {len(page_texts)} pages."
        )

        # ----------------------------------------------------
        # Questions
        # ----------------------------------------------------

        for item in questions:

            question_id = item["id"]
            category = item["category"]
            expected_evidence = item["expected_evidence"]

            print()
            print(f"[{question_id}] {item['question']}")

            # ------------------------------------------------
            # Unanswerable
            # ------------------------------------------------

            if category == "unanswerable":

                total_unanswerable += 1
                paper_result[
                    "unanswerable_questions"
                ] += 1

                print(
                    "   🚫 UNANSWERABLE "
                    "— skipped evidence validation"
                )

                continue

            # ------------------------------------------------
            # Answerable
            # ------------------------------------------------

            total_answerable += 1
            paper_result[
                "answerable_questions"
            ] += 1

            for evidence in expected_evidence:

                total_evidence += 1
                paper_result["total_evidence"] += 1

                pages = find_evidence_pages(
                    evidence,
                    page_texts
                )

                found = len(pages) > 0

                evidence_result = {
                    "question_id": question_id,
                    "evidence": evidence,
                    "found": found,
                    "pages": pages,
                }

                paper_result["evidence"].append(
                    evidence_result
                )

                if found:

                    total_found += 1
                    paper_result[
                        "found_evidence"
                    ] += 1

                    print(
                        f"   ✅ {evidence}"
                        f"  → page(s): {pages}"
                    )

                else:

                    total_missing += 1
                    paper_result[
                        "missing_evidence"
                    ] += 1

                    print(
                        f"   ❌ {evidence}"
                        f"  → NOT FOUND"
                    )

        # ----------------------------------------------------
        # Paper coverage
        # ----------------------------------------------------

        if paper_result["total_evidence"]:

            coverage = (
                paper_result["found_evidence"]
                / paper_result["total_evidence"]
            )

        else:
            coverage = 0.0

        paper_result["evidence_coverage"] = coverage

        validation_results["papers"][
            paper_name
        ] = paper_result

        print()
        print("-" * 90)
        print(f"Paper Evidence Coverage: {coverage:.3f}")
        print(
            f"Found: "
            f"{paper_result['found_evidence']}"
            f"/{paper_result['total_evidence']}"
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("SCHOLARRAG DATASET VALIDATION SUMMARY")
    print("=" * 90)

    if total_evidence:

        overall_coverage = (
            total_found / total_evidence
        )

    else:
        overall_coverage = 0.0

    print()
    print(f"Total Papers          : {len(questions_by_paper)}")
    print(f"Total Questions       : {len(dataset)}")
    print(f"Answerable Questions  : {total_answerable}")
    print(f"Unanswerable Questions: {total_unanswerable}")

    print()
    print("--- Evidence Validation ---")
    print(f"Evidence Items        : {total_evidence}")
    print(f"Evidence Found        : {total_found}")
    print(f"Evidence Missing      : {total_missing}")
    print(f"Evidence Coverage     : {overall_coverage:.3f}")

    print()
    print("--- Structure ---")
    print(
        f"Structure Issues      : "
        f"{len(structure_issues)}"
    )

    # --------------------------------------------------------
    # Per-paper summary
    # --------------------------------------------------------

    print()
    print("--- Per-Paper Coverage ---")

    for paper_name, result in validation_results[
        "papers"
    ].items():

        if not result["exists"]:
            print(
                f"{paper_name:<20}: PDF MISSING"
            )
            continue

        coverage = result.get(
            "evidence_coverage",
            0.0
        )

        print(
            f"{paper_name:<20}: "
            f"{coverage:.3f} "
            f"({result['found_evidence']}/"
            f"{result['total_evidence']})"
        )

    # --------------------------------------------------------
    # Missing evidence
    # --------------------------------------------------------

    missing_items = []

    for paper_name, result in validation_results[
        "papers"
    ].items():

        for evidence in result.get("evidence", []):

            if not evidence["found"]:

                missing_items.append(
                    {
                        "paper": paper_name,
                        **evidence,
                    }
                )

    print()
    print("--- Missing Evidence ---")

    if not missing_items:

        print(
            "✅ Every expected evidence phrase "
            "was found in its assigned PDF."
        )

    else:

        for item in missing_items:

            print(
                f"❌ {item['question_id']} | "
                f"{item['paper']} | "
                f"{item['evidence']}"
            )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    validation_results["summary"] = {
        "answerable_questions": total_answerable,
        "unanswerable_questions": total_unanswerable,
        "total_evidence": total_evidence,
        "found_evidence": total_found,
        "missing_evidence": total_missing,
        "evidence_coverage": overall_coverage,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            validation_results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"💾 Results saved to:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("VALIDATION DECISION")
    print("=" * 90)

    missing_pdfs = [
        paper
        for paper, result
        in validation_results["papers"].items()
        if not result["exists"]
    ]

    if missing_pdfs:

        print("❌ DATASET NOT READY")
        print(
            "One or more referenced PDFs are missing."
        )

    elif structure_issues:

        print("❌ DATASET NOT READY")
        print(
            "Fix structural issues before evaluation."
        )

    elif total_missing:

        print("⚠️ DATASET NEEDS REVIEW")
        print(
            f"{total_missing} expected evidence "
            f"item(s) were not found."
        )
        print(
            "Review those labels before running "
            "the final retrieval benchmark."
        )

    else:

        print("✅ DATASET READY")
        print(
            "All expected evidence was found in "
            "the corresponding PDFs."
        )
        print(
            "You can proceed to the multi-paper "
            "retrieval benchmark."
        )

    print()
    print("✅ Dataset validation complete.")


if __name__ == "__main__":
    main()
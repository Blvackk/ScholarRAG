# src/services/upload_service.py

import os
import shutil

from config import Config


def save_uploaded_pdf(pdf_path: str):
    """
    Validate and save a PDF uploaded through the UI.

    Returns
    -------
    tuple
        (success, message, saved_path)
    """

    if not pdf_path:
        return (
            False,
            "❌ Please select a PDF first.",
            None
        )

    filename = os.path.basename(pdf_path)

    extension = os.path.splitext(filename)[1].lower()

    if extension not in Config.ALLOWED_EXTENSIONS:
        return (
            False,
            "❌ Only PDF files are supported.",
            None
        )

    try:
        file_size_mb = (
            os.path.getsize(pdf_path)
            / (1024 * 1024)
        )

    except OSError as e:
        return (
            False,
            f"❌ Could not read the uploaded file: {e}",
            None
        )

    if file_size_mb > Config.MAX_UPLOAD_SIZE_MB:
        return (
            False,
            (
                f"❌ File is too large. Maximum size is "
                f"{Config.MAX_UPLOAD_SIZE_MB} MB."
            ),
            None
        )

    os.makedirs(
        Config.UPLOAD_DIR,
        exist_ok=True
    )

    destination = os.path.join(
        Config.UPLOAD_DIR,
        filename
    )

    try:

        # Replace the existing copy if the same paper
        # is uploaded again.
        shutil.copy2(
            pdf_path,
            destination
        )

        return (
            True,
            f"📄 {filename}",
            destination
        )

    except Exception as e:

        return (
            False,
            f"❌ Upload failed: {e}",
            None
        )
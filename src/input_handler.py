"""
Input Handler
=============

Loads and validates PaddleOCR JSON files from the input directory.

Responsibilities:
1. Discover JSON files.
2. Load JSON documents.
3. Validate document, page, and text-region structures.
4. Collect errors without interrupting other files.
5. Return valid documents without modifying their contents.
"""

import json
import logging
import math

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Raised when an OCR JSON document has an invalid structure."""


def is_valid_number(value: Any) -> bool:
    """
    Check whether a value is a finite integer or float.

    Boolean values are explicitly excluded because bool is a
    subclass of int in Python.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_bounding_box(
    bounding_box: Any,
    region_id: int
) -> None:
    """
    Validate an OCR bounding box.

    Expected format:
        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ]

    Raises:
        InputValidationError: If the bounding box is invalid.
    """
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise InputValidationError(
            f"Detection {region_id}: bounding_box must contain exactly 4 points."
        )

    for point_index, point in enumerate(bounding_box):
        if not isinstance(point, list) or len(point) != 2:
            raise InputValidationError(
                f"Detection {region_id}: point {point_index + 1} "
                "must contain exactly 2 coordinates."
            )

        x, y = point

        if not is_valid_number(x) or not is_valid_number(y):
            raise InputValidationError(
                f"Detection {region_id}: coordinates must be finite numbers."
            )


def validate_text_region(region: Any, page_number: int) -> None:
    """
    Validate an individual OCR text detection.

    Required fields:
        id
        text
        confidence
        bounding_box
    """
    if not isinstance(region, dict):
        raise InputValidationError(
            f"Page {page_number}: text region must be an object."
        )

    region_id = region.get("id")

    if not isinstance(region_id, int) or isinstance(region_id, bool):
        raise InputValidationError(
            f"Page {page_number}: detection has an invalid ID."
        )

    text = region.get("text")

    if not isinstance(text, str):
        raise InputValidationError(
            f"Page {page_number}, detection {region_id}: "
            "text must be a string."
        )

    confidence = region.get("confidence")

    if (
        not is_valid_number(confidence)
        or not 0 <= confidence <= 1
    ):
        raise InputValidationError(
            f"Page {page_number}, detection {region_id}: "
            "confidence must be between 0 and 1."
        )

    validate_bounding_box(
        region.get("bounding_box"),
        region_id
    )


def validate_page(page: Any, page_index: int) -> int:
    """
    Validate an individual page.

    Returns:
        Number of text regions on the page.
    """
    if not isinstance(page, dict):
        raise InputValidationError(
            f"Page at index {page_index} must be an object."
        )

    page_number = page.get("page_number")

    if (
        not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or page_number <= 0
    ):
        raise InputValidationError(
            f"Page at index {page_index}: invalid page_number."
        )

    # Page dimensions are optional. When supplied, they must be
    # finite positive numbers. The extraction pipeline uses OCR
    # bounding-box coordinates directly and does not require them.
    for dimension in ("page_width_px", "page_height_px"):
        if dimension not in page:
            continue

        value = page[dimension]

        if not is_valid_number(value) or value <= 0:
            raise InputValidationError(
                f"Page {page_number}: {dimension} "
                "must be a positive finite number when provided."
            )

    text_regions = page.get("text_regions")

    if not isinstance(text_regions, list):
        raise InputValidationError(
            f"Page {page_number}: text_regions must be a list."
        )

    for region in text_regions:
        validate_text_region(region, page_number)

    return len(text_regions)


def validate_document(document: Any) -> None:
    """
    Validate the complete PaddleOCR document structure.

    Expected hierarchy:
        document
            document_name
            total_text_regions
            pages[]
                page_number
                page_width_px (optional)
                page_height_px (optional)
                text_regions[]
    """
    if not isinstance(document, dict):
        raise InputValidationError(
            "The JSON root must be an object."
        )

    document_name = document.get("document_name")

    if not isinstance(document_name, str) or not document_name.strip():
        raise InputValidationError(
            "Missing or invalid document_name."
        )

    total_regions = document.get("total_text_regions")

    if (
        not isinstance(total_regions, int)
        or isinstance(total_regions, bool)
        or total_regions < 0
    ):
        raise InputValidationError(
            "total_text_regions must be a non-negative integer."
        )

    pages = document.get("pages")

    if not isinstance(pages, list) or not pages:
        raise InputValidationError(
            "pages must be a non-empty list."
        )

    actual_total_regions = 0

    for page_index, page in enumerate(pages):
        actual_total_regions += validate_page(page, page_index)

    if total_regions != actual_total_regions:
        logger.warning(
            "Document '%s': total_text_regions is %d, "
            "but actual detection count is %d.",
            document_name,
            total_regions,
            actual_total_regions
        )


def load_json_file(file_path: Path) -> dict[str, Any]:
    """
    Read and validate a single OCR JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Validated OCR document.

    Raises:
        InputValidationError: If the document structure is invalid.
        json.JSONDecodeError: If JSON syntax is invalid.
        OSError: If the file cannot be accessed.
    """
    try:
        with file_path.open("r", encoding="utf-8-sig") as file:
            document = json.load(file)

    except json.JSONDecodeError:
        logger.exception(
            "Invalid JSON syntax in file: %s",
            file_path.name
        )
        raise

    except OSError:
        logger.exception(
            "Unable to read file: %s",
            file_path.name
        )
        raise

    validate_document(document)

    logger.info(
        "Successfully loaded OCR document: %s",
        file_path.name
    )

    return document


def load_documents(
    input_directory: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Load all JSON files from the specified input directory.

    Each file is processed independently. An invalid file
    does not interrupt processing of other files.

    Args:
        input_directory: Directory containing OCR JSON files.

    Returns:
        Tuple containing:
            valid_documents: Successfully loaded documents.
            errors: Failed filenames and corresponding errors.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If the input path is not a directory.
        OSError: If the directory cannot be accessed.
    """
    input_path = Path(input_directory)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_path}"
        )

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_path}"
        )

    json_files = sorted(input_path.glob("*.json"))

    valid_documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if not json_files:
        logger.warning(
            "No JSON files found in input directory: %s",
            input_path
        )
        return valid_documents, errors

    logger.info(
        "Found %d JSON files in input directory.",
        len(json_files)
    )

    for file_path in json_files:
        try:
            document = load_json_file(file_path)
            valid_documents.append(document)

        except (
            json.JSONDecodeError,
            InputValidationError,
            OSError,
            UnicodeError
        ) as error:
            logger.error(
                "Failed to process %s: %s",
                file_path.name,
                error
            )

            errors.append({
                "file_name": file_path.name,
                "error": str(error)
            })

    logger.info(
        "Input processing completed. Successful: %d | Failed: %d",
        len(valid_documents),
        len(errors)
    )

    return valid_documents, errors

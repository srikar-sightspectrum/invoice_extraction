
"""Main entry point for the simplified invoice extraction pipeline.

Run from the project root:
    python -m src.main

The pipeline always attempts to generate final JSON after extraction and
validation. Missing fields and validation warnings do not trigger manual review
or block output.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any


# Support execution as: python -m src.main from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.input_handler import load_documents
from src.coordinate_processor import process_document
from src.header_extraction.header_mapper import map_headers
from src.header_extraction.header_validator import validate_headers
from src.table_extraction.table_detector import detect_tables
from src.table_extraction.column_mapper import map_columns
from src.table_extraction.row_grouper import group_rows
from src.table_extraction.cell_mapper import map_cells
from src.table_extraction.line_item_extractor import extract_line_items
from src.validation.extraction_validator import validate_extraction
from src.output.json_generator import generate_invoice_json


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"

with CONFIG_FILE.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

PATHS = CONFIG["paths"]

INPUT_DIR = PROJECT_ROOT / PATHS["input_folder"]
EXTRACTION_DIR = PROJECT_ROOT / PATHS["extraction_folder"]
FINAL_DIR = PROJECT_ROOT / PATHS["final_folder"]
LOG_DIR = PROJECT_ROOT / PATHS["log_folder"]

HEADER_ALIASES_FILE = CONFIG_DIR / "header_aliases.json"
COLUMN_ALIASES_FILE = CONFIG_DIR / "column_aliases.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging, configuration, and output helpers
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure console and file logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOG_DIR / "invoice_extraction.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def create_output_directories() -> None:
    """Create directories used for intermediate and final output."""
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_config_file(path: Path) -> dict[str, list[str]]:
    """Load and validate an alias configuration JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration must contain a JSON object: {path}"
        )

    for field, aliases in data.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError(
                f"Invalid field name in configuration: {path}"
            )

        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip()
            for alias in aliases
        ):
            raise ValueError(
                f"Aliases for {field!r} must be a list of non-empty strings."
            )

    return data


def safe_filename(value: Any) -> str:
    """Convert a document name into a safe filename stem."""
    name = str(value or "invoice").strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip(" ._")
    return name or "invoice"


def save_intermediate_extraction(
    extraction_document: dict[str, Any],
) -> Path:
    """Save the complete validated extraction for troubleshooting."""
    filename = (
        f"{safe_filename(extraction_document.get('document_name'))}.json"
    )
    output_path = EXTRACTION_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            extraction_document,
            file,
            ensure_ascii=False,
            indent=4,
            default=str,
        )
        file.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# Invoice extraction pipeline
# ---------------------------------------------------------------------------

def process_one_document(
    document: dict[str, Any],
    header_aliases: dict[str, list[str]],
    column_aliases: dict[str, list[str]],
) -> str:
    """Extract, validate, and generate final JSON for one OCR document."""
    document_name = document.get("document_name", "invoice")
    logger.info("Starting extraction for: %s", document_name)

    # 1. Calculate and attach geometry to OCR detections.
    processed_document = process_document(document)

    # 2. Extract and validate invoice header fields.
    mapped_headers = map_headers(
        processed_document,
        header_aliases,
    )
    validated_headers = validate_headers(mapped_headers)

    # 3. Detect the line-item table and map its columns, rows, and cells.
    detected_tables = detect_tables(
        processed_document,
        column_aliases,
    )
    mapped_columns = map_columns(
        detected_tables,
        column_aliases,
    )
    grouped_rows = group_rows(mapped_columns)
    mapped_cells = map_cells(grouped_rows)
    extracted_line_items = extract_line_items(mapped_cells)

    # 4. Combine the two extraction branches.
    extraction_document = {
        "document_name": document_name,
        "header_fields": validated_headers.get(
            "header_fields",
            {},
        ),
        "line_item_extraction": extracted_line_items.get(
            "line_item_extraction",
            {},
        ),
    }

    # 5. Validation produces warnings and does not block JSON generation.
    validated_extraction = validate_extraction(extraction_document)

    # 6. Save the full intermediate result.
    intermediate_path = save_intermediate_extraction(
        validated_extraction
    )
    logger.info(
        "Saved intermediate extraction: %s",
        intermediate_path,
    )

    # 7. Always attempt to generate the final JSON.
    final_path = generate_invoice_json(
        validated_extraction,
        FINAL_DIR,
    )
    logger.info(
        "Final invoice JSON generated: %s",
        final_path,
    )

    return "completed"


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Load input invoices and process each document independently."""
    configure_logging()
    create_output_directories()

    try:
        header_aliases = load_config_file(HEADER_ALIASES_FILE)
        column_aliases = load_config_file(COLUMN_ALIASES_FILE)

        documents, input_errors = load_documents(INPUT_DIR)

        for error in input_errors:
            logger.error(
                "Input file skipped: %s | %s",
                error.get("file_name", "unknown"),
                error.get("error", "Unknown input error"),
            )

        if not documents:
            logger.warning(
                "No valid OCR JSON documents found in: %s",
                INPUT_DIR,
            )
            return 1 if input_errors else 0

        completed = 0
        failed = 0

        for document in documents:
            try:
                process_one_document(
                    document,
                    header_aliases,
                    column_aliases,
                )
                completed += 1

            except Exception:
                failed += 1
                logger.exception(
                    "Pipeline failed for document: %s",
                    document.get("document_name", "unknown"),
                )

        logger.info(
            "Pipeline run finished | completed=%d | failed=%d | input_errors=%d",
            completed,
            failed,
            len(input_errors),
        )

        return 1 if failed or input_errors else 0

    except Exception:
        logger.exception("Pipeline could not start.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

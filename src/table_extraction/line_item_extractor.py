"""Convert mapped invoice cells into structured line-item records.

OCR values are preserved as strings. Missing cells become None. Incomplete
rows are retained, and extraction warnings never block the output.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

LINE_ITEM_FIELDS: tuple[str, ...] = (
    "description",
    "hsn_sac",
    "quantity",
    "unit",
    "rate",
    "discount",
    "tax_rate",
    "tax_amount",
    "amount",
)


class LineItemExtractionError(ValueError):
    """Raised when line-item extraction input has an invalid structure."""


def _read_cell_value(cell: Any) -> str | None:
    """Return the original non-empty OCR value from a mapped cell."""
    if cell is None:
        return None
    if not isinstance(cell, dict):
        raise LineItemExtractionError("Each mapped cell must be an object or null.")

    value = cell.get("value")
    if value is None:
        return None
    if not isinstance(value, str):
        raise LineItemExtractionError("Mapped cell values must be strings or null.")
    return value if value.strip() else None


def _cell_source_ids(cell: Any) -> list[Any]:
    """Return source OCR detection IDs for a cell."""
    if not isinstance(cell, dict):
        return []
    source_ids = cell.get("source_detection_ids", [])
    return list(source_ids) if isinstance(source_ids, list) else []


def _extract_row(
    row: dict[str, Any],
    table_id: Any,
    page_number: Any,
    line_item_index: int,
) -> dict[str, Any]:
    """Convert one mapped row into the standard line-item structure."""
    cells = row.get("cells")
    if not isinstance(cells, dict):
        raise LineItemExtractionError(
            f"Row {row.get('row_id')!r} must contain a cells object."
        )

    line_item = {
        field: _read_cell_value(cells.get(field))
        for field in LINE_ITEM_FIELDS
    }

    source_ids_by_field = {
        field: _cell_source_ids(cells.get(field))
        for field in LINE_ITEM_FIELDS
    }
    source_ids = list(dict.fromkeys(
        detection_id
        for field_ids in source_ids_by_field.values()
        for detection_id in field_ids
    ))

    warnings: list[str] = []
    if line_item["description"] is None:
        warnings.append("Description is missing or empty; the row was retained.")

    unmapped = row.get("unmapped_detections", [])
    if isinstance(unmapped, list) and unmapped:
        warnings.append("One or more OCR detections could not be assigned to a cell.")

    row_warnings = row.get("warnings", [])
    if isinstance(row_warnings, list):
        warnings.extend(
            warning for warning in row_warnings
            if isinstance(warning, str) and warning.strip()
        )

    description_cell = cells.get("description")
    continuation = (
        description_cell.get("continuation_candidate")
        if isinstance(description_cell, dict)
        else None
    )
    if isinstance(continuation, dict):
        warnings.append("Possible multi-line description continuation was not merged.")

    metadata = {
        "line_item_index": line_item_index,
        "source_table_id": table_id,
        "source_page_number": page_number,
        "source_row_id": row.get("row_id"),
        "source_detection_ids": source_ids,
        "source_detection_ids_by_field": source_ids_by_field,
        "unmapped_detections": deepcopy(unmapped) if isinstance(unmapped, list) else [],
        "continuation_candidate": deepcopy(continuation),
        "warnings": list(dict.fromkeys(warnings)),
    }

    return {"line_item": line_item, "extraction_metadata": metadata}


def extract_line_items(cell_mapped_document: dict[str, Any]) -> dict[str, Any]:
    """Assemble mapped cells into structured line items without blocking output."""
    if not isinstance(cell_mapped_document, dict):
        raise LineItemExtractionError("cell_mapped_document must be a dictionary.")

    cell_mapping = cell_mapped_document.get("cell_mapping")
    if not isinstance(cell_mapping, dict):
        raise LineItemExtractionError(
            "Document is missing cell_mapping. Run cell_mapper.py first."
        )

    tables = cell_mapping.get("primary_line_item_tables")
    if not isinstance(tables, list):
        raise LineItemExtractionError(
            "cell_mapping.primary_line_item_tables must be a list."
        )

    output_document = deepcopy(cell_mapped_document)
    extracted_tables: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    total_line_items = 0

    for table in tables:
        if not isinstance(table, dict):
            raise LineItemExtractionError(
                "Each primary line-item table result must be an object."
            )
        rows = table.get("rows")
        if not isinstance(rows, list):
            raise LineItemExtractionError(
                f"Rows for table {table.get('table_id')!r} must be a list."
            )

        extracted_items = []
        table_warnings: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            extracted = _extract_row(
                row=row,
                table_id=table.get("table_id"),
                page_number=table.get("page_number"),
                line_item_index=total_line_items + 1,
            )
            extracted_items.append(extracted)
            total_line_items += 1
            table_warnings.extend(extracted["extraction_metadata"]["warnings"])

        if not extracted_items:
            table_warnings.append("No grouped rows were available for line-item extraction.")
        table_warnings = list(dict.fromkeys(table_warnings))
        all_warnings.extend(table_warnings)

        extracted_tables.append({
            "table_id": table.get("table_id"),
            "page_number": table.get("page_number"),
            "line_items": extracted_items,
            "line_item_count": len(extracted_items),
            "warnings": table_warnings,
        })

    if not tables:
        all_warnings.append("No cell-mapped primary line-item tables were available.")

    output_document["line_item_extraction"] = {
        "tables": extracted_tables,
        "total_line_item_count": total_line_items,
        "warnings": list(dict.fromkeys(all_warnings)),
    }

    logger.info(
        "Extracted %d line item(s) from %d table(s) in document %s",
        total_line_items,
        len(extracted_tables),
        cell_mapped_document.get("document_name", "<unknown>"),
    )
    return output_document

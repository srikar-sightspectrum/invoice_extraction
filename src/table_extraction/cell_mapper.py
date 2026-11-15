"""Map grouped OCR detections to invoice line-item cells.

Assigns detections to mapped column intervals, joins text in reading order,
and preserves source IDs. Unmapped detections and continuation candidates are
reported as non-blocking warnings. No business-value validation is performed.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


class CellMappingError(ValueError):
    """Raised when cell-mapping input has an invalid structure."""


def _column_membership(
    geometry: dict[str, Any], columns: list[dict[str, Any]]
) -> tuple[str | None, list[str]]:
    """Return the unique mapped field whose interval contains the detection centre."""
    centre_x = geometry.get("center_x")
    if not isinstance(centre_x, (int, float)):
        return None, []

    fields = sorted({
        str(column["standard_field"])
        for column in columns
        if isinstance(column, dict)
        and isinstance(column.get("standard_field"), str)
        and isinstance(column.get("x_min"), (int, float))
        and isinstance(column.get("x_max"), (int, float))
        and float(column["x_min"]) <= float(centre_x) <= float(column["x_max"])
    })
    return (fields[0], fields) if len(fields) == 1 else (None, fields)


def _reading_order(detection: dict[str, Any]) -> tuple[float, float, str]:
    """Sort text detections top-to-bottom and then left-to-right."""
    geometry = detection.get("geometry", {})
    return (
        float(geometry.get("center_y", 0.0)),
        float(geometry.get("x_min", 0.0)),
        str(detection.get("detection_id", "")),
    )


def _validate_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep detections with usable text and horizontal geometry."""
    detections = row.get("detections")
    if not isinstance(detections, list):
        raise CellMappingError(f"Row {row.get('row_id')!r} must contain a detections list.")

    valid = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        geometry = detection.get("geometry")
        if (
            isinstance(detection.get("text"), str)
            and detection["text"].strip()
            and isinstance(geometry, dict)
            and isinstance(geometry.get("center_x"), (int, float))
            and isinstance(geometry.get("center_y"), (int, float))
            and isinstance(geometry.get("x_min"), (int, float))
        ):
            valid.append(detection)
    return valid


def _map_row(
    row: dict[str, Any],
    columns: list[dict[str, Any]],
    continuation_by_row: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    """Map one grouped row into canonical cells."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    unmapped: list[dict[str, Any]] = []

    for detection in _validate_row(row):
        field, possible_fields = _column_membership(detection["geometry"], columns)
        # Use row-grouper's candidate only when no interval matched at all.
        candidate = detection.get("candidate_column")
        if field is None and not possible_fields and isinstance(candidate, str):
            if any(
                isinstance(column, dict) and column.get("standard_field") == candidate
                for column in columns
            ):
                field = candidate
                possible_fields = [candidate]

        if field is None:
            unmapped.append({
                "detection_id": detection.get("detection_id"),
                "text": detection["text"],
                "possible_fields": possible_fields,
                "warning": (
                    "Detection falls within overlapping column boundaries."
                    if len(possible_fields) > 1
                    else "Detection is outside all mapped column boundaries."
                ),
            })
        else:
            grouped.setdefault(field, []).append(detection)

    canonical_fields = list(dict.fromkeys(
        column["standard_field"]
        for column in columns
        if isinstance(column, dict) and isinstance(column.get("standard_field"), str)
    ))

    cells: dict[str, Any] = {}
    for field in canonical_fields:
        detections = sorted(grouped.get(field, []), key=_reading_order)
        text_parts = [item["text"].strip() for item in detections if item["text"].strip()]
        cells[field] = {
            "value": " ".join(text_parts) if text_parts else None,
            "source_detection_ids": [
                item.get("detection_id") for item in detections
                if item.get("detection_id") is not None
            ],
            "raw_text_parts": [item["text"] for item in detections],
        }

    row_id = row.get("row_id")
    continuation = continuation_by_row.get(row_id)
    if continuation is not None and "description" in cells:
        cells["description"]["continuation_candidate"] = {
            "possible_parent_row_id": continuation.get("possible_parent_row_id"),
            "source_detection_ids": continuation.get("source_detection_ids", []),
            "auto_merged": False,
        }

    warnings = [item["warning"] for item in unmapped]
    if continuation is not None:
        warnings.append("Possible multi-line description continuation was not merged.")

    return {
        "row_id": row_id,
        "cells": cells,
        "unmapped_detections": unmapped,
        "warnings": list(dict.fromkeys(warnings)),
    }


def map_cells(row_grouped_document: dict[str, Any]) -> dict[str, Any]:
    """Map grouped rows into cells for each primary line-item table."""
    if not isinstance(row_grouped_document, dict):
        raise CellMappingError("row_grouped_document must be a dictionary.")

    row_grouping = row_grouped_document.get("row_grouping")
    column_mapping = row_grouped_document.get("column_mapping")
    if not isinstance(row_grouping, dict):
        raise CellMappingError("Document is missing row_grouping.")
    if not isinstance(column_mapping, dict):
        raise CellMappingError("Document is missing column_mapping.")

    grouped_tables = row_grouping.get("primary_line_item_tables")
    mapped_tables = column_mapping.get("primary_line_item_tables")
    if not isinstance(grouped_tables, list) or not isinstance(mapped_tables, list):
        raise CellMappingError("Expected primary_line_item_tables lists in row_grouping and column_mapping.")

    output_document = deepcopy(row_grouped_document)
    mapped_by_id = {
        table.get("table_id"): table
        for table in mapped_tables if isinstance(table, dict)
    }
    table_results = []
    all_warnings: list[str] = []

    for grouped_table in grouped_tables:
        if not isinstance(grouped_table, dict):
            continue
        table_id = grouped_table.get("table_id")
        mapping = mapped_by_id.get(table_id)
        if mapping is None:
            raise CellMappingError(f"No column mapping found for table_id {table_id!r}.")

        columns = mapping.get("columns", [])
        rows = grouped_table.get("rows", [])
        continuations = grouped_table.get("continuation_candidates", [])
        if not isinstance(columns, list) or not isinstance(rows, list):
            raise CellMappingError(f"Columns and rows for table_id {table_id!r} must be lists.")
        if not isinstance(continuations, list):
            raise CellMappingError(f"Continuation candidates for table_id {table_id!r} must be a list.")

        continuation_by_row = {
            candidate.get("row_id"): candidate
            for candidate in continuations if isinstance(candidate, dict)
        }
        mapped_rows = [
            _map_row(row, columns, continuation_by_row)
            for row in rows if isinstance(row, dict)
        ]
        table_warnings = [
            warning for row in mapped_rows for warning in row["warnings"]
        ]
        if not mapped_rows:
            table_warnings.append("No grouped rows were available for cell mapping.")
        table_warnings = list(dict.fromkeys(table_warnings))
        all_warnings.extend(table_warnings)

        table_results.append({
            "table_id": table_id,
            "page_number": grouped_table.get("page_number"),
            "rows": mapped_rows,
            "row_count": len(mapped_rows),
            "warnings": table_warnings,
        })

    if not grouped_tables:
        all_warnings.append("No grouped primary line-item tables were available.")

    output_document["cell_mapping"] = {
        "primary_line_item_tables": table_results,
        "warnings": list(dict.fromkeys(all_warnings)),
    }
    logger.info(
        "Mapped cells in %d primary line-item table(s) for document %s",
        len(table_results), row_grouped_document.get("document_name", "<unknown>"),
    )
    return output_document

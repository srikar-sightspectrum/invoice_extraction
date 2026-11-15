"""Detect the primary line-item table in an OCR-processed invoice."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "description": [
        "description", "description of goods", "description of services",
        "particulars", "item description", "product description", "goods",
    ],
    "hsn_sac": ["hsn", "hsn code", "sac", "hsn sac", "hsn/sac"],
    "quantity": ["qty", "quantity"],
    "unit": ["unit", "uom", "unit of measure"],
    "rate": ["rate", "unit price", "price", "rate per unit"],
    "discount": ["discount", "discount amount"],
    "tax_rate": ["tax rate", "gst rate", "tax %", "gst %", "rate of tax"],
    "tax_amount": ["tax amount", "gst amount", "tax value"],
    "amount": ["amount", "total amount", "line total", "net amount", "value", "total"],
}

SUMMARY_TERMS = {
    "subtotal", "sub total", "grand total", "total amount", "invoice total",
    "amount payable", "amount in words", "total tax", "taxable amount",
    "cgst", "sgst", "igst", "cess", "round off", "shipping charges",
    "freight charges", "discount total",
}


class TableDetectionError(ValueError):
    """Raised when the document or OCR page structure is invalid."""


def _normalise(text: str) -> str:
    """Normalise text for conservative heading and summary matching."""
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[/_#-]+", " ", text)
    text = re.sub(r"[^a-z0-9% ]+", " ", text)
    return " ".join(text.split())


def _get_geometry(region: dict[str, Any]) -> dict[str, float] | None:
    """Return coordinate-processor geometry or derive it from the OCR box."""
    geometry = region.get("geometry")
    keys = (
        "x_min", "x_max", "y_min", "y_max",
        "center_x", "center_y", "width", "height",
    )
    if isinstance(geometry, dict) and all(
        isinstance(geometry.get(key), (int, float)) for key in keys
    ):
        return {key: float(geometry[key]) for key in keys}

    box = region.get("bounding_box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "center_x": (x_min + x_max) / 2,
        "center_y": (y_min + y_max) / 2,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def _prepare_regions(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep usable OCR detections and attach temporary geometry."""
    raw_regions = page.get("text_regions", [])
    if not isinstance(raw_regions, list):
        raise TableDetectionError("Page text_regions must be a list.")

    regions = []
    for region in raw_regions:
        if not isinstance(region, dict):
            continue
        text = region.get("text")
        geometry = _get_geometry(region)
        if not isinstance(text, str) or not text.strip() or geometry is None:
            continue
        item = dict(region)
        item["_geometry"] = geometry
        item["_normalised_text"] = _normalise(text)
        regions.append(item)
    return regions


def _group_rows(regions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group detections on approximately the same horizontal text line."""
    if not regions:
        return []

    typical_height = median(max(1.0, r["_geometry"]["height"]) for r in regions)
    tolerance = max(2.0, typical_height * 0.65)
    ordered = sorted(
        regions,
        key=lambda r: (r["_geometry"]["center_y"], r["_geometry"]["x_min"]),
    )

    rows: list[list[dict[str, Any]]] = []
    centres: list[float] = []
    for region in ordered:
        centre_y = region["_geometry"]["center_y"]
        matching = [
            (abs(centre_y - centre), index)
            for index, centre in enumerate(centres)
            if abs(centre_y - centre) <= tolerance
        ]
        if not matching:
            rows.append([region])
            centres.append(centre_y)
            continue

        _, row_index = min(matching)
        rows[row_index].append(region)
        centres[row_index] = sum(
            item["_geometry"]["center_y"] for item in rows[row_index]
        ) / len(rows[row_index])

    for row in rows:
        row.sort(key=lambda item: item["_geometry"]["x_min"])
    return sorted(rows, key=lambda row: min(
        item["_geometry"]["center_y"] for item in row
    ))


def _match_fields(row: list[dict[str, Any]], aliases: dict[str, list[str]]) -> set[str]:
    """Return canonical fields represented by headings in this OCR row."""
    fields: set[str] = set()
    for region in row:
        text = region["_normalised_text"]
        for field, values in aliases.items():
            for alias in values:
                normalised_alias = _normalise(alias)
                if normalised_alias and (
                    text == normalised_alias
                    or text.startswith(normalised_alias + " ")
                    or f" {normalised_alias} " in f" {text} "
                ):
                    fields.add(field)
                    break
    return fields


def _is_summary_row(row: list[dict[str, Any]]) -> bool:
    """Identify common invoice summary rows so they are not included as items."""
    text = " ".join(region["_normalised_text"] for region in row)
    return any(
        text == term or text.startswith(term + " ")
        for term in SUMMARY_TERMS
    )


def _row_bounds(row: list[dict[str, Any]]) -> dict[str, float]:
    geometries = [region["_geometry"] for region in row]
    return {
        "x_min": min(g["x_min"] for g in geometries),
        "y_min": min(g["y_min"] for g in geometries),
        "x_max": max(g["x_max"] for g in geometries),
        "y_max": max(g["y_max"] for g in geometries),
    }


def _region_bounds(regions: list[dict[str, Any]]) -> dict[str, float]:
    geometries = [region["_geometry"] for region in regions]
    return {
        "x_min": min(g["x_min"] for g in geometries),
        "y_min": min(g["y_min"] for g in geometries),
        "x_max": max(g["x_max"] for g in geometries),
        "y_max": max(g["y_max"] for g in geometries),
    }


def _is_line_item_heading(fields: set[str]) -> bool:
    """Require a description heading plus another field, or qty/rate/amount."""
    return (
        ("description" in fields and len(fields) >= 2)
        or {"quantity", "rate", "amount"}.issubset(fields)
    )


def _build_candidate(
    rows: list[list[dict[str, Any]]],
    heading_index: int,
    page_number: int,
    aliases: dict[str, list[str]],
) -> dict[str, Any]:
    """Build a table candidate from a heading row and nearby item rows."""
    heading_row = rows[heading_index]
    heading_bounds = _row_bounds(heading_row)
    heights = [max(1.0, item["_geometry"]["height"]) for item in heading_row]
    typical_height = median(heights) if heights else 1.0

    selected_rows = [heading_row]
    previous_bottom = heading_bounds["y_max"]
    for row in rows[heading_index + 1:]:
        bounds = _row_bounds(row)
        if _is_summary_row(row):
            break
        if bounds["y_min"] - previous_bottom > typical_height * 3.2:
            break

        # Avoid extending the table into a separate block well outside its columns.
        horizontal_overlap = min(heading_bounds["x_max"], bounds["x_max"]) - max(
            heading_bounds["x_min"], bounds["x_min"]
        )
        if horizontal_overlap <= 0 and bounds["x_min"] > heading_bounds["x_max"] + typical_height * 2:
            break

        selected_rows.append(row)
        previous_bottom = max(previous_bottom, bounds["y_max"])

    selected_regions = [region for row in selected_rows for region in row]
    heading_ids = [
        region.get("id") for region in heading_row if region.get("id") is not None
    ]
    source_ids = [
        region.get("id") for region in selected_regions if region.get("id") is not None
    ]
    fields = sorted(_match_fields(heading_row, aliases))

    return {
        "page_number": page_number,
        "table_type": "line_items",
        "is_primary": False,
        "detection_method": "recognised_column_headings",
        "heading_fields": fields,
        "heading_text": [region["text"] for region in heading_row],
        "heading_detection_ids": heading_ids,
        "row_count_estimate": max(0, len(selected_rows) - 1),
        "source_detection_ids": source_ids,
        "bounding_box": _region_bounds(selected_regions),
    }


def _merge_aliases(custom_aliases: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Extend default aliases with optional per-run aliases."""
    aliases = deepcopy(DEFAULT_COLUMN_ALIASES)
    if custom_aliases is None:
        return aliases
    if not isinstance(custom_aliases, dict):
        raise TableDetectionError("column_aliases must be a dictionary.")

    for field, values in custom_aliases.items():
        if not isinstance(field, str) or not field.strip():
            raise TableDetectionError("Canonical field names must be non-empty strings.")
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise TableDetectionError(f"Aliases for {field!r} must be a list of non-empty strings.")
        aliases[field] = list(dict.fromkeys(aliases.get(field, []) + values))
    return aliases


def detect_tables(
    processed_document: dict[str, Any],
    column_aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Detect line-item tables and return their bounds and OCR source IDs.

    The detector looks for a single OCR row containing at least two recognised
    column headings. A candidate is considered a line-item table when it has a
    description heading plus another recognised heading, or quantity, rate,
    and amount headings together. It selects at most one primary table per page.
    """
    if not isinstance(processed_document, dict):
        raise TableDetectionError("processed_document must be a dictionary.")
    pages = processed_document.get("pages")
    if not isinstance(pages, list):
        raise TableDetectionError("Document must contain a pages list.")

    aliases = _merge_aliases(column_aliases)
    output_document = deepcopy(processed_document)
    tables: list[dict[str, Any]] = []
    page_summaries: list[dict[str, Any]] = []

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise TableDetectionError(f"Page at index {page_index} must be a dictionary.")
        page_number = page.get("page_number", page_index + 1)
        rows = _group_rows(_prepare_regions(page))
        page_candidates: list[dict[str, Any]] = []

        for row_index, row in enumerate(rows):
            fields = _match_fields(row, aliases)
            if len(fields) < 2 or not _is_line_item_heading(fields):
                continue
            page_candidates.append(_build_candidate(rows, row_index, page_number, aliases))

        # If more than one candidate exists, prefer the one with the most
        # recognised columns; ties are resolved by top-to-bottom position.
        page_candidates.sort(
            key=lambda table: (
                -len(table["heading_fields"]),
                table["bounding_box"]["y_min"],
            )
        )
        if page_candidates:
            page_candidates[0]["is_primary"] = True

        for candidate in page_candidates:
            candidate["table_id"] = len(tables) + 1
            tables.append(candidate)

        page_summaries.append({
            "page_number": page_number,
            "candidate_table_count": len(page_candidates),
            "primary_line_item_table_found": bool(page_candidates),
        })

    primary_tables = [
        {"page_number": table["page_number"], "table_id": table["table_id"]}
        for table in tables if table["is_primary"]
    ]

    output_document["table_detection"] = {
        "tables": tables,
        "primary_line_item_tables": primary_tables,
        "pages": page_summaries,
        # Retained as neutral compatibility fields; no review workflow is used.
        "review_required": False,
        "review_reasons": [],
    }

    logger.info(
        "Detected %d line-item table candidate(s) across %d page(s) in %s",
        len(tables), len(pages), processed_document.get("document_name", "<unknown>"),
    )
    return output_document

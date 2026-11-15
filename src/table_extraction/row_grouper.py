"""Group OCR detections into invoice line-item rows using their coordinates."""

from __future__ import annotations

import logging
from copy import deepcopy
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)


class RowGroupingError(ValueError):
    """Raised when row-grouping input has an invalid structure."""


SUMMARY_TERMS = (
    "subtotal", "sub total", "grand total", "invoice total",
    "total amount", "amount payable", "amount in words", "total tax",
    "taxable amount", "taxable value", "cgst", "sgst", "igst",
    "cess", "round off", "shipping charges", "freight charges",
)


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace(":", " ").split())


def _is_summary_label(text: str) -> bool:
    value = _normalise(text)
    return any(value == term or value.startswith(term + " ") for term in SUMMARY_TERMS)


def _geometry(region: dict[str, Any]) -> dict[str, float] | None:
    """Return a consistent geometry dictionary for an OCR detection."""
    geom = region.get("geometry")
    keys = ("x_min", "x_max", "y_min", "y_max", "center_x", "center_y", "width", "height")
    if isinstance(geom, dict) and all(isinstance(geom.get(k), (int, float)) for k in keys):
        return {k: float(geom[k]) for k in keys}

    box = region.get("bounding_box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None

    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return {
        "x_min": left, "x_max": right, "y_min": top, "y_max": bottom,
        "center_x": (left + right) / 2, "center_y": (top + bottom) / 2,
        "width": right - left, "height": bottom - top,
    }


def _find_page(document: dict[str, Any], page_number: int) -> dict[str, Any] | None:
    for index, page in enumerate(document.get("pages", [])):
        if page.get("page_number", index + 1) == page_number:
            return page
    return None


def _column_for_detection(geom: dict[str, float], columns: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    """Use the detection's horizontal centre to identify its mapped column."""
    centre = geom["center_x"]
    fields = sorted({
        str(col["standard_field"])
        for col in columns
        if isinstance(col, dict)
        and isinstance(col.get("x_min"), (int, float))
        and isinstance(col.get("x_max"), (int, float))
        and float(col["x_min"]) <= centre <= float(col["x_max"])
        and isinstance(col.get("standard_field"), str)
    })
    return (fields[0], fields) if len(fields) == 1 else (None, fields)


def _collect_detections(
    page: dict[str, Any], table: dict[str, Any], columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bounds = table.get("bounding_box")
    if not isinstance(bounds, dict):
        raise RowGroupingError("Detected table is missing bounding_box.")
    try:
        left, top = float(bounds["x_min"]), float(bounds["y_min"])
        right, bottom = float(bounds["x_max"]), float(bounds["y_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RowGroupingError("Detected table has invalid bounding-box coordinates.") from exc

    regions = page.get("text_regions", [])
    if not isinstance(regions, list):
        raise RowGroupingError("Page text_regions must be a list.")

    output = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        text = region.get("text")
        geom = _geometry(region)
        if not isinstance(text, str) or not text.strip() or geom is None:
            continue
        inside = (
            geom["x_max"] >= left and geom["x_min"] <= right
            and geom["y_max"] >= top and geom["y_min"] <= bottom
        )
        if not inside:
            continue
        field, possible_fields = _column_for_detection(geom, columns)
        output.append({
            "detection_id": region.get("id"),
            "text": text,
            "confidence": region.get("confidence"),
            "bounding_box": region.get("bounding_box"),
            "geometry": geom,
            "candidate_column": field,
            "possible_columns": possible_fields,
            "is_summary_label": _is_summary_label(text),
        })
    return output


def _remove_headings(records: list[dict[str, Any]], table: dict[str, Any]) -> list[dict[str, Any]]:
    heading_ids = set(table.get("heading_detection_ids", []))
    heading_texts = {
        _normalise(value) for value in table.get("heading_text", [])
        if isinstance(value, str)
    }
    table_top = table.get("bounding_box", {}).get("y_min")
    heights = [max(1.0, r["geometry"]["height"]) for r in records]
    tolerance = max(2.0, median(heights) * 0.8) if heights else 2.0

    kept = []
    for record in records:
        if record["detection_id"] is not None and record["detection_id"] in heading_ids:
            continue
        same_heading = (
            _normalise(record["text"]) in heading_texts
            and isinstance(table_top, (int, float))
            and abs(record["geometry"]["y_min"] - float(table_top)) <= tolerance
        )
        if not same_heading:
            kept.append(record)
    return kept


def _group_rows(records: list[dict[str, Any]], tolerance_factor: float) -> list[list[dict[str, Any]]]:
    """Group detections whose vertical centres are within a text-height tolerance."""
    if not records:
        return []
    typical_height = median(max(1.0, r["geometry"]["height"]) for r in records)
    tolerance = max(2.0, typical_height * tolerance_factor)
    ordered = sorted(records, key=lambda r: (r["geometry"]["center_y"], r["geometry"]["x_min"]))

    rows: list[list[dict[str, Any]]] = []
    for record in ordered:
        centre_y = record["geometry"]["center_y"]
        if not rows:
            rows.append([record])
            continue
        last_row = rows[-1]
        row_centre = sum(item["geometry"]["center_y"] for item in last_row) / len(last_row)
        if abs(centre_y - row_centre) <= tolerance:
            last_row.append(record)
        else:
            rows.append([record])

    for row in rows:
        row.sort(key=lambda r: r["geometry"]["x_min"])
    return rows


def _serialise_rows(rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows, start=1):
        details = [{
            "detection_id": r["detection_id"],
            "text": r["text"],
            "confidence": r["confidence"],
            "candidate_column": r["candidate_column"],
            "possible_columns": r["possible_columns"],
            "bounding_box": r["bounding_box"],
            "geometry": r["geometry"],
        } for r in row]
        ids = [r["detection_id"] for r in row if r["detection_id"] is not None]
        result.append({
            "row_id": index,
            "source_detection_ids": ids,
            "detections": details,
            "row_bounds": {
                "x_min": min(r["geometry"]["x_min"] for r in row),
                "y_min": min(r["geometry"]["y_min"] for r in row),
                "x_max": max(r["geometry"]["x_max"] for r in row),
                "y_max": max(r["geometry"]["y_max"] for r in row),
            },
            # Kept for downstream compatibility; these flags do not block JSON output.
            "review_required": False,
            "review_reasons": [],
        })
    return result


def _continuation_candidates(rows: list[list[dict[str, Any]]], typical_height: float) -> list[dict[str, Any]]:
    """Flag a nearby description-only row; do not merge it automatically."""
    candidates = []
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        previous_fields = {r["candidate_column"] for r in previous}
        current_fields = {r["candidate_column"] for r in current}
        gap = max(
            0.0,
            min(r["geometry"]["y_min"] for r in current)
            - max(r["geometry"]["y_max"] for r in previous),
        )
        if (
            "description" in previous_fields
            and "description" in current_fields
            and not current_fields.intersection({"quantity", "rate", "amount"})
            and gap <= typical_height * 1.8
        ):
            candidates.append({
                "row_id": index + 1,
                "possible_parent_row_id": index,
                "reason": "Nearby description text has no quantity, rate, or amount evidence.",
                "source_detection_ids": [r["detection_id"] for r in current if r["detection_id"] is not None],
                "review_required": False,
            })
    return candidates


def group_rows(mapped_document: dict[str, Any], row_tolerance_factor: float = 0.65) -> dict[str, Any]:
    """Group detections into rows for each primary line-item table."""
    if not isinstance(mapped_document, dict):
        raise RowGroupingError("mapped_document must be a dictionary.")
    if not isinstance(row_tolerance_factor, (int, float)) or row_tolerance_factor <= 0:
        raise RowGroupingError("row_tolerance_factor must be a positive number.")

    table_detection = mapped_document.get("table_detection")
    column_mapping = mapped_document.get("column_mapping")
    if not isinstance(table_detection, dict) or not isinstance(column_mapping, dict):
        raise RowGroupingError("Document must contain table_detection and column_mapping.")
    detected_tables = table_detection.get("tables")
    mapped_tables = column_mapping.get("primary_line_item_tables")
    if not isinstance(detected_tables, list) or not isinstance(mapped_tables, list):
        raise RowGroupingError("Expected table_detection.tables and column_mapping.primary_line_item_tables lists.")

    output_document = deepcopy(mapped_document)
    tables_by_id = {t.get("table_id"): t for t in detected_tables if isinstance(t, dict)}
    results = []

    for mapping in mapped_tables:
        if not isinstance(mapping, dict):
            continue
        table_id = mapping.get("table_id")
        table = tables_by_id.get(table_id)
        if table is None:
            raise RowGroupingError(f"Mapped table_id {table_id!r} was not found in table_detection.")
        page_number = mapping.get("page_number", table.get("page_number"))
        page = _find_page(output_document, page_number)
        if page is None:
            raise RowGroupingError(f"Could not find page {page_number!r} for table {table_id!r}.")
        columns = mapping.get("columns", [])
        if not isinstance(columns, list):
            raise RowGroupingError(f"Columns for table {table_id!r} must be a list.")

        records = _remove_headings(_collect_detections(page, table, columns), table)
        summary_records = [r for r in records if r["is_summary_label"]]
        if records:
            typical_height = median(max(1.0, r["geometry"]["height"]) for r in records)
        else:
            typical_height = 1.0
        summary_tolerance = max(2.0, typical_height * 0.65)
        summary_centres = [r["geometry"]["center_y"] for r in summary_records]

        line_item_records = []
        for record in records:
            if record["is_summary_label"]:
                continue
            if any(abs(record["geometry"]["center_y"] - y) <= summary_tolerance for y in summary_centres):
                summary_records.append(record)
            else:
                line_item_records.append(record)

        grouped = _group_rows(line_item_records, float(row_tolerance_factor))
        rows = _serialise_rows(grouped)
        continuations = _continuation_candidates(grouped, typical_height)
        summary_details = [{
            "detection_id": r["detection_id"],
            "text": r["text"],
            "bounding_box": r["bounding_box"],
        } for r in summary_records]

        warnings = []
        if not rows:
            warnings.append("No line-item rows were grouped inside the primary table bounds.")
        if continuations:
            warnings.append("Possible multi-line descriptions were detected; they were not merged.")

        results.append({
            "table_id": table_id,
            "page_number": page_number,
            "rows": rows,
            "row_count": len(rows),
            "continuation_candidates": continuations,
            "summary_detections": summary_details,
            "review_required": False,
            "review_reasons": [],
            "warnings": warnings,
        })

    overall_warnings = [warning for result in results for warning in result["warnings"]]
    output_document["row_grouping"] = {
        "primary_line_item_tables": results,
        "review_required": False,
        "review_reasons": [],
        "warnings": overall_warnings,
    }
    logger.info("Grouped rows in %d primary line-item table(s) for document %s", len(results), mapped_document.get("document_name", "<unknown>"))
    return output_document

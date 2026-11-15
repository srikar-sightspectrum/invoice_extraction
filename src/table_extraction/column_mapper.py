"""Map invoice line-item table headings to canonical column names."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "description": ["description", "description of goods", "description of services", "particulars", "item description", "product description", "goods"],
    "hsn_sac": ["hsn", "hsn code", "sac", "hsn sac", "hsn/sac"],
    "quantity": ["qty", "quantity"],
    "unit": ["unit", "uom", "unit of measure"],
    "rate": ["rate", "unit price", "price", "rate per unit"],
    "discount": ["discount", "discount amount"],
    "tax_rate": ["tax rate", "gst rate", "tax %", "gst %", "rate of tax"],
    "tax_amount": ["tax amount", "gst amount", "tax value"],
    "amount": ["amount", "total amount", "line total", "net amount", "value", "total"],
}


class ColumnMappingError(ValueError):
    """Raised when the column-mapping input has an invalid structure."""


def _normalise(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[/_#-]+", " ", value)
    value = re.sub(r"[^a-z0-9% ]+", " ", value)
    return " ".join(value.split())


def _geometry(region: dict[str, Any]) -> dict[str, float] | None:
    geom = region.get("geometry")
    keys = ("x_min", "x_max", "y_min", "y_max", "center_x", "center_y", "width", "height")
    if isinstance(geom, dict) and all(isinstance(geom.get(k), (int, float)) for k in keys):
        return {k: float(geom[k]) for k in keys}
    box = region.get("bounding_box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
    except (TypeError, ValueError, IndexError):
        return None
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return {"x_min": x1, "x_max": x2, "y_min": y1, "y_max": y2,
            "center_x": (x1+x2)/2, "center_y": (y1+y2)/2,
            "width": x2-x1, "height": y2-y1}


def _merge_aliases(custom: dict[str, list[str]] | None) -> dict[str, list[str]]:
    aliases = deepcopy(DEFAULT_COLUMN_ALIASES)
    if custom is None:
        return aliases
    if not isinstance(custom, dict):
        raise ColumnMappingError("column_aliases must be a dictionary.")
    for field, values in custom.items():
        if not isinstance(field, str) or not field.strip() or not isinstance(values, list):
            raise ColumnMappingError("Each canonical field must have a list of aliases.")
        if not all(isinstance(v, str) and v.strip() for v in values):
            raise ColumnMappingError(f"Aliases for {field!r} must be non-empty strings.")
        aliases[field] = list(dict.fromkeys(aliases.get(field, []) + values))
    return aliases


def _match_heading(heading: str, aliases: dict[str, list[str]]) -> tuple[str | None, str | None]:
    normalised = _normalise(heading)
    matches: list[tuple[str, int]] = []
    for field, values in aliases.items():
        for alias in values:
            candidate = _normalise(alias)
            if not candidate:
                continue
            if normalised == candidate:
                matches.append((field, 2))
            elif f" {candidate} " in f" {normalised} ":
                matches.append((field, 1))
    if not matches:
        return None, None
    # Prefer exact matches. If fields conflict, do not guess.
    strongest = max(score for _, score in matches)
    fields = {field for field, score in matches if score == strongest}
    if len(fields) != 1:
        return None, None
    field = next(iter(fields))
    return field, "exact_alias" if strongest == 2 else "phrase_alias"


def _page_regions(document: dict[str, Any], page_number: Any) -> list[dict[str, Any]]:
    for index, page in enumerate(document.get("pages", [])):
        if page.get("page_number", index + 1) != page_number:
            continue
        result = []
        for region in page.get("text_regions", []):
            if not isinstance(region, dict) or not isinstance(region.get("text"), str) or not region["text"].strip():
                continue
            geom = _geometry(region)
            if geom is not None:
                result.append({"id": region.get("id"), "text": region["text"],
                               "geometry": geom, "bounding_box": region.get("bounding_box")})
        return result
    return []


def _heading_regions(table: dict[str, Any], regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The detector's source IDs are authoritative; use text/position only as fallback.
    heading_ids = set(table.get("heading_detection_ids", []))
    if heading_ids:
        found = [r for r in regions if r.get("id") in heading_ids]
        if found:
            return sorted(found, key=lambda r: r["geometry"]["x_min"])

    heading_texts = {_normalise(t) for t in table.get("heading_text", []) if isinstance(t, str)}
    bounds = table.get("bounding_box", {})
    top = bounds.get("y_min") if isinstance(bounds, dict) else None
    if not heading_texts or not isinstance(top, (int, float)):
        return []
    heights = [max(1.0, r["geometry"]["height"]) for r in regions]
    tolerance = max(3.0, median(heights) * 1.5) if heights else 3.0
    return sorted([r for r in regions
                   if _normalise(r["text"]) in heading_texts
                   and abs(r["geometry"]["y_min"] - float(top)) <= tolerance],
                  key=lambda r: r["geometry"]["x_min"])


def _set_column_boundaries(columns: list[dict[str, Any]], bounds: dict[str, Any]) -> None:
    if not columns:
        return
    ordered = sorted(columns, key=lambda c: c["_center_x"])
    left = float(bounds.get("x_min", min(c["_left"] for c in ordered)))
    right = float(bounds.get("x_max", max(c["_right"] for c in ordered)))
    centers = [c["_center_x"] for c in ordered]
    for i, col in enumerate(ordered):
        col["x_min"] = round(left if i == 0 else (centers[i-1] + centers[i])/2, 2)
        col["x_max"] = round(right if i == len(ordered)-1 else (centers[i] + centers[i+1])/2, 2)
        for key in ("_center_x", "_left", "_right"):
            col.pop(key, None)


def _map_table(document: dict[str, Any], table: dict[str, Any], aliases: dict[str, list[str]]) -> dict[str, Any]:
    regions = _page_regions(document, table.get("page_number"))
    headings = _heading_regions(table, regions)
    columns: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    warnings: list[str] = []

    for region in headings:
        field, method = _match_heading(region["text"], aliases)
        if field is None:
            unmapped.append({"source_heading": region["text"],
                             "source_detection_id": region.get("id")})
            warnings.append(f"Unmapped table heading: {region['text']!r}.")
            continue
        geom = region["geometry"]
        columns.append({"source_heading": region["text"],
                        "source_detection_id": region.get("id"),
                        "standard_field": field,
                        "matched_alias": method,
                        "mapping_method": method,
                        "x_min": None, "x_max": None,
                        "_center_x": geom["center_x"],
                        "_left": geom["x_min"], "_right": geom["x_max"]})

    counts: dict[str, int] = {}
    for col in columns:
        counts[col["standard_field"]] = counts.get(col["standard_field"], 0) + 1
    for field, count in counts.items():
        if count > 1:
            warnings.append(f"Multiple headings map to {field!r}; both mappings were retained.")

    bounds = table.get("bounding_box", {})
    _set_column_boundaries(columns, bounds if isinstance(bounds, dict) else {})
    required = {"description", "quantity", "amount"}
    missing = sorted(required - {c["standard_field"] for c in columns})
    if missing:
        warnings.append("Core line-item headings not mapped: " + ", ".join(missing) + ".")
    if not headings:
        warnings.append("Could not locate table heading detections.")

    return {"table_id": table.get("table_id"),
            "page_number": table.get("page_number"),
            "table_type": table.get("table_type"),
            "columns": columns,
            "unmapped_headings": unmapped,
            "missing_core_fields": missing,
            "warnings": list(dict.fromkeys(warnings))}


def map_columns(detected_document: dict[str, Any], column_aliases: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Map headings for primary line-item tables and return a copied document."""
    if not isinstance(detected_document, dict):
        raise ColumnMappingError("detected_document must be a dictionary.")
    detection = detected_document.get("table_detection")
    if not isinstance(detection, dict) or not isinstance(detection.get("tables"), list):
        raise ColumnMappingError("Document must contain table_detection.tables.")

    aliases = _merge_aliases(column_aliases)
    output = deepcopy(detected_document)
    primary = [t for t in detection["tables"] if isinstance(t, dict)
               and t.get("is_primary") is True and t.get("table_type") == "line_items"]
    results = [_map_table(output, table, aliases) for table in primary]
    all_warnings = [warning for result in results for warning in result["warnings"]]
    if not primary:
        all_warnings.append("No primary line-item table was identified.")

    # Keep the structure consumed by row_grouper.py; warnings are informational
    # and never block subsequent extraction stages.
    output["column_mapping"] = {
        "primary_line_item_tables": results,
        "warnings": list(dict.fromkeys(all_warnings)),
    }
    logger.info("Mapped columns for %d primary line-item table(s) in %s",
                len(results), detected_document.get("document_name", "<unknown>"))
    return output

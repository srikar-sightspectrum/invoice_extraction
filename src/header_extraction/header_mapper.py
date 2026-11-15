"""
Simplified invoice header mapper.

Maps configured header aliases to nearby OCR text using basic geometry.
Requires a document processed by coordinate_processor.process_document().
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


class HeaderMappingError(Exception):
    """Raised when the OCR document or alias configuration is invalid."""


def _normalise(text: str) -> str:
    """Normalize text for straightforward alias comparison."""
    text = text.casefold().strip()
    text = re.sub(r"[\s\-_/#:：]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _get_geometry(region: Mapping[str, Any], page_number: int) -> Mapping[str, Any]:
    """Get the geometry added by coordinate_processor."""
    geometry = region.get("geometry")
    required = ("x_min", "x_max", "y_min", "y_max", "center_x", "center_y", "width", "height")

    if not isinstance(geometry, Mapping):
        raise HeaderMappingError(
            f"Page {page_number}, detection {region.get('id')}: "
            "geometry is missing. Run coordinate_processor first."
        )

    for key in required:
        value = geometry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HeaderMappingError(
                f"Page {page_number}, detection {region.get('id')}: "
                f"geometry.{key} is missing or invalid."
            )
    return geometry


def _find_inline_value(text: str, aliases: Sequence[str]) -> tuple[str, str] | None:
    """Find values written in the same OCR region as a label, e.g. Invoice No: A123."""
    for alias in sorted(aliases, key=len, reverse=True):
        pattern = rf"^\s*{re.escape(alias)}\s*[:：]\s*(.*?)\s*$"
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).strip():
            return alias, match.group(1).strip()
    return None


def _is_generic_date_alias(alias: str) -> bool:
    """A standalone 'Date' is not sufficient evidence of invoice date."""
    return _normalise(alias) == "date"


def _find_nearby_value(
    label_region: Mapping[str, Any],
    page_regions: Sequence[Mapping[str, Any]],
    label_ids: set[Any],
) -> dict[str, Any] | None:
    """
    Find the nearest OCR region to the right on the same line,
    or below the label when no right-side value is available.
    """
    label_geometry = label_region["geometry"]
    label_height = max(float(label_geometry["height"]), 1.0)
    label_width = max(float(label_geometry["width"]), 1.0)

    right_candidates: list[tuple[float, Mapping[str, Any]]] = []
    below_candidates: list[tuple[float, Mapping[str, Any]]] = []

    for candidate in page_regions:
        if candidate.get("id") in label_ids or candidate.get("id") == label_region.get("id"):
            continue

        text = candidate.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        geometry = candidate["geometry"]
        candidate_height = max(float(geometry["height"]), 1.0)

        # Same-line candidate to the right.
        vertical_difference = abs(
            float(geometry["center_y"]) - float(label_geometry["center_y"])
        )
        horizontal_gap = max(
            0.0,
            float(geometry["x_min"]) - float(label_geometry["x_max"]),
        )
        if (
            float(geometry["center_x"]) > float(label_geometry["center_x"])
            and vertical_difference <= 0.8 * max(label_height, candidate_height)
            and horizontal_gap <= 8.0 * max(label_height, candidate_height)
        ):
            distance = horizontal_gap + vertical_difference
            right_candidates.append((distance, candidate))
            continue

        # Candidate below the label, roughly aligned horizontally.
        vertical_gap = max(
            0.0,
            float(geometry["y_min"]) - float(label_geometry["y_max"]),
        )
        horizontal_difference = abs(
            float(geometry["center_x"]) - float(label_geometry["center_x"])
        )
        if (
            float(geometry["center_y"]) > float(label_geometry["center_y"])
            and vertical_gap <= 8.0 * max(label_height, candidate_height)
            and horizontal_difference <= max(label_width, float(geometry["width"]))
        ):
            distance = vertical_gap + horizontal_difference
            below_candidates.append((distance, candidate))

    candidates = right_candidates if right_candidates else below_candidates
    if not candidates:
        return None

    _, selected = min(candidates, key=lambda item: item[0])
    direction = "right" if right_candidates else "below"

    return {
        "page_number": label_region["page_number"],
        "detection_id": selected["id"],
        "text": selected["text"].strip(),
        "ocr_confidence": selected.get("confidence"),
        "direction": direction,
        "mapping_confidence": None,
        "label_source_detection_ids": [label_region["id"]],
    }


def map_headers(
    document: Mapping[str, Any],
    aliases: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """
    Map invoice header labels to nearby OCR values.

    The mapper uses exact normalized aliases and simple spatial proximity.
    It does not rank candidates or calculate mapping confidence.
    """
    if not isinstance(document, Mapping):
        raise HeaderMappingError("Document must be a mapping.")
    if not isinstance(aliases, Mapping) or not aliases:
        raise HeaderMappingError("Aliases must be a non-empty mapping.")

    pages = document.get("pages")
    if not isinstance(pages, list) or not pages:
        raise HeaderMappingError("Document must contain a non-empty pages list.")

    alias_lookup: dict[str, tuple[str, str]] = {}
    aliases_by_field: dict[str, list[str]] = {}

    for field, field_aliases in aliases.items():
        if not isinstance(field, str) or not field.strip():
            raise HeaderMappingError("Alias configuration contains an invalid field name.")
        if isinstance(field_aliases, str) or not isinstance(field_aliases, Sequence):
            raise HeaderMappingError(f"Aliases for '{field}' must be a list of strings.")
        aliases_by_field[field] = []
        for alias in field_aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise HeaderMappingError(f"Invalid alias configured for '{field}'.")
            aliases_by_field[field].append(alias)
            alias_lookup[_normalise(alias)] = (field, alias)

    prepared_pages: list[dict[str, Any]] = []
    all_label_ids_by_page: dict[int, set[Any]] = {}
    labels_by_field: dict[str, list[dict[str, Any]]] = {
        field: [] for field in aliases
    }

    # Validate geometry and identify exact label matches.
    for page in pages:
        page_number = page.get("page_number")
        regions = page.get("text_regions")
        if not isinstance(regions, list):
            raise HeaderMappingError(f"Page {page_number} must contain text_regions.")

        prepared_regions: list[dict[str, Any]] = []
        for region in regions:
            if not isinstance(region, Mapping):
                raise HeaderMappingError(f"Page {page_number} contains an invalid text region.")
            if not isinstance(region.get("text"), str):
                raise HeaderMappingError(
                    f"Page {page_number}, detection {region.get('id')}: text must be a string."
                )
            prepared = dict(region)
            prepared["geometry"] = _get_geometry(region, page_number)
            prepared["page_number"] = page_number
            prepared_regions.append(prepared)

        prepared_pages.append({"page_number": page_number, "text_regions": prepared_regions})
        all_label_ids_by_page[page_number] = set()

        for region in prepared_regions:
            text = region["text"].strip()
            normalized = _normalise(text)

            # Exact label in its own OCR detection.
            alias_entry = alias_lookup.get(normalized)
            if alias_entry:
                field, alias = alias_entry
                label = {
                    "field": field,
                    "alias": alias,
                    "region": region,
                    "inline_value": None,
                }
                labels_by_field[field].append(label)
                all_label_ids_by_page[page_number].add(region["id"])
                continue

            # Label and value in one OCR detection.
            for field, field_aliases in aliases_by_field.items():
                inline_match = _find_inline_value(text, field_aliases)
                if inline_match:
                    alias, inline_value = inline_match
                    label = {
                        "field": field,
                        "alias": alias,
                        "region": region,
                        "inline_value": inline_value,
                    }
                    labels_by_field[field].append(label)
                    all_label_ids_by_page[page_number].add(region["id"])
                    break

    results: dict[str, Any] = {
        "document_name": document.get("document_name"),
        "header_fields": {},
    }

    for field, field_aliases in aliases_by_field.items():
        field_labels = labels_by_field[field]
        field_result: dict[str, Any] = {
            "value": None,
            "mapping_confidence": None,
            "label_matches": [],
            "candidates": [],
            "selected_candidate": None,
            "review_required": False,  # Retained for compatibility; review is disabled in the MVP.
            "review_reasons": [],
            "warnings": [],
            "status": "missing",
        }

        if not field_labels:
            results["header_fields"][field] = field_result
            continue

        page_lookup = {
            page["page_number"]: page["text_regions"] for page in prepared_pages
        }

        for label in field_labels:
            region = label["region"]
            alias = label["alias"]
            field_result["label_matches"].append({
                "standard_field": field,
                "matched_alias": alias,
                "matched_text": region["text"],
                "page_number": region["page_number"],
                "source_detection_ids": [region["id"]],
                "is_reconstructed": False,
            })

            # Avoid assigning a generic Date label to invoice_date.
            if field == "invoice_date" and _is_generic_date_alias(alias):
                field_result["warnings"].append("generic_date_label_is_ambiguous")
                continue

            if label["inline_value"]:
                candidate = {
                    "page_number": region["page_number"],
                    "detection_id": region["id"],
                    "text": label["inline_value"],
                    "ocr_confidence": region.get("confidence"),
                    "direction": "inline",
                    "mapping_confidence": None,
                    "label_source_detection_ids": [region["id"]],
                }
            else:
                candidate = _find_nearby_value(
                    region,
                    page_lookup[region["page_number"]],
                    all_label_ids_by_page[region["page_number"]],
                )

            if candidate:
                field_result["candidates"].append(candidate)

        if field_result["candidates"]:
            # Prefer the first matched label's nearest value. This is a simple,
            # deterministic MVP rule, not a confidence-based decision.
            selected = field_result["candidates"][0]
            field_result["value"] = selected["text"]
            field_result["selected_candidate"] = selected
            field_result["status"] = "mapped"
            if len(field_result["candidates"]) > 1:
                field_result["warnings"].append("multiple_header_values_found")
        elif field_result["warnings"]:
            field_result["status"] = "ambiguous"
        else:
            field_result["status"] = "label_found_value_not_found"
            field_result["warnings"].append("label_found_but_value_not_found")

        results["header_fields"][field] = field_result

    logger.info(
        "Header mapping completed for document '%s'. Fields mapped: %d of %d.",
        document.get("document_name", "unknown"),
        sum(
            1 for item in results["header_fields"].values()
            if item["value"] is not None
        ),
        len(results["header_fields"]),
    )
    return results

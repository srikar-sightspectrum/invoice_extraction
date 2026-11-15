"""
Coordinate Processor
====================

Calculates geometric properties for PaddleOCR text detections.

The original OCR data and bounding boxes are preserved.
Derived properties are added under a new "geometry" field.
"""

import copy
import logging
import math

from typing import Any


logger = logging.getLogger(__name__)


class CoordinateProcessingError(Exception):
    """Raised when OCR coordinate data cannot be processed."""


def _is_valid_coordinate(value: Any) -> bool:
    """Check whether a coordinate is a finite integer or float."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _calculate_geometry(
    bounding_box: list
) -> dict[str, int | float]:
    """
    Calculate geometric properties from a four-point bounding box.

    Args:
        bounding_box: Four corner points in [x, y] format.

    Returns:
        Axis-aligned geometry properties.

    Raises:
        CoordinateProcessingError: If the bounding box is invalid.
    """
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise CoordinateProcessingError(
            "Bounding box must contain exactly four corner points."
        )

    x_coordinates = []
    y_coordinates = []

    for point_index, point in enumerate(bounding_box):
        if not isinstance(point, list) or len(point) != 2:
            raise CoordinateProcessingError(
                f"Invalid coordinate pair at point {point_index + 1}."
            )

        x, y = point

        if not _is_valid_coordinate(x) or not _is_valid_coordinate(y):
            raise CoordinateProcessingError(
                f"Invalid coordinate values at point {point_index + 1}."
            )

        x_coordinates.append(x)
        y_coordinates.append(y)

    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "center_x": (x_min + x_max) / 2,
        "center_y": (y_min + y_max) / 2,
        "width": x_max - x_min,
        "height": y_max - y_min
    }


def process_document(document: dict[str, Any]) -> dict[str, Any]:
    """
    Process all OCR detections in every page of a document.

    The original document is not modified. A deep copy is created,
    and derived geometry is added to each text region.

    Args:
        document: OCR document dictionary.

    Returns:
        A new document dictionary containing geometry properties.

    Raises:
        CoordinateProcessingError: If the document structure or
            coordinate data is invalid.
    """
    if not isinstance(document, dict):
        raise CoordinateProcessingError(
            "Document must be a dictionary."
        )

    pages = document.get("pages")

    if not isinstance(pages, list):
        raise CoordinateProcessingError(
            "Document must contain a valid pages list."
        )

    processed_document = copy.deepcopy(document)
    total_processed = 0

    for page_index, page in enumerate(processed_document["pages"]):
        if not isinstance(page, dict):
            raise CoordinateProcessingError(
                f"Invalid page structure at index {page_index}."
            )

        page_number = page.get("page_number", page_index + 1)
        text_regions = page.get("text_regions")

        if not isinstance(text_regions, list):
            raise CoordinateProcessingError(
                f"Page {page_number}: text_regions must be a list."
            )

        for region in text_regions:
            if not isinstance(region, dict):
                raise CoordinateProcessingError(
                    f"Page {page_number}: invalid text region."
                )

            region_id = region.get("id", "unknown")
            bounding_box = region.get("bounding_box")

            try:
                geometry = _calculate_geometry(bounding_box)
            except CoordinateProcessingError as error:
                raise CoordinateProcessingError(
                    f"Page {page_number}, detection {region_id}: {error}"
                ) from error

            region["geometry"] = geometry
            total_processed += 1

    logger.debug(
        "Processed document '%s': %d text regions across %d pages.",
        document.get("document_name", "unknown"),
        total_processed,
        len(pages)
    )

    return processed_document

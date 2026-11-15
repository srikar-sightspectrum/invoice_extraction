"""
Simplified invoice header validator.

Preserves OCR-extracted values and adds basic normalized values and
validation statuses. Missing or invalid fields do not stop JSON generation.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class HeaderValidationError(Exception):
    """Raised when header validation input has an invalid structure."""


DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)

GSTIN_PATTERN = re.compile(
    r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$"
)
AMOUNT_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")

DATE_FIELDS = {"invoice_date", "due_date"}
AMOUNT_FIELDS = {"subtotal", "tax_amount", "discount", "total_amount"}
TEXT_FIELDS = {
    "invoice_number",
    "po_number",
    "supplier_name",
    "buyer_name",
    "payment_terms",
}


def _normalize_date(value: str) -> tuple[str | None, list[str]]:
    """Parse a supported date and normalize it to YYYY-MM-DD."""
    candidate = value.strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, date_format).date().isoformat(), []
        except ValueError:
            continue
    return None, ["invalid_or_unsupported_date"]


def _normalize_amount(value: str) -> tuple[str | None, list[str]]:
    """Parse a basic invoice amount without changing its numeric meaning."""
    candidate = value.strip()
    negative = candidate.startswith("(") and candidate.endswith(")")
    if negative:
        candidate = candidate[1:-1].strip()

    candidate = re.sub(
        r"(?i)^(?:INR|USD|EUR|GBP|RS\.?|RUPEES)\s*",
        "",
        candidate,
    )
    candidate = re.sub(r"^[₹$€£]\s*", "", candidate)
    candidate = candidate.replace(",", "").replace(" ", "")
    candidate = re.sub(r"(?i)(?:INR|USD|EUR|GBP|RS\.?|RUPEES)$", "", candidate)
    candidate = candidate.rstrip("₹$€£")

    if not AMOUNT_PATTERN.fullmatch(candidate):
        return None, ["invalid_numeric_amount"]

    try:
        amount = Decimal(candidate)
    except InvalidOperation:
        return None, ["invalid_numeric_amount"]

    if negative:
        amount = -abs(amount)
    return format(amount, "f"), []


def _validate_value(field_name: str, value: Any) -> dict[str, Any]:
    """Return normalized value, validation status and warnings for one field."""
    result = {
        "original_value": value,
        "normalized_value": None,
        "validation": {
            "is_valid": None,
            "checks": [],
        },
        "warnings": [],
        "review_required": False,  # Retained for compatibility; review is disabled.
    }

    if value is None or (isinstance(value, str) and not value.strip()):
        result["validation"]["checks"] = ["missing_value"]
        result["warnings"].append("value_missing")
        return result

    if not isinstance(value, str):
        result["validation"]["is_valid"] = False
        result["validation"]["checks"] = ["value_is_not_text"]
        result["warnings"].append("value_is_not_text")
        return result

    candidate = value.strip()
    field = field_name.casefold()

    if field in DATE_FIELDS:
        normalized, warnings = _normalize_date(candidate)
        result["normalized_value"] = normalized
        result["validation"]["is_valid"] = normalized is not None
        result["validation"]["checks"] = (
            ["valid_date"] if normalized is not None else warnings
        )
        result["warnings"].extend(warnings)

    elif field == "gstin":
        normalized = re.sub(r"\s+", "", candidate).upper()
        is_valid = bool(GSTIN_PATTERN.fullmatch(normalized))
        result["normalized_value"] = normalized
        result["validation"]["is_valid"] = is_valid
        result["validation"]["checks"] = (
            ["valid_gstin_structure"] if is_valid else ["invalid_gstin_structure"]
        )
        if not is_valid:
            result["warnings"].append("invalid_gstin_structure")

    elif field in AMOUNT_FIELDS:
        normalized, warnings = _normalize_amount(candidate)
        result["normalized_value"] = normalized
        result["validation"]["is_valid"] = normalized is not None
        result["validation"]["checks"] = (
            ["valid_numeric_amount"] if normalized is not None else warnings
        )
        result["warnings"].extend(warnings)

    elif field == "currency":
        result["normalized_value"] = candidate.upper()
        result["validation"]["is_valid"] = None
        result["validation"]["checks"] = ["currency_format_not_verified"]

    else:
        result["normalized_value"] = candidate
        result["validation"]["is_valid"] = True
        result["validation"]["checks"] = ["non_empty_text"]

    return result


def validate_headers(header_results: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate header values from map_headers().

    Missing and invalid values are reported as statuses/warnings. They do not
    trigger human review and do not prevent downstream JSON generation.
    """
    if not isinstance(header_results, Mapping):
        raise HeaderValidationError("Header results must be a mapping.")

    header_fields = header_results.get("header_fields")
    if not isinstance(header_fields, Mapping):
        raise HeaderValidationError(
            "Header results must contain a 'header_fields' mapping."
        )

    validated = deepcopy(dict(header_results))

    for field_name, field_result in validated["header_fields"].items():
        if not isinstance(field_result, dict):
            raise HeaderValidationError(
                f"Header field '{field_name}' must contain an object."
            )

        original_value = field_result.get("value")
        mapper_warnings = list(field_result.get("warnings", []))
        mapper_status = field_result.get("status")
        validation = _validate_value(field_name, original_value)

        # Preserve mapper evidence and combine mapping/validation warnings.
        field_result.update(validation)
        field_result["warnings"] = list(dict.fromkeys(
            mapper_warnings + validation["warnings"]
        ))

        is_valid = field_result["validation"]["is_valid"]
        if mapper_status == "ambiguous" and original_value is None:
            field_result["status"] = "ambiguous"
        elif original_value is None or (
            isinstance(original_value, str) and not original_value.strip()
        ):
            field_result["status"] = "missing"
        elif is_valid is False:
            field_result["status"] = "invalid"
        elif is_valid is None:
            field_result["status"] = "unverified"
        else:
            field_result["status"] = "validated"

        # Explicitly disable review-related behaviour for this MVP.
        field_result["review_required"] = False
        field_result.setdefault("review_reasons", [])

    logger.info(
        "Header validation completed for document '%s'.",
        validated.get("document_name", "unknown"),
    )
    return validated

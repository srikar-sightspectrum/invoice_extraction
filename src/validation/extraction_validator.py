"""Validate invoice headers and line items without blocking JSON output.

The validator checks required values, numeric fields and selected arithmetic
relationships. It never changes extracted business values. Every validation
issue is reported as a warning; no human-review workflow is used.
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_HEADER_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "invoice_date",
    "total_amount",
)
REQUIRED_LINE_ITEM_FIELDS: tuple[str, ...] = ("description", "amount")
NUMERIC_LINE_ITEM_FIELDS: tuple[str, ...] = (
    "quantity",
    "rate",
    "discount",
    "tax_rate",
    "tax_amount",
    "amount",
)


class ExtractionValidationError(ValueError):
    """Raised when validation input has an invalid structure."""


def _issue(
    *, code: str, message: str, location: dict[str, Any]
) -> dict[str, Any]:
    """Build a structured, non-blocking validation warning."""
    return {
        "code": code,
        "severity": "warning",
        "message": message,
        "location": location,
    }


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _parse_decimal(value: Any) -> Decimal | None:
    """Parse supported invoice numeric text without changing the source value."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None

    negative_parentheses = candidate.startswith("(") and candidate.endswith(")")
    if negative_parentheses:
        candidate = candidate[1:-1].strip()

    candidate = re.sub(
        r"(?i)^(?:INR|USD|EUR|GBP|RS\.?|RUPEES)\s*", "", candidate
    )
    candidate = re.sub(r"^[₹$€£]\s*", "", candidate)
    candidate = candidate.replace(",", "").replace(" ", "")
    candidate = re.sub(
        r"(?i)(?:INR|USD|EUR|GBP|RS\.?|RUPEES)$", "", candidate
    )
    candidate = candidate.rstrip("₹$€£")

    if not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
        return None
    try:
        parsed = Decimal(candidate)
    except InvalidOperation:
        return None
    return -abs(parsed) if negative_parentheses else parsed


def _validate_headers(
    document: Mapping[str, Any],
    required_fields: tuple[str, ...],
    issues: list[dict[str, Any]],
) -> None:
    """Check required header values and upstream field-validation results."""
    header_fields = document.get("header_fields")
    if header_fields is None:
        if required_fields:
            issues.append(_issue(
                code="header_fields_missing",
                message="The document does not contain a header_fields mapping.",
                location={"section": "header"},
            ))
        return
    if not isinstance(header_fields, Mapping):
        raise ExtractionValidationError("header_fields must be an object/mapping.")

    for field_name in required_fields:
        result = header_fields.get(field_name)
        location = {"section": "header", "field": field_name}
        if not isinstance(result, Mapping):
            issues.append(_issue(
                code="required_header_field_missing",
                message=f"Required header field '{field_name}' is missing.",
                location=location,
            ))
            continue

        value = result.get("normalized_value")
        if value is None:
            value = result.get("value")
        if not _has_value(value):
            issues.append(_issue(
                code="required_header_value_empty",
                message=f"Required header field '{field_name}' has no value.",
                location=location,
            ))

        validation = result.get("validation")
        if isinstance(validation, Mapping) and validation.get("is_valid") is False:
            issues.append(_issue(
                code="header_field_validation_failed",
                message=f"Header field '{field_name}' failed its field-level validation.",
                location=location,
            ))


def _validate_line_items(
    document: Mapping[str, Any], issues: list[dict[str, Any]]
) -> list[Decimal]:
    """Validate line-item values and return parseable line-item amounts."""
    extraction = document.get("line_item_extraction")
    if extraction is None:
        issues.append(_issue(
            code="line_item_extraction_missing",
            message="No line_item_extraction section was found.",
            location={"section": "line_items"},
        ))
        return []
    if not isinstance(extraction, Mapping):
        raise ExtractionValidationError("line_item_extraction must be an object/mapping.")

    tables = extraction.get("tables")
    if not isinstance(tables, list):
        raise ExtractionValidationError("line_item_extraction.tables must be a list.")
    if not tables:
        issues.append(_issue(
            code="line_item_tables_empty",
            message="No primary line-item tables were available for validation.",
            location={"section": "line_items"},
        ))
        return []

    parsed_amounts: list[Decimal] = []
    for table_index, table in enumerate(tables):
        if not isinstance(table, Mapping):
            raise ExtractionValidationError(
                f"Line-item table at index {table_index} must be an object."
            )
        table_id = table.get("table_id")
        page_number = table.get("page_number")
        line_items = table.get("line_items")
        if not isinstance(line_items, list):
            raise ExtractionValidationError(
                f"line_items in table index {table_index} must be a list."
            )

        for item_index, item in enumerate(line_items):
            if not isinstance(item, Mapping):
                raise ExtractionValidationError(
                    f"Line item {item_index} in table index {table_index} must be an object."
                )
            business_data = item.get("line_item")
            metadata = item.get("extraction_metadata", {})
            if not isinstance(business_data, Mapping):
                raise ExtractionValidationError(
                    "Each extracted line item must contain a line_item object."
                )
            if metadata is not None and not isinstance(metadata, Mapping):
                raise ExtractionValidationError(
                    "extraction_metadata must be an object when provided."
                )
            metadata = metadata if isinstance(metadata, Mapping) else {}
            location = {
                "section": "line_items",
                "table_id": table_id,
                "page_number": page_number,
                "line_item_index": metadata.get("line_item_index", item_index + 1),
                "source_row_id": metadata.get("source_row_id"),
            }

            for field_name in REQUIRED_LINE_ITEM_FIELDS:
                if not _has_value(business_data.get(field_name)):
                    issues.append(_issue(
                        code="required_line_item_value_missing",
                        message=f"Required line-item field '{field_name}' is missing.",
                        location={**location, "field": field_name},
                    ))

            parsed_values: dict[str, Decimal] = {}
            for field_name in NUMERIC_LINE_ITEM_FIELDS:
                value = business_data.get(field_name)
                if not _has_value(value):
                    continue
                parsed = _parse_decimal(value)
                if parsed is None:
                    issues.append(_issue(
                        code="line_item_numeric_value_invalid",
                        message=f"Line-item field '{field_name}' is not a supported numeric value.",
                        location={**location, "field": field_name},
                    ))
                    continue
                parsed_values[field_name] = parsed

            if "amount" in parsed_values:
                parsed_amounts.append(parsed_values["amount"])

            # Avoid assuming a calculation formula when invoice adjustments exist.
            has_adjustments = any(
                _has_value(business_data.get(field))
                for field in ("discount", "tax_rate", "tax_amount")
            )
            if (
                not has_adjustments
                and all(field in parsed_values for field in ("quantity", "rate", "amount"))
            ):
                expected = parsed_values["quantity"] * parsed_values["rate"]
                actual = parsed_values["amount"]
                if abs(expected - actual) > Decimal("0.02"):
                    issues.append(_issue(
                        code="line_item_arithmetic_mismatch",
                        message=(
                            "Quantity multiplied by rate does not match the extracted "
                            "line-item amount within the 0.02 tolerance."
                        ),
                        location=location,
                    ))

    return parsed_amounts


def _validate_subtotal(
    document: Mapping[str, Any],
    line_item_amounts: list[Decimal],
    issues: list[dict[str, Any]],
    tolerance: Decimal,
) -> None:
    """Compare the sum of extracted line-item amounts with header subtotal."""
    header_fields = document.get("header_fields")
    if not isinstance(header_fields, Mapping):
        return
    subtotal_result = header_fields.get("subtotal")
    if not isinstance(subtotal_result, Mapping):
        return
    subtotal_value = subtotal_result.get("normalized_value")
    if subtotal_value is None:
        subtotal_value = subtotal_result.get("value")
    subtotal = _parse_decimal(subtotal_value)
    if subtotal is None or not line_item_amounts:
        return

    extracted_sum = sum(line_item_amounts, Decimal("0"))
    if abs(extracted_sum - subtotal) > tolerance:
        issues.append(_issue(
            code="subtotal_line_items_mismatch",
            message=(
                "The sum of extracted line-item amounts differs from the invoice "
                "subtotal. Check discounts, taxes, rounding, and whether all line "
                "items were extracted."
            ),
            location={"section": "header", "field": "subtotal"},
        ))


def validate_extraction(
    extraction_document: Mapping[str, Any],
    *,
    required_header_fields: tuple[str, ...] = DEFAULT_REQUIRED_HEADER_FIELDS,
    arithmetic_tolerance: Decimal = Decimal("0.02"),
) -> dict[str, Any]:
    """Return a copy with validation warnings; never block downstream JSON output."""
    if not isinstance(extraction_document, Mapping):
        raise ExtractionValidationError("extraction_document must be a mapping.")
    if (
        isinstance(required_header_fields, str)
        or not isinstance(required_header_fields, tuple)
        or any(not isinstance(field, str) or not field.strip() for field in required_header_fields)
    ):
        raise ExtractionValidationError(
            "required_header_fields must be a tuple of non-empty field names."
        )
    try:
        tolerance = Decimal(str(arithmetic_tolerance))
    except (InvalidOperation, ValueError) as exc:
        raise ExtractionValidationError(
            "arithmetic_tolerance must be a non-negative decimal value."
        ) from exc
    if not tolerance.is_finite() or tolerance < 0:
        raise ExtractionValidationError(
            "arithmetic_tolerance must be a finite, non-negative decimal value."
        )

    output_document = deepcopy(dict(extraction_document))
    issues: list[dict[str, Any]] = []
    _validate_headers(output_document, required_header_fields, issues)
    line_item_amounts = _validate_line_items(output_document, issues)
    _validate_subtotal(output_document, line_item_amounts, issues, tolerance)

    warning_count = len(issues)
    output_document["extraction_validation"] = {
        "status": "passed" if not issues else "issues_found",
        "is_valid": not issues,
        "error_count": 0,
        "warning_count": warning_count,
        "issue_count": len(issues),
        "issues": issues,
    }

    logger.info(
        "Extraction validation completed for '%s': %d warning(s).",
        output_document.get("document_name", "unknown"),
        warning_count,
    )
    return output_document

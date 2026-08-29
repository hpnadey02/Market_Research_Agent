# ==============================================================================
# validators.py — Fail-Safe Input Validation
# ==============================================================================
# Validates all user inputs with clear, human-readable error messages.
# Returns a structured result dict so the UI can display precise stage failures.
# ==============================================================================

from typing import Dict, Any, Tuple
from src.logger import get_logger

logger = get_logger("Validator")


def validate_inputs(
    metal: str, country: str, price: str, weekly_change: str
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Validate user inputs for the market research pipeline.

    Args:
        metal: Metal name string.
        country: Country name string.
        price: Current price (as string from UI).
        weekly_change: Weekly percentage change (as string from UI).

    Returns:
        Tuple of:
            - is_valid (bool): True if all inputs are valid.
            - validated_data (dict): Cleaned and typed data if valid, else empty.
            - error_message (str): Human-readable error if invalid, else empty.
    """
    errors = []

    # ── Metal Name ───────────────────────────────────────────────────────────
    metal_clean = metal.strip() if metal else ""
    if not metal_clean:
        errors.append("Metal name cannot be empty. Please enter a valid metal "
                       "(e.g., Steel, Copper, Aluminum).")

    # ── Country ──────────────────────────────────────────────────────────────
    country_clean = country.strip() if country else ""
    if not country_clean:
        errors.append("Country name cannot be empty. Please enter a valid country "
                       "(e.g., India, USA, China).")

    # ── Price ────────────────────────────────────────────────────────────────
    price_value = None
    price_str = price.strip() if price else ""
    if not price_str:
        errors.append("Current price cannot be empty. Please enter a numeric value.")
    else:
        try:
            price_value = float(price_str.replace(",", ""))
            if price_value <= 0:
                errors.append(
                    f"Price must be a positive number. You entered: {price_str}"
                )
        except ValueError:
            errors.append(
                f"Price must be a valid number. '{price_str}' is not numeric. "
                "Please remove any currency symbols and try again."
           )

    # ── Weekly % Change ──────────────────────────────────────────────────────
    change_value = None
    change_str = weekly_change.strip() if weekly_change else ""
    if not change_str:
        errors.append("Weekly % change cannot be empty. Please enter a numeric value "
                       "(e.g., -2.5 or 3.1).")
    else:
        # Strip trailing % if user included it
        change_str_clean = change_str.rstrip("%").strip()
        try:
            change_value = float(change_str_clean)
        except ValueError:
            errors.append(
                f"Weekly % change must be a valid number. '{weekly_change}' is not "
                "numeric. Please enter a value like -2.5 or 3.1."
            )

     # ── Result ───────────────────────────────────────────────────────────────
    if errors:
        combined = " | ".join(errors)
        logger.warning("Input validation failed: %s", combined)
        return False, {}, combined

    validated = {
        "metal": metal_clean.title(),
        "country": country_clean.title(),
        "price": price_value,
        "weekly_change": change_value,
    }
    logger.info("Input validation passed: %s", validated)
    return True, validated, ""

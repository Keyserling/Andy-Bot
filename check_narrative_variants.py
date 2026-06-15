"""Simple local check for Andy Bot narrative variety."""

from __future__ import annotations

import random

from app import build_email
from narratives import NARRATIVE_LIBRARY, PERSONAS

FORBIDDEN_EMAIL_TEXT = (
    "For [persona] teams",
    "Metabolon helps",
    "could be valuable",
    "align with",
    "[Your Name]",
)


def main() -> None:
    random.seed(7)
    contacts = [
        build_email(f"Test Contact {index}", "ExampleCo", "Discovery")
        for index in range(20)
    ]
    variant_ids = {contact.narrative_variant_id for contact in contacts}

    if len(PERSONAS) != 6:
        raise AssertionError(f"Expected 6 personas, found {len(PERSONAS)}")

    for persona in PERSONAS:
        narrative_set = NARRATIVE_LIBRARY[persona]
        if len(narrative_set["subjects"]) != 10:
            raise AssertionError(f"{persona} must have 10 subject lines")
        if len(narrative_set["narratives"]) != 10:
            raise AssertionError(f"{persona} must have 10 narratives")

    for contact in contacts:
        for phrase in FORBIDDEN_EMAIL_TEXT:
            if phrase in contact.email:
                raise AssertionError(
                    f"Forbidden phrase {phrase!r} found in {contact.narrative_variant_id}"
                )

    if len(variant_ids) < 6:
        raise AssertionError(
            f"Expected at least 6 different narrative_variant_id values, found {len(variant_ids)}"
        )

    print(f"Generated 20 emails with {len(variant_ids)} narrative_variant_id values.")


if __name__ == "__main__":
    main()

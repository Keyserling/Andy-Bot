"""Simple local check for Andy Bot narrative variety."""

from __future__ import annotations

import random

import pandas as pd

from app import build_email, classify_persona
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

    if len(PERSONAS) != 7:
        raise AssertionError(f"Expected 7 personas, found {len(PERSONAS)}")

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

    expected_greetings = {
        "Mélanie Dupont": "Dear Mélanie,",
        "José Alvarez": "Dear José,",
        "María Garcia": "Dear María,",
        "A Smith": "Dear Colleague,",
    }
    for name, greeting in expected_greetings.items():
        generated = build_email(name, "ExampleCo", "Discovery")
        actual_greeting = generated.email.split("\n", 1)[0]
        if actual_greeting != greeting:
            raise AssertionError(f"Expected {greeting!r}, got {actual_greeting!r}")

    biomarker_contact = pd.Series(
        {"Title": "Director, Translational Biomarker Clinical Development"}
    )
    biomarker_classification = classify_persona(biomarker_contact, ["Title"])
    if biomarker_classification.persona != "Biomarkers / Bioanalysis":
        raise AssertionError("Biomarker keywords must outrank clinical development")

    operations_contact = pd.Series({"Title": "Process Excellence Lead"})
    operations_classification = classify_persona(operations_contact, ["Title"])
    operations_email = build_email(
        "Jordan Example",
        "ExampleCo",
        operations_classification.persona,
        operations_classification.confidence_score,
        operations_classification.matched_keyword,
    )
    if operations_email.email != "Review manually":
        raise AssertionError("Operations contacts must be marked for manual review")

    pharmacology_contact = pd.Series({"Title": "Clinical Pharmacology PK/PD Lead"})
    pharmacology_classification = classify_persona(pharmacology_contact, ["Title"])
    if pharmacology_classification.persona != "Clinical Pharmacology":
        raise AssertionError("Clinical Pharmacology should use its dedicated persona")

    print(f"Generated 20 emails with {len(variant_ids)} narrative_variant_id values.")


if __name__ == "__main__":
    main()

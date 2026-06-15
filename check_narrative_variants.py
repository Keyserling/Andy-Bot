"""Simple local check for Andy Bot narrative variety."""

from __future__ import annotations

import random
from email import message_from_bytes
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from app import (
    build_email,
    classify_persona,
    generate_outreach_table,
    validate_contact_integrity,
)
from metabolon_knowledge import recommend_metabolon_story
from draft_exports import EMLDraftProvider, SENDER_NOT_CONFIGURED_NOTE
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

    if len(PERSONAS) != 8:
        raise AssertionError(f"Expected 8 personas, found {len(PERSONAS)}")

    for persona in PERSONAS:
        narrative_set = NARRATIVE_LIBRARY[persona]
        if len(narrative_set["subjects"]) != 10:
            raise AssertionError(f"{persona} must have 10 subject lines")
        if len(narrative_set["use_cases"]) != 10:
            raise AssertionError(f"{persona} must have 10 use cases")
        if len(narrative_set["benefits"]) != 3:
            raise AssertionError(f"{persona} must have 3 benefits")

    for contact in contacts:
        for phrase in FORBIDDEN_EMAIL_TEXT:
            if phrase in contact.email:
                raise AssertionError(
                    f"Forbidden phrase {phrase!r} found in {contact.narrative_variant_id}"
                )

    linkedin_text = (
        "LinkedIn profile summary with metabolomics-relevant publication history "
        "and translational research leadership."
    )
    linkedin_rows = generate_outreach_table(
        pd.DataFrame(
            [
                {
                    "Name": "LinkedIn Example",
                    "Company": "ExampleCo",
                    "Email": "linkedin@example.com",
                    "Title": "Discovery Scientist",
                    "LinkedIn Content": linkedin_text,
                }
            ]
        )
    )
    if linkedin_rows.loc[0, "LinkedIn Content Available"] != "Yes":
        raise AssertionError("LinkedIn Content rows must be marked available")
    if linkedin_rows.loc[0, "LinkedIn Content Preview"] != linkedin_text[:200]:
        raise AssertionError(
            "LinkedIn Content preview must contain the first 200 characters"
        )

    normal_rows = generate_outreach_table(
        pd.DataFrame(
            [
                {
                    "Name": "Normal Example",
                    "Company": "ExampleCo",
                    "Email": "normal@example.com",
                    "Title": "Discovery Scientist",
                }
            ]
        )
    )
    if normal_rows.loc[0, "LinkedIn Content Available"] != "No":
        raise AssertionError("Rows without LinkedIn Content must still work")
    if normal_rows.loc[0, "LinkedIn Content Preview"] != "":
        raise AssertionError("Rows without LinkedIn Content must have an empty preview")

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
        persona_confidence_score=operations_classification.confidence_score,
        matched_keyword=operations_classification.matched_keyword,
    )
    if operations_email.email != "Review manually":
        raise AssertionError("Operations contacts must be marked for manual review")

    pharmacology_contact = pd.Series({"Title": "Clinical Pharmacology PK/PD Lead"})
    pharmacology_classification = classify_persona(pharmacology_contact, ["Title"])
    if pharmacology_classification.persona != "Clinical Pharmacology":
        raise AssertionError("Clinical Pharmacology should use its dedicated persona")


    story_cases = (
        (
            "Frank Oellien",
            "Computational Biology",
            "Computational Drug Discovery",
            "",
            "drug discovery",
            "Bioinformatics / Multiomics Software",
        ),
        (
            "Max Eberle",
            "Discovery",
            "Principal Scientist Lab Head",
            "",
            "principal scientist",
            "Global Discovery Panel",
        ),
        (
            "Alex Phipps",
            "Clinical Pharmacology",
            "Clinical Pharmacology Oncology",
            "Oncology",
            "clinical pharmacology",
            "Biopharma Services",
        ),
        (
            "Hayato Yamazaki inflammation",
            "Immunology",
            "Immunology Clinical Development",
            "inflammation",
            "immunology",
            "Lipidomics",
        ),
        (
            "Hayato Yamazaki broad",
            "Immunology",
            "Immunology Clinical Development",
            "autoimmune",
            "immunology",
            "Global Discovery Panel",
        ),
        (
            "Zeljana Koletic",
            "Safety / Quality",
            "Risk Management",
            "",
            "risk management",
            "Global Discovery Panel",
        ),
    )
    for label, persona, title, therapeutic_area, matched_keyword, expected in story_cases:
        story = recommend_metabolon_story(
            persona, title, therapeutic_area, matched_keyword
        )
        if story.recommended_offering != expected:
            raise AssertionError(
                f"{label} expected {expected}, got {story.recommended_offering}"
            )

    operations_story = recommend_metabolon_story(
        "Operations / Low Priority", "Process Excellence Lead", "", "process excellence"
    )
    if operations_story.recommended_offering:
        raise AssertionError("Operations contacts must not receive an offering")

    integrity_cases = (
        ("Mario", "Bayer", "mario@bayer.com", "", "GREEN"),
        ("AbbVie exact", "AbbVie", "lead@abbvie.com", "", "GREEN"),
        ("AbbVie suffix", "Abbvie Deutschland Gmbh", "lead@abbvie.com", "", "GREEN"),
        ("Missing email", "Novartis", "", "Novartis", "YELLOW"),
        ("Alex Phipps", "AstraZeneca", "alex.phipps@bayer.com", "", "RED"),
        ("Alex Phipps suggested", "Bayer", "alex.phipps@bayer.com", "", "GREEN"),
    )
    for label, company, email, linkedin_company, expected_status in integrity_cases:
        integrity = validate_contact_integrity(company, email, linkedin_company)
        if integrity.status != expected_status:
            raise AssertionError(
                f"{label} expected {expected_status}, got {integrity.status}: {integrity.reason}"
            )

    sample_email = build_email("Taylor Example", "ExampleCo", "Discovery").email
    required_lines = (
        "My name is Helmut von Keyserling, and I support ExampleCo as Strategic Account Manager at Metabolon.",
        "Many discovery teams are using metabolomics to ",
        "This type of data can help:",
        "For this contact, the best Metabolon angle is Global Discovery Panel",
        "Metabolon can support generating broad biochemical evidence for target biology",
        "If this is of interest, I would be happy to briefly introduce our approach and learn how your team is thinking about this area.",
    )
    for required_line in required_lines:
        if required_line not in sample_email:
            raise AssertionError(f"Missing required email text: {required_line!r}")

    draft_rows = pd.DataFrame(
        [{"To": "recipient@example.com", "Subject": "Subject", "Body": sample_email}]
    )
    unset_zip = EMLDraftProvider(sender_email="").export(draft_rows)
    with ZipFile(BytesIO(unset_zip)) as archive:
        unset_message = message_from_bytes(archive.read("contact_001.eml"))
    if unset_message.get("From") is not None:
        raise AssertionError("EML must not set From when sender is not configured")
    if SENDER_NOT_CONFIGURED_NOTE not in unset_message.get_payload(decode=True).decode():
        raise AssertionError("EML must include sender selection note when unset")

    configured_zip = EMLDraftProvider(sender_email="configured").export(draft_rows)
    with ZipFile(BytesIO(configured_zip)) as archive:
        configured_message = message_from_bytes(archive.read("contact_001.eml"))
    expected_from = "Helmut von Keyserling <helmut.vonkeyserling@metabolon.com>"
    if configured_message.get("From") != expected_from:
        raise AssertionError("EML must set Helmut as From when sender is configured")

    print(f"Generated 20 emails with {len(variant_ids)} narrative_variant_id values.")


if __name__ == "__main__":
    main()

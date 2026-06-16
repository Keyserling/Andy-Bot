"""Local checks for Outreach Wording Engine V2."""

from __future__ import annotations

from email import message_from_bytes
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from app import (
    build_email,
    build_outlook_graph_draft_table,
    classify_persona,
    generate_outreach_table,
    validate_contact_integrity,
)
from draft_exports import EMLDraftProvider, SENDER_NOT_CONFIGURED_NOTE
from metabolon_knowledge import recommend_metabolon_story

FORBIDDEN_EMAIL_TEXT = (
    "your work appears connected to",
    "your responsibilities suggest",
    "your role appears close to",
    "Your role at",
    "appears connected",
    "For your team, that may be useful",
    "Recommended offering",
    "Persona name",
)


def main() -> None:
    sample = build_email("Taylor Example", "ExampleCo", "Discovery")
    if sample.narrative_variant_id != "ENGINE-V2":
        raise AssertionError("Engine V2 must use one deterministic variant id")
    if sample.email != build_email("Taylor Example", "ExampleCo", "Discovery").email:
        raise AssertionError("Identical inputs must generate identical email text")
    if "+49 176 61356899" not in sample.email:
        raise AssertionError("Signature must include Helmut's phone number")
    if "Given your role at ExampleCo, I thought this might be relevant." not in sample.email:
        raise AssertionError("Missing required no-LinkedIn fallback observation")
    for phrase in FORBIDDEN_EMAIL_TEXT:
        if phrase in sample.email:
            raise AssertionError(f"Forbidden phrase found: {phrase!r}")

    linkedin_email = build_email(
        "Casey Example",
        "Bayer Aktiengesellschaft",
        "Clinical Pharmacology",
        linkedin_content_available="Yes",
        linkedin_content_preview="Recent focus on PK/PD strategy in oncology and target engagement.",
    )
    if "Your recent focus on PK/PD caught my attention." not in linkedin_email.email:
        raise AssertionError("LinkedIn-derived observation should lead the email")
    if linkedin_email.personalization_source != "LinkedIn":
        raise AssertionError("LinkedIn source should be recorded when a signal is used")

    rows = generate_outreach_table(
        pd.DataFrame(
            [
                {
                    "Name": "LinkedIn Example",
                    "Company": "ExampleCo",
                    "Email": "linkedin@example.com",
                    "Title": "Discovery Scientist",
                    "LinkedIn Content": "Published work on biomarker discovery.",
                }
            ]
        )
    )
    if rows.loc[0, "LinkedIn Content Available"] != "Yes":
        raise AssertionError("LinkedIn Content rows must be marked available")
    if rows.loc[0, "LinkedIn Content Present"] != "TRUE":
        raise AssertionError("LinkedIn debug content-present flag must be TRUE")

    no_linkedin_rows = generate_outreach_table(
        pd.DataFrame(
            [
                {
                    "Name": "Normal Example",
                    "Company": "AbbVie Deutschland Gmbh",
                    "Email": "normal@abbvie.com",
                    "Title": "Discovery Scientist",
                }
            ]
        )
    )
    if no_linkedin_rows.loc[0, "Personalization Source"] != "LinkedIn fallback":
        raise AssertionError("Rows without LinkedIn signal must show LinkedIn fallback")
    if "Given your role at AbbVie, I thought this might be relevant." not in no_linkedin_rows.loc[0, "Email"]:
        raise AssertionError("Fallback observation must use the display company brand")

    biomarker_classification = classify_persona(
        pd.Series({"Title": "Director, Translational Biomarker Clinical Development"}),
        ["Title"],
    )
    if biomarker_classification.persona != "Biomarkers / Bioanalysis":
        raise AssertionError("Biomarker keywords must outrank clinical development")

    operations_email = build_email("Jordan Example", "ExampleCo", "Operations / Low Priority")
    if operations_email.email != "Review manually":
        raise AssertionError("Operations contacts must be marked for manual review")

    story_cases = (
        ("Computational Biology", "Computational Drug Discovery", "", "drug discovery", "Multiomics"),
        ("Discovery", "Principal Scientist Lab Head", "", "principal scientist", "Global Discovery Panel"),
        ("Clinical Pharmacology", "Clinical Pharmacology Oncology", "Oncology", "clinical pharmacology", "Biopharma Services"),
        ("Immunology", "Immunology Clinical Development", "inflammation", "immunology", "Lipidomics"),
    )
    for persona, title, area, keyword, expected in story_cases:
        story = recommend_metabolon_story(persona, title, area, keyword)
        if story.recommended_offering != expected:
            raise AssertionError(f"{persona} expected {expected}, got {story.recommended_offering}")

    integrity = validate_contact_integrity("AstraZeneca", "alex.phipps@bayer.com", "")
    if integrity.status != "RED":
        raise AssertionError("Conflicting company/email data must be RED")

    draft_rows = pd.DataFrame([{"To": "recipient@example.com", "Subject": "Subject", "Body": sample.email}])
    unset_zip = EMLDraftProvider(sender_email="").export(draft_rows)
    with ZipFile(BytesIO(unset_zip)) as archive:
        unset_message = message_from_bytes(archive.read("contact_001.eml"))
    if SENDER_NOT_CONFIGURED_NOTE not in unset_message.get_payload(decode=True).decode():
        raise AssertionError("EML must include sender selection note when unset")

    graph_rows = pd.DataFrame(
        [
            {"To": "ok@example.com", "Subject": "Subject", "Email": sample.email, "Integrity Status": "GREEN"},
            {"To": "red@example.com", "Subject": "Review Required", "Email": "Review Required", "Integrity Status": "RED"},
        ]
    )
    graph_drafts = build_outlook_graph_draft_table(graph_rows)
    if graph_drafts.to_dict("records") != [{"To": "ok@example.com", "Subject": "Subject", "Body": sample.email}]:
        raise AssertionError("Microsoft Graph draft creation must skip RED contacts")

    print("Outreach Wording Engine V2 checks passed.")


if __name__ == "__main__":
    main()

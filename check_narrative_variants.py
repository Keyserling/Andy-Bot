"""Local checks for Outreach Engine V5."""

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
    "Metabolomics helps",
    "We offer",
    "Our platform can",
)


def main() -> None:
    sample = build_email("Taylor Example", "ExampleCo", "Discovery")
    if sample.narrative_variant_id != "ENGINE-V5":
        raise AssertionError("Engine V5 must use one deterministic variant id")
    if sample.email != build_email("Taylor Example", "ExampleCo", "Discovery").email:
        raise AssertionError("Identical inputs must generate identical email text")
    if not sample.subject.startswith("Metabolon |"):
        raise AssertionError(
            "Generated subjects must use persona-specific Metabolon subject lines"
        )
    if "+49 176 61356899" not in sample.email:
        raise AssertionError("Signature must include Helmut's phone number")
    if "Strategic Account Manager, Pharma International" not in sample.email:
        raise AssertionError("Signature must include Helmut's title")
    if "hvonkeyserling@metabolon.com" not in sample.email:
        raise AssertionError("Signature must include Helmut's email address")
    if "\nHelmut\n\nStrategic Account Manager" not in sample.email:
        raise AssertionError("First-name greetings must sign with Helmut")
    if (
        "Given your role at ExampleCo, I thought this might be relevant."
        not in sample.email
    ):
        raise AssertionError("Missing required no-LinkedIn fallback observation")
    if "Are we doing enough" in sample.email:
        raise AssertionError(
            "Email should create FOMO without using the goal phrase literally"
        )
    if (
        "What has surprised me is not the growing interest in metabolomics and multiomics"
        in sample.email
    ):
        raise AssertionError("Universal multiomics problem paragraph must be removed")
    if (
        "generating molecular data is usually easier than deciding which biological signals should influence a program"
        not in sample.email
    ):
        raise AssertionError(
            "Default scientific story must use a scientific problem block"
        )
    if (
        "compare notes and share what we have learned across other programs"
        not in sample.email
    ):
        raise AssertionError("CTA should invite discussion rather than sell services")
    if "currently thinks about this area" in sample.email:
        raise AssertionError("Generic company-thinking CTA must be replaced")
    if "not all metabolomics platforms are equivalent" not in sample.email:
        raise AssertionError(
            "Email must challenge equivalence assumptions before the CTA"
        )
    if sample.email.index(
        "not all metabolomics platforms are equivalent"
    ) > sample.email.index("compare notes"):
        raise AssertionError("Differentiation narrative must appear before the CTA")

    persona_problem_cases = (
        (
            "Biomarker Director",
            "Biomarkers / Bioanalysis",
            "biomarker programs generate large amounts of molecular data",
        ),
        (
            "PKPD Lead",
            "Clinical Pharmacology",
            "exposure-response relationships are often easier to quantify",
        ),
        (
            "Oncology Scientist",
            "Oncology",
            "tumor response and resistance are increasingly understood as biological processes",
        ),
        (
            "Immunology Scientist",
            "Immunology",
            "patients with similar clinical presentations can exhibit very different underlying biology",
        ),
        (
            "Translational Leader",
            "Translational / Clinical Development",
            "promising biological findings do not always translate cleanly",
        ),
    )
    persona_problem_texts = []
    for name, persona, expected_problem in persona_problem_cases:
        persona_email = build_email(name, "ExampleCo", persona)
        if expected_problem not in persona_email.email:
            raise AssertionError(
                f"Missing persona-specific problem block for {persona}"
            )
        if (
            "What has surprised me is not the growing interest in metabolomics and multiomics"
            in persona_email.email
        ):
            raise AssertionError(f"Universal multiomics paragraph found for {persona}")
        persona_problem_texts.append(expected_problem)
    if len(set(persona_problem_texts)) != len(persona_problem_texts):
        raise AssertionError("Persona scientific problem blocks must be distinct")
    if "analytical depth, standardization, reproducibility" not in sample.email:
        raise AssertionError(
            "Differentiation narrative must emphasize platform-quality factors"
        )
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
    if (
        "Interesting to see your work at the intersection of oncology and PK/PD strategy."
        not in linkedin_email.email
    ):
        raise AssertionError(
            "LinkedIn-derived observation should use the highest-priority true signal"
        )
    if linkedin_email.personalization_source != "LinkedIn":
        raise AssertionError("LinkedIn source should be recorded when a signal is used")
    if linkedin_email.linkedin_observation_confidence != "High":
        raise AssertionError("LinkedIn observation confidence should be recorded")

    companion_diagnostics_email = build_email(
        "Riley Example",
        "ExampleCo",
        "Biomarkers / Bioanalysis",
        linkedin_content_available="Yes",
        linkedin_content_preview=(
            "Biomarker assay development, Companion Diagnostics, Personalized "
            "Medicine, Patient Stratification, Translational Assay Technologies, "
            "Oncology, Regulatory Strategy, Molecular Diagnostics, Clinical Development"
        ),
    )
    if (
        "Your work across biomarker assay development and companion diagnostics caught my attention."
        not in companion_diagnostics_email.email
    ):
        raise AssertionError(
            "LinkedIn extraction must preserve companion diagnostics and biomarker assay signals"
        )
    if (
        "Your focus on clinical development caught my attention."
        in companion_diagnostics_email.email
    ):
        raise AssertionError(
            "LinkedIn extraction must not collapse specific diagnostics signals into clinical development"
        )

    cdx_email = build_email(
        "Diagnostic Signal Example",
        "ExampleCo",
        "Translational / Clinical Development",
        linkedin_content_available="Yes",
        linkedin_content_preview=(
            "Clinical Development leader with experience in CDx, biomarker assay, "
            "assay validation, diagnostic strategy, and personalized medicine."
        ),
    )
    if (
        "Your work across biomarker assay development and companion diagnostics "
        "caught my attention." not in cdx_email.email
    ):
        raise AssertionError(
            "CDx and biomarker-assay signals must receive a distinctive LinkedIn observation"
        )
    if "Your focus on clinical development caught my attention." in cdx_email.email:
        raise AssertionError(
            "Diagnostic and biomarker signals must outrank generic clinical development"
        )

    full_text_observation = (
        "Your work across biomarker assay development and companion diagnostics "
        "caught my attention."
    )
    truncated_linkedin_rows = generate_outreach_table(
        pd.DataFrame(
            [
                {
                    "Name": "Full Text Example",
                    "Company": "ExampleCo",
                    "Email": "fulltext@example.com",
                    "Title": "Clinical Development Leader",
                    "LinkedIn Content": (
                        "Clinical Development " * 20
                        + " biomarker assay development and companion diagnostics"
                    ),
                }
            ]
        )
    )
    truncated_email = truncated_linkedin_rows.loc[0, "Email"]
    if full_text_observation not in truncated_email:
        raise AssertionError(
            "Email opener must reuse the full-text LinkedIn observation instead of recomputing from the preview"
        )
    if truncated_linkedin_rows.loc[0, "LinkedIn Observation"] != full_text_observation:
        raise AssertionError(
            "Exported LinkedIn Observation must match the email opener"
        )
    if truncated_linkedin_rows.loc[0, "LinkedIn Hook"] != full_text_observation:
        raise AssertionError(
            "Diagnostic LinkedIn Hook must match the exported observation"
        )

    if (
        truncated_linkedin_rows.loc[0, "Selected LinkedIn Signal"]
        != "biomarker assay development / companion diagnostics"
    ):
        raise AssertionError(
            "Selected LinkedIn Signal must come from the full-text observation source"
        )
    if truncated_linkedin_rows.loc[0, "LinkedIn Signal #1"] != "companion diagnostics":
        raise AssertionError(
            "LinkedIn debug ranking must use full LinkedIn text, not the truncated preview"
        )

    formal_email = build_email("Dr. Morgan Smith", "ExampleCo", "Discovery")
    if not formal_email.email.startswith("Dear Dr. Smith,"):
        raise AssertionError("Formal titled contacts must receive a formal greeting")
    if "\nHelmut von Keyserling\n\nStrategic Account Manager" not in formal_email.email:
        raise AssertionError(
            "Formal titled contacts must receive Helmut's full-name signature"
        )

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
    for column in (
        "LinkedIn Observation",
        "LinkedIn Observation Source",
        "LinkedIn Observation Confidence",
        "LinkedIn Signal #1",
        "LinkedIn Signal Score #1",
        "Selected Signal",
        "Selection Reason",
    ):
        if column not in rows.columns:
            raise AssertionError(f"Missing LinkedIn debug column: {column}")

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
    if (
        "Given your role at AbbVie, I thought this might be relevant."
        not in no_linkedin_rows.loc[0, "Email"]
    ):
        raise AssertionError("Fallback observation must use the display company brand")

    biomarker_classification = classify_persona(
        pd.Series({"Title": "Director, Translational Biomarker Clinical Development"}),
        ["Title"],
    )
    if biomarker_classification.persona != "Biomarkers / Bioanalysis":
        raise AssertionError("Biomarker keywords must outrank clinical development")

    operations_email = build_email(
        "Jordan Example", "ExampleCo", "Operations / Low Priority"
    )
    if operations_email.email != "Review manually":
        raise AssertionError("Operations contacts must be marked for manual review")

    story_cases = (
        (
            "Computational Biology",
            "Computational Drug Discovery",
            "",
            "drug discovery",
            "Multiomics",
        ),
        (
            "Discovery",
            "Principal Scientist Lab Head",
            "",
            "principal scientist",
            "Global Discovery Panel",
        ),
        (
            "Clinical Pharmacology",
            "Clinical Pharmacology Oncology",
            "Oncology",
            "clinical pharmacology",
            "Biopharma Services",
        ),
        (
            "Immunology",
            "Immunology Clinical Development",
            "inflammation",
            "immunology",
            "Lipidomics",
        ),
    )
    for persona, title, area, keyword, expected in story_cases:
        story = recommend_metabolon_story(persona, title, area, keyword)
        if story.recommended_offering != expected:
            raise AssertionError(
                f"{persona} expected {expected}, got {story.recommended_offering}"
            )

    integrity = validate_contact_integrity("AstraZeneca", "alex.phipps@bayer.com", "")
    if integrity.status != "RED":
        raise AssertionError("Conflicting company/email data must be RED")

    draft_rows = pd.DataFrame(
        [{"To": "recipient@example.com", "Subject": "Subject", "Body": sample.email}]
    )
    unset_zip = EMLDraftProvider(sender_email="").export(draft_rows)
    with ZipFile(BytesIO(unset_zip)) as archive:
        unset_message = message_from_bytes(archive.read("contact_001.eml"))
    if (
        SENDER_NOT_CONFIGURED_NOTE
        not in unset_message.get_payload(decode=True).decode()
    ):
        raise AssertionError("EML must include sender selection note when unset")

    graph_rows = pd.DataFrame(
        [
            {
                "To": "ok@example.com",
                "Subject": "Subject",
                "Email": sample.email,
                "Integrity Status": "GREEN",
            },
            {
                "To": "red@example.com",
                "Subject": "Review Required",
                "Email": "Review Required",
                "Integrity Status": "RED",
            },
        ]
    )
    graph_drafts = build_outlook_graph_draft_table(graph_rows)
    if graph_drafts.to_dict("records") != [
        {"To": "ok@example.com", "Subject": "Subject", "Body": sample.email}
    ]:
        raise AssertionError("Microsoft Graph draft creation must skip RED contacts")

    print("Outreach Engine V5 checks passed.")


if __name__ == "__main__":
    main()

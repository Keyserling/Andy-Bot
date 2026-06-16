"""Metabolon offering knowledge layer for outreach story selection."""

from __future__ import annotations

from typing import NamedTuple


class MetabolonStory(NamedTuple):
    """Recommended Metabolon story for one contact."""

    primary_capability: str
    recommended_offering: str
    scientific_problem: str
    email_story: str
    confidence: float


OFFERING_FAMILIES = {
    "Biopharma Services": {
        "positioning": (
            "Metabolon supports biopharma teams in interpreting patient sample biology, "
            "drug mechanisms, target biology, biomarker discovery, drug efficacy, "
            "and translational development decisions."
        ),
        "use_cases": (
            "biomarker discovery",
            "drug mechanism / mechanism of action",
            "target biology",
            "drug efficacy",
            "pharmacodynamic biology",
            "evidence for clinical development decisions",
            "patient stratification",
            "responder / non-responder biology",
            "clinical sample interpretation",
        ),
        "best_personas": (
            "Translational / Clinical Development",
            "Clinical Pharmacology",
            "Clinical Biomarkers",
            "Medical Affairs",
            "Oncology",
            "Immunology",
        ),
    },
    "Global Discovery Panel": {
        "positioning": (
            "Broad untargeted metabolomics for hypothesis generation, broad biochemical "
            "measurement, biomarker discovery, mechanism understanding, and "
            "discovery/translational research."
        ),
        "use_cases": (
            "broad biochemical profiling",
            "pathway activity",
            "mechanism of action",
            "target modulation",
            "lead optimization",
            "biomarker discovery",
            "model-to-human translation",
            "patient heterogeneity",
            "functional biology beyond genes/proteins",
        ),
        "best_personas": (
            "Discovery",
            "Translational / Clinical Development",
            "Clinical Biomarkers",
            "Oncology",
            "Immunology",
            "Safety / Quality",
        ),
    },
    "Lipidomics": {
        "positioning": (
            "Focused lipid biology for cardiometabolic disease, inflammation, "
            "obesity, CVRM, MASH/NASH, immunology, and lipid-mediated disease mechanisms."
        ),
        "use_cases": (
            "lipid signaling",
            "inflammatory mechanisms",
            "cardiometabolic risk",
            "obesity biology",
            "MASH/NASH biology",
            "immune-metabolic response",
            "treatment response",
            "patient stratification",
        ),
        "best_personas": (
            "Immunology",
            "Clinical Development",
            "Translational Medicine",
            "CVRM",
            "Obesity",
            "Medical Affairs",
        ),
    },
    "Multiomics": {
        "positioning": (
            "Software and analysis workflows that help integrate metabolomics with "
            "transcriptomics, proteomics, genomics, and other omics data to interpret "
            "biochemical activity and systems-level mechanisms."
        ),
        "use_cases": (
            "multiomics integration",
            "pathway analysis",
            "systems biology",
            "biological interpretation",
            "transcriptomics/proteomics/metabolomics integration",
            "computational biology support",
            "target/mechanism prioritization",
            "data interpretation for AI/computational teams",
        ),
        "best_personas": (
            "Computational Biology",
            "Bioinformatics",
            "Discovery",
            "Translational Informatics",
            "Systems Biology",
        ),
    },
}

LIPID_INFLAMMATION_TERMS = (
    "lipid",
    "lipidomics",
    "inflammation",
    "inflammatory",
    "immune metabolism",
    "immunometabolism",
    "obesity",
    "mash",
    "nash",
    "cvrm",
    "cardiometabolic",
    "cardiovascular",
)

CLINICAL_PHARMACOLOGY_TERMS = (
    "clinical pharmacology",
    "pk/pd",
    "pk",
    "pd",
    "pharmacokinetic",
    "pharmacodynamic",
    "dose",
    "exposure response",
    "drug efficacy",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _story_for(offering: str, problem: str, confidence: float) -> MetabolonStory:
    return MetabolonStory(
        primary_capability=OFFERING_FAMILIES[offering]["positioning"],
        recommended_offering=offering,
        scientific_problem=problem,
        email_story=(
            f"Use {offering.lower()} for {problem}, connecting sample data "
            "to biological mechanisms and clearer program decisions."
        ),
        confidence=confidence,
    )


def recommend_metabolon_story(
    persona: str, title: str, therapeutic_area: str, matched_keyword: str
) -> MetabolonStory:
    """Recommend the Metabolon capability and concise outreach story for a contact."""
    persona_text = (persona or "").lower()
    combined_text = " ".join(
        part for part in (persona, title, therapeutic_area, matched_keyword) if part
    ).lower()

    if "operations" in persona_text or "low priority" in persona_text:
        return MetabolonStory("", "", "", "", 0.0)

    if "computational" in combined_text or "bioinformatics" in combined_text:
        return _story_for(
            "Multiomics",
            "integrating metabolomics with other omics data for pathway and mechanism interpretation",
            0.92,
        )

    if "clinical pharmacology" in persona_text or _contains_any(
        combined_text, CLINICAL_PHARMACOLOGY_TERMS
    ):
        return _story_for(
            "Biopharma Services",
            "connecting PK/PD, drug efficacy, and pharmacodynamic biology in clinical samples",
            0.91,
        )

    if (
        "immunology" in persona_text
        or "immunology" in combined_text
        or "immune" in combined_text
    ):
        if _contains_any(combined_text, LIPID_INFLAMMATION_TERMS):
            return _story_for(
                "Lipidomics",
                "understanding inflammatory, immune-metabolic, or lipid-mediated mechanisms",
                0.88,
            )
        return _story_for(
            "Global Discovery Panel",
            "profiling immune activity, patient heterogeneity, and treatment response",
            0.78,
        )

    if "safety" in persona_text or "quality" in persona_text or "risk" in combined_text:
        return _story_for(
            "Global Discovery Panel",
            "interpreting mechanism and safety biology behind risk, tolerability, or quality findings",
            0.84,
        )

    if "clinical biomarker" in combined_text or "biomarker" in combined_text:
        return _story_for(
            "Global Discovery Panel",
            "discovering and prioritizing biomarkers tied to biological mechanisms and patient stratification",
            0.86,
        )

    if (
        "oncology" in persona_text
        or "oncology" in combined_text
        or "cancer" in combined_text
    ):
        return _story_for(
            "Global Discovery Panel",
            "studying tumor metabolism, treatment response, resistance, and patient heterogeneity",
            0.83,
        )

    if "discovery" in persona_text or "discovery" in combined_text:
        return _story_for(
            "Global Discovery Panel",
            "generating broad biochemical evidence for target biology, mechanisms, and discovery decisions",
            0.87,
        )

    if "translational" in persona_text or "clinical development" in persona_text:
        return _story_for(
            "Biopharma Services",
            "interpreting patient sample biology and treatment-associated metabolic changes",
            0.82,
        )

    return _story_for(
        "Global Discovery Panel",
        "using broad metabolomics to understand biological mechanisms and generate actionable hypotheses",
        0.62,
    )

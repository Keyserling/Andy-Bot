"""Andy Bot Streamlit application for persona-based Metabolon outreach.

Workflow:
1. Upload a CSV contact list.
2. Classify each contact into one persona.
3. Randomly select from persona-specific subject and use-case variants.
4. Generate Name, Company, Persona, Subject, and Email for CSV export.
"""

from __future__ import annotations

import random
import re
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st

from draft_exports import CSVDraftProvider, EMLDraftProvider
from metabolon_knowledge import MetabolonStory, recommend_metabolon_story
from narratives import get_narrative_set

OLD_TO_NEW_PERSONA = {
    "Translational Research": "Translational / Clinical Development",
    "Clinical Development": "Translational / Clinical Development",
    "Clinical Biomarkers": "Biomarkers / Bioanalysis",
    "Bioanalysis": "Biomarkers / Bioanalysis",
    "Safety/Risk": "Safety / Quality",
}

PERSONA_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Biomarkers / Bioanalysis",
        (
            "regulated bioanalysis",
            "translational biomarker",
            "clinical biomarker",
            "companion diagnostic",
            "pd biomarker",
            "pk/pd biomarker",
            "biomarkers",
            "biomarker",
            "bioanalysis",
            "bioanalytical",
            "assays",
            "assay",
            "glp",
            "gclp",
            "biomarker strategy",
        ),
    ),
    (
        "Clinical Pharmacology",
        (
            "clinical pharmacology",
            "exposure response",
            "dose optimization",
            "pharmacokinetics",
            "pharmacodynamics",
            "pk/pd",
            "pharmacokinetic",
            "pharmacodynamic",
            "dmpk",
        ),
    ),
    (
        "Operations / Low Priority",
        (
            "process excellence",
            "operational excellence",
            "site contracting",
            "clinical operations",
            "study operations",
            "quality systems",
            "business process",
            "portfolio operations",
        ),
    ),
    (
        "Computational Biology",
        (
            "computational",
            "ai",
            "in silico",
            "drug discovery",
            "modeling",
            "modelling",
            "cheminformatics",
            "bioinformatics",
            "data science",
            "systems biology",
        ),
    ),
    (
        "Discovery",
        (
            "discovery",
            "principal scientist",
            "lab head",
            "screening",
            "target identification",
            "target discovery",
            "early research",
            "research scientist",
            "biology",
            "multiomics",
            "multi-omics",
        ),
    ),
    (
        "Medical Affairs",
        ("medical affairs",),
    ),
    (
        "Immunology",
        (
            "immunology",
            "autoimmune",
            "inflammation",
            "immune",
        ),
    ),
    (
        "Oncology",
        (
            "oncology",
            "hematology-oncology",
            "haematology-oncology",
            "cancer",
            "tumor",
            "tumour",
            "immuno-oncology",
        ),
    ),
    (
        "Safety / Quality",
        (
            "safety",
            "quality",
            "pharmacovigilance",
            "risk management",
            "patient safety",
            "drug safety",
            "benefit risk",
            "toxicology",
            "toxicity",
        ),
    ),
    (
        "Translational / Clinical Development",
        (
            "clinical development",
            "clinical trial",
            "clinical study",
            "clinical research",
            "development lead",
            "translational medicine",
            "translational science",
            "translational",
            "bench to bedside",
            "human biology",
        ),
    ),
)


def map_persona(persona: str) -> str:
    """Map legacy persona names into the six active personas."""
    return OLD_TO_NEW_PERSONA.get(persona, persona)


class PersonaClassification(NamedTuple):
    """Persona classification details for one contact."""

    persona: str
    confidence_score: float
    matched_keyword: str


class ContactIntegrity(NamedTuple):
    """Contact-data integrity validation details for one contact."""

    status: str
    reason: str
    suggested_company: str
    suggested_title: str


class ContactOutreach(NamedTuple):
    """Generated outreach output for one contact."""

    name: str
    company: str
    to: str
    persona: str
    persona_confidence_score: float
    integrity_status: str
    integrity_reason: str
    suggested_company: str
    suggested_title: str
    subject: str
    email: str
    matched_keyword: str
    narrative_variant_id: str
    metabolon_capability: str
    recommended_offering: str
    scientific_problem: str
    email_story: str
    linkedin_content_available: str
    linkedin_content_preview: str


def find_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find a column by normalized candidate names."""
    if dataframe.empty:
        return None

    normalized = {column.lower().strip(): column for column in dataframe.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def find_name_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely contact-name column in uploaded contact data."""
    direct_match = find_column(
        dataframe,
        ("name", "full_name", "full name", "contact", "contact_name", "contact name"),
    )
    if direct_match:
        return direct_match

    text_columns = [
        column for column in dataframe.columns if dataframe[column].dtype == "object"
    ]
    if text_columns:
        return text_columns[0]
    return dataframe.columns[0] if len(dataframe.columns) else None


def find_company_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely company or account column in uploaded contact data."""
    return find_column(
        dataframe,
        (
            "company",
            "company name",
            "company_name",
            "account",
            "account name",
            "account_name",
            "organization",
            "organisation",
            "employer",
        ),
    )


def find_email_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely recipient email-address column."""
    return find_column(
        dataframe,
        (
            "email",
            "email address",
            "email_address",
            "e-mail",
            "e-mail address",
            "work email",
            "work_email",
            "business email",
            "business_email",
        ),
    )


def find_linkedin_company_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely LinkedIn current-company column."""
    return find_column(
        dataframe,
        (
            "linkedin current company",
            "linkedin_current_company",
            "linkedin company",
            "linkedin_company",
            "current company",
            "current_company",
            "current employer",
            "current_employer",
        ),
    )


def find_linkedin_title_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely LinkedIn current-title column."""
    return find_column(
        dataframe,
        (
            "linkedin current title",
            "linkedin_current_title",
            "linkedin title",
            "linkedin_title",
            "current title",
            "current_title",
            "title",
            "job title",
            "job_title",
        ),
    )


def find_linkedin_content_column(dataframe: pd.DataFrame) -> str | None:
    """Find the optional LinkedIn content column for future personalization hooks."""
    return find_column(
        dataframe,
        (
            "linkedin text",
            "linkedin content",
            "linkedin profile text",
        ),
    )


def find_role_columns(dataframe: pd.DataFrame) -> list[str]:
    """Find title and role columns that should drive persona selection."""
    role_terms = ("title", "role", "position", "job", "function", "department")
    matches = [
        column
        for column in dataframe.columns
        if any(term in column.lower() for term in role_terms)
    ]
    return matches or list(dataframe.columns)


def get_cell_value(contact: pd.Series, column: str | None, fallback: str) -> str:
    """Return a cleaned string value from a contact row."""
    if not column:
        return fallback

    value: Any = contact.get(column, "")
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()


def find_first_name_column(dataframe: pd.DataFrame) -> str | None:
    """Find an explicit first-name column in uploaded contact data."""
    return find_column(
        dataframe,
        ("first name", "first_name", "firstname", "given name", "given_name"),
    )


def is_safe_first_name(first_name: str) -> bool:
    """Return whether a first-name value is safe to use in a greeting."""
    return (
        bool(re.fullmatch(r"[^\W\d_](?:[^\W\d_]|[.'’\-])*", first_name, re.UNICODE))
        and len(first_name) > 1
    )


def get_contact_first_name(name: str, source_first_name: str = "") -> str:
    """Return a non-truncated Unicode first name from contact source data."""
    for candidate in (source_first_name,):
        candidate = candidate.strip()
        if is_safe_first_name(candidate):
            return candidate

    if name == "Unnamed Contact":
        return "Colleague"

    name_without_email = re.sub(r"<[^>]+>", "", name).strip()
    name_without_parentheses = re.sub(r"\([^)]*\)", " ", name_without_email).strip()
    name_parts = re.findall(
        r"[^\W\d_](?:[^\W\d_]|[.'’\-])*", name_without_parentheses, re.UNICODE
    )
    if name_parts and is_safe_first_name(name_parts[0]):
        return name_parts[0]

    return "Colleague"


FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
}

COMPANY_SUFFIXES = (
    "incorporated",
    "inc",
    "limited",
    "ltd",
    "llc",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "group",
    "ag",
    "gmbh",
    "sa",
    "bv",
)

COMMON_PHARMA_COMPANY_DOMAINS = {
    "abbvie": ("abbvie.com",),
    "astrazeneca": ("astrazeneca.com",),
    "bayer": ("bayer.com",),
    "boehringer ingelheim": ("boehringer-ingelheim.com",),
    "bristol myers squibb": ("bms.com",),
    "eli lilly": ("lilly.com",),
    "gilead": ("gilead.com",),
    "glaxosmithkline": ("gsk.com",),
    "johnson johnson": ("jnj.com",),
    "merck": ("merck.com", "msd.com"),
    "novartis": ("novartis.com",),
    "pfizer": ("pfizer.com",),
    "roche": ("roche.com",),
    "sanofi": ("sanofi.com",),
    "takeda": ("takeda.com",),
}

DOMAIN_TO_COMMON_PHARMA_COMPANY = {
    domain: company
    for company, domains in COMMON_PHARMA_COMPANY_DOMAINS.items()
    for domain in domains
}


def extract_email_domain(email: str) -> str:
    """Return a normalized domain from an email address, or an empty string."""
    match = re.search(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", email, re.I)
    return match.group(1).lower() if match else ""


def normalize_company_name(company: str) -> str:
    """Normalize company names and domains for integrity comparisons."""
    normalized = re.sub(r"[^a-z0-9]+", " ", company.lower()).strip()
    tokens = [token for token in normalized.split() if token not in COMPANY_SUFFIXES]
    return " ".join(tokens)


def domain_to_company_key(domain: str) -> str:
    """Return a normalized company key for a known or obvious email domain."""
    if not domain or domain in FREE_EMAIL_DOMAINS:
        return ""
    if domain in DOMAIN_TO_COMMON_PHARMA_COMPANY:
        return DOMAIN_TO_COMMON_PHARMA_COMPANY[domain]
    for known_domain, company in DOMAIN_TO_COMMON_PHARMA_COMPANY.items():
        if domain.endswith(f".{known_domain}"):
            return company
    domain_root = domain.split(".")[0]
    return normalize_company_name(domain_root)


def company_to_common_pharma_key(company: str) -> str:
    """Map common pharma company variants to their canonical integrity key."""
    company_key = normalize_company_name(company)
    compact_company_key = company_key.replace(" ", "")
    for common_company in COMMON_PHARMA_COMPANY_DOMAINS:
        compact_common_company = common_company.replace(" ", "")
        if (
            compact_common_company
            and compact_company_key
            and (
                compact_common_company in compact_company_key
                or compact_company_key in compact_common_company
            )
        ):
            return common_company
    return company_key


def company_keys_match(first_key: str, second_key: str) -> bool:
    """Return whether two normalized company keys clearly match."""
    compact_first = first_key.replace(" ", "")
    compact_second = second_key.replace(" ", "")
    return bool(
        compact_first
        and compact_second
        and (compact_first in compact_second or compact_second in compact_first)
    )


def company_matches_domain(company: str, domain: str) -> bool:
    """Return whether a company name appears to match an email domain."""
    return company_keys_match(
        company_to_common_pharma_key(company), domain_to_company_key(domain)
    )


def company_matches_company(company: str, linkedin_company: str) -> bool:
    """Return whether the uploaded company and LinkedIn company match."""
    return company_keys_match(
        company_to_common_pharma_key(company),
        company_to_common_pharma_key(linkedin_company),
    )


def has_multiple_current_employers(linkedin_company: str) -> bool:
    """Detect obvious multiple-current-employer LinkedIn values."""
    return bool(re.search(r"\b(and|/)\b|;|,", linkedin_company, re.I))


def has_recent_employer_change(linkedin_title: str) -> bool:
    """Detect obvious recent employer-change signals in current-title text."""
    return bool(
        re.search(
            r"\b(recent|new role|joined|formerly|previously|ex-)\b",
            linkedin_title,
            re.I,
        )
    )


def validate_contact_integrity(
    company: str, email: str, linkedin_company: str = "", linkedin_title: str = ""
) -> ContactIntegrity:
    """Validate contact/company consistency before email generation."""
    domain = extract_email_domain(email)
    company_key = company_to_common_pharma_key(company)
    email_company_key = domain_to_company_key(domain)
    linkedin_company_key = company_to_common_pharma_key(linkedin_company)
    email_matches = company_keys_match(company_key, email_company_key)
    linkedin_matches = company_keys_match(company_key, linkedin_company_key)
    linkedin_matches_email = company_keys_match(email_company_key, linkedin_company_key)
    email_known = bool(domain and domain not in FREE_EMAIL_DOMAINS)
    linkedin_known = bool(linkedin_company)
    email_conflicts = bool(
        email_known
        and company_key
        and email_company_key
        and not email_matches
        and (
            company_key in COMMON_PHARMA_COMPANY_DOMAINS
            or email_company_key in COMMON_PHARMA_COMPANY_DOMAINS
        )
    )

    if has_multiple_current_employers(linkedin_company):
        return ContactIntegrity(
            "RED", "Multiple current employers detected.", "", linkedin_title
        )
    if has_recent_employer_change(linkedin_title):
        return ContactIntegrity(
            "RED", "Recent employer change detected.", linkedin_company, linkedin_title
        )

    if email_conflicts:
        return ContactIntegrity(
            "RED",
            "Email domain conflicts with company.",
            email_company_key.title() or domain.split(".")[0].title(),
            linkedin_title,
        )
    if (
        email_matches
        and linkedin_known
        and not linkedin_matches
        and not linkedin_matches_email
    ):
        return ContactIntegrity(
            "RED",
            "LinkedIn company conflicts with company and email domain.",
            linkedin_company,
            linkedin_title,
        )
    if email_matches and linkedin_matches:
        return ContactIntegrity(
            "GREEN",
            "Company, email domain, and LinkedIn company align.",
            company,
            linkedin_title,
        )

    if email_matches:
        return ContactIntegrity(
            "GREEN",
            "Company confirmed by email domain.",
            company,
            linkedin_title,
        )
    if linkedin_matches and not email_known:
        return ContactIntegrity(
            "YELLOW",
            "Email domain missing or unconfirmed; LinkedIn company matches.",
            company,
            linkedin_title,
        )
    return ContactIntegrity(
        "YELLOW",
        "Cannot verify current employer.",
        linkedin_company or company,
        linkedin_title,
    )


def row_to_text(contact: pd.Series, columns: list[str] | None = None) -> str:
    """Format contact fields as compact text for prompts."""
    selected_columns = columns or list(contact.index)
    lines: list[str] = []
    for field in selected_columns:
        value = contact.get(field, "")
        if pd.isna(value) or str(value).strip() == "":
            continue
        lines.append(f"{field}: {str(value).strip()}")
    return "\n".join(lines) or "Not provided"


def pattern_matches(search_text: str, pattern: str) -> bool:
    """Return whether a persona pattern appears as a phrase or acronym."""
    pattern_regex = rf"(?<![\w]){re.escape(pattern)}(?![\w])"
    return re.search(pattern_regex, search_text, re.UNICODE) is not None


def persona_confidence(match_count: int, role_match_count: int) -> float:
    """Convert match counts into a conservative persona confidence score."""
    if match_count == 0:
        return 0.35
    score = 0.64 + min(match_count, 3) * 0.08 + min(role_match_count, 2) * 0.06
    return round(min(score, 0.98), 2)


def classify_persona(
    contact: pd.Series, role_columns: list[str]
) -> PersonaClassification:
    """Classify a contact into one explicit persona and return confidence."""
    role_text = " ".join(
        str(contact.get(column, ""))
        for column in role_columns
        if not pd.isna(contact.get(column, ""))
    ).lower()
    all_text = row_to_text(contact).lower()
    search_text = f"{role_text} {all_text}"

    for persona, patterns in PERSONA_PATTERNS:
        role_match_count = sum(
            1 for pattern in patterns if pattern_matches(role_text, pattern)
        )
        all_match_count = sum(
            1 for pattern in patterns if pattern_matches(search_text, pattern)
        )
        if all_match_count:
            matched_keyword = next(
                pattern for pattern in patterns if pattern_matches(search_text, pattern)
            )
            return PersonaClassification(
                persona,
                persona_confidence(all_match_count, role_match_count),
                matched_keyword,
            )
    return PersonaClassification("Discovery", persona_confidence(0, 0), "")


def identify_persona(contact: pd.Series, role_columns: list[str]) -> str:
    """Classify a contact into one explicit persona label."""
    return classify_persona(contact, role_columns).persona


def _story_persona(persona: str, story: MetabolonStory) -> str:
    """Choose the narrative library persona from the recommended Metabolon story."""
    if story.recommended_offering == "Bioinformatics / Multiomics Software":
        return "Discovery"
    if story.recommended_offering == "Lipidomics":
        return "Immunology" if "Immunology" in get_narrative_personas() else "Discovery"
    if (
        story.recommended_offering == "Biopharma Services"
        and persona == "Clinical Pharmacology"
    ):
        return "Clinical Pharmacology"
    if story.recommended_offering == "Global Discovery Panel" and persona in (
        "Oncology",
        "Safety / Quality",
    ):
        return persona
    if persona in get_narrative_personas():
        return persona
    return "Discovery"


def get_narrative_personas() -> tuple[str, ...]:
    """Return supported narrative personas without exposing library internals here."""
    from narratives import PERSONAS

    return PERSONAS


def build_email(
    name: str,
    company: str,
    persona: str,
    to: str = "",
    persona_confidence_score: float = 0.35,
    matched_keyword: str = "",
    source_first_name: str = "",
    used_emails: set[str] | None = None,
    integrity: ContactIntegrity | None = None,
    title: str = "",
    therapeutic_area: str = "",
    linkedin_content_available: str = "No",
    linkedin_content_preview: str = "",
) -> ContactOutreach:
    """Build outreach from one random subject and one persona-specific use case."""
    integrity = integrity or ContactIntegrity(
        "YELLOW", "Company cannot be confirmed.", company, ""
    )
    first_name = get_contact_first_name(name, source_first_name)
    company_text = company or "your organization"
    if integrity.status == "RED":
        return ContactOutreach(
            name,
            company,
            to,
            persona,
            persona_confidence_score,
            integrity.status,
            integrity.reason,
            integrity.suggested_company,
            integrity.suggested_title,
            "Review Required",
            "Review Required",
            matched_keyword,
            "REVIEW-REQUIRED",
            "",
            "",
            "",
            "",
            linkedin_content_available,
            linkedin_content_preview,
        )
    active_persona = map_persona(persona)
    if active_persona == "Operations / Low Priority":
        return ContactOutreach(
            name,
            company,
            to,
            active_persona,
            persona_confidence_score,
            integrity.status,
            integrity.reason,
            integrity.suggested_company,
            integrity.suggested_title,
            "Review manually",
            "Review manually",
            matched_keyword,
            "MANUAL-REVIEW",
            "",
            "",
            "",
            "",
            linkedin_content_available,
            linkedin_content_preview,
        )

    metabolon_story = recommend_metabolon_story(
        active_persona, title, therapeutic_area, matched_keyword
    )
    narrative_persona = _story_persona(active_persona, metabolon_story)
    variant_set = get_narrative_set(narrative_persona)
    used_emails = used_emails if used_emails is not None else set()

    use_case_options = list(enumerate(variant_set["use_cases"], start=1))
    subject_options = list(enumerate(variant_set["subjects"], start=1))
    random.shuffle(use_case_options)
    random.shuffle(subject_options)

    persona_label = narrative_persona.lower()
    benefits = "\n".join(f"• {benefit}" for benefit in variant_set["benefits"])

    for use_case_index, use_case in use_case_options:
        for subject_index, subject_template in subject_options:
            subject = subject_template.format(company=company_text)
            email = (
                f"Dear {first_name},\n\n"
                "My name is Helmut von Keyserling, and I support "
                f"{company_text} as Strategic Account Manager at Metabolon.\n\n"
                f"Many {persona_label} teams are using metabolomics to {use_case}.\n\n"
                f"For this contact, the best Metabolon angle is {metabolon_story.recommended_offering}: "
                f"{metabolon_story.scientific_problem}.\n\n"
                f"This type of data can help:\n{benefits}\n\n"
                f"{metabolon_story.email_story}\n\n"
                "If this is of interest, I would be happy to briefly introduce our approach "
                "and learn how your team is thinking about this area.\n\n"
                "Best regards,\n\n"
                "Helmut von Keyserling\n"
                "Strategic Account Manager"
            )
            if email not in used_emails:
                used_emails.add(email)
                variant_id = (
                    f"{active_persona}-{use_case_index:02d}-S{subject_index:02d}"
                )
                return ContactOutreach(
                    name,
                    company,
                    to,
                    persona,
                    persona_confidence_score,
                    integrity.status,
                    integrity.reason,
                    integrity.suggested_company,
                    integrity.suggested_title,
                    subject,
                    email,
                    matched_keyword,
                    variant_id,
                    metabolon_story.primary_capability,
                    metabolon_story.recommended_offering,
                    metabolon_story.scientific_problem,
                    metabolon_story.email_story,
                    linkedin_content_available,
                    linkedin_content_preview,
                )

    raise ValueError(
        f"Could not generate a unique {active_persona} email for {name} at {company_text}."
    )


def read_contacts(uploaded_file: Any) -> pd.DataFrame:
    """Read contacts from a CSV or Excel upload."""
    filename = uploaded_file.name.lower()
    if filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    if filename.endswith(".csv"):
        encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")
        last_error: Exception | None = None
        for encoding in encodings:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error:
            raise last_error
    raise ValueError("Upload a .csv, .xls, or .xlsx file.")


def empty_output_table() -> pd.DataFrame:
    """Return an empty output table with the required export columns."""
    return pd.DataFrame(
        columns=[
            "Name",
            "Company",
            "To",
            "Persona",
            "Persona Confidence Score",
            "Integrity Status",
            "Integrity Reason",
            "Suggested Company",
            "Suggested Title",
            "Subject",
            "Email",
            "Matched Keyword",
            "Narrative Variant ID",
            "Metabolon Capability",
            "Recommended Offering",
            "Scientific Problem",
            "Email Story",
            "LinkedIn Content Available",
            "LinkedIn Content Preview",
        ]
    )


def initialize_session_state() -> None:
    """Initialize workflow session state keys."""
    defaults = {
        "contacts": None,
        "uploaded_filename": "",
        "generated_emails": empty_output_table(),
        "outlook_draft_csv": b"",
        "outlook_draft_zip": b"",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def generate_outreach_table(contacts: pd.DataFrame) -> pd.DataFrame:
    """Generate one persona-based email row for every uploaded contact."""
    name_column = find_name_column(contacts)
    if not name_column:
        raise ValueError("Could not find any columns in the uploaded file.")

    company_column = find_company_column(contacts)
    email_column = find_email_column(contacts)
    linkedin_company_column = find_linkedin_company_column(contacts)
    linkedin_title_column = find_linkedin_title_column(contacts)
    linkedin_content_column = find_linkedin_content_column(contacts)
    first_name_column = find_first_name_column(contacts)
    role_columns = find_role_columns(contacts)
    rows: list[ContactOutreach] = []
    used_emails: set[str] = set()
    progress = st.progress(
        0,
        text="Validating contact integrity, classifying contacts, and generating emails...",
    )

    total_contacts = len(contacts)
    for position, (_, contact) in enumerate(contacts.iterrows(), start=1):
        name = get_cell_value(contact, name_column, "Unnamed Contact")
        company = get_cell_value(contact, company_column, "")
        to = get_cell_value(contact, email_column, "")
        source_first_name = get_cell_value(contact, first_name_column, "")
        linkedin_company = get_cell_value(contact, linkedin_company_column, "")
        linkedin_title = get_cell_value(contact, linkedin_title_column, "")
        linkedin_content = get_cell_value(contact, linkedin_content_column, "")
        linkedin_content_available = "Yes" if linkedin_content else "No"
        linkedin_content_preview = linkedin_content[:200]
        integrity = validate_contact_integrity(
            company, to, linkedin_company, linkedin_title
        )
        classification = classify_persona(contact, role_columns)
        rows.append(
            build_email(
                name,
                company,
                classification.persona,
                to,
                classification.confidence_score,
                classification.matched_keyword,
                source_first_name,
                used_emails,
                integrity,
                linkedin_title or row_to_text(contact, role_columns),
                row_to_text(contact),
                linkedin_content_available,
                linkedin_content_preview,
            )
        )
        progress.progress(
            position / total_contacts,
            text=f"Generated {position} of {total_contacts} emails",
        )

    progress.empty()
    return pd.DataFrame(
        rows,
        columns=[
            "Name",
            "Company",
            "To",
            "Persona",
            "Persona Confidence Score",
            "Integrity Status",
            "Integrity Reason",
            "Suggested Company",
            "Suggested Title",
            "Subject",
            "Email",
            "Matched Keyword",
            "Narrative Variant ID",
            "Metabolon Capability",
            "Recommended Offering",
            "Scientific Problem",
            "Email Story",
            "LinkedIn Content Available",
            "LinkedIn Content Preview",
        ],
    )


def build_draft_table(generated_emails: pd.DataFrame) -> pd.DataFrame:
    """Convert generated outreach rows into draft export rows."""
    return generated_emails.rename(columns={"Email": "Body"})[
        ["To", "Subject", "Body"]
    ].copy()


def build_outlook_draft_exports(generated_emails: pd.DataFrame) -> tuple[bytes, bytes]:
    """Create Outlook-compatible CSV and zipped EML draft exports."""
    draft_table = build_draft_table(generated_emails)
    return (
        CSVDraftProvider().export(draft_table),
        EMLDraftProvider().export(draft_table),
    )


def main() -> None:
    st.set_page_config(page_title="Andy Bot", page_icon="🤖", layout="wide")
    initialize_session_state()

    st.title("🤖 Andy Bot")
    st.caption(
        "Upload a CSV contact list to classify each contact into a persona and export one varied email per contact."
    )

    with st.sidebar:
        st.header("Settings")
        st.markdown("**Personas**")
        st.write(", ".join(persona for persona, _ in PERSONA_PATTERNS))

    uploaded_file = st.file_uploader(
        "Upload contacts file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"]
    )
    if (
        uploaded_file is not None
        and uploaded_file.name != st.session_state.uploaded_filename
    ):
        try:
            contacts = read_contacts(uploaded_file)
        except (
            Exception
        ) as exc:  # pandas parsing errors vary by file content and version
            st.error(f"Could not read contacts file: {exc}")
            return

        if contacts.empty:
            st.error("The uploaded file has no rows.")
            return

        st.session_state.contacts = contacts
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.generated_emails = empty_output_table()
        st.session_state.outlook_draft_csv = b""
        st.session_state.outlook_draft_zip = b""
        st.success(
            f"Imported {len(contacts)} contacts. Andy Bot will export one email per contact."
        )

    contacts = st.session_state.contacts
    if contacts is None:
        st.warning("Upload a CSV file to get started.")
        return

    st.subheader("Contacts to Process")
    st.dataframe(contacts, use_container_width=True)

    if st.button("Generate emails", type="primary"):
        try:
            st.session_state.generated_emails = generate_outreach_table(contacts)
            st.session_state.outlook_draft_csv = b""
            st.session_state.outlook_draft_zip = b""
        except Exception as exc:
            st.error(f"Could not generate emails: {exc}")
            return

    if not st.session_state.generated_emails.empty:
        st.subheader("Generated Emails")
        st.dataframe(
            st.session_state.generated_emails, use_container_width=True, hide_index=True
        )
        st.subheader("Processing Summary")
        summary_table = st.session_state.generated_emails
        st.write(f"Contacts processed: {len(summary_table)}")
        st.write("Contact integrity")
        integrity_counts = summary_table["Integrity Status"].value_counts()
        green_contacts = int(integrity_counts.get("GREEN", 0))
        yellow_contacts = int(integrity_counts.get("YELLOW", 0))
        red_contacts = int(integrity_counts.get("RED", 0))
        col_green, col_yellow, col_red = st.columns(3)
        col_green.metric("GREEN contacts", green_contacts)
        col_yellow.metric("YELLOW contacts", yellow_contacts)
        col_red.metric("RED contacts", red_contacts)
        if yellow_contacts:
            st.warning("YELLOW contacts will generate email with a warning.")
        if red_contacts:
            st.error(
                'RED contacts are marked "Review Required" and emails are not generated.'
            )
        st.write("Persona distribution")
        st.dataframe(
            summary_table["Persona"]
            .value_counts()
            .rename_axis("Persona")
            .reset_index(name="Contacts"),
            use_container_width=True,
            hide_index=True,
        )
        low_confidence = summary_table[summary_table["Persona Confidence Score"] < 0.75]
        operations = summary_table[
            summary_table["Persona"] == "Operations / Low Priority"
        ]
        st.write(f"Low-confidence contacts (<0.75): {len(low_confidence)}")
        if not low_confidence.empty:
            st.dataframe(
                low_confidence[
                    [
                        "Name",
                        "Company",
                        "Persona",
                        "Persona Confidence Score",
                        "Matched Keyword",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.write(f"Operations / Low Priority contacts: {len(operations)}")
        if not operations.empty:
            st.dataframe(
                operations[["Name", "Company", "Persona", "Matched Keyword"]],
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Export all emails as CSV",
            data=st.session_state.generated_emails.to_csv(index=False),
            file_name="andy_bot_persona_emails.csv",
            mime="text/csv",
        )

        st.subheader("Outlook Draft Export")
        st.caption(
            "Create Outlook-ready draft files with To, Subject, and Body for review before sending."
        )
        if st.button("Create Outlook Drafts"):
            try:
                (
                    st.session_state.outlook_draft_csv,
                    st.session_state.outlook_draft_zip,
                ) = build_outlook_draft_exports(st.session_state.generated_emails)
            except Exception as exc:
                st.error(f"Could not create Outlook drafts: {exc}")
                return

        if st.session_state.outlook_draft_csv and st.session_state.outlook_draft_zip:
            st.download_button(
                "Download Outlook import CSV",
                data=st.session_state.outlook_draft_csv,
                file_name="outlook_drafts.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Outlook Drafts ZIP",
                data=st.session_state.outlook_draft_zip,
                file_name="outlook_drafts.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()

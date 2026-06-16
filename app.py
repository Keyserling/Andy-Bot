"""Andy Bot Streamlit application for persona-based Metabolon outreach.

Workflow:
1. Upload a CSV contact list.
2. Classify each contact into one persona.
3. Generate deterministic outreach with one LinkedIn observation and one Metabolon knowledge story.
4. Generate Name, Company, Persona, Subject, and Email for CSV export.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st

from draft_exports import (
    CSVDraftProvider,
    EMLDraftProvider,
    OutlookGraphAuthRequired,
    OutlookGraphDraftProvider,
    is_outlook_graph_configured,
)
from metabolon_knowledge import MetabolonStory, recommend_metabolon_story

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
    contact_narrative: str
    contact_narrative_confidence: float
    matched_keyword: str
    narrative_variant_id: str
    metabolon_capability: str
    recommended_offering: str
    scientific_problem: str
    email_story: str
    linkedin_content_available: str
    linkedin_hook: str
    linkedin_hook_type: str
    linkedin_hook_used: str
    linkedin_content_preview: str
    linkedin_content_present: str
    linkedin_summary: str
    personalization_source: str


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


def find_linkedin_url_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely personal LinkedIn profile URL column."""
    return find_column(
        dataframe,
        (
            "person linkedin url",
            "person_linkedin_url",
            "linkedin url",
            "linkedin_url",
            "linkedin",
            "profile linkedin url",
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

COMPANY_BRAND_NAMES = {
    "abbvie": "AbbVie",
    "astrazeneca": "AstraZeneca",
    "bayer": "Bayer",
    "boehringer ingelheim": "Boehringer Ingelheim",
    "bristol myers squibb": "Bristol Myers Squibb",
    "eli lilly": "Eli Lilly",
    "gilead": "Gilead",
    "glaxosmithkline": "GSK",
    "johnson johnson": "Johnson & Johnson",
    "merck": "Merck",
    "novartis": "Novartis",
    "pfizer": "Pfizer",
    "roche": "Roche",
    "sanofi": "Sanofi",
    "takeda": "Takeda",
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


CARE_STATEMENTS = {
    "Computational Biology": "Multiomics interpretation for computational biology.",
    "Clinical Pharmacology": "PK/PD interpretation in clinical samples.",
    "Immunology": "Patient stratification in immunology.",
    "Oncology": "Translational decision making in oncology.",
    "Safety / Quality": "Mechanistic context for safety and risk signals.",
    "Biomarkers / Bioanalysis": "Biomarker discovery and patient stratification.",
    "Translational / Clinical Development": "Translational decision making.",
    "Discovery": "Biomarker discovery and mechanism understanding.",
    "Medical Affairs": "Biological evidence for scientific exchange.",
}


def generate_contact_narrative(
    persona: str,
    title: str,
    therapeutic_area: str,
    matched_keyword: str,
    persona_confidence_score: float,
) -> tuple[str, float]:
    """Return one deterministic sentence for what this person likely cares about."""
    active_persona = map_persona(persona)
    combined_text = " ".join(
        part for part in (active_persona, title, therapeutic_area, matched_keyword) if part
    ).lower()
    if "pk/pd" in combined_text or "pharmacology" in combined_text:
        narrative = "PK/PD interpretation in clinical samples."
    elif "immun" in combined_text or "inflammation" in combined_text:
        narrative = "Patient stratification in immunology."
    elif "oncology" in combined_text or "cancer" in combined_text:
        narrative = "PK/PD interpretation in oncology."
    elif "biomarker" in combined_text or "bioanalysis" in combined_text:
        narrative = "Biomarker discovery."
    elif "translational" in combined_text or "clinical development" in combined_text:
        narrative = "Translational decision making."
    else:
        narrative = CARE_STATEMENTS.get(active_persona, "Translational decision making.")
    confidence = round(min(persona_confidence_score + (0.04 if matched_keyword else 0), 0.98), 2)
    return narrative, confidence


def build_scientific_story(story: MetabolonStory) -> str:
    """Build a challenger narrative around industry adoption and biological insight gaps."""
    offering = (story.recommended_offering or "Global Discovery Panel").strip()
    problem = story.scientific_problem or "extracting biological insight from available samples"

    if offering == "Biopharma Services":
        return (
            "Across US pharma, I am increasingly seeing metabolomics and multiomics move "
            "from specialist projects into routine translational, biomarker, and PK/PD decision making. "
            f"What has surprised me is not the interest in these data, but how differently teams approach {problem}. "
            "Many groups already have samples that could answer important biological questions; the bottleneck is often turning those samples into interpretable biology before the next program decision."
        )
    if offering == "Lipidomics":
        return (
            "Across US pharma, I am increasingly seeing metabolomics and multiomics move "
            "from specialist projects into routine biomarker, translational, and patient-stratification workstreams. "
            f"What has surprised me is how often {problem} becomes a decision gap rather than a data-generation gap. "
            "Many groups already have the right cohorts or stored biospecimens; the question is whether they are extracting enough biological signal from them."
        )
    if offering == "Multiomics":
        return (
            "Across US pharma, I am increasingly seeing multiomics become part of routine mechanism, biomarker, and translational decision making. "
            f"What has surprised me is not the volume of data being generated, but how unevenly organizations approach {problem}. "
            "The advantage increasingly goes to teams that can connect existing sample sets across molecular layers and convert them into clear biological interpretation."
        )
    return (
        "Across US pharma, I am increasingly seeing metabolomics and multiomics move "
        "from exploratory science into standard biomarker, mechanism, translational, and patient-stratification workstreams. "
        f"What has surprised me is how often the hard part is {problem}, not generating another dataset. "
        "Many groups already have samples that could answer important biological questions; the challenge is whether those samples are being used aggressively enough to inform development decisions."
    )


def display_company_brand(company: str) -> str:
    """Return a recipient-facing company brand instead of legal-entity wording."""
    company = company.strip()
    company_key = company_to_common_pharma_key(company)
    if company_key in COMPANY_BRAND_NAMES:
        return COMPANY_BRAND_NAMES[company_key]
    return company or "your organization"


def clean_signal(signal: str) -> str:
    """Normalize a detected profile signal for use in a human sentence."""
    signal = signal.strip()
    signal_display = {
        "pk/pd": "PK/PD",
        "multiomics": "multiomics",
        "multi-omics": "multiomics",
        "ai": "AI",
    }.get(signal.lower(), signal.lower())
    return signal_display


def extract_role_change_observation(linkedin_text: str) -> str:
    """Return a careful role-change observation when the text explicitly supports it."""
    compact = " ".join(linkedin_text.split())
    match = re.search(
        r"(?:joined|moved from|move from|transitioned from)\s+([A-Z][A-Za-z& .-]{1,45}?)\s+(?:to|at)\s+([A-Z][A-Za-z& .-]{1,45})(?:[.;,]|$)",
        compact,
        re.I,
    )
    if match:
        previous_company = display_company_brand(match.group(1).strip())
        current_company = display_company_brand(match.group(2).strip())
        if previous_company and current_company and previous_company != current_company:
            return f"Congratulations on your recent move from {previous_company} to {current_company}."
    match = re.search(
        r"(?:new role|recently joined|joined)\s+(?:at\s+)?([A-Z][A-Za-z& .-]{1,45})(?:[.;,]|$)",
        compact,
        re.I,
    )
    if match:
        current_company = display_company_brand(match.group(1).strip())
        return f"Congratulations on your recent move to {current_company}."
    return ""


PERSONAL_OBSERVATION_PATTERNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "Conference participation",
        ("conference", "congress", "asco", "aacr", "eular", "ash", "esmo"),
        "Your recent conference participation around {signal} caught my attention.",
    ),
    (
        "Publication",
        ("publication", "published", "manuscript", "paper", "author", "co-author"),
        "Your recent publication activity around {signal} caught my attention.",
    ),
    (
        "Scientific interest",
        (
            "pk/pd",
            "biomarker",
            "patient stratification",
            "precision medicine",
            "target engagement",
            "drug discovery",
            "multiomics",
            "multi-omics",
            "systems biology",
            "federated ai",
        ),
        "Your recent focus on {signal} caught my attention.",
    ),
    (
        "Therapeutic focus",
        (
            "oncology",
            "cancer",
            "immunology",
            "autoimmune",
            "inflammation",
            "rare disease",
            "neurology",
            "cardiometabolic",
            "metabolic disease",
        ),
        "Interesting to see your work in {signal}.",
    ),
)


def build_personal_observation(
    linkedin_text: str,
    company_text: str,
) -> tuple[str, str, str]:
    """Return one observation from LinkedIn, or the required company-role fallback."""
    if linkedin_text.strip():
        role_change = extract_role_change_observation(linkedin_text)
        if role_change:
            return role_change, "Role change", "Yes"
        normalized_text = linkedin_text.lower()
        for hook_type, signals, template in PERSONAL_OBSERVATION_PATTERNS:
            for signal in signals:
                if pattern_matches(normalized_text, signal):
                    return template.format(signal=clean_signal(signal)), hook_type, "Yes"
    return f"Given your role at {company_text}, I thought this might be relevant.", "LinkedIn fallback", "No"

def summarize_linkedin_content(linkedin_text: str) -> str:
    """Return a compact debug-only summary of LinkedIn content signals."""
    compact_text = " ".join(linkedin_text.split())
    if not compact_text:
        return ""

    normalized_text = compact_text.lower()
    for _, signals, _ in PERSONAL_OBSERVATION_PATTERNS:
        for signal in signals:
            if pattern_matches(normalized_text, signal):
                return f"Focus on {clean_signal(signal)}"
    return compact_text[:200]


def linkedin_content_present_flag(linkedin_text: str) -> str:
    """Return the debug flag for whether LinkedIn content reached generation."""
    return "TRUE" if linkedin_text.strip() else "FALSE"


def extract_linkedin_hook(linkedin_text: str) -> tuple[str, str, str]:
    """Return a short observation when LinkedIn text contains an eligible signal."""
    observation, hook_type, hook_used = build_personal_observation(
        linkedin_text, "your organization"
    )
    if hook_used == "Yes":
        return observation, hook_type, hook_used
    return "", "", "No"


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
    linkedin_hook: str = "",
    linkedin_hook_type: str = "",
    linkedin_hook_used: str = "No",
) -> ContactOutreach:
    """Build deterministic Challenger Outreach Engine V4 email copy."""
    integrity = integrity or ContactIntegrity(
        "YELLOW", "Company cannot be confirmed.", company, ""
    )
    first_name = get_contact_first_name(name, source_first_name)
    company_text = display_company_brand(company)
    linkedin_content_present = linkedin_content_present_flag(linkedin_content_preview)
    linkedin_summary = summarize_linkedin_content(linkedin_content_preview)
    initial_personalization_source = (
        "LinkedIn" if linkedin_hook_used == "Yes" else "LinkedIn fallback"
    )
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
            "",
            0.0,
            matched_keyword,
            "REVIEW-REQUIRED",
            "",
            "",
            "",
            "",
            linkedin_content_available,
            linkedin_hook,
            linkedin_hook_type,
            linkedin_hook_used,
            linkedin_content_preview,
            linkedin_content_present,
            linkedin_summary,
            initial_personalization_source,
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
            "",
            0.0,
            matched_keyword,
            "MANUAL-REVIEW",
            "",
            "",
            "",
            "",
            linkedin_content_available,
            linkedin_hook,
            linkedin_hook_type,
            linkedin_hook_used,
            linkedin_content_preview,
            linkedin_content_present,
            linkedin_summary,
            initial_personalization_source,
        )

    metabolon_story = recommend_metabolon_story(
        active_persona, title, therapeutic_area, matched_keyword
    )
    contact_narrative, contact_narrative_score = generate_contact_narrative(
        active_persona,
        title,
        therapeutic_area,
        matched_keyword,
        persona_confidence_score,
    )
    observation, hook_type, hook_used = build_personal_observation(
        linkedin_content_preview, company_text
    )
    scientific_story = build_scientific_story(metabolon_story)
    subject = f"A question on {contact_narrative.rstrip('.').lower()}"
    email = (
        f"Dear {first_name},\n\n"
        f"{observation}\n\n"
        "I work with pharmaceutical R&D organizations at Metabolon, where we sit close to "
        "how large development teams are standardizing metabolomics and multiomics across programs.\n\n"
        f"{scientific_story}\n\n"
        f"I would be interested in how {company_text} currently thinks about this area, particularly given how quickly adoption appears to be increasing across the industry. "
        "Would it be worth comparing notes?\n\n"
        "Best regards,\n\n"
        "Helmut von Keyserling\n"
        "+49 176 61356899"
    )
    if used_emails is not None:
        used_emails.add(email)
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
        contact_narrative,
        contact_narrative_score,
        matched_keyword,
        "ENGINE-V4",
        metabolon_story.primary_capability,
        metabolon_story.recommended_offering,
        metabolon_story.scientific_problem,
        scientific_story,
        linkedin_content_available,
        observation,
        hook_type,
        hook_used,
        linkedin_content_preview,
        linkedin_content_present,
        linkedin_summary,
        "LinkedIn" if hook_used == "Yes" else "LinkedIn fallback",
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
            "Contact Narrative",
            "Contact Narrative Confidence",
            "Matched Keyword",
            "Narrative Variant ID",
            "Metabolon Capability",
            "Recommended Offering",
            "Scientific Problem",
            "Email Story",
            "LinkedIn Content Available",
            "LinkedIn Hook",
            "LinkedIn Hook Type",
            "LinkedIn Hook Used",
            "LinkedIn Content Preview",
            "LinkedIn Content Present",
            "LinkedIn Summary",
            "Personalization Source",
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
        "outlook_graph_device_flow": None,
        "outlook_graph_created_count": 0,
        "outlook_graph_connected": False,
        "linkedin_text_by_contact": {},
        "selected_linkedin_contact": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_contact_key(index: int, email: str) -> str:
    """Return a stable session-state key for a contact."""
    return email.lower().strip() if email else f"row-{index}"


def generate_contact_outreach(
    contact: pd.Series,
    index: int,
    contacts: pd.DataFrame,
    linkedin_text_by_contact: dict[str, str] | None = None,
    used_emails: set[str] | None = None,
) -> ContactOutreach:
    """Generate outreach for a single contact without changing other rows."""
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

    name = get_cell_value(contact, name_column, "Unnamed Contact")
    company = get_cell_value(contact, company_column, "")
    to = get_cell_value(contact, email_column, "")
    source_first_name = get_cell_value(contact, first_name_column, "")
    linkedin_company = get_cell_value(contact, linkedin_company_column, "")
    linkedin_title = get_cell_value(contact, linkedin_title_column, "")
    linkedin_content = get_cell_value(contact, linkedin_content_column, "")
    contact_key = get_contact_key(index, to)
    linkedin_text_by_contact = linkedin_text_by_contact or {}
    if linkedin_text_by_contact.get(contact_key, "").strip():
        linkedin_content = linkedin_text_by_contact[contact_key].strip()
    linkedin_content_available = "Yes" if linkedin_content else "No"
    linkedin_hook, linkedin_hook_type, linkedin_hook_used = extract_linkedin_hook(
        linkedin_content
    )
    integrity = validate_contact_integrity(
        company, to, linkedin_company, linkedin_title
    )
    classification = classify_persona(contact, role_columns)
    return build_email(
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
        linkedin_content[:200],
        linkedin_hook,
        linkedin_hook_type,
        linkedin_hook_used,
    )


def generate_outreach_table(
    contacts: pd.DataFrame, linkedin_text_by_contact: dict[str, str] | None = None
) -> pd.DataFrame:
    """Generate one persona-based email row for every uploaded contact."""
    if not find_name_column(contacts):
        raise ValueError("Could not find any columns in the uploaded file.")

    linkedin_text_by_contact = linkedin_text_by_contact or {}
    rows: list[ContactOutreach] = []
    used_emails: set[str] = set()
    progress = st.progress(
        0,
        text="Validating contact integrity, classifying contacts, and generating emails...",
    )

    total_contacts = len(contacts)
    for position, (_, contact) in enumerate(contacts.iterrows(), start=1):
        rows.append(
            generate_contact_outreach(
                contact, position - 1, contacts, linkedin_text_by_contact, used_emails
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
            "Contact Narrative",
            "Contact Narrative Confidence",
            "Matched Keyword",
            "Narrative Variant ID",
            "Metabolon Capability",
            "Recommended Offering",
            "Scientific Problem",
            "Email Story",
            "LinkedIn Content Available",
            "LinkedIn Hook",
            "LinkedIn Hook Type",
            "LinkedIn Hook Used",
            "LinkedIn Content Preview",
            "LinkedIn Content Present",
            "LinkedIn Summary",
            "Personalization Source",
        ],
    )


def build_draft_table(generated_emails: pd.DataFrame) -> pd.DataFrame:
    """Convert generated outreach rows into draft export rows."""
    return generated_emails.rename(columns={"Email": "Body"})[
        ["To", "Subject", "Body"]
    ].copy()


def build_outlook_graph_draft_table(generated_emails: pd.DataFrame) -> pd.DataFrame:
    """Return only contacts eligible for direct Microsoft Graph draft creation."""
    eligible = generated_emails.copy()
    if "Integrity Status" in eligible.columns:
        eligible = eligible[
            eligible["Integrity Status"].astype(str).str.upper() != "RED"
        ]
    eligible = eligible[
        ~eligible["Email"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"review required", "review manually", ""})
    ]
    return build_draft_table(eligible)


def build_outlook_draft_exports(generated_emails: pd.DataFrame) -> tuple[bytes, bytes]:
    """Create Outlook-compatible CSV and zipped EML draft exports."""
    draft_table = build_draft_table(generated_emails)
    return (
        CSVDraftProvider().export(draft_table),
        EMLDraftProvider().export(draft_table),
    )


def render_linkedin_personalization(contacts: pd.DataFrame) -> None:
    """Render per-contact LinkedIn personalization controls."""
    if st.session_state.generated_emails.empty:
        return

    name_column = find_name_column(contacts)
    company_column = find_company_column(contacts)
    email_column = find_email_column(contacts)
    title_column = find_linkedin_title_column(contacts)
    linkedin_url_column = find_linkedin_url_column(contacts)

    contact_options: list[tuple[str, int, str]] = []
    for index, contact in contacts.reset_index(drop=True).iterrows():
        name = get_cell_value(contact, name_column, "Unnamed Contact")
        email = get_cell_value(contact, email_column, "")
        company = get_cell_value(contact, company_column, "")
        key = get_contact_key(index, email)
        contact_options.append(
            (f"{index + 1}. {name} — {company} — {email or key}", index, key)
        )

    st.subheader("Select contact for LinkedIn personalization")
    labels = [option[0] for option in contact_options]
    selected_label = st.selectbox(
        "Select contact for LinkedIn personalization",
        labels,
        label_visibility="collapsed",
    )
    selected_index, selected_key = next(
        (index, key) for label, index, key in contact_options if label == selected_label
    )
    contact = contacts.reset_index(drop=True).iloc[selected_index]
    generated = st.session_state.generated_emails.iloc[selected_index]

    st.write(
        f"**Full Name:** {get_cell_value(contact, name_column, 'Unnamed Contact')}"
    )
    st.write(f"**Title:** {get_cell_value(contact, title_column, '')}")
    st.write(f"**Company:** {get_cell_value(contact, company_column, '')}")
    st.write(f"**Email:** {get_cell_value(contact, email_column, '')}")
    linkedin_url = get_cell_value(contact, linkedin_url_column, "")
    if linkedin_url:
        st.markdown(f"**Person Linkedin Url:** [Open LinkedIn profile]({linkedin_url})")
    else:
        st.write("**Person Linkedin Url:** Not provided")
    st.write(f"**Current generated Subject:** {generated['Subject']}")
    st.text_area(
        "Current generated Email",
        value=str(generated["Email"]),
        height=260,
        disabled=True,
    )

    text_value = st.text_area(
        "Paste LinkedIn profile text for this contact",
        value=st.session_state.linkedin_text_by_contact.get(selected_key, ""),
        key=f"linkedin_text_input_{selected_key}",
        height=180,
    )
    if st.button("Save LinkedIn text and regenerate this email"):
        st.session_state.linkedin_text_by_contact[selected_key] = text_value.strip()
        used_emails = set(
            st.session_state.generated_emails["Email"].dropna().astype(str)
        )
        used_emails.discard(str(generated["Email"]))
        updated = generate_contact_outreach(
            contact,
            selected_index,
            contacts,
            st.session_state.linkedin_text_by_contact,
            used_emails,
        )
        st.session_state.generated_emails.iloc[selected_index] = list(updated)
        st.session_state.outlook_draft_csv = b""
        st.session_state.outlook_draft_zip = b""
        st.success(
            "Saved LinkedIn text and regenerated only the selected contact's email."
        )


def main() -> None:
    st.set_page_config(page_title="Andy Bot", page_icon="🤖", layout="wide")
    initialize_session_state()

    st.title("🤖 Andy Bot")
    st.caption(
        "Upload a CSV contact list to classify each contact into a persona and export one deterministic email per contact."
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
        st.session_state.outlook_graph_device_flow = None
        st.session_state.outlook_graph_created_count = 0
        st.session_state.linkedin_text_by_contact = {}
        st.session_state.selected_linkedin_contact = ""
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
            st.session_state.generated_emails = generate_outreach_table(
                contacts, st.session_state.linkedin_text_by_contact
            )
            st.session_state.outlook_draft_csv = b""
            st.session_state.outlook_draft_zip = b""
            st.session_state.outlook_graph_device_flow = None
            st.session_state.outlook_graph_created_count = 0
        except Exception as exc:
            st.error(f"Could not generate emails: {exc}")
            return

    if not st.session_state.generated_emails.empty:
        render_linkedin_personalization(contacts)

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
            "Create real unsent Outlook drafts with Microsoft Graph. Download EML ZIP remains available as a fallback."
        )

        graph_configured = is_outlook_graph_configured()
        provider = None
        if graph_configured:
            try:
                provider = OutlookGraphDraftProvider()
                if provider.has_cached_account():
                    st.session_state.outlook_graph_connected = True
            except Exception as exc:
                st.session_state.outlook_graph_connected = False
                st.error(
                    f"Could not initialize Microsoft Graph Outlook integration: {exc}"
                )

        if graph_configured and provider:
            status = (
                "Connected"
                if st.session_state.outlook_graph_connected
                else "Not connected"
            )
            st.write(f"**Outlook connection status:** {status}")

            if st.button("Connect Outlook"):
                try:
                    if st.session_state.outlook_graph_device_flow:
                        provider.complete_device_authentication(
                            st.session_state.outlook_graph_device_flow
                        )
                        st.session_state.outlook_graph_device_flow = None
                        st.session_state.outlook_graph_connected = True
                        st.success("Outlook connected")
                    else:
                        provider.connect()
                        st.session_state.outlook_graph_connected = True
                        st.success("Outlook connected")
                except OutlookGraphAuthRequired as exc:
                    st.session_state.outlook_graph_device_flow = exc.flow
                    st.session_state.outlook_graph_connected = False
                    st.info(str(exc))
                except Exception as exc:
                    st.session_state.outlook_graph_connected = False
                    st.error(f"Could not connect Outlook: {exc}")

            if st.session_state.outlook_graph_device_flow:
                flow = st.session_state.outlook_graph_device_flow
                st.warning(
                    "Outlook authentication pending. Open "
                    f"{flow['verification_uri']} and enter code {flow['user_code']}, "
                    "then click Connect Outlook."
                )

            if st.button(
                "Create drafts in Outlook",
                disabled=not st.session_state.outlook_graph_connected,
            ):
                try:
                    draft_table = build_outlook_graph_draft_table(
                        st.session_state.generated_emails
                    )
                    created_count = provider.create_drafts(draft_table)
                    st.session_state.outlook_graph_created_count = created_count
                    st.success(f"{created_count} Outlook drafts created")
                except OutlookGraphAuthRequired as exc:
                    st.session_state.outlook_graph_device_flow = exc.flow
                    st.session_state.outlook_graph_connected = False
                    st.info(str(exc))
                except Exception as exc:
                    st.error(f"Could not create Outlook drafts: {exc}")
        else:
            st.write("**Outlook connection status:** Microsoft Graph not configured")
            st.info(
                "To enable direct Outlook draft creation, register a Microsoft Entra public-client app with delegated Microsoft Graph Mail.ReadWrite permission, then set MS_GRAPH_CLIENT_ID before starting Streamlit. Until then, use Download EML ZIP as the fallback."
            )

        if st.session_state.outlook_graph_created_count:
            st.success(
                f"{st.session_state.outlook_graph_created_count} Outlook drafts created"
            )

        if (
            not st.session_state.outlook_draft_csv
            or not st.session_state.outlook_draft_zip
        ):
            (
                st.session_state.outlook_draft_csv,
                st.session_state.outlook_draft_zip,
            ) = build_outlook_draft_exports(st.session_state.generated_emails)

        if st.session_state.outlook_draft_csv and st.session_state.outlook_draft_zip:
            st.download_button(
                "Download Outlook import CSV",
                data=st.session_state.outlook_draft_csv,
                file_name="outlook_drafts.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download EML ZIP",
                data=st.session_state.outlook_draft_zip,
                file_name="outlook_drafts.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()

"""Andy Bot Streamlit application for persona-template Metabolon outreach.

Workflow:
1. Upload a CSV contact list.
2. Classify each contact into one persona template.
3. Select the matching Metabolon email template.
4. Generate Name, Company, Persona, Subject, and Email for CSV export.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st

PERSONAS = (
    "Discovery",
    "Translational Research",
    "Clinical Biomarkers",
    "Clinical Development",
    "Medical Affairs",
    "Oncology",
    "Immunology",
    "Safety/Risk",
    "Bioanalysis",
    "Computational Biology",
)

PERSONA_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Safety/Risk",
        (
            "safety",
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
        "Medical Affairs",
        (
            "medical affairs",
            "medical science liaison",
            "msl",
            "field medical",
            "medical director",
            "scientific affairs",
        ),
    ),
    (
        "Clinical Biomarkers",
        (
            "clinical biomarker",
            "biomarker",
            "patient stratification",
            "companion diagnostic",
            "diagnostic",
            "pharmacodynamic biomarker",
        ),
    ),
    (
        "Clinical Development",
        (
            "clinical development",
            "clinical trial",
            "clinical operations",
            "clinical study",
            "clinical research",
            "development lead",
        ),
    ),
    (
        "Bioanalysis",
        (
            "bioanalysis",
            "bioanalytical",
            "dmpk",
            "pk/pd",
            "pharmacokinetic",
            "pharmacodynamic",
            "assay",
            "regulated bioanalysis",
        ),
    ),
    (
        "Computational Biology",
        (
            "computational biology",
            "bioinformatics",
            "data science",
            "systems biology",
            "multiomics",
            "multi-omics",
            "machine learning",
            "ai/ml",
        ),
    ),
    (
        "Oncology",
        ("oncology", "cancer", "tumor", "tumour", "immuno-oncology", "io research"),
    ),
    (
        "Immunology",
        ("immunology", "inflammation", "autoimmune", "immune", "immunotherapy"),
    ),
    (
        "Translational Research",
        (
            "translational",
            "translational medicine",
            "translational science",
            "bench to bedside",
            "human biology",
        ),
    ),
    (
        "Discovery",
        (
            "discovery",
            "drug discovery",
            "target discovery",
            "early research",
            "research scientist",
            "principal scientist",
            "biology",
        ),
    ),
)

EMAIL_TEMPLATES = {
    "Discovery": {
        "subject": "Metabolomics for discovery work at {company}",
        "use_case": (
            "For discovery teams, Metabolon helps connect target biology to pathway-level "
            "biochemistry in cells, models, and early translational samples. This can help "
            "prioritize mechanisms, identify metabolic shifts linked to phenotype, and make "
            "earlier decisions about which programs should move forward."
        ),
    },
    "Translational Research": {
        "subject": "Metabolomics for translational research at {company}",
        "use_case": (
            "For translational research teams, Metabolon helps compare biology across models "
            "and human samples using a consistent biochemical readout. This can support mechanism "
            "of action work, pharmacology interpretation, and decisions about which signals are "
            "most relevant for clinical studies."
        ),
    },
    "Clinical Biomarkers": {
        "subject": "Metabolomics for clinical biomarker work at {company}",
        "use_case": (
            "For clinical biomarker teams, Metabolon helps identify biochemical markers tied to "
            "drug response, disease activity, and patient segmentation. This can support "
            "pharmacodynamic readouts and biomarker strategies that need clear biological context."
        ),
    },
    "Clinical Development": {
        "subject": "Metabolomics for clinical development at {company}",
        "use_case": (
            "For clinical development teams, Metabolon helps read drug response, disease biology, "
            "and patient heterogeneity directly from clinical samples. This can add biological "
            "context to endpoints, dose decisions, and responder analyses without creating a large "
            "operational burden for the study."
        ),
    },
    "Medical Affairs": {
        "subject": "Metabolomics for medical affairs at {company}",
        "use_case": (
            "For medical affairs teams, Metabolon helps explain disease biology and treatment "
            "response through measurable biochemical differences. This can support scientific "
            "exchange with clinicians, researchers, and external experts when the discussion needs "
            "clear evidence rather than broad claims."
        ),
    },
    "Oncology": {
        "subject": "Metabolomics for oncology work at {company}",
        "use_case": (
            "For oncology teams, Metabolon helps characterize tumor metabolism, host response, "
            "treatment effect, and resistance biology across research and clinical samples. This "
            "can support mechanism work, patient stratification, and interpretation of response in "
            "complex oncology studies."
        ),
    },
    "Immunology": {
        "subject": "Metabolomics for immunology work at {company}",
        "use_case": (
            "For immunology teams, Metabolon helps connect immune activity with the biochemical "
            "pathways that reflect inflammation, disease activity, and treatment response. This can "
            "support work in autoimmune and immune-mediated disease where pathway context is needed "
            "alongside cellular or cytokine readouts."
        ),
    },
    "Safety/Risk": {
        "subject": "Metabolomics for safety and risk work at {company}",
        "use_case": (
            "For safety and risk teams, Metabolon helps detect biochemical changes that may explain "
            "toxicity, off-target effects, or emerging risk signals. This can support earlier "
            "interpretation of organ-specific effects and help distinguish adaptive changes from "
            "signals that need closer follow-up."
        ),
    },
    "Bioanalysis": {
        "subject": "Metabolomics alongside bioanalysis at {company}",
        "use_case": (
            "For bioanalysis teams, Metabolon adds broad biochemical context alongside targeted "
            "assays, PK/PD work, and regulated sample analysis. This can help connect exposure and "
            "response to pathway-level changes when standard analyte panels do not explain the "
            "biology."
        ),
    },
    "Computational Biology": {
        "subject": "Metabolomics for computational biology at {company}",
        "use_case": (
            "For computational biology teams, Metabolon provides a high-dimensional phenotype layer "
            "that can strengthen multi-omics models and biological interpretation. This can help "
            "connect genomic, transcriptomic, or proteomic patterns to measured biochemical activity "
            "in the same disease or treatment context."
        ),
    },
}


class ContactOutreach(NamedTuple):
    """Generated outreach output for one contact."""

    name: str
    company: str
    persona: str
    subject: str
    email: str


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


def get_contact_first_name(name: str) -> str:
    """Return the first name from a contact display name."""
    if name == "Unnamed Contact":
        return "Colleague"

    name_without_email = re.sub(r"<[^>]+>", "", name).strip()
    name_parts = re.findall(r"[A-Za-z][A-Za-z'’-]*", name_without_email)
    return name_parts[0] if name_parts else "Colleague"


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


def identify_persona(contact: pd.Series, role_columns: list[str]) -> str:
    """Classify a contact into exactly one persona template."""
    role_text = " ".join(
        str(contact.get(column, ""))
        for column in role_columns
        if not pd.isna(contact.get(column, ""))
    ).lower()
    all_text = row_to_text(contact).lower()
    search_text = f"{role_text} {all_text}"

    for persona, patterns in PERSONA_PATTERNS:
        if any(pattern in search_text for pattern in patterns):
            return persona
    return "Discovery"


def build_email(name: str, company: str, persona: str) -> ContactOutreach:
    """Build deterministic outreach from the selected persona template."""
    first_name = get_contact_first_name(name)
    company_text = company or "your organization"
    template = EMAIL_TEMPLATES[persona]
    subject = template["subject"].format(company=company_text)
    email = (
        f"Dear {first_name},\n\n"
        "My name is Helmut von Keyserling, and I support "
        f"{company_text} as Strategic Account Manager at Metabolon.\n\n"
        f"{template['use_case']}\n\n"
        "If this is relevant to your work, I would welcome a brief 20-minute conversation.\n\n"
        "Best regards,\n"
        "Helmut von Keyserling\n"
        "Strategic Account Manager"
    )
    return ContactOutreach(name, company, persona, subject, email)


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
    return pd.DataFrame(columns=["Name", "Company", "Persona", "Subject", "Email"])


def initialize_session_state() -> None:
    """Initialize workflow session state keys."""
    defaults = {
        "contacts": None,
        "uploaded_filename": "",
        "generated_emails": empty_output_table(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def generate_outreach_table(contacts: pd.DataFrame) -> pd.DataFrame:
    """Generate one persona-template email row for every uploaded contact."""
    name_column = find_name_column(contacts)
    if not name_column:
        raise ValueError("Could not find any columns in the uploaded file.")

    company_column = find_company_column(contacts)
    role_columns = find_role_columns(contacts)
    rows: list[ContactOutreach] = []
    progress = st.progress(0, text="Classifying contacts and generating emails...")

    total_contacts = len(contacts)
    for position, (_, contact) in enumerate(contacts.iterrows(), start=1):
        name = get_cell_value(contact, name_column, "Unnamed Contact")
        company = get_cell_value(contact, company_column, "")
        persona = identify_persona(contact, role_columns)
        rows.append(build_email(name, company, persona))
        progress.progress(
            position / total_contacts,
            text=f"Generated {position} of {total_contacts} emails",
        )

    progress.empty()
    return pd.DataFrame(
        rows, columns=["Name", "Company", "Persona", "Subject", "Email"]
    )


def main() -> None:
    st.set_page_config(page_title="Andy Bot", page_icon="🤖", layout="wide")
    initialize_session_state()

    st.title("🤖 Andy Bot")
    st.caption(
        "Upload a CSV contact list to classify each contact into a persona template and export one email per contact."
    )

    with st.sidebar:
        st.header("Settings")
        st.markdown("**Personas**")
        st.write(", ".join(PERSONAS))

    uploaded_file = st.file_uploader(
        "Upload contacts file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"]
    )
    if (
        uploaded_file is not None
        and uploaded_file.name != st.session_state.uploaded_filename
    ):
        try:
            contacts = read_contacts(uploaded_file)
        except Exception as exc:  # pandas parsing errors vary by file content and version
            st.error(f"Could not read contacts file: {exc}")
            return

        if contacts.empty:
            st.error("The uploaded file has no rows.")
            return

        st.session_state.contacts = contacts
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.generated_emails = empty_output_table()
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
        except Exception as exc:
            st.error(f"Could not generate emails: {exc}")
            return

    if not st.session_state.generated_emails.empty:
        st.subheader("Generated Emails")
        st.dataframe(
            st.session_state.generated_emails, use_container_width=True, hide_index=True
        )
        st.download_button(
            "Export all emails as CSV",
            data=st.session_state.generated_emails.to_csv(index=False),
            file_name="andy_bot_persona_emails.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

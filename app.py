"""Andy Bot Streamlit application for persona-based Metabolon outreach.

Workflow:
1. Upload a CSV contact list.
2. Classify each contact into one persona.
3. Randomly select from persona-specific narrative and subject variants.
4. Generate Name, Company, Persona, Subject, and Email for CSV export.
"""

from __future__ import annotations

import random
import re
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st

from narratives import PERSONAS, get_narrative_set


OLD_TO_NEW_PERSONA = {
    "Translational Research": "Translational / Clinical Development",
    "Clinical Development": "Translational / Clinical Development",
    "Clinical Biomarkers": "Biomarkers / Bioanalysis",
    "Bioanalysis": "Biomarkers / Bioanalysis",
    "Safety/Risk": "Safety / Quality",
    "Immunology": "Discovery",
    "Computational Biology": "Discovery",
}

PERSONA_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
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
        "Biomarkers / Bioanalysis",
        (
            "clinical biomarker",
            "biomarker",
            "patient stratification",
            "companion diagnostic",
            "diagnostic",
            "pharmacodynamic biomarker",
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
        "Translational / Clinical Development",
        (
            "clinical development",
            "clinical trial",
            "clinical operations",
            "clinical study",
            "clinical research",
            "development lead",
            "translational",
            "translational medicine",
            "translational science",
            "bench to bedside",
            "human biology",
        ),
    ),
    (
        "Oncology",
        ("oncology", "cancer", "tumor", "tumour", "immuno-oncology", "io research"),
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
            "immunology",
            "inflammation",
            "autoimmune",
            "immune",
            "computational biology",
            "bioinformatics",
            "data science",
            "systems biology",
            "multiomics",
            "multi-omics",
        ),
    ),
)


def map_persona(persona: str) -> str:
    """Map legacy persona names into the six active personas."""
    return OLD_TO_NEW_PERSONA.get(persona, persona)


class ContactOutreach(NamedTuple):
    """Generated outreach output for one contact."""

    name: str
    company: str
    persona: str
    subject: str
    email: str
    narrative_variant_id: str


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
            return map_persona(persona)
    return "Discovery"


def build_email(
    name: str,
    company: str,
    persona: str,
    used_emails: set[str] | None = None,
) -> ContactOutreach:
    """Build outreach from one random subject and one random narrative."""
    first_name = get_contact_first_name(name)
    company_text = company or "your organization"
    active_persona = map_persona(persona)
    variant_set = get_narrative_set(active_persona)
    used_emails = used_emails if used_emails is not None else set()

    narrative_options = list(enumerate(variant_set["narratives"], start=1))
    subject_options = list(enumerate(variant_set["subjects"], start=1))
    random.shuffle(narrative_options)
    random.shuffle(subject_options)

    for narrative_index, narrative in narrative_options:
        for subject_index, subject_template in subject_options:
            subject = subject_template.format(company=company_text)
            email = (
                f"Dear {first_name},\n\n"
                "My name is Helmut von Keyserling, and I support "
                f"{company_text} as Strategic Account Manager at Metabolon.\n\n"
                f"{narrative}\n\n"
                "If this is relevant to your work, I would welcome a brief conversation.\n\n"
                "Best regards,\n\n"
                "Helmut von Keyserling\n"
                "Strategic Account Manager"
            )
            if email not in used_emails:
                used_emails.add(email)
                variant_id = f"{active_persona}-{narrative_index:02d}-S{subject_index:02d}"
                return ContactOutreach(
                    name, company, active_persona, subject, email, variant_id
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
            "Persona",
            "Subject",
            "Email",
            "narrative_variant_id",
        ]
    )


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
    """Generate one persona-based email row for every uploaded contact."""
    name_column = find_name_column(contacts)
    if not name_column:
        raise ValueError("Could not find any columns in the uploaded file.")

    company_column = find_company_column(contacts)
    role_columns = find_role_columns(contacts)
    rows: list[ContactOutreach] = []
    used_emails: set[str] = set()
    progress = st.progress(0, text="Classifying contacts and generating emails...")

    total_contacts = len(contacts)
    for position, (_, contact) in enumerate(contacts.iterrows(), start=1):
        name = get_cell_value(contact, name_column, "Unnamed Contact")
        company = get_cell_value(contact, company_column, "")
        persona = identify_persona(contact, role_columns)
        rows.append(build_email(name, company, persona, used_emails))
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
            "Persona",
            "Subject",
            "Email",
            "narrative_variant_id",
        ],
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

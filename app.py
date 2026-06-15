"""Andy Bot Streamlit application for persona-template Metabolon outreach.

Workflow:
1. Upload a CSV contact list.
2. Classify each contact into one persona template.
3. Select the matching Metabolon narrative.
4. Generate Name, Company, Persona, Subject, and Email for CSV export.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

DEFAULT_MODEL = "gpt-4.1-mini"

PERSONAS = (
    "Discovery",
    "Translational Research",
    "Clinical Development",
    "Clinical Biomarkers",
    "Bioanalysis",
    "Medical Affairs",
    "Epidemiology",
    "Safety / Risk",
    "Computational Biology",
    "Oncology Research",
    "Immunology Research",
)

PERSONA_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Epidemiology",
        (
            "epidemiology",
            "epidemiologist",
            "heor",
            "health economics",
            "outcomes research",
            "real world evidence",
            "rwe",
            "population health",
            "cohort",
        ),
    ),
    (
        "Safety / Risk",
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
        "Oncology Research",
        ("oncology", "cancer", "tumor", "tumour", "immuno-oncology", "io research"),
    ),
    (
        "Immunology Research",
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

NARRATIVES = {
    "Discovery": "Metabolomics can connect target biology, mechanism, and pathway-level phenotype early enough to shape discovery decisions.",
    "Translational Research": "Metabolomics can bridge model systems and human samples to clarify mechanism, pharmacology, and translational relevance.",
    "Clinical Development": "Metabolomics can read drug response, disease biology, and patient heterogeneity within clinical studies without adding a large operational burden.",
    "Clinical Biomarkers": "Metabolomics can support biomarker discovery and validation tied to pharmacodynamic response, patient segmentation, and disease activity.",
    "Bioanalysis": "Broad LC-MS metabolomics can add biochemical context alongside targeted bioanalytical and PK/PD work.",
    "Medical Affairs": "Metabolomics evidence can help explain disease biology, treatment response, and clinically meaningful biochemical differences to scientific stakeholders.",
    "Epidemiology": "Metabolomics can add biological depth to cohorts, outcomes research, RWE, and population-level disease stratification.",
    "Safety / Risk": "Metabolomics can help interpret toxicity, off-target effects, and risk signals through earlier biochemical change detection.",
    "Computational Biology": "Metabolomics can serve as a high-dimensional phenotype layer that strengthens multi-omics models and biological interpretation.",
    "Oncology Research": "Metabolomics can characterize tumor metabolism, host response, treatment effect, and resistance biology across oncology studies.",
    "Immunology Research": "Metabolomics can clarify immune cell state, inflammatory pathways, disease activity, and treatment response in immune-mediated disease.",
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


def build_email_prompt(
    contact: pd.Series,
    name: str,
    company: str,
    persona: str,
    narrative: str,
    role_columns: list[str],
) -> str:
    """Create a strict prompt for one persona-template outreach email."""
    first_name = get_contact_first_name(name)
    return f"""
Generate one Metabolon outreach email from the selected persona template.

Output only valid JSON with exactly these keys:
{{"subject":"...","email":"..."}}

Contact name: {name}
Recipient first name: {first_name}
Company: {company}
Persona: {persona}
Selected Metabolon narrative: {narrative}

Title and role evidence:
{row_to_text(contact, role_columns)}

Other CSV context:
{row_to_text(contact)}

Style requirements:
- Similar in style to Andrew Noel outreach emails: direct, plain-spoken, scientifically literate, specific, and restrained.
- Simple and scientifically relevant to the persona.
- Ask for a brief meeting.
- No marketing language.
- No reports, long reports, contact intelligence reports, why-this-person outputs, confidence scores, explanations, reasoning, or markdown.

Email rules:
- 80-140 words.
- Start exactly with: Dear {first_name},
- Mention Metabolon at most once.
- Do not start with "My name is", "I noticed", "I came across", or "I saw".
- Avoid vendor language and do not use: capabilities, platform, solution, leverage, synergy, value proposition, actionable insights, discuss further.
- Close with a simple request for a brief meeting or 20-minute conversation.
""".strip()


def generate_text(prompt: str, model: str) -> str:
    """Generate text using the OpenAI Responses API."""
    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    return str(response).strip()


def parse_outreach(
    raw_output: str, fallback_name: str, fallback_company: str, persona: str
) -> ContactOutreach:
    """Parse model JSON into stable table fields."""
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return ContactOutreach(fallback_name, fallback_company, persona, "", cleaned)

    return ContactOutreach(
        fallback_name,
        fallback_company,
        persona,
        str(payload.get("subject") or "").strip(),
        str(payload.get("email") or "").strip(),
    )


def read_contacts(uploaded_file: Any) -> pd.DataFrame:
    """Read contacts from a CSV upload."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Upload a CSV file.")


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


def generate_outreach_table(contacts: pd.DataFrame, model: str) -> pd.DataFrame:
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
        narrative = NARRATIVES[persona]
        raw_output = generate_text(
            build_email_prompt(contact, name, company, persona, narrative, role_columns),
            model,
        )
        rows.append(parse_outreach(raw_output, name, company, persona))
        progress.progress(
            position / total_contacts,
            text=f"Generated {position} of {total_contacts} emails",
        )

    progress.empty()
    return pd.DataFrame(
        rows, columns=["Name", "Company", "Persona", "Subject", "Email"]
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Andy Bot", page_icon="🤖", layout="wide")
    initialize_session_state()

    st.title("🤖 Andy Bot")
    st.caption(
        "Upload a CSV contact list to classify each contact into a persona template and export one email per contact."
    )

    with st.sidebar:
        st.header("Settings")
        model = st.text_input(
            "OpenAI model",
            value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            help="Override with any model available to your API key.",
        )
        st.markdown("**Personas**")
        st.write(", ".join(PERSONAS))

    uploaded_file = st.file_uploader("Upload contacts CSV", type=["csv"])
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
        if not model.strip():
            st.error("Enter an OpenAI model before generating emails.")
            return

        try:
            st.session_state.generated_emails = generate_outreach_table(
                contacts, model.strip()
            )
        except OpenAIError as exc:
            st.error(f"OpenAI request failed: {exc}")
            return
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

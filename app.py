"""Andy Bot Streamlit application for rapid Metabolon outreach generation.

Workflow:
1. Upload a CSV or XLSX contact list.
2. Identify each contact persona from title and role fields.
3. Select the appropriate Metabolon outreach narrative.
4. Generate only Name, Subject, and Email in a copy/paste table.
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
MAX_CONTACTS = 20

PERSONAS = (
    "Discovery",
    "Translational Research",
    "Clinical Development",
    "Clinical Biomarkers",
    "Bioanalysis",
    "Medical Affairs",
    "Epidemiology / HEOR",
    "Safety / Risk Management",
    "Computational Biology",
    "Oncology Research",
    "Immunology Research",
)

PERSONA_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Epidemiology / HEOR",
        (
            "epidemiology",
            "epidemiologist",
            "heor",
            "health economics",
            "outcomes research",
            "real world evidence",
            "rwe",
            "market access",
        ),
    ),
    (
        "Safety / Risk Management",
        (
            "safety",
            "pharmacovigilance",
            "risk management",
            "patient safety",
            "drug safety",
            "benefit risk",
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
    "Discovery": "Use metabolomics to connect target biology, mechanism, and pathway-level phenotype early enough to shape discovery decisions.",
    "Translational Research": "Use metabolomics as a bridge between model systems and human samples to clarify mechanism, pharmacology, and translational relevance.",
    "Clinical Development": "Use metabolomics to read drug response, disease biology, and patient heterogeneity within clinical studies without adding a large operational burden.",
    "Clinical Biomarkers": "Use metabolomics to discover and validate biomarkers tied to pharmacodynamic response, patient segmentation, and disease activity.",
    "Bioanalysis": "Use Metabolon's LC-MS metabolomics experience to add broad biochemical context alongside targeted bioanalytical and PK/PD work.",
    "Medical Affairs": "Use metabolomics evidence to help explain disease biology, treatment response, and clinically meaningful biochemical differences to scientific stakeholders.",
    "Epidemiology / HEOR": "Use metabolomics to add biological depth to cohorts, outcomes research, RWE, and population-level disease stratification.",
    "Safety / Risk Management": "Use metabolomics to detect biochemical changes that may help interpret toxicity, off-target effects, and risk signals earlier.",
    "Computational Biology": "Use metabolomics as a high-dimensional phenotype layer that can strengthen multi-omics models and biological interpretation.",
    "Oncology Research": "Use metabolomics to characterize tumor metabolism, host response, treatment effect, and resistance biology across oncology studies.",
    "Immunology Research": "Use metabolomics to understand immune cell state, inflammatory pathways, disease activity, and treatment response in immune-mediated disease.",
}


class ContactOutreach(NamedTuple):
    """Generated outreach output for one contact."""

    name: str
    subject: str
    email: str


def find_name_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely contact-name column in uploaded contact data."""
    if dataframe.empty:
        return None

    normalized = {column.lower().strip(): column for column in dataframe.columns}
    for candidate in (
        "name",
        "full_name",
        "full name",
        "contact",
        "contact_name",
        "contact name",
    ):
        if candidate in normalized:
            return normalized[candidate]

    text_columns = [
        column for column in dataframe.columns if dataframe[column].dtype == "object"
    ]
    return text_columns[0] if text_columns else dataframe.columns[0]


def find_role_columns(dataframe: pd.DataFrame) -> list[str]:
    """Find title and role columns that should drive persona selection."""
    role_terms = ("title", "role", "position", "job", "function", "department")
    matches = [
        column
        for column in dataframe.columns
        if any(term in column.lower() for term in role_terms)
    ]
    return matches or list(dataframe.columns)


def get_contact_name(contact: pd.Series, name_column: str) -> str:
    """Return the selected contact's display name."""
    value: Any = contact.get(name_column, "")
    if pd.isna(value) or str(value).strip() == "":
        return "Unnamed Contact"
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
    """Identify the primary outreach persona from title and role fields."""
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
    contact: pd.Series, name: str, persona: str, narrative: str, role_columns: list[str]
) -> str:
    """Create a strict prompt for one persona-specific outreach email."""
    first_name = get_contact_first_name(name)
    return f"""
Generate one Metabolon outreach email.

Output only valid JSON with exactly these keys:
{{"name":"{name}","subject":"...","email":"..."}}

Contact name: {name}
Recipient first name: {first_name}
Identified persona: {persona}
Metabolon outreach narrative: {narrative}

Title and role evidence:
{row_to_text(contact, role_columns)}

Other CSV context:
{row_to_text(contact)}

Style requirements:
- Similar in style to Andrew Noel outreach emails: direct, plain-spoken, scientifically literate, specific, and restrained.
- Persona-specific and scientifically relevant to the identified persona.
- Simple meeting request.
- No reports, explanations, scores, reasoning, or markdown.

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


def parse_outreach(raw_output: str, fallback_name: str) -> ContactOutreach:
    """Parse model JSON into stable table fields."""
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return ContactOutreach(fallback_name, "", cleaned)

    return ContactOutreach(
        str(payload.get("name") or fallback_name).strip(),
        str(payload.get("subject") or "").strip(),
        str(payload.get("email") or "").strip(),
    )


def read_contacts(uploaded_file: Any) -> pd.DataFrame:
    """Read contacts from a CSV or XLSX upload."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a CSV or XLSX file.")


def initialize_session_state() -> None:
    """Initialize workflow session state keys."""
    defaults = {
        "contacts": None,
        "uploaded_filename": "",
        "generated_emails": pd.DataFrame(columns=["Name", "Subject", "Email"]),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def generate_outreach_table(contacts: pd.DataFrame, model: str) -> pd.DataFrame:
    """Generate outreach for up to 20 contacts and return copy/paste table columns only."""
    name_column = find_name_column(contacts)
    if not name_column:
        raise ValueError("Could not find any columns in the uploaded file.")

    role_columns = find_role_columns(contacts)
    rows: list[ContactOutreach] = []
    progress = st.progress(0, text="Generating emails...")

    contacts_to_process = contacts.head(MAX_CONTACTS)
    for position, (_, contact) in enumerate(contacts_to_process.iterrows(), start=1):
        name = get_contact_name(contact, name_column)
        persona = identify_persona(contact, role_columns)
        narrative = NARRATIVES[persona]
        raw_output = generate_text(
            build_email_prompt(contact, name, persona, narrative, role_columns), model
        )
        rows.append(parse_outreach(raw_output, name))
        progress.progress(
            position / len(contacts_to_process),
            text=f"Generated {position} of {len(contacts_to_process)} emails",
        )

    progress.empty()
    return pd.DataFrame(rows, columns=["Name", "Subject", "Email"])


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Andy Bot", page_icon="🤖", layout="wide")
    initialize_session_state()

    st.title("🤖 Andy Bot")
    st.caption(
        "Upload a CSV contact list and generate Name, Subject, and Email for up to 20 contacts."
    )

    with st.sidebar:
        st.header("Settings")
        model = st.text_input(
            "OpenAI model",
            value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            help="Override with any model available to your API key.",
        )

    uploaded_file = st.file_uploader(
        "Upload contacts CSV or XLSX", type=["csv", "xlsx", "xls"]
    )
    if (
        uploaded_file is not None
        and uploaded_file.name != st.session_state.uploaded_filename
    ):
        try:
            contacts = read_contacts(uploaded_file)
        except Exception as exc:  # pandas parsing errors vary by file type and version
            st.error(f"Could not read contacts file: {exc}")
            return

        if contacts.empty:
            st.error("The uploaded file has no rows.")
            return

        st.session_state.contacts = contacts
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.generated_emails = pd.DataFrame(
            columns=["Name", "Subject", "Email"]
        )
        st.success(
            f"Imported {len(contacts)} contacts. Andy Bot will process the first {min(len(contacts), MAX_CONTACTS)}."
        )

    contacts = st.session_state.contacts
    if contacts is None:
        st.warning("Upload a CSV or XLSX file to get started.")
        return

    preview_contacts = contacts.head(MAX_CONTACTS)
    st.subheader("Contacts to Process")
    st.dataframe(preview_contacts, use_container_width=True)

    if len(contacts) > MAX_CONTACTS:
        st.info(
            f"Only the first {MAX_CONTACTS} contacts will be processed to keep the workflow under 10 minutes."
        )

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
            "Download CSV",
            data=st.session_state.generated_emails.to_csv(index=False),
            file_name="andy_bot_generated_emails.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

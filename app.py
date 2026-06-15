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

EMAIL_VARIANTS = {
    "Discovery": {
        "subjects": (
            "Metabolomics in discovery programs at {company}",
            "Biochemical context for discovery work at {company}",
            "A metabolomics lens on early biology at {company}",
            "Connecting phenotype and pathway biology at {company}",
            "Metabolomics support for discovery decisions at {company}",
        ),
        "narratives": (
            "Discovery groups often need a practical way to connect phenotype, target biology, and pathway-level biochemistry before programs move too far downstream.",
            "Metabolomics can help early research teams see which biochemical pathways shift when a target, model, or compound changes the underlying biology.",
            "A common challenge in discovery is deciding whether an observed phenotype reflects a durable mechanism or a narrower model-specific effect.",
            "Many discovery programs use metabolomics to add functional context when genetics, transcriptomics, or cellular assays do not fully explain the biology.",
            "Early research decisions are often stronger when teams can measure broad biochemical activity rather than rely on a single marker or endpoint.",
        ),
    },
    "Translational Research": {
        "subjects": (
            "Metabolomics for translational research at {company}",
            "Connecting model and human biology at {company}",
            "Translational metabolomics support for {company}",
            "Biochemical readouts across translational studies at {company}",
            "Metabolomics across preclinical and clinical biology at {company}",
        ),
        "narratives": (
            "Many translational teams are using metabolomics to compare biology across model systems and human samples.",
            "One area where metabolomics has become useful is connecting preclinical findings with what is observed clinically.",
            "Many groups are looking for additional ways to understand whether biological signals observed in models persist in patients.",
            "Metabolomics is increasingly being used to provide a functional readout that spans discovery and clinical research.",
            "A recurring challenge in translational work is deciding which biological signals are most likely to matter clinically.",
        ),
    },
    "Clinical Biomarkers": {
        "subjects": (
            "Metabolomics for biomarker work at {company}",
            "Biochemical context for biomarkers at {company}",
            "Clinical biomarker support for {company}",
            "Metabolomics in patient stratification at {company}",
            "Functional biomarker readouts at {company}",
        ),
        "narratives": (
            "Biomarker teams often need functional evidence that connects patient biology with response, disease activity, or mechanism.",
            "Metabolomics can help identify biochemical markers that add context to pharmacodynamic signals and patient segmentation strategies.",
            "A recurring biomarker challenge is finding signals that are measurable, biologically interpretable, and useful across clinical samples.",
            "Many clinical biomarker groups use metabolomics to understand which pathway changes may explain response or disease heterogeneity.",
            "Metabolomics can provide a broad biochemical readout when single-analyte markers do not capture enough of the patient biology.",
        ),
    },
    "Clinical Development": {
        "subjects": (
            "Metabolomics for clinical development at {company}",
            "Biochemical readouts in clinical studies at {company}",
            "Clinical metabolomics support for {company}",
            "Adding biology to development decisions at {company}",
            "Metabolomics across clinical samples at {company}",
        ),
        "narratives": (
            "Clinical development teams often need more biological context around response, dose, patient heterogeneity, and endpoint interpretation.",
            "Metabolomics can help read drug effect and disease biology directly from clinical samples without adding a large operational burden.",
            "One challenge in development is understanding why patients respond differently when standard endpoints only show part of the picture.",
            "Many clinical groups use metabolomics to add functional pathway evidence around treatment effect, progression, or responder analyses.",
            "Clinical samples can carry useful biochemical information that helps clarify mechanism, response patterns, and decisions between study stages.",
        ),
    },
    "Medical Affairs": {
        "subjects": (
            "Metabolomics for medical affairs at {company}",
            "Biochemical evidence for scientific exchange at {company}",
            "Disease biology context for {company}",
            "Metabolomics in medical discussions at {company}",
            "Scientific evidence generation support for {company}",
        ),
        "narratives": (
            "Medical affairs teams often need clear biological evidence to support scientific exchange with clinicians, researchers, and external experts.",
            "Metabolomics can help explain disease biology and treatment response through measurable biochemical differences rather than broad claims.",
            "One useful role for metabolomics is building evidence that makes mechanism and patient biology easier to discuss in a scientific setting.",
            "Many medical teams look for ways to connect emerging data with disease pathways that clinicians and investigators can interpret.",
            "A common need in medical affairs is credible, biologically grounded evidence that supports education, data generation, and external engagement.",
        ),
    },
    "Oncology": {
        "subjects": (
            "Metabolomics for oncology work at {company}",
            "Tumor metabolism and response biology at {company}",
            "Oncology metabolomics support for {company}",
            "Biochemical context in oncology studies at {company}",
            "Metabolomics across oncology samples at {company}",
        ),
        "narratives": (
            "Oncology teams often need to understand tumor metabolism, host response, resistance biology, and treatment effect across complex samples.",
            "Metabolomics can help clarify which biochemical pathways shift with therapy, progression, or differences between responder groups.",
            "A recurring challenge in oncology is connecting molecular findings with functional biology that may explain response or resistance.",
            "Many oncology groups use metabolomics to add pathway-level evidence around mechanism, stratification, and clinical sample interpretation.",
            "Cancer studies often generate many signals, and metabolomics can help show which changes reflect active biochemical biology.",
        ),
    },
    "Immunology": {
        "subjects": (
            "Metabolomics for immunology work at {company}",
            "Biochemical context for immune biology at {company}",
            "Immunology metabolomics support for {company}",
            "Metabolomics in inflammation studies at {company}",
            "Functional readouts for immune-mediated disease at {company}",
        ),
        "narratives": (
            "Immunology teams often need to connect immune activity with biochemical pathways tied to inflammation, disease activity, and response.",
            "Metabolomics can add functional context when cytokine, cellular, or transcriptomic readouts do not fully explain immune biology.",
            "A common challenge in immune-mediated disease is understanding which pathway changes reflect disease state versus treatment effect.",
            "Many immunology groups use metabolomics to study inflammation, autoimmune biology, and response patterns in clinical or model systems.",
            "Metabolic pathways can provide useful evidence about immune activation and resolution when teams need a broader biological readout.",
        ),
    },
    "Safety/Risk": {
        "subjects": (
            "Metabolomics for safety and risk work at {company}",
            "Biochemical context for safety signals at {company}",
            "Metabolomics in risk interpretation at {company}",
            "Safety biology support for {company}",
            "Understanding toxicity signals at {company}",
        ),
        "narratives": (
            "Safety teams often need to understand whether biochemical changes point to adaptive biology, off-target effects, or signals that require follow-up.",
            "Metabolomics can help interpret toxicity and organ-specific effects by showing pathway-level changes behind observed safety findings.",
            "A recurring challenge in risk assessment is deciding which early biological signals are meaningful enough to influence program decisions.",
            "Many safety groups use metabolomics to add mechanistic context around adverse findings, exposure effects, and emerging risk signals.",
            "Broad biochemical profiling can help distinguish nonspecific changes from patterns that suggest a clearer safety or risk mechanism.",
        ),
    },
    "Bioanalysis": {
        "subjects": (
            "Metabolomics alongside bioanalysis at {company}",
            "Biochemical context for PK/PD work at {company}",
            "Bioanalysis and metabolomics support for {company}",
            "Connecting exposure and response at {company}",
            "Metabolomics around targeted assays at {company}",
        ),
        "narratives": (
            "Bioanalysis teams often need broader biochemical context when targeted assays or PK/PD data do not fully explain response biology.",
            "Metabolomics can complement established analyte panels by showing pathway-level changes linked to exposure, response, or disease state.",
            "A common challenge in bioanalysis is connecting precise measurements with the wider biology occurring in the same samples.",
            "Many groups use metabolomics alongside targeted work to understand whether exposure is translating into expected biochemical effects.",
            "Broad profiling can help place specific assay results into a functional context that is easier for research and clinical teams to interpret.",
        ),
    },
    "Computational Biology": {
        "subjects": (
            "Metabolomics for computational biology at {company}",
            "A phenotype layer for multi-omics at {company}",
            "Biochemical data for computational models at {company}",
            "Connecting omics signals to metabolism at {company}",
            "Metabolomics in systems biology work at {company}",
        ),
        "narratives": (
            "Computational biology teams often need a measured phenotype layer that connects upstream omics patterns with functional biochemical activity.",
            "Metabolomics can strengthen multi-omics interpretation by showing which pathways are active in a disease, model, or treatment context.",
            "A recurring challenge in computational work is translating genomic or transcriptomic signals into biology that can be measured directly.",
            "Many systems biology groups use metabolomics to anchor models in biochemical readouts from the same samples or study context.",
            "Broad metabolite data can help connect algorithms, pathway models, and biological hypotheses to observable changes in physiology.",
        ),
    },
}

CTA_VARIANTS = (
    "If this is relevant, I would welcome a brief 20-minute conversation.",
    "If useful, I would be glad to compare notes in a short 20-minute call.",
    "If this aligns with current work, I would welcome a brief conversation.",
    "If helpful, I would be glad to discuss where this may fit in a 20-minute call.",
    "If there is interest, I would welcome a short conversation to share more detail.",
)



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


def build_email(
    name: str,
    company: str,
    persona: str,
    used_emails: set[str] | None = None,
) -> ContactOutreach:
    """Build outreach from a random persona narrative while avoiding batch duplicates."""
    first_name = get_contact_first_name(name)
    company_text = company or "your organization"
    variant_set = EMAIL_VARIANTS[persona]
    used_emails = used_emails if used_emails is not None else set()

    combinations = [
        (subject_template, narrative, cta)
        for subject_template in variant_set["subjects"]
        for narrative in variant_set["narratives"]
        for cta in CTA_VARIANTS
    ]
    random.shuffle(combinations)

    for subject_template, narrative, cta in combinations:
        subject = subject_template.format(company=company_text)
        email = (
            f"Dear {first_name},\n\n"
            "My name is Helmut von Keyserling, and I support "
            f"{company_text} as Strategic Account Manager at Metabolon.\n\n"
            f"{narrative} Metabolon helps teams measure broad biochemical activity "
            "from research and clinical samples, with results that are designed to be "
            f"interpretable for scientific decision-making across study designs and existing scientific questions.\n\n"
            f"{cta}\n\n"
            "Best regards,\n"
            "Helmut von Keyserling\n"
            "Strategic Account Manager"
        )
        if email not in used_emails:
            used_emails.add(email)
            return ContactOutreach(name, company, persona, subject, email)

    raise ValueError(
        f"Could not generate a unique {persona} email for {name} at {company_text}."
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
        rows, columns=["Name", "Company", "Persona", "Subject", "Email"]
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

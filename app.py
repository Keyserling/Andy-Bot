"""Andy Bot V1 Streamlit application.

A simple local contact intelligence workflow:
1. Upload a CSV or XLSX file of contacts.
2. Import contacts into Streamlit session state.
3. Search and select one contact.
4. Generate a reusable markdown intelligence report with OpenAI.
5. Generate an outreach email from the saved report.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL = "gpt-4.1-mini"

DEFAULT_SENDER_NAME = "Helmut von Keyserling"
DEFAULT_SENDER_TITLE = "Strategic Account Manager"
DEFAULT_SENDER_COMPANY = "Metabolon"
DEFAULT_SUPPORTED_ACCOUNT = "the account"

SCIENTIFIC_RELEVANCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Discovery Research", r"\b(discovery research|drug discovery|early discovery|target discovery)\b"),
    ("Translational Research", r"\b(translational research|translational science|translational medicine)\b"),
    ("Biomarkers", r"\b(biomarker|biomarkers|pharmacodynamic biomarker|clinical biomarker)\b"),
    ("Pharmacology", r"\b(pharmacology|pharmacodynamic|pharmacokinetic|pk/pd)\b"),
    ("Clinical Development", r"\b(clinical development|clinical trial|clinical trials|clinical study|clinical studies)\b"),
    ("Bioanalytics", r"\b(bioanalytical|bioanalytics|bioanalysis|bioanalytical sciences)\b"),
    ("Omics", r"\b(omics|metabolomics|proteomics|genomics|transcriptomics|lipidomics)\b"),
    ("Disease Biology", r"\b(disease biology|biology|immunology|oncology|neuroscience|metabolism|respiratory biology)\b"),
)

ORGANIZATIONAL_FIT_KEYWORDS = (
    "portfolio",
    "operations",
    "pmo",
    "project management",
    "program management",
    "risk management",
    "risk",
    "procurement",
    "sourcing",
    "strategy",
    "strategic",
    "finance",
    "financial",
    "legal",
    "compliance",
)


SenderFields = dict[str, str]
FitLevel = Literal["High", "Medium", "Low"]


class ScientificRelevance(NamedTuple):
    """Deterministic scientific relevance gate based only on explicit profile text."""

    is_relevant: bool
    evidence: tuple[str, ...]


def clean_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "contact"


def find_name_column(dataframe: pd.DataFrame) -> str | None:
    """Find the most likely contact-name column in uploaded contact data."""
    if dataframe.empty:
        return None

    normalized = {column.lower().strip(): column for column in dataframe.columns}
    for candidate in ("name", "full_name", "full name", "contact", "contact_name", "contact name"):
        if candidate in normalized:
            return normalized[candidate]

    text_columns = [column for column in dataframe.columns if dataframe[column].dtype == "object"]
    return text_columns[0] if text_columns else dataframe.columns[0]


def row_to_markdown(contact: pd.Series) -> str:
    """Format a selected contact row as markdown bullets."""
    lines: list[str] = []
    for field, value in contact.items():
        if pd.isna(value) or str(value).strip() == "":
            display_value = "Not provided"
        else:
            display_value = str(value).strip()
        lines.append(f"- **{field}:** {display_value}")
    return "\n".join(lines)


def assess_scientific_relevance(contact: pd.Series, linkedin_profile_text: str = "") -> ScientificRelevance:
    """Return explicit profile signals that earn scientific relevance claims."""
    evidence: list[str] = []
    for field, value in contact.items():
        if pd.isna(value) or str(value).strip() == "":
            continue
        display_value = str(value).strip()
        for label, pattern in SCIENTIFIC_RELEVANCE_PATTERNS:
            if re.search(pattern, display_value, flags=re.IGNORECASE):
                evidence.append(f"{label}: {field} = {display_value}")
                break

    linkedin_profile_text = linkedin_profile_text.strip()
    if linkedin_profile_text:
        for label, pattern in SCIENTIFIC_RELEVANCE_PATTERNS:
            if re.search(pattern, linkedin_profile_text, flags=re.IGNORECASE):
                evidence.append(f"{label}: LinkedIn Profile Text = {linkedin_profile_text}")
                break

    return ScientificRelevance(bool(evidence), tuple(dict.fromkeys(evidence)))


def format_linkedin_profile_context(linkedin_profile_text: str) -> str:
    """Format pasted LinkedIn profile content for model prompts."""
    linkedin_profile_text = linkedin_profile_text.strip()
    if not linkedin_profile_text:
        return "LinkedIn Profile Text: Not provided"
    return f"LinkedIn Profile Text:\n{linkedin_profile_text}"


def format_scientific_relevance_report(assessment: ScientificRelevance) -> str:
    """Format the relevance gate as a visible markdown field for reports and prompts."""
    evidence_lines = "\n".join(f"   - {item}" for item in assessment.evidence)
    if not evidence_lines:
        evidence_lines = "   - None"
    return (
        "Scientific Relevance:\n"
        f"- Scientific Relevance: {'TRUE' if assessment.is_relevant else 'FALSE'}\n"
        "- Evidence:\n"
        f"{evidence_lines}"
    )


def attach_scientific_relevance(report: str, assessment: ScientificRelevance) -> str:
    """Prepend the deterministic relevance gate without relying on model-generated evidence."""
    if report.lstrip().lower().startswith("scientific relevance:"):
        return report
    return f"{format_scientific_relevance_report(assessment)}\n\n{report.strip()}"


def build_report_prompt(contact: pd.Series, linkedin_profile_text: str = "") -> str:
    """Create a prompt for Metabolon-focused contact intelligence."""
    return f"""
Create a concise, reusable contact intelligence report from the structured contact details below.

Goal:
Replicate the outreach style used by Andrew Noel at Metabolon. The report should identify the
contact's scientific or business identity only when directly supported by the contact details.

Use uploaded contact fields and pasted LinkedIn Profile Text. Do not browse LinkedIn or external websites.
Do not use Outlook, Gmail, web browsing, or any external enrichment. Do not claim facts that are not present in the data. If information is missing,
say "Not provided" or "Insufficient information."

Do NOT generate psychological profiles, buying signals, strategic risk assessments, personality
analyses, or speculative business conclusions.

Extract and return:
- Current role
- Seniority level
- Company
- Functional area
- Therapeutic area(s)
- Top 10-20 recurring keywords
- Scientific/business themes
- Likely Metabolon-relevant interests only when direct evidence exists
- Recommended outreach angle only when direct scientific evidence exists

Assign exactly one primary persona from this list:
- Discovery Research
- Translational Research
- Biomarkers
- Clinical Development
- Bioanalytical Sciences
- Pharmacology
- Portfolio Strategy
- Immunology
- Oncology
- Respiratory
- Metabolism
- Neuroscience
- Infectious Disease
- Other

Select exactly one recommended outreach angle from this list:
- Mechanism of Action
- Translational Biomarkers
- Pharmacodynamic Biomarkers
- Pathway Biology
- Host Response Biology
- Metabolic Phenotyping
- Patient Stratification
- Quantitative Biology
- Clinical Biomarkers
- Discovery Research Support
- Portfolio Decision Support

Before assigning relevance, determine whether the profile contains direct evidence of involvement in
Discovery Research, Translational Research, Biomarkers, Pharmacology, Clinical Development,
Bioanalytics, Omics, or Disease Biology. Do not treat portfolio, risk, strategy, operations, PMO,
procurement, finance, legal, or leadership language as scientific relevance.

Clearly separate observed facts from inferred themes. Inferences must be conservative and directly
grounded in the provided contact data. Do not invent interests or priorities.

Return markdown with exactly these sections:
1. Intelligence Report
   - Observed Facts
   - Inferred Themes
   - Primary Persona
   - Recommended Outreach Angle
2. Scientific Relevance
   - Scientific Relevance: TRUE or FALSE
   - Evidence: exact profile signals used, or None

Contact details:
{row_to_markdown(contact)}

Additional contact context:
{format_linkedin_profile_context(linkedin_profile_text)}
""".strip()


def build_email_prompt(
    report: str,
    contact: pd.Series,
    first_name: str,
    fit: FitLevel,
    fit_reason: str,
    scientific_relevance: ScientificRelevance,
    sender: SenderFields,
    linkedin_profile_text: str = "",
) -> str:
    """Create an Andrew Noel-style outreach email based on a saved report."""
    greeting_name = first_name or "Colleague"
    return f"""
Generate a credible, personalized Metabolon outreach email using only the contact data and saved
contact intelligence report below. Write in Andrew Noel style: calm, professional, direct,
non-salesy, and without hype. Prioritize evidence from the pasted LinkedIn Profile Text over job title alone.

Sender context:
{format_sender_context(sender)}

Recipient first name for greeting: {greeting_name}
Fit classification: {fit}
Reason for fit: {fit_reason}
{format_scientific_relevance_report(scientific_relevance)}

Relevance discipline:
- Scientific Relevance must be earned before any scientific relevance claim is allowed.
- Scientific Relevance TRUE requires exact evidence listed above for Discovery Research, Translational
  Research, Biomarkers, Pharmacology, Clinical Development, Bioanalytics, Omics, or Disease Biology.
- Scientific Relevance FALSE means the goal is referral discovery, not scientific selling.
- The email may only use scientific evidence explicitly listed above.

Contact details:
{row_to_markdown(contact)}

Additional contact context:
{format_linkedin_profile_context(linkedin_profile_text)}

Requirements:
- Output exactly these markdown sections:
  1. Fit Classification
  2. Reason for Fit
  3. Scientific Relevance
  4. Evidence
  5. Generated Outreach Email
- The Generated Outreach Email must sound like a thoughtful human reaching out to another
  professional.
- The email body must be 120 words or fewer.
- The email body must be 6 sentences or fewer.
- Start with the recipient, never with Metabolon.
- Mention one specific aspect of the person's role, career, responsibility, or background.
- Do not flatter, praise, or use marketing language.
- Do not describe Metabolon for more than one sentence.
- Avoid broad claims.
- Do not mention LinkedIn, Outlook, Gmail, external enrichment, or unavailable facts.
- Do not use these words or phrases in the email: "capabilities", "solutions", "platform",
  "precision medicine", "support your efforts", "align with objectives", "value", "leverage",
  "synergy", "discussion", "I hope this message finds you well", "transformational",
  "revolutionary", "game-changing", "game changing", "save millions", "maximize value".
- Do not infer portfolio risk reduction, strategic risk management benefits, commercial value,
  cost savings, or decision-grade evidence unless those facts are explicitly present in the data.
- Sign off with {sender['name']} and {sender['title']} without adding extra claims.

Email structure:
1. Sentence 1: Specific observation about the person.
2. Sentence 2: A real problem or question relevant to that role.
3. Sentence 3: One sentence about how metabolomics can sometimes help illuminate that problem.
4. Sentence 4: Why I thought of them specifically.
5. Sentence 5: Simple meeting request; do not use the word "discussion".
6. Sentence 6: Redirect request if someone else owns the topic.

Scientific relevance handling:
- If Scientific Relevance is TRUE, ground the email only in exact evidence listed above.
- If Scientific Relevance is FALSE, do not propose a specific Metabolon use case and do not infer
  scientific needs. Keep the note to referral discovery using the same six-sentence structure.
- When Scientific Relevance is FALSE, the redirect request should ask who owns translational
  research, biomarkers, discovery, pharmacology, clinical development, or omics activities.

Forbidden when Scientific Relevance is FALSE:
- Do not propose a specific Metabolon use case.
- Do not mention pathway biology, disease biology, translational research needs, biomarker needs,
  pharmacology support, clinical development support, portfolio decision support, risk reduction,
  strategic decision making, risk evaluation, or portfolio decisions.
- Do not say Metabolon can support risk management, improve portfolio decisions, reduce development
  risk, strengthen strategic decision making, improve operations, lower costs, or solve procurement,
  finance, legal, strategy, PMO, or operational problems.

Before finalizing the email, silently ask: "Would a senior pharma executive believe this was
personally written?" If the answer is no, rewrite automatically before returning the final output.

Saved intelligence report:
{report}
""".strip()


def generate_text(prompt: str, model: str) -> str:
    """Generate text using the OpenAI Responses API."""
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    return str(response).strip()


def save_markdown(contact_name: str, content: str, suffix: str) -> Path:
    """Save markdown content into outputs/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{timestamp}_{clean_filename(contact_name)}_{suffix}.md"
    path.write_text(content + "\n", encoding="utf-8")
    return path


def get_contact_name(contact: pd.Series, name_column: str) -> str:
    """Return the selected contact's display name."""
    value: Any = contact.get(name_column, "")
    if pd.isna(value) or str(value).strip() == "":
        return "Unnamed Contact"
    return str(value).strip()


def get_contact_first_name(contact: pd.Series, name_column: str) -> str:
    """Return the contact's first name when a name value is available."""
    full_name = get_contact_name(contact, name_column)
    if full_name == "Unnamed Contact":
        return ""

    name_without_email = re.sub(r"<[^>]+>", "", full_name).strip()
    name_parts = re.findall(r"[A-Za-z][A-Za-z'’-]*", name_without_email)
    return name_parts[0] if name_parts else ""


def classify_contact_fit(
    contact: pd.Series, report: str = "", linkedin_profile_text: str = ""
) -> tuple[FitLevel, str]:
    """Classify outreach fit from explicit contact evidence only."""
    assessment = assess_scientific_relevance(contact, linkedin_profile_text)
    contact_text = f"{row_to_markdown(contact)}\n{linkedin_profile_text}".lower()
    organizational_matches = [keyword for keyword in ORGANIZATIONAL_FIT_KEYWORDS if keyword in contact_text]

    if assessment.is_relevant:
        return (
            "High",
            "Scientific Relevance = TRUE. Exact evidence: " + "; ".join(assessment.evidence) + ".",
        )
    if organizational_matches:
        return (
            "Medium",
            "Scientific Relevance = FALSE. Organizational role signals are present, but no direct "
            "Discovery Research, Translational Research, Biomarkers, Pharmacology, Clinical Development, "
            "Bioanalytics, Omics, or Disease Biology evidence is shown. Organizational signals: "
            + ", ".join(organizational_matches[:4])
            + ".",
        )
    if any(keyword in contact_text for keyword in ("pharma", "biotech", "life science", "medical")):
        return (
            "Medium",
            "Scientific Relevance = FALSE. Only broad organizational or industry relevance is present; "
            "no direct scientific role is shown.",
        )
    return (
        "Low",
        "Scientific Relevance = FALSE. The provided contact data has insufficient direct evidence of "
        "Discovery Research, Translational Research, Biomarkers, Pharmacology, Clinical Development, "
        "Bioanalytics, Omics, or Disease Biology involvement.",
    )


def format_sender_context(sender: SenderFields) -> str:
    """Format sender configuration for prompt grounding."""
    return "\n".join(
        [
            f"- Sender Name: {sender['name']}",
            f"- Sender Title: {sender['title']}",
            f"- Sender Company: {sender['company']}",
            f"- Supported Account / Customer: {sender['account']}",
        ]
    )


def read_contacts(uploaded_file: Any) -> pd.DataFrame:
    """Read contacts from a CSV or XLSX upload."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a CSV or XLSX file.")


def reset_generated_outputs() -> None:
    """Clear generated artifacts that depend on the selected contact."""
    st.session_state.report = ""
    st.session_state.report_path = ""
    st.session_state.email = ""
    st.session_state.email_path = ""


def initialize_session_state() -> None:
    """Initialize workflow session state keys."""
    defaults = {
        "contacts": None,
        "uploaded_filename": "",
        "selected_contact_index": None,
        "report": "",
        "report_path": "",
        "email": "",
        "email_path": "",
        "linkedin_profile_text_by_contact": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def filter_contacts(contacts: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return contacts with any cell containing the search query."""
    if not query.strip():
        return contacts

    normalized_query = query.strip().lower()
    row_matches = contacts.astype(str).apply(
        lambda row: row.str.lower().str.contains(normalized_query, na=False, regex=False).any(),
        axis=1,
    )
    return contacts[row_matches]


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Andy Bot V1", page_icon="🤖", layout="wide")
    initialize_session_state()

    st.title("🤖 Andy Bot V1")
    st.caption("Upload contacts, select one contact, generate a reusable intelligence report, then draft outreach.")

    with st.sidebar:
        st.header("Settings")
        model = st.text_input(
            "OpenAI model",
            value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            help="Override with any model available to your API key.",
        )
        st.subheader("Sender")
        sender_name = st.text_input("Sender Name", value=DEFAULT_SENDER_NAME)
        sender_title = st.text_input("Sender Title", value=DEFAULT_SENDER_TITLE)
        sender_company = st.text_input("Sender Company", value=DEFAULT_SENDER_COMPANY)
        supported_account = st.text_input("Supported Account / Customer", value=DEFAULT_SUPPORTED_ACCOUNT)
        st.info("Outlook, LinkedIn scraping, and Gmail integrations are intentionally not included in V1.")

    uploaded_file = st.file_uploader("Upload contacts CSV or XLSX", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_filename:
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
        st.session_state.selected_contact_index = None
        st.session_state.linkedin_profile_text_by_contact = {}
        reset_generated_outputs()
        st.success(f"Imported {len(contacts)} contacts into this session.")

    contacts = st.session_state.contacts
    if contacts is None:
        st.warning("Upload a CSV or XLSX file to get started.")
        return

    name_column = find_name_column(contacts)
    if not name_column:
        st.error("Could not find any columns in the uploaded file.")
        return

    st.subheader("Searchable Contacts Table")
    search_query = st.text_input("Search contacts", placeholder="Search any field in the uploaded contacts...")
    filtered_contacts = filter_contacts(contacts, search_query)
    st.dataframe(filtered_contacts, use_container_width=True)

    if filtered_contacts.empty:
        st.warning("No contacts match your search.")
        return

    contact_options = {
        f"{get_contact_name(row, name_column)} (row {index + 1})": index
        for index, row in filtered_contacts.iterrows()
    }
    selected_label = st.selectbox("Select one contact", list(contact_options.keys()))
    selected_index = contact_options[selected_label]

    if st.session_state.selected_contact_index != selected_index:
        st.session_state.selected_contact_index = selected_index
        reset_generated_outputs()

    selected_contact = contacts.loc[selected_index]
    selected_name = get_contact_name(selected_contact, name_column)
    selected_first_name = get_contact_first_name(selected_contact, name_column)
    sender: SenderFields = {
        "name": sender_name.strip() or DEFAULT_SENDER_NAME,
        "title": sender_title.strip() or DEFAULT_SENDER_TITLE,
        "company": sender_company.strip() or DEFAULT_SENDER_COMPANY,
        "account": supported_account.strip() or DEFAULT_SUPPORTED_ACCOUNT,
    }

    st.subheader("Selected Contact Details")
    details = pd.DataFrame(
        {
            "Field": selected_contact.index,
            "Value": ["" if pd.isna(value) else str(value) for value in selected_contact.values],
        }
    )
    st.table(details)

    linkedin_profile_texts: dict[int, str] = st.session_state.linkedin_profile_text_by_contact
    stored_linkedin_profile_text = linkedin_profile_texts.get(selected_index, "")
    linkedin_profile_text = st.text_area(
        "LinkedIn Profile Text",
        value=stored_linkedin_profile_text,
        key=f"linkedin_profile_text_{st.session_state.uploaded_filename}_{selected_index}",
        height=240,
        placeholder="Paste copied LinkedIn profile content here...",
        help="Optional. This text is used only as added context for the selected contact during this session.",
    )
    if linkedin_profile_text != stored_linkedin_profile_text:
        linkedin_profile_texts[selected_index] = linkedin_profile_text
        reset_generated_outputs()

    if st.button("Generate reusable intelligence report", type="primary"):
        if not model.strip():
            st.error("Enter an OpenAI model before generating a report.")
            return

        with st.spinner("Generating report with OpenAI..."):
            try:
                scientific_relevance = assess_scientific_relevance(selected_contact, linkedin_profile_text)
                generated_report = generate_text(
                    build_report_prompt(selected_contact, linkedin_profile_text), model.strip()
                )
                report = attach_scientific_relevance(generated_report, scientific_relevance)
                report_path = save_markdown(selected_name, report, "report")
            except OpenAIError as exc:
                st.error(f"OpenAI request failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Could not generate or save the report: {exc}")
                return

        st.session_state.report = report
        st.session_state.report_path = str(report_path)
        st.session_state.email = ""
        st.session_state.email_path = ""
        st.success(f"Report saved to {report_path}")

    if st.session_state.report:
        st.subheader("Saved Intelligence Report")
        st.markdown(st.session_state.report)
        st.download_button(
            "Download markdown report",
            data=st.session_state.report,
            file_name=Path(st.session_state.report_path).name,
            mime="text/markdown",
        )

        if st.button("Generate outreach email from saved report"):
            if not model.strip():
                st.error("Enter an OpenAI model before generating an email.")
                return

            with st.spinner("Generating outreach email with OpenAI..."):
                try:
                    scientific_relevance = assess_scientific_relevance(selected_contact, linkedin_profile_text)
                    fit, fit_reason = classify_contact_fit(
                        selected_contact, st.session_state.report, linkedin_profile_text
                    )
                    email = generate_text(
                        build_email_prompt(
                            st.session_state.report,
                            selected_contact,
                            selected_first_name,
                            fit,
                            fit_reason,
                            scientific_relevance,
                            sender,
                            linkedin_profile_text,
                        ),
                        model.strip(),
                    )
                    email_path = save_markdown(selected_name, email, "outreach_email")
                except OpenAIError as exc:
                    st.error(f"OpenAI request failed: {exc}")
                    return
                except Exception as exc:
                    st.error(f"Could not generate or save the outreach email: {exc}")
                    return

            st.session_state.email = email
            st.session_state.email_path = str(email_path)
            st.success(f"Outreach email saved to {email_path}")

    if st.session_state.email:
        st.subheader("Generated Outreach Email")
        st.markdown(st.session_state.email)
        st.download_button(
            "Download outreach email",
            data=st.session_state.email,
            file_name=Path(st.session_state.email_path).name,
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()

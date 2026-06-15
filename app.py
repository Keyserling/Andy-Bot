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
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL = "gpt-4.1-mini"


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


def build_report_prompt(contact: pd.Series) -> str:
    """Create a compact prompt for a basic contact intelligence report."""
    return f"""
Create a concise, reusable contact intelligence report from the structured contact details below.

Do not claim facts that are not present in the data. Do not use LinkedIn, Outlook, Gmail,
web browsing, or any external enrichment. If information is missing, say so.

Return markdown with these sections:
1. Executive Summary
2. Known Contact Details
3. Relationship / Outreach Signals
4. Suggested Talking Points
5. Follow-up Questions
6. Data Gaps

Contact details:
{row_to_markdown(contact)}
""".strip()


def build_email_prompt(report: str) -> str:
    """Create a prompt for an outreach email based on a saved report."""
    return f"""
Write a concise, professional outreach email using only the saved contact intelligence report below.

Do not mention unavailable facts, LinkedIn research, Outlook, Gmail, or external enrichment.
Keep the email warm, specific to the available data, and easy to customize.

Return markdown with these sections:
1. Subject
2. Email Body
3. Personalization Notes

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

    st.subheader("Selected Contact Details")
    details = pd.DataFrame(
        {
            "Field": selected_contact.index,
            "Value": ["" if pd.isna(value) else str(value) for value in selected_contact.values],
        }
    )
    st.table(details)

    if st.button("Generate reusable intelligence report", type="primary"):
        if not model.strip():
            st.error("Enter an OpenAI model before generating a report.")
            return

        with st.spinner("Generating report with OpenAI..."):
            try:
                report = generate_text(build_report_prompt(selected_contact), model.strip())
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
                    email = generate_text(build_email_prompt(st.session_state.report), model.strip())
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

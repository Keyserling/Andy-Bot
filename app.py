"""Andy Bot V1 Streamlit application.

A simple contact intelligence workflow:
1. Upload a CSV of contacts.
2. Select one contact by name.
3. Generate and save a markdown intelligence report with OpenAI.
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
    """Find the most likely contact-name column in a CSV."""
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


def build_prompt(contact: pd.Series) -> str:
    """Create a compact prompt for a basic contact intelligence report."""
    return f"""
Create a concise contact intelligence report from the structured contact details below.

Do not claim facts that are not present in the data. Do not use LinkedIn, Gmail, web browsing,
or any external enrichment. If information is missing, say so.

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


def generate_report(contact: pd.Series, model: str) -> str:
    """Generate a contact intelligence report using the OpenAI Responses API."""
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=build_prompt(contact),
    )
    report = getattr(response, "output_text", None)
    if report:
        return str(report).strip()
    return str(response).strip()


def save_report(contact_name: str, report: str) -> Path:
    """Save a markdown report into outputs/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{timestamp}_{clean_filename(contact_name)}.md"
    path.write_text(report + "\n", encoding="utf-8")
    return path


def get_contact_name(contact: pd.Series, name_column: str) -> str:
    """Return the selected contact's display name."""
    value: Any = contact.get(name_column, "")
    if pd.isna(value) or str(value).strip() == "":
        return "Unnamed Contact"
    return str(value).strip()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Andy Bot V1", page_icon="🤖", layout="wide")

    st.title("🤖 Andy Bot V1")
    st.caption("Upload contacts, review one contact, and generate a basic OpenAI-powered intelligence report.")

    with st.sidebar:
        st.header("Settings")
        model = st.text_input(
            "OpenAI model",
            value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            help="Override with any model available to your API key.",
        )
        st.info("LinkedIn scraping and Gmail integrations are intentionally not included in V1.")

    uploaded_file = st.file_uploader("Upload a contacts CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("Upload a CSV file to get started.")
        return

    try:
        contacts = pd.read_csv(uploaded_file)
    except Exception as exc:  # pandas CSV parsing errors vary by version
        st.error(f"Could not read CSV file: {exc}")
        return

    if contacts.empty:
        st.error("The uploaded CSV has no rows.")
        return

    st.subheader("Contacts Table")
    st.dataframe(contacts, use_container_width=True)

    name_column = find_name_column(contacts)
    if not name_column:
        st.error("Could not find any columns in the uploaded CSV.")
        return

    contact_labels = []
    for index, row in contacts.iterrows():
        name = get_contact_name(row, name_column)
        contact_labels.append(f"{name} (row {index + 1})")

    selected_label = st.selectbox("Select one contact by name", contact_labels)
    selected_index = contact_labels.index(selected_label)
    selected_contact = contacts.iloc[selected_index]
    selected_name = get_contact_name(selected_contact, name_column)

    st.subheader("Contact Details")
    details = pd.DataFrame(
        {
            "Field": selected_contact.index,
            "Value": ["" if pd.isna(value) else str(value) for value in selected_contact.values],
        }
    )
    st.table(details)

    if st.button("Generate contact intelligence report", type="primary"):
        if not model.strip():
            st.error("Enter an OpenAI model before generating a report.")
            return

        with st.spinner("Generating report with OpenAI..."):
            try:
                report = generate_report(selected_contact, model.strip())
                report_path = save_report(selected_name, report)
            except OpenAIError as exc:
                st.error(f"OpenAI request failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Could not generate or save the report: {exc}")
                return

        st.success(f"Report saved to {report_path}")
        st.subheader("Generated Report")
        st.markdown(report)
        st.download_button(
            "Download markdown report",
            data=report,
            file_name=report_path.name,
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()

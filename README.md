# Andy Bot

Andy Bot is a Streamlit app for generating varied persona-based Metabolon outreach from an uploaded CSV contact list.

## Workflow

1. Upload a CSV contact list.
2. Andy Bot classifies every contact into exactly one persona:
   - Discovery
   - Translational Research
   - Clinical Biomarkers
   - Clinical Development
   - Medical Affairs
   - Oncology
   - Immunology
   - Safety/Risk
   - Bioanalysis
   - Computational Biology
3. It randomly selects one of five narrative variants and one of five subject-line variants for that persona.
4. It generates and exports a table with one row per contact:
   - Name
   - Company
   - Persona
   - Subject
   - Email

Generated emails keep persona classification, persona assignment, and the CSV workflow, while varying the outreach copy within each persona. Each persona has five narrative variants, five subject-line variants, and several concise call-to-action variants. Emails start with `Dear FirstName,`, introduce Helmut von Keyserling as Strategic Account Manager at Metabolon, use simple scientific language, avoid marketing-heavy personalization, and close with Helmut von Keyserling's signature. The app avoids starting the narrative paragraph with `For X teams...` and tracks generated email bodies so identical emails are not repeated within the same batch. The app does not use fake flattery, over-personalization, LinkedIn profile analysis, long reports, contact intelligence reports, why-this-person outputs, or confidence scores.

## Setup

Install the runtime dependencies before running the app:

```bash
pip install -r requirements.txt
```

No OpenAI API key is required because generated emails come from local templates.

## Run the app

```bash
streamlit run app.py
```

## Local testing

A lightweight syntax check can be run with:

```bash
python -m py_compile app.py
```

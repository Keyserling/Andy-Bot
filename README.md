# Andy Bot

Andy Bot is a Streamlit app for generating persona-template Metabolon outreach from an uploaded CSV contact list.

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
3. It selects the matching Metabolon email template for that persona.
4. It generates and exports a table with one row per contact:
   - Name
   - Company
   - Persona
   - Subject
   - Email

Generated emails are deterministic template output. Each email starts with `Dear FirstName,`, introduces Helmut von Keyserling as Strategic Account Manager at Metabolon, includes a persona-specific Metabolon use case paragraph, asks for a brief 20-minute conversation, and closes with Helmut von Keyserling's signature. The app does not use creative free-writing, fake flattery, over-personalization, LinkedIn profile analysis, long reports, contact intelligence reports, why-this-person outputs, or confidence scores.

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

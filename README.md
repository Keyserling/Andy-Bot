# Andy Bot

Andy Bot is a Streamlit app for generating persona-template Metabolon outreach from an uploaded CSV contact list.

## Workflow

1. Upload a CSV contact list.
2. Andy Bot classifies every contact into exactly one persona:
   - Discovery
   - Translational Research
   - Clinical Development
   - Clinical Biomarkers
   - Bioanalysis
   - Medical Affairs
   - Epidemiology
   - Safety / Risk
   - Computational Biology
   - Oncology Research
   - Immunology Research
3. It selects the matching Metabolon narrative for that persona.
4. It generates and exports a table with one row per contact:
   - Name
   - Company
   - Persona
   - Subject
   - Email

Generated emails are constrained to 80-140 words, start with `Dear FirstName,`, use simple scientifically relevant language, and include a brief meeting request. The app does not generate LinkedIn profile analysis, long reports, contact intelligence reports, why-this-person outputs, or confidence scores.

## Setup

Install the runtime dependencies before running the app:

```bash
pip install -r requirements.txt
```

Create a local environment file from the example and add your OpenAI API key:

```bash
cp .env.example .env
```

## Run the app

```bash
streamlit run app.py
```

## Local testing

A lightweight syntax check can be run with:

```bash
python -m py_compile app.py
```

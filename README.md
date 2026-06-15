# Andy Bot

Andy Bot is a Streamlit app for generating varied persona-based Metabolon outreach from an uploaded CSV or Excel contact list.

## Workflow

1. Upload a CSV or Excel contact list.
2. Andy Bot classifies every contact into exactly one of six active personas:
   - Discovery
   - Translational / Clinical Development
   - Biomarkers / Bioanalysis
   - Medical Affairs
   - Safety / Quality
   - Oncology
3. It randomly selects one of ten subject lines and one of ten complete narrative paragraphs for that persona.
4. It generates and exports a table with one row per contact:
   - Name
   - Company
   - Persona
   - Subject
   - Email
   - narrative_variant_id

Generated emails keep the existing upload/import workflow and persona classification logic, with legacy persona matches mapped into the six active personas. The email body uses a fixed greeting, Helmut von Keyserling introduction, one selected narrative paragraph, the requested relevance sentence, and Helmut's signature.

## Setup

Install the runtime dependencies before running the app:

```bash
pip install -r requirements.txt
```

No OpenAI API key is required because generated emails come from the local narrative library.

## Run the app

```bash
streamlit run app.py
```

## Local checks

Run syntax checks:

```bash
python -m py_compile app.py narratives.py check_narrative_variants.py
```

Run the narrative variety check:

```bash
python check_narrative_variants.py
```

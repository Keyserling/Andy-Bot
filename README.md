# Andy Bot

Andy Bot is a Streamlit app for generating varied persona-based Metabolon outreach from an uploaded CSV or Excel contact list.

## Workflow

1. Upload a CSV or Excel contact list.
2. Andy Bot classifies every contact into exactly one explicit persona using ordered routing rules and returns a confidence score:
   - Biomarkers / Bioanalysis
   - Clinical Pharmacology
   - Operations / Low Priority
   - Computational Biology
   - Discovery
   - Medical Affairs
   - Immunology
   - Oncology
   - Safety / Quality
   - Translational / Clinical Development
3. It randomly selects one of ten subject lines and one of ten persona-specific use cases.
4. It generates and exports a table with one row per contact:
   - Name
   - Company
   - Persona
   - Persona Confidence Score
   - Subject
   - Email
   - Matched Keyword
   - Narrative Variant ID

Operations / Low Priority contacts are flagged for manual review instead of automatic outreach. Generated emails keep the existing upload/import workflow. Explicit persona labels such as Biomarkers / Bioanalysis, Clinical Pharmacology, Computational Biology, Operations / Low Priority, and Immunology are preserved in the export, while related legacy or specialist labels are mapped to the closest available content family for email copy. Medical Affairs is routed only by an explicit `medical affairs` match. The email body uses a fixed greeting, Helmut von Keyserling introduction, persona-specific use-case sentence, three persona-specific bullets, Metabolon dataset sentence, requested interest sentence, and Helmut's signature. EML export sets Helmut as the From header when `METABOLON_SENDER_EMAIL` is configured; otherwise each EML begins with `Open as draft and choose sender in Outlook.`

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
python -m py_compile app.py narratives.py draft_exports.py check_narrative_variants.py
```

Run the narrative variety check:

```bash
python check_narrative_variants.py
```

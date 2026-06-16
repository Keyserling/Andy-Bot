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
3. Before email generation, Andy Bot validates contact integrity using Company, Email Address, LinkedIn Current Company, and LinkedIn Current Title when those fields are available.
4. It generates a Contact Narrative of 20 words or fewer that summarizes what the person most likely cares about in their current role, plus a confidence score.
5. It randomly selects one of ten subject lines and one of ten persona-specific use cases for contacts that pass integrity checks.
6. It generates and exports a table with one row per contact:
   - Name
   - Company
   - Persona
   - Persona Confidence Score
   - Integrity Status
   - Integrity Reason
   - Suggested Company
   - Suggested Title
   - Subject
   - Email
   - Contact Narrative
   - Contact Narrative Confidence
   - Matched Keyword
   - Narrative Variant ID

Contact integrity statuses are GREEN, YELLOW, and RED. GREEN contacts generate normally, YELLOW contacts generate with a warning in the processing summary, and RED contacts are marked `Review Required` instead of receiving generated outreach copy. Operations / Low Priority contacts are flagged for manual review instead of automatic outreach. Generated emails keep the existing upload/import workflow. Explicit persona labels such as Biomarkers / Bioanalysis, Clinical Pharmacology, Computational Biology, Operations / Low Priority, and Immunology are preserved in the export, while related legacy or specialist labels are mapped to the closest available content family for email copy. Medical Affairs is routed only by an explicit `medical affairs` match. The email body uses a fixed greeting, the Contact Narrative opening, Helmut von Keyserling introduction, a short role-relevant Metabolon story based on the Contact Narrative plus internal persona/offering selection, a meeting-oriented CTA, and Helmut's signature. Internal persona labels, recommended offering names, and routing logic are not written into the email body. EML export sets Helmut as the From header when `METABOLON_SENDER_EMAIL` is configured; otherwise each EML begins with `Open as draft and choose sender in Outlook.`

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

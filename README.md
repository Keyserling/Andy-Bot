# Andy Bot

Andy Bot is a Streamlit app for generating deterministic Metabolon outreach from an uploaded CSV or Excel contact list.

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
4. Outreach Engine V5 generates one sentence for what the person likely cares about.
5. It generates exactly one LinkedIn observation. If no LinkedIn-derived observation exists, it uses: `Given your role at [Company], I thought this might be relevant.`
6. It selects one Metabolon story from the Metabolon Knowledge Database and creates a deterministic industry reality → Metabolon perspective → discussion story.
7. It assembles the email as LinkedIn observation, Helmut's introduction, industry reality, Metabolon perspective, CTA, and Helmut's full signature with role, company, email, and `+49 176 61356899`.
8. It generates and exports a table with one row per contact, including:
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
   - Metabolon Capability
   - Recommended Offering
   - Scientific Problem
   - Email Story
   - LinkedIn debug fields

Contact integrity statuses are GREEN, YELLOW, and RED. GREEN contacts generate normally, YELLOW contacts generate with a warning in the processing summary, and RED contacts are marked `Review Required` instead of receiving generated outreach copy. Operations / Low Priority contacts are flagged for manual review instead of automatic outreach. Generated emails keep the existing upload/import workflow. Explicit persona labels are preserved in the export, while related legacy labels are mapped to the closest active persona for classification. Medical Affairs is routed only by an explicit `medical affairs` match. The **Create Outlook Drafts** button creates unsent Outlook drafts directly through Microsoft Graph using the generated To, Subject, and Body fields. CSV and EML ZIP downloads remain available as a fallback. EML export sets Helmut as the From header when `METABOLON_SENDER_EMAIL` is configured.

## Setup

Install the runtime dependencies before running the app:

```bash
pip install -r requirements.txt
```

No OpenAI API key is required because generated emails come from deterministic local logic and the Metabolon Knowledge Database.

To create drafts directly in Outlook, register or use a Microsoft Entra public-client application with delegated Microsoft Graph `Mail.ReadWrite` permission and set its client ID before starting Streamlit:

```bash
export MS_GRAPH_CLIENT_ID=your-public-client-application-id
```

The first Outlook draft creation starts Microsoft device-code authentication. Andy Bot stores the Microsoft Graph token cache in the operating-system keyring when available, with a user-only `~/.andy_bot/ms_graph_token_cache.json` fallback for environments without a keyring backend. Draft creation calls Microsoft Graph `/me/messages`, which saves messages to Drafts and does not send them.

## Run the app

```bash
streamlit run app.py
```

## Local checks

Run syntax checks:

```bash
python -m py_compile app.py metabolon_knowledge.py draft_exports.py check_narrative_variants.py
```

Run the Outreach Engine V5 check:

```bash
python check_narrative_variants.py
```

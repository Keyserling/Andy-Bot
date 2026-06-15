# Andy Bot

Andy Bot is a Streamlit app for rapid Metabolon outreach generation from an uploaded contact list.

## Workflow

1. Upload a CSV or XLSX contact list.
2. Andy Bot processes up to 20 contacts.
3. It identifies each contact persona from title and role fields.
4. It selects a persona-specific Metabolon outreach narrative.
5. It generates a copy/paste table with only:
   - Name
   - Subject
   - Email

Generated emails are constrained to 80-140 words, start with `Dear FirstName,`, use persona-specific scientific context, and include a simple meeting request.

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

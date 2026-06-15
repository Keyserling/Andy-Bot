# Andy Bot V1

Andy Bot V1 is a Streamlit app for generating a basic markdown contact intelligence report from an uploaded contacts CSV.

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

Local tests and checks are optional for this small app, but they require the same dependencies used by the application. Before running local testing commands, install the requirements:

```bash
pip install -r requirements.txt
```

A lightweight syntax check can be run with:

```bash
python -m py_compile app.py
```

`pytest` is not required at runtime and is intentionally not included in `requirements.txt`. If you add pytest-based tests later, install pytest separately in your development environment.

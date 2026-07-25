# Policy & Claims Check

Policy & Claims Check is an AI-powered insurance assistant that uses Retrieval-Augmented Generation (RAG) to answer questions strictly from uploaded policy PDF documents. It supports:

- Policy coverage and exclusions
- Waiting periods and limits
- Claim submission steps and required documents
- Preliminary claim pre-check based on uploaded policy text

## Architecture

```text
Uploaded PDF
  -> PDF Loader
  -> Text Chunking
  -> Embeddings (Sentence Transformers)
  -> FAISS Vector Store
  -> Gemini LLM
  -> Streamlit UI
```

## Installation

1. Create and activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file using `.env.example` and add your Gemini API key.

## How to Get a Gemini API Key

1. Visit Google AI Studio.
2. Create an API key.
3. Paste it into the `.env` file.

## Run Locally

```bash
streamlit run app.py
```

## Folder Structure

```text
Policy_Claims_Check/
├── app.py
├── rag_pipeline.py
├── utils.py
├── prompt_template.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── assets/
├── uploaded_policies/
├── vector_store/
└── css/
    style.css
```

## Screenshots

Placeholder for screenshots.

## Future Improvements

- Multi-policy support
- Document citation highlighting
- Admin dashboard for policy versions
- Better claim scoring logic

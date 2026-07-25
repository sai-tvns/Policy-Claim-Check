# Policy & Claims Check

Policy & Claims Check is an AI-powered assistant for insurance policy support and claims pre-check. It uses Retrieval-Augmented Generation (RAG) to answer questions only from uploaded policy PDF documents.

The system helps users and agents quickly find:
- policy coverage details
- exclusions
- waiting periods
- limits and sub-limits
- claim submission steps
- required documents
- a preliminary claim eligibility assessment

## Project Overview

This project is built as a local web application using Streamlit. It processes uploaded policy PDFs, converts them into searchable chunks, stores them in a vector database, and uses Gemini AI to generate grounded answers based only on the uploaded policy content.

The app is designed to:
- answer questions from policy documents
- avoid hallucinations
- provide source page numbers
- support claims pre-check for preliminary assessment only

## Features

- Upload insurance policy PDF
- Build vector database from PDF
- Ask policy-related questions
- Get grounded answers with source page references
- Check claim eligibility preliminarily
- Maintain chat history in the app
- Download chat conversation

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- python-dotenv

## Project Structure

```text
Policy_Claims_Copilot/
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

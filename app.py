import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from rag_pipeline import (
    build_vector_store_from_pdf,
    get_vector_store_status,
    get_pdf_page_count,
    load_existing_vector_store,
    query_policy,
    run_claim_precheck,
)
from utils import (
    create_download_chat,
    format_sources,
    load_css,
    validate_claim_form,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STYLESHEET_PATH = PROJECT_ROOT / "css" / "style.css"
UPLOAD_DIR = PROJECT_ROOT / "uploaded_policies"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "page_count" not in st.session_state:
        st.session_state.page_count = 0
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def display_chat_history() -> None:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(message["content"])


def handle_chat_query(question: str) -> None:
    if not question.strip():
        st.warning("Please enter a question before sending it.")
        return

    if not st.session_state.vector_store_ready:
        st.warning("Please upload a policy PDF and build the vector database first.")
        return

    with st.spinner("Searching the policy and preparing an answer..."):
        result = query_policy(question)

    if result.get("error"):
        st.error(result["error"])
        return

    answer_text = result["answer"]
    sources = format_sources(result.get("sources", []))
    if sources:
        answer_text = f"{answer_text}\n\n**Source**\n{sources}"

    add_message("user", question)
    add_message("assistant", answer_text)
    st.session_state.last_query = question


def handle_claim_precheck(form_data: Dict[str, Any]) -> None:
    error_message = validate_claim_form(form_data)
    if error_message:
        st.error(error_message)
        return

    if not st.session_state.vector_store_ready:
        st.error("Please upload a policy PDF and build the vector database first.")
        return

    with st.spinner("Running the preliminary claim assessment..."):
        result = run_claim_precheck(form_data)

    if result.get("error"):
        st.error(result["error"])
        return

    assessment = result["answer"]
    add_message("user", f"Claim pre-check for {form_data.get('disease', 'the claim')}")
    add_message("assistant", assessment)


def main() -> None:
    st.set_page_config(page_title="Policy & Claims Check", page_icon="🛡️", layout="wide")
    init_session_state()
    load_css(STYLESHEET_PATH)

    st.markdown("<h1 class='main-title'>Policy & Claims Check</h1>", unsafe_allow_html=True)
    st.caption("AI-powered insurance policy support and claims pre-check using RAG")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload a policy PDF", type=["pdf"])

        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            st.session_state.pdf_name = uploaded_file.name

            save_path = UPLOAD_DIR / uploaded_file.name
            with save_path.open("wb") as handle:
                handle.write(uploaded_file.getvalue())

            page_count = get_pdf_page_count(save_path)
            st.session_state.page_count = page_count
            st.metric("PDF Pages", page_count)

        if st.button("Build Vector Database", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please upload a PDF before building the vector database.")
            else:
                save_path = UPLOAD_DIR / uploaded_file.name
                with st.spinner("Processing the PDF and creating embeddings..."):
                    status = build_vector_store_from_pdf(save_path)
                if status.get("error"):
                    st.error(status["error"])
                else:
                    st.session_state.vector_store_ready = True
                    st.session_state.chunk_count = status.get("chunks_created", 0)
                    st.success("Vector database built successfully.")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_query = ""
            st.success("Chat history cleared.")

        if st.button("Download Chat", use_container_width=True):
            chat_text = create_download_chat(st.session_state.messages)
            st.download_button(
                label="Download chat",
                data=chat_text,
                file_name="policy_claims_chat.txt",
                mime="text/plain",
            )

        st.divider()
        st.subheader("Policy Statistics")
        vector_status = get_vector_store_status()
        st.metric("Embedding Status", vector_status.get("embedding_status", "Not Ready"))
        st.metric("Vector DB Status", vector_status.get("vector_db_status", "Not Ready"))
        st.metric("Chunks Created", st.session_state.chunk_count)
        st.metric("PDF Pages", st.session_state.page_count)

    example_questions = [
        "Is cataract surgery covered?",
        "What is the waiting period for cataract surgery?",
        "How do I submit a claim?",
        "What documents are needed for a hospitalization claim?",
    ]

    col1, col2, col3, col4 = st.columns(4)
    for column, question in zip([col1, col2, col3, col4], example_questions):
        with column:
            if st.button(question, key=f"q_{question}", use_container_width=True):
                handle_chat_query(question)

    st.divider()

    display_chat_history()

    st.divider()
    st.subheader("Claim Pre-check")
    with st.form("claim_precheck_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            policy_age_years = st.number_input("Policy age (years)", min_value=0, max_value=50, value=0)
            policy_age_months = st.number_input("Policy age (months)", min_value=0, max_value=11, value=0)
            disease = st.text_input("Disease", placeholder="e.g. Cataract")
            hospitalized = st.selectbox("Hospitalized?", ["Yes", "No"])
            surgery_name = st.text_input("Surgery name", placeholder="e.g. Knee replacement")
        with col_b:
            hospital_bill = st.selectbox("Hospital bill available?", ["Yes", "No"])
            doctor_certificate = st.selectbox("Doctor certificate available?", ["Yes", "No"])
            discharge_summary = st.selectbox("Discharge summary available?", ["Yes", "No"])
            admission_date = st.date_input("Admission date")
            discharge_date = st.date_input("Discharge date")
            policy_start_date = st.date_input("Policy start date")

        submitted = st.form_submit_button("Analyze Claim")
        if submitted:
            form_data = {
                "policy_age_years": int(policy_age_years),
                "policy_age_months": int(policy_age_months),
                "disease": disease,
                "hospitalized": hospitalized,
                "surgery_name": surgery_name,
                "hospital_bill": hospital_bill,
                "doctor_certificate": doctor_certificate,
                "discharge_summary": discharge_summary,
                "admission_date": admission_date,
                "discharge_date": discharge_date,
                "policy_start_date": policy_start_date,
            }
            handle_claim_precheck(form_data)

    st.divider()
    user_input = st.chat_input("Ask about coverage, exclusions, waiting periods, limits, or claim steps")
    if user_input:
        handle_chat_query(user_input)


if __name__ == "__main__":
    main()

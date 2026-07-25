from pathlib import Path
from typing import Any, Dict, List


def load_css(stylesheet_path: Path) -> None:
    import streamlit as st

    if stylesheet_path.exists():
        with stylesheet_path.open("r", encoding="utf-8") as handle:
            st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)


def format_sources(sources: List[Any]) -> str:
    if not sources:
        return "No source page available."
    unique_pages = sorted({str(source) for source in sources})
    return ", ".join(f"Page {page}" for page in unique_pages)


def create_download_chat(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}: {message['content']}")
        lines.append("")
    return "\n".join(lines)


def validate_claim_form(form_data: Dict[str, Any]) -> str:
    disease = (form_data.get("disease") or "").strip()
    if not disease:
        return "Please enter the disease name before analyzing the claim."

    if form_data.get("admission_date") and form_data.get("discharge_date"):
        if form_data["discharge_date"] < form_data["admission_date"]:
            return "Discharge date cannot be earlier than the admission date."

    if form_data.get("policy_start_date") and form_data.get("admission_date"):
        if form_data["policy_start_date"] > form_data["admission_date"]:
            return "Policy start date cannot be later than the admission date."

    return ""

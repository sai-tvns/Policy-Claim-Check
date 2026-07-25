import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader


def load_environment_file() -> None:
    env_candidates = [PROJECT_ROOT / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


from prompt_template import build_chat_prompt, build_precheck_prompt

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploaded_policies"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ensure_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    if env_path.exists():
        return
    if example_path.exists():
        shutil.copyfile(example_path, env_path)
    else:
        env_path.write_text("GEMINI_API_KEY=your_api_key_here\n", encoding="utf-8")


ensure_env_file()
load_environment_file()
load_dotenv(PROJECT_ROOT / ".env", override=False)


class SimpleDocument:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_gemini_model() -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or still set to the placeholder value. "
            "Create or edit the .env file in the project root and replace it with your real Gemini API key from Google AI Studio."
        )

    genai.configure(api_key=api_key)
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            return genai.GenerativeModel(model_name)
        except Exception:
            continue

    return genai.GenerativeModel("gemini-2.0-flash")


def load_policy_documents(pdf_path: Path) -> List[SimpleDocument]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        raise ValueError("The uploaded PDF does not contain any readable text.")

    documents: List[SimpleDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                SimpleDocument(
                    page_content=text,
                    metadata={"page": page_number, "source": str(pdf_path.name)},
                )
            )

    if not documents:
        raise ValueError("The uploaded PDF does not contain any readable text.")
    return documents


def split_documents(documents: List[SimpleDocument]) -> List[SimpleDocument]:
    chunks: List[SimpleDocument] = []
    for document in documents:
        text = document.page_content
        words = text.split()
        chunk_size = 250
        chunk_overlap = 50

        for index in range(0, len(words), chunk_size - chunk_overlap):
            segment = words[index:index + chunk_size]
            if not segment:
                continue
            chunk_text = " ".join(segment)
            metadata = dict(document.metadata)
            metadata["chunk_index"] = len(chunks)
            chunks.append(SimpleDocument(page_content=chunk_text, metadata=metadata))
    return chunks


def get_embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDINGS_MODEL)


def build_vector_store_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    ensure_directories()
    try:
        documents = load_policy_documents(pdf_path)
        chunks = split_documents(documents)
        embedder = get_embedder()
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, str(VECTOR_STORE_DIR / "faiss_index.faiss"))
        with (VECTOR_STORE_DIR / "chunks.json").open("w", encoding="utf-8") as handle:
            json.dump(
                [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks],
                handle,
                indent=2,
            )

        return {
            "status": "success",
            "chunks_created": len(chunks),
            "pdf_path": str(pdf_path),
        }
    except Exception as exc:  # pragma: no cover - exercised during runtime
        return {"status": "error", "error": f"Failed to build the vector store: {exc}"}


def load_existing_vector_store() -> Optional[Dict[str, Any]]:
    ensure_directories()
    index_path = VECTOR_STORE_DIR / "faiss_index.faiss"
    chunks_path = VECTOR_STORE_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        return None

    try:
        index = faiss.read_index(str(index_path))
        with chunks_path.open("r", encoding="utf-8") as handle:
            chunks = json.load(handle)
        embedder = get_embedder()
        return {"index": index, "chunks": chunks, "embedder": embedder}
    except Exception:
        return None


def get_vector_store_status() -> Dict[str, str]:
    ensure_directories()
    if (VECTOR_STORE_DIR / "faiss_index.faiss").exists() and (VECTOR_STORE_DIR / "chunks.json").exists():
        return {
            "embedding_status": "Ready",
            "vector_db_status": "Ready",
        }
    return {
        "embedding_status": "Not Ready",
        "vector_db_status": "Not Ready",
    }


def query_policy(question: str) -> Dict[str, Any]:
    try:
        vector_store = load_existing_vector_store()
        if vector_store is None:
            return {"answer": "Please build the vector database first.", "sources": [], "error": None}

        embedder = vector_store["embedder"]
        index = vector_store["index"]
        chunks = vector_store["chunks"]
        query_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_embedding, k=4)

        relevant_chunks = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(chunks):
                continue
            relevant_chunks.append(chunks[int(idx)])

        context_text = "\n\n".join(
            f"Page {chunk['metadata'].get('page', 'unknown')}: {chunk['page_content']}" for chunk in relevant_chunks
        )

        model = get_gemini_model()
        prompt = build_chat_prompt().format(question=question, context=context_text)
        response = model.generate_content(prompt)
        answer = response.text if hasattr(response, "text") else str(response)
        sources = sorted({chunk["metadata"].get("page", "unknown") for chunk in relevant_chunks})
        return {"answer": answer, "sources": sources, "error": None}
    except Exception as exc:  # pragma: no cover - exercised during runtime
        message = str(exc)
        if "GEMINI_API_KEY" in message or "Google AI Studio" in message:
            return {
                "answer": (
                    "The Gemini API key is not configured yet. Please open the .env file in the project folder, "
                    "replace the placeholder with your real Gemini API key, and try again."
                ),
                "sources": [],
                "error": None,
            }
        return {"answer": "", "sources": [], "error": f"Unable to answer the question right now: {exc}"}


def run_claim_precheck(form_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        vector_store = load_existing_vector_store()
        if vector_store is None:
            return {"answer": "Please build the vector database first.", "sources": [], "error": None}

        embedder = vector_store["embedder"]
        index = vector_store["index"]
        chunks = vector_store["chunks"]
        query = f"Claim pre-check for {form_data.get('disease', '')} and {form_data.get('surgery_name', '')}"
        query_embedding = embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_embedding, k=4)

        relevant_chunks = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(chunks):
                continue
            relevant_chunks.append(chunks[int(idx)])

        context_text = "\n\n".join(
            f"Page {chunk['metadata'].get('page', 'unknown')}: {chunk['page_content']}" for chunk in relevant_chunks
        )

        model = get_gemini_model()
        prompt = build_precheck_prompt().format(form_data=form_data, context=context_text)
        response = model.generate_content(prompt)
        answer = response.text if hasattr(response, "text") else str(response)
        sources = sorted({chunk["metadata"].get("page", "unknown") for chunk in relevant_chunks})
        return {"answer": answer, "sources": sources, "error": None}
    except Exception as exc:  # pragma: no cover - exercised during runtime
        message = str(exc)
        if "GEMINI_API_KEY" in message or "Google AI Studio" in message:
            return {
                "answer": (
                    "The Gemini API key is not configured yet. Please open the .env file in the project folder, "
                    "replace the placeholder with your real Gemini API key, and try again."
                ),
                "sources": [],
                "error": None,
            }
        return {"answer": "", "sources": [], "error": f"Unable to run the pre-check right now: {exc}"}


def get_pdf_page_count(pdf_path: Path) -> int:
    if not pdf_path.exists():
        return 0
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)

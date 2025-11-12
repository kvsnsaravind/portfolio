# backend/main.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
from pathlib import Path
import asyncio
import httpx
import re
from typing import List, Dict, Tuple

# optional deps you already had
import numpy as np  # noqa: F401
from bs4 import BeautifulSoup

import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
import cohere
from pypdf import PdfReader  # <-- NEW: read resume.pdf


# ========= Env & Paths =========

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

COHERE_API_KEY = os.getenv("CO_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("CO_API_KEY not set in environment or .env")

co = cohere.Client(COHERE_API_KEY)

app = FastAPI(title="Portfolio Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= Helpers: clean, chunk, extract =========

def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def chunk_text(label: str, text: str, max_chars: int = 800) -> List[str]:
    """Chunks an arbitrary string and labels each chunk."""
    chunks: List[str] = []
    buf = []
    cur = 0
    for sent in re.split(r"(?<=[\.\?!])\s+", text):
        if not sent:
            continue
        if cur + len(sent) + 1 > max_chars and buf:
            chunks.append(f"[SECTION: {label}] " + " ".join(buf))
            buf = [sent]
            cur = len(sent) + 1
        else:
            buf.append(sent)
            cur += len(sent) + 1
    if buf:
        chunks.append(f"[SECTION: {label}] " + " ".join(buf))
    return chunks

def extract_sections_from_html(html_path: Path) -> Dict[str, List[str]]:
    """Return section_id -> list of textual lines from index.html."""
    if not html_path.exists():
        return {}
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    sections: Dict[str, List[str]] = {}
    for sec in soup.find_all("section"):
        sec_id = sec.get("id", "unknown")
        lines: List[str] = []
        for tag in sec.find_all(["h1", "h2", "h3", "p", "li", "strong", "em", "a", "div", "span"]):
            txt = _clean_text(tag.get_text(" ", strip=True))
            if txt and len(txt) > 2:
                lines.append(txt)
        if lines:
            sections[sec_id] = lines
    return sections

def chunk_lines_with_labels(sections: Dict[str, List[str]], max_chars: int = 800) -> List[str]:
    """Make labeled chunks from section lines."""
    chunks: List[str] = []
    for sec_id, lines in sections.items():
        buf, cur = [], 0
        for ln in lines:
            if cur + len(ln) + 1 > max_chars and buf:
                chunks.append(f"[SECTION: {sec_id}] " + " ".join(buf))
                buf, cur = [ln], len(ln) + 1
            else:
                buf.append(ln)
                cur += len(ln) + 1
        if buf:
            chunks.append(f"[SECTION: {sec_id}] " + " ".join(buf))
    return chunks

def extract_recent_projects(html_path: Path) -> List[Tuple[str, str, str]]:
    """Grab project cards: (title, description, link)."""
    out: List[Tuple[str, str, str]] = []
    if not html_path.exists():
        return out
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    projects_section = soup.find("section", {"id": "projects"})
    if not projects_section:
        return out
    for card in projects_section.select(".project-card"):
        title_tag = card.find(["strong", "h3"])
        desc_tag = card.find("p")
        link_tag = card.find("a", href=True)
        title = _clean_text(title_tag.get_text()) if title_tag else ""
        desc = _clean_text(desc_tag.get_text()) if desc_tag else ""
        href = link_tag["href"] if link_tag else ""
        if title or desc:
            out.append((title, desc, href))
    return out

def extract_pdf_text(pdf_path: Path) -> str:
    """Return full text of resume.pdf."""
    if not pdf_path.exists():
        return ""
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        pages.append(_clean_text(p.extract_text() or ""))
    return "\n".join(pages)


# ========= ChromaDB (RAG) =========

chroma_client = chromadb.Client()
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
)
COLLECTION_NAME = "aravind-knowledge"
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=cohere_ef,
)

# small cached facts
CURRENT_EMPLOYER: str | None = None
CURRENT_ROLE: str | None = None
CURRENT_LOCATION: str | None = None   # <-- NEW
PROJECTS_INDEX: List[Tuple[str, str, str]] = []


@app.on_event("startup")
async def build_collection() -> None:
    global CURRENT_EMPLOYER, CURRENT_ROLE, CURRENT_LOCATION, PROJECTS_INDEX

    try:
        html_path = PROJECT_ROOT / "index.html"
        resume_path = PROJECT_ROOT / "resume.pdf"

        # HTML → labeled chunks
        sections = extract_sections_from_html(html_path)
        html_chunks = chunk_lines_with_labels(sections, max_chars=800)

        # Resume → raw text → chunks
        resume_text = extract_pdf_text(resume_path)
        resume_chunks = chunk_text("resume", resume_text, max_chars=800) if resume_text else []

        # Cache easy facts from either source (HTML lines + resume text)
        lines_for_scan = [ln for lines in sections.values() for ln in lines]
        blob = " ".join(lines_for_scan + [resume_text.lower()])

        # Current employer / role
        if re.search(r"(amazon|aws)", blob, re.I) and re.search(r"present", blob, re.I):
            CURRENT_EMPLOYER = "Amazon Web Services (AWS)"
            if re.search(r"software development engineer", blob, re.I):
                CURRENT_ROLE = "Software Development Engineer"

        # Current location (very simple heuristic; extend as you like)
        # Look for common city/state patterns you use.
        m_loc = re.search(r"(Bellevue|Seattle|Redmond|Kirkland)\s*,?\s*(WA|Washington)", blob, re.I)
        if m_loc:
            city = m_loc.group(1).title()
            state = "WA"
            CURRENT_LOCATION = f"{city}, {state}"

        PROJECTS_INDEX = extract_recent_projects(html_path)

        # Load both sources into Chroma (idempotent on a fresh collection)
        if collection.count() == 0:
            docs = html_chunks + resume_chunks
            if docs:
                collection.add(documents=docs, ids=[str(i) for i in range(len(docs))])
                print(f"Chroma: added {len(docs)} chunks (html+resume)")
        print(f"Chroma ready (count={collection.count()})")
        print(f"Cached employer={CURRENT_EMPLOYER}, role={CURRENT_ROLE}, location={CURRENT_LOCATION}")
        if PROJECTS_INDEX:
            print(f"Found {len(PROJECTS_INDEX)} project cards; latest={PROJECTS_INDEX[0][0]}")
    except Exception as e:
        print("Startup error:", e)


# ========= GitHub (optional) =========

GITHUB_USERNAME = "kvsnsaravind"

async def fetch_github_profile():
    user_url = f"https://api.github.com/users/{GITHUB_USERNAME}"
    repos_url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        user_resp, repos_resp = await asyncio.gather(
            client.get(user_url),
            client.get(repos_url),
        )
        user_resp.raise_for_status()
        repos_resp.raise_for_status()
        return user_resp.json(), repos_resp.json()


# ========= Request model =========

class ChatbotRequest(BaseModel):
    question: str


# ========= LLM helper (strict grounding) =========

def get_llm_response(context: str, question: str) -> str:
    prompt = (
        "You are Aravind's portfolio assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, reply exactly: 'I don't know.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    resp = co.chat(
        model="command-a-03-2025",
        message=prompt,
        documents=[{"text": context}] if context else [],
        temperature=0.2,
        max_tokens=220,
    )
    return (resp.text or "").strip()


# ========= Routes =========

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chatbot")
async def chatbot(req: ChatbotRequest):
    q_raw = (req.question or "").strip()
    q = q_raw.lower()
    print("Q:", q_raw)

    try:
        # ---- Direct intents (from cached facts) ----
        if any(k in q for k in ["where is aravind working", "where do you work",
                                "working now", "current employer", "current company"]):
            if CURRENT_EMPLOYER:
                role = f" as a {CURRENT_ROLE}" if CURRENT_ROLE else ""
                return {"answer": f"Aravind is working at {CURRENT_EMPLOYER}{role}."}

        if any(k in q for k in ["where does aravind live", "current location",
                                "where is he living", "work location", "location"]):
            if CURRENT_LOCATION:
                return {"answer": f"Current location: {CURRENT_LOCATION}."}

        if any(k in q for k in ["latest project", "recent project", "newest project", "recent projects"]):
            if PROJECTS_INDEX:
                title, desc, link = PROJECTS_INDEX[0]
                msg = f"Latest project: {title}. {desc}"
                if link:
                    msg += f" (Link: {link})"
                return {"answer": msg}

        # GitHub (optional)
        if "github" in q:
            user_data, repos_data = await fetch_github_profile()
            if "repo" in q or "project" in q:
                names = [r.get("name", "") for r in repos_data if r.get("name")]
                ans = f"My GitHub repositories include: {', '.join(names[:5])}." if names else "I have no public repositories."
            elif "star" in q:
                stars = sum(int(r.get("stargazers_count", 0) or 0) for r in repos_data)
                ans = f"My repositories have a total of {stars} stars."
            elif "language" in q:
                langs = sorted({r.get("language") for r in repos_data if r.get("language")})
                ans = f"I use these languages in my GitHub repos: {', '.join(langs)}." if langs else "No languages detected."
            else:
                ans = f"My GitHub profile: {user_data.get('html_url', '')} with {user_data.get('public_repos', 0)} public repos."
            return {"answer": ans}

        # ---- RAG with Chroma ----
        results = collection.query(query_texts=[q_raw], n_results=8)
        docs_lists = results.get("documents") or []
        flat_docs: List[str] = []
        for lst in docs_lists:
            if lst:
                flat_docs.extend([d for d in lst if d])
        context = "\n\n".join(flat_docs)

        if not context:
            return {"answer": "I don't know."}

        llm_answer = get_llm_response(context=context, question=q_raw)
        return {"answer": llm_answer or "I don't know."}

    except Exception as e:
        print("Error:", e)
        return {"answer": f"Error: {str(e)}"}
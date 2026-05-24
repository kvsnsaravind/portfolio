from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
from pathlib import Path
import asyncio
import httpx
import re
import time
from typing import List, Dict, Tuple, Optional

import numpy as np  # noqa: F401
from bs4 import BeautifulSoup

import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
import cohere

# safe import for PDF reader (optional)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


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

# serve frontend and static files on a dedicated path
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static")


@app.get("/")
def root():
    return FileResponse(PROJECT_ROOT / "index.html")


# ========= Helpers: clean, chunk, extract =========

def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def chunk_text(label: str, text: str, max_chars: int = 800) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
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
    chunks: List[str] = []
    for sec_id, lines in sections.items():
        buf: List[str] = []
        cur = 0
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
    if PdfReader is None:
        return ""
    if not pdf_path.exists():
        return ""
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        pages.append(_clean_text(p.extract_text() or ""))
    return "\n".join(pages)


def extract_social_links(html_path: Path, resume_text: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    if html_path.exists():
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "github.com" in href and "github" not in out:
                out["github"] = href
            if "linkedin.com" in href and "linkedin" not in out:
                out["linkedin"] = href
    if resume_text:
        for m in re.findall(r"https?://[^\s)]+", resume_text):
            if "github.com" in m and "github" not in out:
                out["github"] = m
            if "linkedin.com" in m and "linkedin" not in out:
                out["linkedin"] = m
    return out


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

CURRENT_EMPLOYER: Optional[str] = None
CURRENT_ROLE: Optional[str] = None
CURRENT_LOCATION: Optional[str] = None
PROJECTS_INDEX: List[Tuple[str, str, str]] = []
GITHUB_URL: Optional[str] = None
LINKEDIN_URL: Optional[str] = None


@app.on_event("startup")
async def build_collection() -> None:
    global CURRENT_EMPLOYER, CURRENT_ROLE, CURRENT_LOCATION, PROJECTS_INDEX, GITHUB_URL, LINKEDIN_URL
    try:
        html_path = PROJECT_ROOT / "index.html"
        # try a few resume filenames
        possible_resumes = ["resume.pdf", "Resume Aravind Kollipara.pdf", "resume_Aravind.pdf"]
        resume_path = next((PROJECT_ROOT / p for p in possible_resumes if (PROJECT_ROOT / p).exists()), PROJECT_ROOT / "resume.pdf")

        sections = extract_sections_from_html(html_path)
        html_chunks = chunk_lines_with_labels(sections, max_chars=800)

        resume_text = extract_pdf_text(resume_path)
        resume_chunks = chunk_text("resume", resume_text, max_chars=800) if resume_text else []

        lines_for_scan = [ln for lines in sections.values() for ln in lines]
        blob = " ".join(lines_for_scan + [resume_text.lower() if resume_text else ""])

        # simple heuristics for employer/role/location
        if re.search(r"(oracle|oracle cloud|oracle cloud infrastructure)", blob, re.I) and re.search(r"present", blob, re.I):
            CURRENT_EMPLOYER = "Oracle Cloud Infrastructure"
            if re.search(r"senior member of technical staff", blob, re.I):
                CURRENT_ROLE = "Senior Member of Technical Staff"

        m_loc = re.search(r"(Austin)\s*,?\s*(TX|Texas)", blob, re.I)
        if m_loc:
            city = m_loc.group(1).title()
            state = "TX" if re.search(r"(TX|Texas)", m_loc.group(0), re.I) else "Texas"
            CURRENT_LOCATION = f"{city}, {state}"

        PROJECTS_INDEX = extract_recent_projects(html_path)

        social = extract_social_links(html_path, resume_text)
        GITHUB_URL = social.get("github")
        LINKEDIN_URL = social.get("linkedin")

        if collection.count() == 0:
            docs = html_chunks + resume_chunks
            if docs:
                batch_size = 10
                for i in range(0, len(docs), batch_size):
                    batch = docs[i : i + batch_size]
                    ids = [str(j) for j in range(i, i + len(batch))]
                    for attempt in range(3):
                        try:
                            collection.add(documents=batch, ids=ids)
                            break
                        except Exception as e:
                            print(f"Error adding batch {i}-{i+len(batch)}: {e} (attempt {attempt+1})")
                            time.sleep(1 + attempt * 2)
        print(f"Chroma ready (count={collection.count()})")
        print(f"Cached employer={CURRENT_EMPLOYER}, role={CURRENT_ROLE}, location={CURRENT_LOCATION}")
        if PROJECTS_INDEX:
            print(f"Found {len(PROJECTS_INDEX)} project cards; latest={PROJECTS_INDEX[0][0]}")
        if GITHUB_URL or LINKEDIN_URL:
            print(f"Social links: github={GITHUB_URL}, linkedin={LINKEDIN_URL}")
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
    if not context:
        prompt = (
            "You are Aravind's portfolio assistant. Answer concisely and honestly. "
            "If you don't know a fact, say 'I don't know.'\n\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        prompt = (
            "You are Aravind's portfolio assistant. Answer ONLY using the provided context. "
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

    # greeting handler
    if re.search(r"\b(hi|hello|hey|greetings|what's up)\b", q):
        return {"answer": "Hi — I'm Aravind's portfolio assistant. I can answer questions about Aravind's work, projects, and skills based on his resume and website. Ask me anything!"}

    # capability question
    if re.search(r"\b(what can you do|what does this chatbot do|what this chatbot can do|what can you tell)\b", q):
        return {"answer": "I answer questions using Aravind's portfolio and resume: work history, projects, skills, and links (GitHub/LinkedIn) when available."}

    # social link intents (accept variants like 'linked in', 'git hub')
    if re.search(r"\b(github|git[\s-]?hub)\b", q) and re.search(r"\b(account|profile|link|url)\b", q):
        if GITHUB_URL:
            return {"answer": f"My GitHub profile: {GITHUB_URL}"}
        try:
            user_data, _ = await fetch_github_profile()
            return {"answer": f"My GitHub profile: {user_data.get('html_url', '')}"}
        except Exception:
            return {"answer": "I don't know."}

    if re.search(r"\b(linkedin|linked[\s-]?in)\b", q) and re.search(r"\b(profile|link|url)\b", q):
        if LINKEDIN_URL:
            return {"answer": f"My LinkedIn profile: {LINKEDIN_URL}"}
        return {"answer": "I don't know."}

    # cached facts
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

    # RAG with Chroma
    try:
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
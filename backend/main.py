from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cohere
import numpy as np
import httpx
import os

# Set the environment variable for Cohere API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your knowledge base
docs = [
    "Aravind Kollipara is a full-stack developer with 4+ years of experience.",
    "He worked at Amazon Web Services as a Software Development Engineer.",
    "He is passionate about AI/ML and Generative AI.",
    "Aravind Kollipara's LinkedIn profile is https://linkedin.com/in/aravind-kollipara.",
    "Aravind Kollipara's GitHub profile is https://github.com/kvsnsaravind.",
    "He has a highest degree of Masters in Computer Science from the State University of New York at Buffalo."
    "and also a Bachelors in Information Technology from SRM University, Chennai, India.",
    "Aravind Kollipara is Originally from India and currently lives in the USA, Seattle.",
    "If the user uses 'he' or 'his', they refer to Aravind Kollipara.",
    "Aravind Kollipara has worked on various projects including web applications, APIs, and cloud solutions.",
    "He worked in Amazon Web Services for 2 years as a Software Development Engineer. Before that worked in Accenture as a Associate Software Engineer"
    " and BNP Paribas owrked as a Database Developer.",
    # Add more facts about you here
]

def get_embedding(text, input_type="search_document"):
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type=input_type
    )
    return np.array(response.embeddings[0])

def get_llm_response(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=200,
        temperature=0.5
    )
    return response.generations[0].text.strip()

# Precompute embeddings for docs
doc_embeddings = [get_embedding(doc, input_type="search_document") for doc in docs]

GITHUB_USERNAME = "kvsnsaravind"

async def fetch_github_profile():
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(f"https://api.github.com/users/{GITHUB_USERNAME}")
        repos_resp = await client.get(f"https://api.github.com/users/{GITHUB_USERNAME}/repos")
        user_data = user_resp.json()
        repos_data = repos_resp.json()
        return user_data, repos_data

@app.post("/api/chatbot")
async def chatbot(req: Request):
    try:
        print("Received request")
        data = await req.json()
        print("Payload:", data)
        question = data.get("question", "").lower()
        print("Question:", question)

        # If the question is about GitHub, fetch live data
        if "github" in question:
            print("Fetching GitHub profile...")
            user_data, repos_data = await fetch_github_profile()
            print("GitHub data fetched.")
            if "repo" in question or "project" in question:
                repo_names = [repo["name"] for repo in repos_data]
                answer = f"My GitHub repositories include: {', '.join(repo_names[:5])}..." if repo_names else "I have no public repositories."
            elif "star" in question:
                stars = sum(repo.get("stargazers_count", 0) for repo in repos_data)
                answer = f"My repositories have a total of {stars} stars."
            elif "language" in question:
                languages = set()
                for repo in repos_data:
                    if repo.get("language"):
                        languages.add(repo["language"])
                answer = f"I use these languages in my GitHub repos: {', '.join(languages)}." if languages else "No languages detected."
            else:
                answer = f"My GitHub profile: {user_data.get('html_url', '')} with {user_data.get('public_repos', 0)} public repos."
            print("Answer:", answer)
            return JSONResponse({"answer": answer})

        # Otherwise, use RAG/LLM
        print("Using RAG/LLM...")
        q_emb = get_embedding(question, input_type="search_query")
        print("Query embedding computed.")
        sims = [np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)) for d_emb in doc_embeddings]
        print("Similarities computed:", sims)
        top_idx = int(np.argmax(sims))
        context = docs[top_idx]
        print("Context:", context)
        answer = get_llm_response(context, question)
        print("LLM Answer:", answer)
        return {"answer": answer}
    except Exception as e:
        print("Error in chatbot endpoint:", e)
        return {"answer": f"Error: {str(e)}"}
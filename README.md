# Aravind Kollipara Portfolio & Personal AI Chatbot

This is a personal portfolio website for **Venkata Siva Naga Sai Aravind Kollipara** featuring a modern, responsive design and an integrated AI-powered chatbot. The chatbot uses Retrieval-Augmented Generation (RAG) with Cohere LLM and live GitHub data to answer questions about Aravind's career, skills, and projects.

---

## 🚀 Features

- **Responsive Portfolio Website**: Built with HTML, CSS, and JavaScript.
- **AI Chatbot**: Answers questions about Aravind using RAG (Retrieval-Augmented Generation) and Cohere LLM.
- **Live GitHub Integration**: Fetches real-time GitHub profile and repository data.
- **Certifications Section**: Showcases verified certifications with clickable links.
- **Deployed on Render**: Backend hosted on [Render](https://render.com/), frontend can be hosted on any static site host (Vercel, Netlify, GitHub Pages, etc.).

---

## 🛠️ Technologies Used

### **Frontend**
- **HTML5** & **CSS3**: Structure and styling.
- **JavaScript**: Dynamic UI, chatbot integration, fetch API.
- **Font Awesome**: Icons.
- **Google Fonts**: Typography.

### **Backend**
- **Python 3**
- **FastAPI**: High-performance API framework.
- **Uvicorn**: ASGI server for running FastAPI.
- **Cohere API**: For embeddings and LLM-based answers.
- **NumPy**: Vector math for RAG.
- **httpx**: Async HTTP client for GitHub API.

### **RAG (Retrieval-Augmented Generation)**
- **Knowledge Base**: In-memory Python list (`docs`).
- **Embeddings**: Cohere's `embed-english-v3.0` model.
- **Similarity Search**: Cosine similarity with NumPy.
- **LLM**: Cohere's `command-r-plus` model.

### **Live Data**
- **GitHub API**: Fetches profile and repo info in real time.

### **Hosting**
- **Render**: Persistent backend hosting for FastAPI.
- **Frontend**: I used Vercel. Can be hosted on Vercel, Netlify, GitHub Pages, or any static host.

---

## 🗄️ Database

- **How RAG is used in my project:**

When a user asks a question, the backend:
Embeds the question using Cohere’s embedding model.
Finds the most relevant document from in-memory knowledge base (using cosine similarity).
Sends the context and question to the Cohere LLM to generate a final answer.
This is the classic RAG workflow: retrieve relevant info, then generate an answer using that info.

Summary:
Your chatbot uses RAG, but with a simple in-memory list as the knowledge base (not a vector database).

---

## 📂 Project Structure

```
portfolio/
│
├── backend/
│   ├── main.py
│
|__ requirements.txt
├── index.html
├── style.css
└── README.md
```

---

## 💡 How It Works

- The frontend displays Aravind's portfolio and a chatbot widget.
- When you ask a question, the frontend sends it to the FastAPI backend.
- The backend uses Cohere to embed the question, finds the most relevant info from the knowledge base, and generates an answer.
- For GitHub-related queries, the backend fetches live data from the GitHub API.
- The answer is returned and displayed in the chat window.

---

## 📜 License

This project is for personal portfolio and demonstration purposes.

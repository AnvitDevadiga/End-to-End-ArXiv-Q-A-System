End-to-End Question-Answering System for ArXiv Research Papers
This project is a complete, end-to-end system that allows users to ask natural language questions about scientific research papers from ArXiv. It utilizes a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers grounded in the content of the papers.

Features
Dynamic Data Ingestion: Fetches and processes the latest research papers from ArXiv based on a user-defined search query.

Local Vector Store: Uses ChromaDB to store document embeddings locally, making it fast and free to run.

Open-Source LLM Integration: Connects to a locally running LLM (like Llama 2 or a faster alternative) via Ollama for answer generation.

Interactive UI: A simple and intuitive web interface built with Streamlit.

Source Referencing: Displays the exact text chunks used by the LLM to generate the answer, allowing for easy verification.

Architecture (RAG Pipeline)
The system follows a Retrieval-Augmented Generation workflow:

```mermaid
graph TD
    subgraph "1. Ingestion"
        A[User provides topic e.g., 'diffusion models'] --> B(Fetch & Process ArXiv Papers);
        B --> C{ChromaDB Vector Store};
    end

    subgraph "2. Retrieval & Generation"
        D[User asks question] --> E(Embed Question);
        E -- Similarity Search --> C;
        C -- Retrieve Relevant Chunks --> F((Construct Prompt));
        D -- Original Question --> F;
        F --> G[LLM via Ollama];
        G --> H[Generated Answer];
    end

    H --> I[Display Answer & Sources to User];
end

Tech Stack
Application Framework: Streamlit

ArXiv API: arxiv Python library

PDF Processing: pypdf

Vector Database: ChromaDB

Embedding Model: sentence-transformers

LLM Hosting: Ollama

LLM: Llama 2, tinyllama, or any other Ollama-supported model

Screenshots
Initial UI:

Final Result with Answer and Sources:
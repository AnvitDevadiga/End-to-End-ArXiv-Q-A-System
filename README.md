End-to-End Question-Answering System for ArXiv Research Papers
This project is a complete, end-to-end system that allows users to ask natural language questions about scientific research papers from ArXiv. It utilizes a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers grounded in the content of the papers.

Features
Dynamic Data Ingestion: Fetches and processes the latest research papers from ArXiv based on a user-defined search query.

Local Vector Store: Uses ChromaDB to store document embeddings locally, making it fast and free to run.

Open-Source LLM Integration: Connects to a locally running LLM (like Llama 2 or Mistral) via Ollama for answer generation.

Interactive UI: A simple and intuitive web interface built with Streamlit.

Source Referencing: Displays the exact text chunks used by the LLM to generate the answer, allowing for easy verification.

Architecture (RAG Pipeline)
The system follows a Retrieval-Augmented Generation workflow, which is visualized below. This process ensures that the AI's answers are based directly on the content of the indexed documents, rather than its general knowledge.

graph TD
    subgraph "1. Data Ingestion & Indexing"
        A[User provides topic e.g., 'diffusion models'] --> B(Fetch & Process ArXiv Papers);
        B --> C{ChromaDB Vector Store};
    end

    subgraph "2. Query & Answer Generation"
        D[User asks question] --> E(Embed Question);
        E -- Similarity Search --> C;
        C -- Retrieve Relevant Chunks --> F((Construct Prompt));
        D -- Original Question --> F;
        F --> G[LLM via Ollama];
        G --> H[Generated Answer];
        H --> I[Display Answer & Sources to User];
    end

Screenshots
Initial UI for Indexing Papers:

Final Result with Generated Answer and Verifiable Sources:

Tech Stack
Application Framework: Streamlit

ArXiv API: arxiv Python library

PDF Processing: pypdf

Vector Database: ChromaDB

Embedding Model: sentence-transformers

LLM Hosting: Ollama

LLM: Llama 2, Mistral, or any other Ollama-supported model

Setup and Installation
Follow these steps to get the project running on your local machine.

Step 1: Set Up Ollama
This project requires a locally running LLM. Ollama is the easiest way to achieve this.

Install Ollama: Follow the instructions on the official Ollama website.

Pull an LLM model: Open your terminal and pull a model. We recommend llama2 for quality or tinyllama for speed.

ollama pull llama2

Ensure the Ollama application is running in the background before proceeding.

Step 2: Create a Python Environment
It is highly recommended to use a virtual environment to manage project dependencies.

Navigate to your project directory in the terminal.

cd End-to-End-ArXiv-Q-A-System

Create and activate a virtual environment.

# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install the required libraries.

pip install -r requirements.txt

Step 3: Run the Application
Once the setup is complete, run the Streamlit application with a single command:

streamlit run app.py

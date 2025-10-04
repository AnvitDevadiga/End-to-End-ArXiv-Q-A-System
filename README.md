# End-to-End Question-Answering System for ArXiv Research Papers

This project is a complete, end-to-end system that allows users to ask natural language questions about scientific research papers from ArXiv. It utilizes a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers grounded in the content of the papers.

## Features

- **Dynamic Data Ingestion**: Fetches and processes the latest research papers from ArXiv based on a user-defined search query.
- **Local Vector Store**: Uses ChromaDB to store document embeddings locally, making it fast and free to run.
- **Open-Source LLM Integration**: Connects to a locally running LLM (like Llama 2 or Mistral) via Ollama for answer generation.
- **Interactive UI**: A simple and intuitive web interface built with Streamlit.
- **Source Referencing**: Displays the exact text chunks used by the LLM to generate the answer, allowing for easy verification.

## Architecture (RAG Pipeline)

The system follows a Retrieval-Augmented Generation workflow:

1.  **Ingestion**: The user provides a topic. The system searches ArXiv, downloads relevant papers, extracts text, splits it into chunks, and stores vector embeddings of these chunks in ChromaDB.
2.  **Retrieval**: A user asks a question. The question is converted into an embedding.
3.  **Search**: ChromaDB performs a similarity search to find the most relevant text chunks from the indexed papers.
4.  **Augmentation**: The user's question and the retrieved text chunks are combined into a detailed prompt.
5.  **Generation**: The prompt is sent to a local LLM (via Ollama), which generates a final answer based *only* on the provided context.

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
...
    H --> I[Display Answer & Sources to User];
end

## Tech Stack

- **Application Framework**: Streamlit
- **ArXiv API**: `arxiv` Python library
- **PDF Processing**: `pypdf`
- **Vector Database**: ChromaDB
- **Embedding Model**: `sentence-transformers`
- **LLM Hosting**: Ollama
- **LLM**: Llama 2, Mistral, or any other Ollama-supported model

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Step 1: Clone the Repository (or Create the Files)

If this were a Git repository, you would clone it. For now, create a folder named `arxiv-qa-system` and create the files (`app.py`, `requirements.txt`, etc.) inside it with the content provided.

### Step 2: Set Up Ollama

You need a locally running LLM. Ollama is the easiest way to achieve this.

1.  **Install Ollama**: Follow the instructions on the [official Ollama website](https://ollama.com/).
2.  **Pull an LLM model**: Open your terminal and pull a model. We recommend `llama2`.
    ```bash
    ollama pull llama2
    ```
3.  Ensure the Ollama application is running in the background.

### Step 3: Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Navigate to your project directory
cd arxiv-qa-system

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
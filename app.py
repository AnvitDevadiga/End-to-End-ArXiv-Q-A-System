import streamlit as st
import arxiv
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import os
import re

# --- Constants & Configuration ---
DB_DIR = "chroma_db"
COLLECTION_NAME = "arxiv_papers"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2" # Make sure you have this model pulled in Ollama

# --- Helper Functions ---

def safe_filename(filename):
    """Creates a safe filename by removing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def fetch_and_process_papers(search_query: str, max_results: int = 5):
    """
    Fetches papers from ArXiv, downloads them, extracts text, chunks it,
    and returns a list of document chunks.
    """
    st.info(f"Searching ArXiv for '{search_query}'...")
    try:
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())
    except Exception as e:
        st.error(f"Failed to fetch from ArXiv: {e}")
        return []

    if not results:
        st.warning("No papers found on ArXiv for your query.")
        return []
        
    papers_dir = "arxiv_papers"
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)

    all_chunks = []
    
    for i, result in enumerate(results):
        progress_bar.progress((i + 1) / len(results), text=f"Processing: {result.title}")
        try:
            filename = f"{safe_filename(result.title)}.pdf"
            pdf_path = os.path.join(papers_dir, filename)
            
            # Download PDF
            result.download_pdf(dirpath=papers_dir, filename=filename)
            
            # Extract Text
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

            if not text:
                st.warning(f"Could not extract text from '{result.title}'. Skipping.")
                continue

            # Chunk Text
            chunks = chunk_text(text, result.title, result.pdf_url)
            all_chunks.extend(chunks)

        except Exception as e:
            st.error(f"Error processing paper '{result.title}': {e}")
            
    return all_chunks


def chunk_text(text: str, title: str, url: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits a long text into smaller, overlapping chunks.
    Each chunk includes metadata (the paper's title and URL).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "metadata": {"title": title, "url": url}
        })
        start += chunk_size - chunk_overlap
    return chunks

def get_embedding_model():
    """Loads the sentence-transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def setup_vector_db(chunks, embedding_model):
    """
    Initializes ChromaDB, creates embeddings for the text chunks,
    and stores them in a collection.
    """
    if not chunks:
        st.warning("No text chunks to process. Vector DB not created.")
        return None

    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Clear out the old collection if it exists
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    # Process in batches to show progress
    batch_size = 50
    total_chunks = len(chunks)
    st.info(f"Generating embeddings for {total_chunks} chunks...")

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [chunk["text"] for chunk in batch_chunks]
        
        embeddings = embedding_model.encode(batch_texts, show_progress_bar=False).tolist()
        
        ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
        metadatas = [chunk["metadata"] for chunk in batch_chunks]

        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=metadatas,
            ids=ids
        )
        progress_bar.progress((i + len(batch_chunks)) / total_chunks, text=f"Embedding chunks... {i+len(batch_chunks)}/{total_chunks}")
        
    return collection

def query_llm(question: str, context: str):
    """Sends a question and context to the local LLM and streams the response."""
    prompt = f"""
    You are a helpful AI assistant for answering questions about scientific papers.
    Use only the following context to answer the question. Your answer should be concise and directly based on the provided text.
    If the context does not contain the answer, state that the answer is not found in the provided context.

    Context:
    ---
    {context}
    ---

    Question: {question}

    Answer:
    """
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False # Set to False for a single, complete response
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        # Extract the content from the response JSON
        response_json = response.json()
        return response_json.get("response", "Error: Could not parse LLM response.")

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"


# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="ArXiv Q&A System")

st.title("📚 End-to-End Question-Answering System for ArXiv Papers")
st.markdown("This app uses a RAG pipeline to answer questions about ML research papers. Enter a topic, index the papers, and ask a question.")

# --- Session State Initialization ---
if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None


# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("1. Index Papers")
    st.write("Enter a topic to fetch related papers from ArXiv and build a vector index.")
    
    arxiv_query = st.text_input("ArXiv Search Query", "diffusion models")
    max_papers = st.slider("Number of papers to index", 1, 20, 5)

    if st.button("Fetch and Index Papers"):
        with st.spinner("Processing papers... This may take a few minutes."):
            st.session_state.db_ready = False
            progress_bar = st.progress(0.0, text="Starting...")
            
            # Step 1: Fetch and Chunk
            chunks = fetch_and_process_papers(arxiv_query, max_results=max_papers)
            
            if chunks:
                # Step 2: Load Model
                if st.session_state.embedding_model is None:
                     st.session_state.embedding_model = get_embedding_model()

                # Step 3: Setup Vector DB
                st.session_state.collection = setup_vector_db(chunks, st.session_state.embedding_model)
                st.session_state.db_ready = True
                progress_bar.progress(1.0, text="Indexing complete!")
                st.success(f"Successfully indexed {len(chunks)} text chunks from {max_papers} papers.")
            else:
                st.error("Failed to process any papers. Please try a different query.")
                progress_bar.empty()

# --- Main Area for Q&A ---
st.header("2. Ask a Question")

if not st.session_state.db_ready:
    st.warning("Please index some papers first using the sidebar.")
else:
    question = st.text_input("Enter your question about the indexed papers:", "What are the latest techniques in diffusion models?")

    if question:
        with st.spinner("Searching for answers..."):
            # 1. Embed the user's question
            question_embedding = st.session_state.embedding_model.encode(question).tolist()

            # 2. Query the vector database
            try:
                results = st.session_state.collection.query(
                    query_embeddings=[question_embedding],
                    n_results=5 # Retrieve top 5 most relevant chunks
                )
                
                context_docs = results['documents'][0]
                context = "\n\n---\n\n".join(context_docs)
                
                # 3. Query the LLM with the context
                answer = query_llm(question, context)
                
                st.subheader("Answer")
                st.markdown(answer)
                
                # 4. Display sources
                with st.expander("Show Sources"):
                    st.write("The answer was generated based on the following text chunks:")
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                        st.info(f"**From: {meta['title']}**")
                        st.link_button("Go to Paper", meta['url'])
                        st.caption(doc)
            
            except Exception as e:
                st.error(f"An error occurred during the query: {e}")


# Agentic RAG: High-Precision Technical Document Assistant

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline designed for high-accuracy technical analysis. Using the "Attention Is All You Need" research paper as a benchmark, the system demonstrates how to move beyond basic vector search by incorporating **Two-Stage Retrieval (Re-ranking)** and **Automated Fact-Verification**.



## 🧠 Core Engineering Concepts

### 1. Two-Stage Retrieval (Retrieve & Re-rank)
Standard RAG systems often suffer from "noise" because vector databases (Bi-Encoders) only look at mathematical distance between phrases. This project uses a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM`) to perform a deep semantic audit of the top 20 retrieved chunks, selecting only the 3–5 most relevant pieces of evidence for the LLM.

### 2. Multi-Model Embedding Showdown
The system compares two distinct embedding architectures:
* **Nomic Embed Text (768d):** High-dimensionality for capturing complex architectural nuances.
* **All-MiniLM-L6 (384d):** A lightweight, fast model optimized for local hardware with a strict 256-token context window.

### 3. Agentic Verification Layer
To eliminate hallucinations, the system employs a "Self-Correction" loop. After **Llama 3** generates an initial response, a **Verification Agent** (running at $Temperature = 0.0$) audits the answer against the original PDF sources to ensure every claim is explicitly supported by the text.

---

## 🛠️ Technical Stack

* **LLM:** Meta Llama 3 (via Ollama)
* **Embedding Models:** `nomic-embed-text`, `all-minilm`
* **Vector Database:** ChromaDB
* **Re-ranker:** HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Framework:** LangChain & LangChain-Classic
* **Environment:** Jupyter Notebook (VS Code Interface)

---

## 📂 Project Structure

| File | Purpose |
| :--- | :--- |
| `Agentic_RAG.ipynb` | The main execution environment containing the interactive pipeline. |
| `prompts.yaml` | Version-controlled configuration for System and User prompts. |
| `/chroma_prod_db` | Persistent local storage for the document embeddings. |
| `Attention Is All You Need.pdf` | The source technical document (Transformer Research Paper). |

---

## 🚀 Step-by-Step Logic Flow

### Step 1: Document Ingestion & Safe Chunking
The PDF is loaded and split using `RecursiveCharacterTextSplitter`. 
> **Note:** We use a smaller chunk size (400 chars) for MiniLM to prevent "Context Overflow" errors caused by its hard 256-token limit.

### Step 2: Vectorization
The text is transformed into high-dimensional vectors and stored in ChromaDB. 


### Step 3: The Search Pipeline
1.  **Similarity Search:** ChromaDB identifies the top 20 "potential" matches.
2.  **Cross-Encoding:** The Re-ranker scores these 20 matches based on their actual relationship to the user's question.
3.  **Context Assembly:** The top 3 re-ranked chunks are stitched together into a "Golden Context" block.

### Step 4: Guarded Generation
Llama 3 receives the context. If the answer is not present, the model is strictly instructed to return `DECLINE_TO_ANSWER` rather than hallucinating a guess.

### Step 5: Verification Report
A final agentic check is performed, providing a "SUPPORTED: Yes/No" status and a reason for the grade.

---

## 📊 Performance Visualizations
The project includes histogram visualizations of chunk length distributions, helping developers tune the `chunk_size` and `chunk_overlap` parameters to ensure no data is "cut off" mid-sentence.



---

## 🔧 Getting Started

1.  **Install Requirements:**
    `pip install langchain langchain-community chromadb pypdf sentence-transformers langchain-classic`
2.  **Start Ollama:**
    Ensure `ollama serve` is running with `llama3` and `nomic-embed-text` pulled.
3.  **Run the Notebook:**
    Execute cells sequentially to see the Embedding Showdown and the final Verified Answer.

---

Would you like me to add a **"Troubleshooting"** section to this README covering the specific `ValueError 500` and `NameError` fixes we implemented?

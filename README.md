# Advanced RAG Pipeline for Document Q&A

This project implements an advanced **Retrieval-Augmented Generation (RAG)** pipeline designed for querying a local collection of PDF documents. It leverages local language models and embedding models to ensure privacy and control. The system is built with a sophisticated, multi-step agentic workflow that includes query decomposition and reranking to provide accurate, cited answers.

---

## 🚀 Features

- 📄 **Structured PDF Parsing**: Intelligently extracts text by identifying headers and paragraphs to maintain document context.  
- 🧠 **Agentic Workflow**:
  - **Orchestrator Agent**: Analyzes the user's query to decide whether to search the knowledge base or attempt a direct answer.  
  - **Query Decomposition Agent**: Breaks down complex user questions into multiple, targeted sub-queries for more comprehensive retrieval.  
- 🔍 **Multi-Query Retrieval**: Fetches relevant document chunks from a ChromaDB vector store for each sub-query.  
- ✨ **Cross-Encoder Reranking**: Refines the retrieved results by reranking them against the original query for maximum relevance, ensuring the best documents are used as context.  
- 🤖 **100% Local Inference**: Uses GGUF models via llama-cpp-python for all language generation tasks, ensuring data privacy and offline capability.  
- 📝 **Cited Responses**: The final answer is synthesized from the retrieved context, with each piece of information followed by a citation  
  *Example: [Source: document.pdf, Page X]*  

---

## 📂 Project Structure

The codebase is organized into logical modules for clarity and maintainability.
```
Directory of the Project/
├── 📁 documents/ # PDF files here
├── 📁 rag_agents/ # Contains the logic for the RAG pipeline agents
├── 📁 data_processing/ # Modules for PDF parsing and text chunking
├── 📁 llm/ # Wrapper for the local Llama.cpp model
├── 📁 vector_store/ # Functions for managing and querying ChromaDB
├── 📄 demo.ipynb # Jupyter Notebook for running the end-to-end demo
├── 📄 requirements.txt # Project dependencies
└── 📄 README.md # This file
```
---

## 🛠️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Place Your Documents
Add all the PDF files you want to query into the `documents/` folder.

### 3. Configure the Project
Open the demo.ipynb and review the settings. The defaults are chosen for a good balance of performance and resource usage, but you can customize them:

pdf_folder: Path to your documents folder.

CHROMA_PATH & COLLECTION_NAME: Settings for the vector database.

EMBEDDING_MODEL & RERANKER_MODEL: The HuggingFace repo IDs for the Bi-Encoder and the CrossEncoder respectively.

LLM_MODEL_ID & LLM_MODEL_FILE: The HuggingFace repo ID and filename for the GGUF model you wish to use.

The first time you run the code, the model will be automatically downloaded and cached by the huggingface_hub library.

## 🚀 How to Run the Demo
The entire workflow is demonstrated in the demo.ipynb Jupyter Notebook.

### Step 1: Launch Jupyter

Make sure your virtual environment is activated, then start Jupyter:

### Step 2: Build the Vector Database
Open demo.ipynb and run the cells in `Document Processing and DB populating`

This will:
- Parse all PDFs from your documents/ folder.

- Chunk the text contextually.

- Generate embeddings using the sentence-transformer model.

- Store everything in a local ChromaDB database.

Note: You only need to run this step once, unless you add, remove, or change the PDF documents.

### Step 3: Ask Questions!
Proceed to `Retrieve information based on User Queries`.

- Locate the get_rag_response() function call.

- Change the user_query variable to your question or add another block by copy pasting the existing code.

- Run the cell to get a complete, cited answer.

## ⚙️ The RAG Pipeline Explained

When you ask a question, the system follows this step-by-step process:

### 1. Orchestration
- The `run_orchestrator_agent` analyzes your query.  
- If it's simple chit-chat, it may answer directly.  
- If it's knowledge-based, it triggers the **SEARCH** action.  

### 2. Query Decomposition
- For complex questions (e.g., comparisons), the `run_query_decomposition_agent` breaks it into smaller, focused sub-queries.  
- **Example:**  
  - *"Compare A and B"* → *"What is A?"* + *"What is B?"*  

### 3. Retrieval
- Queries the **ChromaDB** vector store for all sub-queries, gathering potentially relevant chunks.  
- Removes duplicates.  

### 4. Reranking
- Uses a **CrossEncoder** model to rerank results by direct relevance to your original query.  
- The most important information is pushed to the top.  

### 5. Synthesis
- Combines the top-ranked documents into a context block.  
- The `run_response_agent` uses this context + query to generate the final answer.  
- Every claim is **cited** with its original source.  

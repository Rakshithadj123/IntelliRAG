# IntelliRAG
Multi-Agent Research Assistant Using Agentic RAG.
# IntelliRAG – Multi-Agent Research Assistant Using Agentic RAG

## Project Overview
IntelliRAG is an AI-powered research assistant designed to help users understand and retrieve information from uploaded PDF documents using locally running Large Language Models (LLMs).

The project uses Ollama to run language models locally and a text-based retrieval approach to find relevant information from documents without requiring an embedding model or vector database.

## Key Features
- PDF document upload.
- Ask questions based on uploaded documents.
- Local LLM integration using Ollama.
- Select from available installed Ollama models.
- Text-based document retrieval.
- Temporary in-memory document storage.
- AI-generated answers based on retrieved document content.

## Technologies Used
- **Programming Language:** Python
- **Frontend:** Streamlit
- **LLM Runtime:** Ollama
- **AI Models:** Locally installed Ollama models
- **PDF Processing:** PyMuPDF
- **Retrieval:** Text-based search and matching

## System Architecture

User uploads PDF documents  
↓  
Document Loading  
↓  
Text Extraction  
↓  
Text Chunking  
↓  
Query Processing  
↓  
Relevant Text Retrieval  
↓  
Context + Prompt  
↓  
Ollama LLM  
↓  
AI-Generated Answer

## Project Structure

```text
IntelliRAG/
│
├── IntelliRAG/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rakshithadj123/IntelliRAG.git
```

### 2. Open the project folder

```bash
cd IntelliRAG
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Install and start Ollama

Download Ollama from:

https://ollama.com/

Pull an example model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running.

### 7. Run the application

```bash
streamlit run IntelliRAG/app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Future Enhancements
- Improved document retrieval and ranking.
- Query expansion for better search results.
- Conversational memory.
- Page-level PDF citations.
- OCR support for scanned PDF documents.
- Advanced multi-agent research workflows.

## Project Goal

To develop a locally running AI research assistant that helps users extract useful information from documents and get answers using open-source language models.

## Author

**Rakshitha DJ**

GitHub: https://github.com/Rakshithadj123


# RAG Document Q&A with Groq and Lama3

This project is a Streamlit-based application for building a **Retrieval-Augmented Generation (RAG)** system. The app allows users to upload PDF and text files, process them into vector embeddings, and query the documents using a large language model (LLM) powered by **Groq** and **Llama3**.

## Features
- Upload and process multiple PDF or text files.
- Create vector embeddings using **OpenAIEmbeddings**.
- Perform document similarity searches with **FAISS**.
- Generate accurate answers to queries using **ChatGroq** LLM.
- Dynamic and interactive UI built with **Streamlit**.

## Installation

### Prerequisites
- Python 3.8 or higher
- A Groq API Key (for ChatGroq)
- OpenAI API Key (for embeddings)

### Clone the Repository
```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa


Install Dependencies
```bash
pip install -r requirements.txt

File Structure
```bash

rag-document-qa/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not included in the repository)
├── research_papers/       # Folder for storing uploaded files
└── README.md              # Project documentation
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from io import BytesIO
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
import openai
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ API Key
api_key = os.getenv("OPENAI_API_KEY")

groq_api_key = os.getenv("groq_api")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to handle document uploads
def upload_and_process_files():
    st.session_state.embeddings = OpenAIEmbeddings(api_key=api_key)
    
    uploaded_files = st.file_uploader(
        "Upload PDF or Text Files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.docs = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.type == "application/pdf" else ".txt") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Use the temporary file path with the appropriate loader
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(tmp_file_path)
            else:
                st.error("Unsupported file format!")
                continue

            st.session_state.docs.extend(loader.load())

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Files processed and vector database is ready!")

# Streamlit App Title
st.title("RAG Document Q&A With Groq And Lama3")

# File Upload Section

upload_and_process_files()

# Query Input Section

user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    response_time = time.process_time() - start

    st.write(f"Response Time: {response_time:.2f} seconds")
    st.write(response['answer'])

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
else:
    if not "vectors" in st.session_state:
        st.warning("Please upload and process documents first.")
    elif user_prompt:
        st.error("Unable to process the query. Please ensure the documents are uploaded.")

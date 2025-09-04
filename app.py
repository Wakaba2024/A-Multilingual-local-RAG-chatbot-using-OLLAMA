import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# PROMPT DEFINITIONS
# ---------------------------

# Query Prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Your task is to take the user’s question
    and, if needed, rephrase it slightly so that it retrieves the most relevant 
    and precise context from a vector database built on the provided Finance Bill document.

    The goal is to maximize the chance of finding exact or very close answers 
    within the document.

    User question: {question}"""
)

# Answer Prompt
ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant for question-answering tasks.
Use only the following context extracted from the Finance Bill PDF:

{context}

Question: {question}

Instructions:
- If the answer is in the context, provide a clear and concise response.
- If the information is NOT in the context, say:
  "The provided Finance Bill document does not contain information to answer that question."

Answer:"""
)


st.title("A PDF(Finance Bill Kenya 2025) Multilingual RAG Chatbot Using Ollama")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Save file locally
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)

    # Embedding + vector DB (using nomic-embed-text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="local-rag")

    # Retriever
    retriever = vector_db.as_retriever()

    # Prompt
    template = """Use the following context to answer the question.
    Context: {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    # Model + chain
    model = ChatOllama(model="llama3.1")  
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
    )

    # Ask user
    user_query = st.text_input("Ask a question about the document:")
    if user_query:
        with st.spinner("Generating answer..."):
            response = chain.invoke(user_query)
            st.write("### Answer:")
            st.write(response.content)

import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# --- INITIAL SETUP ---
st.set_page_config(page_title="Legal Auditor AI", layout="centered")
st.title("⚖️ Legal Contract Red-Flagger")
st.markdown("### Identifying risks in your contracts using RAG")

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a Contract (PDF)", type="pdf")
    process_button = st.button("Analyze Document")

# --- CORE LOGIC ---
if uploaded_file and process_button:
    with st.spinner("Reading and indexing the contract..."):
        # 1. Save the file temporarily
        with open("temp_contract.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. Load the PDF
        loader = PyPDFLoader("temp_contract.pdf")
        data = loader.load()
        
        # 3. Chunking (Splitting text into searchable pieces)
        # This is the "Data Science" part: We split by 1000 characters 
        # so we don't exceed the AI's memory limit.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)
        
        # 4. Create the Vector Brain (Embeddings + FAISS)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # 5. Setup the AI "Expert"
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Faster and great for RAG
            google_api_key=GEMINI_API_KEY, 
            temperature=0 # Zero means no "imagination", stay factual!
        )
        
        # Create the RAG retrieval chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
        )
        st.success("Contract Analysis Ready!")

# --- CHAT / ANALYSIS INTERFACE ---
if "qa_chain" in st.session_state:
    st.divider()
    user_input = st.text_input("Ask a specific question (e.g., 'What are the termination rules?' or 'List any hidden fees'):")
    
    if user_input:
        with st.spinner("Scanning document..."):
            # Customizing the query to force a "Red Flag" persona
            response = st.session_state.qa_chain.invoke(
                f"Act as a strict legal auditor. Answer this question based ONLY on the provided document: {user_input}. "
                "If you find a clause that seems unfair or risky, start your answer with '🚩 RED FLAG:'"
            )
            st.markdown("#### Auditor's Finding:")
            st.write(response["result"])
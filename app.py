import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        
        # 3. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)
        
        # 4. Create the Vector Brain
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # 5. Setup the AI Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY, 
            temperature=0
        )
        
        # Save the active retriever and model to session state
        st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        st.session_state.llm = llm
        
        st.success("Contract Analysis Ready!")

# --- CHAT / ANALYSIS INTERFACE ---
if "retriever" in st.session_state:
    st.divider()
    user_input = st.text_input("Ask a specific question (e.g., 'What are the termination rules?'):")
    
    if user_input:
        with st.spinner("Scanning document..."):
            
            # Step A: RETRIEVAL (Find the 3 most relevant paragraphs)
            docs = st.session_state.retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Step B: AUGMENTATION (Inject the paragraphs into a prompt)
            prompt = f"""
            Act as a strict legal auditor. Answer this question based ONLY on the provided document context. 
            If you find a clause that seems unfair or risky, start your answer with '🚩 RED FLAG:'
            
            CONTEXT FROM CONTRACT:
            {context}
            
            USER QUESTION:
            {user_input}
            """
            
            # Step C: GENERATION (Let Gemini analyze it)
            response = st.session_state.llm.invoke(prompt)
            
            st.markdown("#### Auditor's Finding:")
            # We use .content to get the text out of the modern ChatModel object
            st.write(response.content)
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Streamlit UI
st.set_page_config(page_title="Legal Auditor AI", layout="centered")
st.title("⚖️ Legal Contract Red-Flagger")
st.markdown("### Identifying risks in your contracts using RAG")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a Contract (PDF)", type="pdf")
    process_button = st.button("Analyze Document")

# Document processing and Vector DB initialization
if uploaded_file and process_button:
    with st.spinner("Reading and indexing the contract..."):
        temp_path = "temp_contract.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY, 
            temperature=0
        )
        
        st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        st.session_state.llm = llm
        
        st.success("Contract Analysis Ready!")

# Retrieval and Generation Pipeline
if "retriever" in st.session_state:
    st.divider()
    user_input = st.text_input("Ask a specific question (e.g., 'What are the termination rules?'):")
    
    if user_input:
        with st.spinner("Scanning document..."):
            docs = st.session_state.retriever.invoke(user_input)
            
            context = ""
            citations = []
            
            # Extract content and metadata for citation tracking
            for doc in docs:
                page_num = doc.metadata.get("page", 0) + 1
                context += f"\n--- Excerpt from Page {page_num} ---\n{doc.page_content}\n"
                citations.append(f"Page {page_num}")
            
            unique_citations = list(set(citations))
            
            prompt = f"""
            Act as a strict legal auditor. Answer this question based ONLY on the provided document context. 
            If you find a clause that seems unfair or risky, start your answer with '🚩 RED FLAG:'
            Always mention which page your finding comes from.
            
            CONTEXT FROM CONTRACT:
            {context}
            
            USER QUESTION:
            {user_input}
            """
            
            response = st.session_state.llm.invoke(prompt)
            
            st.markdown("#### Auditor's Finding:")
            st.write(response.content)
            st.caption(f"🔍 **Sources checked by AI:** {', '.join(unique_citations)}")
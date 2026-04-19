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
        
        # Save the full text for the overarching Risk Score calculation
        st.session_state.full_text = "\n".join([doc.page_content for doc in data])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)
        
        # Using the latest embedding model to prevent 404 errors
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
    
    # --- FEATURE 3: RISK SCORE DASHBOARD ---
    st.markdown("### 📊 Contract Risk Dashboard")
    if st.button("Calculate Overall Risk Score"):
        with st.spinner("Analyzing full contract liabilities..."):
            risk_prompt = f"""
            Act as a senior legal risk analyst. Read the following contract and assign a quantitative 'Risk Score' from 1 to 100.
            1 = Perfectly safe, standard terms.
            100 = Highly dangerous, predatory, or illegal terms.
            
            Respond strictly in this exact format, nothing else:
            SCORE: [Number]
            REASON: [One sentence explaining the score]
            
            CONTRACT TEXT:
            {st.session_state.full_text[:50000]} 
            """
            
            try:
                score_response = st.session_state.llm.invoke(risk_prompt).content
                score_str = score_response.split("SCORE:")[1].split("\n")[0].strip()
                reason = score_response.split("REASON:")[1].strip()
                score = int(score_str)
                
                if score < 40:
                    status = "🟢 Low Risk"
                elif score < 75:
                    status = "🟡 Medium Risk"
                else:
                    status = "🔴 HIGH RISK"
                
                st.metric(label=f"Risk Assessment: {status}", value=f"{score}/100")
                st.progress(score / 100.0)
                st.warning(f"**Analyst Note:** {reason}")
                
            except Exception as e:
                st.error("Could not calculate risk score. Ensure the document contains readable text.")
    
    st.divider()

    # --- QUICK AUDITS ---
    st.markdown("### ⚡ Quick Audits")
    col1, col2, col3 = st.columns(3)
    
    button_query = None
    if col1.button("💰 Financial Liabilities"):
        button_query = "Identify any hidden fees, unexpected costs, or financial liabilities in this contract."
    if col2.button("📅 Termination Rules"):
        button_query = "What are the exact rules, notice periods, and penalties for terminating this agreement?"
    if col3.button("🔒 Data & IP Risks"):
        button_query = "Are there any clauses related to data sharing, privacy risks, or Intellectual Property ownership?"
        
    st.markdown("### 💬 Or ask a custom question")
    # Using chat_input to prevent Streamlit SSL network disconnects
    user_input = st.chat_input("Type your specific query here...")
    
    final_query = button_query or user_input
    
    if final_query:
        with st.spinner("Scanning document..."):
            docs = st.session_state.retriever.invoke(final_query)
            
            context = ""
            citations = []
            
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
            {final_query}
            """
            
            response = st.session_state.llm.invoke(prompt)
            
            st.markdown("#### Auditor's Finding:")
            st.info(f"**Query:** {final_query}") 
            st.write(response.content)
            st.caption(f"🔍 **Sources checked by AI:** {', '.join(unique_citations)}")
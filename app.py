import os

# Set fallback flags to prevent transformers from complaining about missing optional framework dependencies
os.environ["TRANSFORMERS_NO_AD_HOC_VISION_MODELS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import streamlit as st
from dotenv import load_dotenv

# Load local .env file securely
load_dotenv()

st.set_page_config(page_title="Zyro Dynamics HR Portal", page_icon="🏢", layout="centered")

# Custom CSS for Premium UI Layout
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .response-card {
        background-color: #1e222b;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #ff4b4b;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏢 Zyro Dynamics HR Support Portal")
st.markdown("Query the company policy database with zero friction.")
st.markdown("---")

# Verify configuration state upfront
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing `GROQ_API_KEY`! Please configure your environment variables or local `.env` file.")
    st.stop()

# Cached Resource Setup to eliminate runtime overhead
@st.cache_resource(show_spinner="Indexing corporate guidelines...")
def initialize_rag_system():
    # Defensive imports inside the cached setup to isolate scanning issues
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq

    corpus_path = "./zyro-dynamics-hr-corpus"

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Directory '{corpus_path}' not found.")

    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No valid PDF data found inside target path.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Passing encode_kwargs ensures it doesn't query a live endpoint for configuration processing
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm_model_name = "llama-3.3-70b-versatile"
    main_llm = ChatGroq(model_name=llm_model_name, temperature=0.1)
    guard_llm = ChatGroq(model_name=llm_model_name, temperature=0.0)

    # Guardrail Prompt Definition
    oos_template = (
        "You are a classification gatekeeper for a company HR support desk.\n"
        "Determine if the incoming user question is related to company policies, benefits, HR, or workplace guidelines.\n"
        "Reply with exactly one word: 'SAFE' if it is in-scope, or 'OUT' if it is out-of-scope.\n\n"
        "User Question: {question}\n"
        "Classification:"
    )
    oos_prompt = ChatPromptTemplate.from_template(oos_template)
    guard_chain = oos_prompt | guard_llm | StrOutputParser()

    # RAG Response Prompt Definition
    prompt_template = (
        "You are an expert HR assistant answering questions based strictly on the provided company policy text.\n"
        "If you do not know the answer or if the information is missing from the context, clearly state that you do not know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    return retriever, main_llm, guard_chain, rag_prompt, ChatPromptTemplate, StrOutputParser

try:
    retriever, main_llm, guard_chain, rag_prompt, ChatPromptTemplate, StrOutputParser = initialize_rag_system()
except Exception as e:
    st.error(f"Failed to bootstrap pipeline: {e}")
    st.stop()

# Clean UI Input Placement
with st.container():
    with st.form(key="query_form", clear_on_submit=False):
        user_query = st.text_input("📝 Enter your question regarding company policies:", placeholder="e.g., What is the company's work from home policy?")
        submit_button = st.form_submit_button(label="Search System")

# Processing and Presentation
if submit_button and user_query:
    st.markdown("### 🔍 System Evaluation & Search Results")

    with st.spinner("Analyzing document context..."):
        # Step A: Evaluate Guardrail Gating
        decision = guard_chain.invoke({"question": user_query}).strip().upper()

        if "OUT" in decision:
            st.markdown(
                '<div class="response-card">'
                '⚠️ <b>Out of Scope Notification</b><br><br>'
                'I am sorry, but I can only assist with questions regarding company policies, HR documentation, and employee guidelines.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            # Step B: Gather relevant document context block
            retrieved_docs = retriever.invoke(user_query)
            context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Format and sort unique citations cleanly
            citations = sorted(list(set(
                f"📄 {os.path.basename(doc.metadata.get('source', 'Unknown'))} — Page {doc.metadata.get('page', 0) + 1}"
                for doc in retrieved_docs
            )))

            # Step C: Generate text completion via core pipeline
            core_chain = rag_prompt | main_llm | StrOutputParser()
            answer = core_chain.invoke({"context": context_str, "question": user_query})

            # Render cleaner block container
            st.markdown(f'<div class="response-card">🤖 <b>Assistant Response:</b><br><br>{answer}</div>', unsafe_allow_html=True)

            # Render Source Expander inside structured hierarchy
            with st.expander("📚 View Document Source Citations (" + str(len(citations)) + " references found)"):
                for src in citations:
                    st.caption(src)
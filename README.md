# 🏢 Zyro Dynamics HR Support Portal

A Retrieval-Augmented Generation (RAG) assistant built with LangChain, Groq (Llama 3.3), and Streamlit to answer employee questions based on corporate HR policy documentation.

## 🚀 Features
- **Semantic Search**: Powered by FAISS and HuggingFace (`all-MiniLM-L6-v2`) embeddings.
- **Safety Guardrails**: Integrated out-of-scope intent validation gatekeeper.
- **Traceability**: Monitored via LangSmith evaluation pipelines.

## 🛠️ Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure your `.env` file with your `GROQ_API_KEY`.
3. Run the app: `streamlit run app.py`
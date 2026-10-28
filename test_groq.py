import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config(
        page_title="Legal AI Assistant", 
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Legal AI Assistant")
    st.markdown("Powered by Document Search + Groq LLM (Direct API)")

    # Load vector store
    @st.cache_resource
    def load_vector_store():
        try:
            import faiss
            import pickle
            from sentence_transformers import SentenceTransformer
            
            index = faiss.read_index("my_vector_store/faiss.index")
            with open("my_vector_store/documents.pkl", "rb") as f:
                documents = pickle.load(f)
            
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            return {
                'index': index,
                'documents': documents,
                'model': embedding_model
            }
        except Exception as e:
            st.error(f"❌ Error loading vector store: {e}")
            return None

    # Direct Groq API call
    def call_groq_api(messages, model="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None, "API key not found"
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": messages,
            "model": model,
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'], None
            else:
                return None, f"API Error {response.status_code}: {response.text}"
        except Exception as e:
            return None, f"Request failed: {str(e)}"

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Legal Assistant. I can search through legal documents and provide AI-powered explanations. Ask me anything about Indian law!"}
        ]

    # Load resources
    with st.spinner("🔄 Loading AI components..."):
        vector_store = load_vector_store()

    if not vector_store:
        st.error("Please run 'ingestion.py' first to process your legal documents.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ AI Configuration")
        
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success(f"✅ GROQ_API_KEY found ({api_key[:10]}...)")
            ai_enabled = st.toggle("Enable AI Responses", value=True)
            
            st.subheader("🤖 AI Model")
            model_options = {
                "Fast (Llama 3.1 8B)": "llama-3.1-8b-instant",
                "Quality (Llama 3.1 70B)": "llama-3.1-70b-versatile", 
                "High Quality (Mixtral)": "mixtral-8x7b-32768"
            }
            selected_model = st.selectbox("Choose model:", list(model_options.keys()), index=0)
        else:
            ai_enabled = False
            st.error("❌ GROQ_API_KEY not found")
        
        st.markdown("---")
        st.header("🔍 Search Settings")
        num_results = st.slider("Number of results", 2, 8, 4)
        
        st.markdown("---")
        st.header("🎯 Quick Legal Questions")
        
        quick_questions = {
            "What is Right to Equality?": "Explain the right to equality under Article 14-18 of Indian Constitution",
            "Fundamental Rights Overview": "Provide an overview of all fundamental rights in the Indian Constitution",
            "Citizen Duties": "What are the fundamental duties of Indian citizens under Article 51A?",
            "Constitutional Remedies": "Explain the constitutional remedies available under Article 32 and 226"
        }
        
        for label, question in quick_questions.items():
            if st.button(label, key=question, use_container_width=True):
                st.session_state.quick_question = question
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help you with Indian law?"}
            ]
            st.rerun()

        st.markdown("---")
        st.header("📊 Document Info")
        st.info(f"📚 {len(vector_store['documents'])} legal document chunks loaded")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check for quick question
    user_input = None
    if hasattr(st.session_state, 'quick_question'):
        user_input = st.session_state.quick_question
        del st.session_state.quick_question

    # Get user input
    if not user_input:
        user_input = st.chat_input("Ask a question about Indian law...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(f"**{user_input}**")

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..." if not ai_enabled else "🤔 Analyzing with AI..."):
                try:
                    # Search for relevant documents
                    query_embedding = vector_store['model'].encode([user_input])
                    import faiss
                    faiss.normalize_L2(query_embedding)
                    
                    k = min(num_results, len(vector_store['documents']))
                    scores, indices = vector_store['index'].search(query_embedding.astype('float32'), k)
                    
                    relevant_docs = []
                    sources = set()
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(vector_store['documents']):
                            doc = vector_store['documents'][idx]
                            relevant_docs.append({
                                'content': doc['content'],
                                'source': doc['source'],
                                'score': float(score)
                            })
                            sources.add(doc['source'])
                    
                    if ai_enabled and api_key and relevant_docs:
                        # AI + Documents approach
                        context = "\n\n".join([
                            f"From {doc['source']} (relevance: {doc['score']:.2f}):\n{doc['content']}" 
                            for doc in relevant_docs
                        ])
                        
                        messages = [
                            {
                                "role": "system",
                                "content": f"""You are an expert legal assistant specializing in Indian law. Use the provided legal documents to answer questions accurately.

INSTRUCTIONS:
- Answer based STRICTLY on the provided legal documents
- Structure your response clearly
- Mention specific articles/sections when available
- Explain legal concepts simply
- Acknowledge if information is incomplete
- Do not invent information not in the documents"""
                            },
                            {
                                "role": "user", 
                                "content": f"""LEGAL DOCUMENTS CONTEXT:
{context}

USER QUESTION: {user_input}

Please provide a comprehensive answer:"""
                            }
                        ]
                        
                        model_name = model_options[selected_model]
                        response, error = call_groq_api(messages, model_name)
                        
                        if error:
                            final_response = f"## ⚠️ AI Error\n\n{error}\n\n**Document Results:**\n\n"
                            # Fall back to document results
                            for i, doc in enumerate(relevant_docs, 1):
                                content = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
                                final_response += f"**{i}. {doc['source']}**\n{content}\n\n"
                        else:
                            final_response = f"{response}\n\n---\n**📚 Source Documents:** {', '.join(sources)}"
                            
                    elif relevant_docs:
                        # Document-only approach
                        final_response = f"## 🔍 Legal Information: {user_input}\n\n"
                        for i, doc in enumerate(relevant_docs, 1):
                            content = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
                            final_response += f"**{i}. 📄 {doc['source']}** (relevance: {doc['score']:.2f})\n{content}\n\n"
                        final_response += f"**📚 Sources:** {', '.join(sources)}"
                        if not ai_enabled:
                            final_response += "\n\n*💡 Enable AI responses for comprehensive explanations*"
                    else:
                        final_response = f"## ❌ No Specific Information Found\n\nNo relevant documents found for: **'{user_input}'**"
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                except Exception as e:
                    error_msg = f"## ⚠️ Error\n\n`{str(e)}`\n\nPlease try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
import streamlit as st
import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables FIRST
load_dotenv()

# Suppress warnings at the very top
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    st.set_page_config(
        page_title="Legal AI Assistant", 
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Legal AI Assistant")
    st.markdown("Powered by Document Search + Groq LLM")

    # Load vector store
    @st.cache_resource
    def load_vector_store():
        try:
            import faiss
            import numpy as np
            import pickle
            from sentence_transformers import SentenceTransformer
            
            # Load the FAISS index and documents
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

    # Load LLM (Groq) with detailed error handling
    @st.cache_resource
    def load_llm():
        try:
            from langchain_groq import ChatGroq
            
            # Get API key
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            if not groq_api_key:
                return None, "GROQ_API_KEY not found in environment variables"
            
            # Test the API key with a simple call
            try:
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=100  # Small token count for testing
                )
                
                # Test the connection with a simple prompt
                test_response = llm.invoke("Say 'Hello' in one word.")
                if hasattr(test_response, 'content'):
                    return llm, "✅ LLM initialized successfully"
                else:
                    return None, "❌ LLM response format unexpected"
                    
            except Exception as api_error:
                return None, f"❌ API Error: {str(api_error)}"
            
        except ImportError as import_error:
            return None, f"❌ Import Error: {str(import_error)}"
        except Exception as e:
            return None, f"❌ Unexpected Error: {str(e)}"

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Legal Assistant. I can search through legal documents and provide AI-powered explanations. Ask me anything about Indian law!"}
        ]

    # Load resources
    with st.spinner("🔄 Loading AI components..."):
        vector_store = load_vector_store()
        llm, llm_status = load_llm()

    if not vector_store:
        st.error("Please run 'ingestion.py' first to process your legal documents.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ AI Configuration")
        
        # Show API key status
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            st.success(f"✅ GROQ_API_KEY found ({groq_api_key[:10]}...)")
            
            # Show LLM status
            st.write(f"**LLM Status:** {llm_status}")
            
            if llm:
                ai_enabled = st.toggle("Enable AI Responses", value=True, help="Use Groq LLM for enhanced responses")
                st.success("✅ Groq LLM Connected")
                
                # Model selection
                st.subheader("🤖 AI Model")
                model_options = {
                    "Fast (Llama 3.1 8B)": "llama-3.1-8b-instant",
                    "Quality (Llama 3.1 70B)": "llama-3.1-70b-versatile", 
                    "High Quality (Mixtral)": "mixtral-8x7b-32768"
                }
                selected_model = st.selectbox("Choose model:", list(model_options.keys()), index=0)
            else:
                ai_enabled = False
                st.error("❌ LLM initialization failed")
                st.info("Check the error message above and see troubleshooting tips below.")
                
        else:
            ai_enabled = False
            st.error("❌ GROQ_API_KEY not found")
            st.info("""
            Add to .env file:
            ```
            GROQ_API_KEY=your_actual_key_here
            ```
            """)
        
        # Troubleshooting section
        st.markdown("---")
        st.header("🔧 Troubleshooting")
        
        with st.expander("Common LLM Issues"):
            st.markdown("""
            **1. Invalid API Key:**
            - Make sure your Groq API key is valid
            - Check if you have credits/usage remaining
            - Get a new key from [Groq Console](https://console.groq.com/)
            
            **2. Network Issues:**
            - Check your internet connection
            - Try again in a few minutes
            
            **3. Package Issues:**
            - Update langchain-groq: `pip install --upgrade langchain-groq`
            - Reinstall packages: `pip install langchain-groq --force-reinstall`
            """)
        
        st.markdown("---")
        
        st.header("🔍 Search Settings")
        num_results = st.slider("Number of results", 2, 8, 4, help="How many document chunks to retrieve")
        
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

        # Document info
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
                    # Step 1: Search for relevant documents
                    query_embedding = vector_store['model'].encode([user_input])
                    import faiss
                    faiss.normalize_L2(query_embedding)
                    
                    # Search in vector store
                    k = min(num_results, len(vector_store['documents']))
                    scores, indices = vector_store['index'].search(query_embedding.astype('float32'), k)
                    
                    # Get relevant documents
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
                    
                    # Step 2: Generate response based on mode
                    if ai_enabled and llm and relevant_docs:
                        # AI + Documents approach
                        context = "\n\n".join([
                            f"From {doc['source']} (relevance: {doc['score']:.2f}):\n{doc['content']}" 
                            for doc in relevant_docs
                        ])
                        
                        # Create sophisticated legal prompt
                        prompt = f"""You are an expert legal assistant specializing in Indian law. Use the provided legal documents to answer the question comprehensively.

LEGAL DOCUMENTS CONTEXT:
{context}

USER QUESTION: {user_input}

INSTRUCTIONS:
1. Provide a comprehensive answer based STRICTLY on the provided legal documents
2. Structure your response with clear headings and organization
3. Mention specific articles, sections, and laws when available
4. Explain legal concepts in simple, understandable language
5. If information is incomplete in the documents, acknowledge this limitation
6. Do not invent or assume information not present in the documents
7. Focus on accuracy, clarity, and practical understanding

RESPONSE STRUCTURE:
- Start with a clear introduction answering the question
- Break down key legal concepts with explanations
- Reference specific articles/sections from the documents
- Provide practical implications or examples when possible
- End with a concise summary

ANSWER:"""
                        
                        response = llm.invoke(prompt)
                        final_response = f"{response.content}\n\n---\n**📚 Source Documents:** {', '.join(sources)}"
                        
                    elif ai_enabled and llm and not relevant_docs:
                        # AI only (no relevant documents found)
                        prompt = f"""You are a legal assistant. The user asked: "{user_input}"

However, no specific information was found in the available legal documents about this topic.

Please provide a general response explaining that you couldn't find specific information in the documents, and suggest alternative approaches or general legal principles that might be relevant.

Be helpful but clear about the limitations."""
                        
                        response = llm.invoke(prompt)
                        final_response = f"{response.content}\n\n---\n*Note: This is a general AI response as no specific documents were found for your query.*"
                        
                    elif relevant_docs:
                        # Document-only approach (no AI)
                        final_response = f"## 🔍 Legal Information: {user_input}\n\n"
                        
                        for i, doc in enumerate(relevant_docs, 1):
                            content = doc['content']
                            source = doc['source']
                            score = doc['score']
                            
                            # Clean and format content
                            if len(content) > 400:
                                content = content[:400] + "..."
                            
                            final_response += f"**{i}. 📄 {source}** (relevance: {score:.2f})\n{content}\n\n"
                        
                        final_response += f"**📚 Sources:** {', '.join(sources)}"
                        final_response += "\n\n*💡 Enable AI responses in sidebar for comprehensive explanations*"
                        
                    else:
                        # No documents found
                        final_response = f"## ❌ No Specific Information Found\n\n"
                        final_response += f"I couldn't find specific information about **'{user_input}'** in the legal documents.\n\n"
                        final_response += "**💡 Suggestions:**\n"
                        final_response += "- Try using different keywords or legal terminology\n"
                        final_response += "- Ask about fundamental rights, constitutional articles, or specific laws\n"
                        final_response += "- Use the quick question buttons in the sidebar\n"
                        if not ai_enabled:
                            final_response += "- Fix LLM configuration to enable AI responses"
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                except Exception as e:
                    error_msg = f"## ⚠️ Error\n\nI encountered an error: `{str(e)}`\n\nPlease try again with a different question."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption("💡 **AI Assistant Features:** Document retrieval + LLM explanations | Source citation | Legal context awareness")
    
    with col2:
        if llm and ai_enabled:
            st.success("🤖 AI Enabled")
        else:
            st.warning("🔍 Search Only")

if __name__ == "__main__":
    main()
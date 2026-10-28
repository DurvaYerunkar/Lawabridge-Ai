import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def embed_and_save_documents():
    """
    Load PDF documents from LEGAL-DATA folder, split them into chunks,
    create embeddings and save to FAISS vector store.
    """
    print("🚀 Starting Legal Document Ingestion Process...")
    print("=" * 50)
    
    # Check if LEGAL-DATA directory exists
    data_dir = Path("./LEGAL-DATA")
    if not data_dir.exists():
        print(f"❌ ERROR: Directory '{data_dir}' not found!")
        print("Please create a 'LEGAL-DATA' folder and add your PDF files.")
        return False
    
    # List all PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ ERROR: No PDF files found in LEGAL-DATA directory!")
        print("Please add PDF files to the LEGAL-DATA folder.")
        return False
    
    print(f"📚 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()
    
    try:
        # Step 1: Initialize embeddings - Use direct import
        print("🔄 Step 1: Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully!")
        
        # Step 2: Load PDF documents using PyPDF directly
        print("\n📄 Step 2: Loading PDF documents...")
        from PyPDF2 import PdfReader
        
        documents = []
        for pdf_file in pdf_files:
            try:
                print(f"   Reading: {pdf_file.name}")
                reader = PdfReader(str(pdf_file))
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_num + 1}: {page_text}\n"
                
                if text.strip():
                    # Create document chunks
                    words = text.split()
                    chunk_size = 500  # words per chunk
                    
                    for i in range(0, len(words), chunk_size):
                        chunk_text = ' '.join(words[i:i + chunk_size])
                        if len(chunk_text.strip()) > 50:  # Only keep substantial chunks
                            documents.append({
                                'content': chunk_text,
                                'source': pdf_file.name,
                                'chunk_id': len(documents)
                            })
                    
                    print(f"   ✅ Created chunks from {pdf_file.name}")
                else:
                    print(f"   ⚠️  No text extracted from: {pdf_file.name}")
                    
            except Exception as e:
                print(f"   ❌ Error loading {pdf_file.name}: {str(e)}")
        
        if not documents:
            print("❌ ERROR: No content could be extracted from PDF files!")
            return False
            
        print(f"✅ Successfully created {len(documents)} text chunks")
        
        # Step 3: Create embeddings
        print("\n🧠 Step 3: Creating embeddings...")
        contents = [doc['content'] for doc in documents]
        embeddings = embedding_model.encode(contents)
        print(f"✅ Created embeddings with shape: {embeddings.shape}")
        
        # Step 4: Create and save FAISS index
        print("\n💾 Step 4: Saving vector store...")
        try:
            import faiss
            import numpy as np
            import pickle
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Using Inner Product for similarity
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            # Save index and documents
            os.makedirs("my_vector_store", exist_ok=True)
            faiss.write_index(index, "my_vector_store/faiss.index")
            
            with open("my_vector_store/documents.pkl", "wb") as f:
                pickle.dump(documents, f)
            
            with open("my_vector_store/embedding_model.pkl", "wb") as f:
                pickle.dump('all-MiniLM-L6-v2', f)
            
            print("✅ Vector store saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")
            # Fallback: Save as pickle
            print("🔄 Using fallback storage method...")
            os.makedirs("my_vector_store", exist_ok=True)
            with open("my_vector_store/documents.pkl", "wb") as f:
                pickle.dump(documents, f)
            with open("my_vector_store/embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)
            print("✅ Documents and embeddings saved as pickle files!")
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Summary:")
        print(f"   - PDF files processed: {len(pdf_files)}")
        print(f"   - Text chunks created: {len(documents)}")
        print(f"   - Embedding dimension: {embeddings.shape[1]}")
        print(f"   - Vector store location: ./my_vector_store/")
        print()
        print("➡️ Next step: Run 'streamlit run app.py' to start the Legal Assistant!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during ingestion process:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check if all required packages are installed"""
    print("🔍 Checking environment...")
    required_packages = {
        'PyPDF2': 'pypdf2',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (install with: pip install {pip_name})")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed!")
    return True

if __name__ == "__main__":
    print("⚖️ Legal Document Ingestion Script")
    print("This script will process PDF files in the LEGAL-DATA folder")
    print()
    
    # Check environment first
    if not check_environment():
        sys.exit(1)
    
    print()
    
    # Run the ingestion process
    success = embed_and_save_documents()
    
    if not success:
        print("\n❌ Ingestion failed! Please check the errors above.")
        sys.exit(1)
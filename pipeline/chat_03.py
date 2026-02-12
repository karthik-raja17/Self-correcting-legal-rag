#!/usr/bin/env python3
import os
import sys
import chromadb
from dotenv import load_dotenv
from groq import Groq

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

def main():
    load_dotenv()
    
    print(f"📁 Chroma DB: {CHROMA_DB_PATH}")
    
    # 1. Load embedding model
    print("🔄 Loading BGE-M3 (1024-dim)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    # 2. Connect to ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        chroma_collection = db.get_collection("solar_ppa_collection")
        print(f"✅ Collection found. Count: {chroma_collection.count()}")
    except:
        print("❌ Collection not found. Run index_02.py first.")
        return
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. Build index (only for retriever)
    print("🔄 Building VectorStoreIndex...")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # 4. Retriever
    retriever = index.as_retriever(similarity_top_k=5)
    
    # 5. Direct Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        return
    
    client = Groq(api_key=groq_api_key)
    print("✅ Direct Groq client ready")
    
    # 6. System prompt
    system_prompt = (
        "You are an expert Solar Energy Legal Consultant. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain the answer, say 'I cannot find this information in the documents.' "
        "Respond in the same language as the question."
    )
    
    print("\n☀️ SOLAR CHAT IS ONLINE (Custom RAG with Groq)")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("💬 Ask a question: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            # --- RETRIEVAL ---
            nodes = retriever.retrieve(query)
            print(f"\n🔍 Retrieved {len(nodes)} relevant passages")
            if len(nodes) == 0:
                print("❌ No relevant documents found.")
                continue
            
            # Format context from top nodes
            context = "\n\n".join([node.text for node in nodes[:3]])
            print(f"   Top passage score: {nodes[0].score:.4f}")
            print(f"   Preview: {nodes[0].text[:150]}...")
            
            # --- GENERATION ---
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            answer = completion.choices[0].message.content
            print(f"\n📢 ANSWER:\n{answer}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

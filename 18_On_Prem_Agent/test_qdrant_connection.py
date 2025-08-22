#!/usr/bin/env python3
"""
Test Qdrant connection and create vector store
"""
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import JSONLoader, DirectoryLoader
import json

def test_qdrant_connection():
    print("Testing Qdrant connection...")
    
    # Test connection to Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient("http://localhost:6333")
        collections = client.get_collections()
        print(f"✅ Successfully connected to Qdrant!")
        print(f"Available collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False

def create_vector_store():
    print("\nCreating vector store...")
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        print("✅ Embeddings initialized")
        
        # Load documents
        json_loader = DirectoryLoader(
            path="./data/data",
            glob="**/*.json",
            loader_cls=JSONLoader,
            loader_kwargs={"jq_schema": "..", "text_content": False}
        )
        json_documents = json_loader.load()
        print(f"✅ Loaded {len(json_documents)} documents")
        
        # Create vector store
        url = "http://127.0.0.1:6333"
        qdrant = QdrantVectorStore.from_documents(
            json_documents,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="DnD_Documents",
        )
        print("✅ Vector store created successfully!")
        
        # Test retrieval
        query = "What is the Agent of Order?"
        docs = qdrant.similarity_search(query, k=2)
        print(f"✅ Test retrieval successful! Found {len(docs)} documents")
        
        return qdrant
        
    except Exception as e:
        print(f"❌ Failed to create vector store: {e}")
        return None

if __name__ == "__main__":
    if test_qdrant_connection():
        vector_store = create_vector_store()
        if vector_store:
            print("\n🎉 Qdrant setup completed successfully!")
        else:
            print("\n❌ Vector store creation failed")
    else:
        print("\n❌ Qdrant connection failed")

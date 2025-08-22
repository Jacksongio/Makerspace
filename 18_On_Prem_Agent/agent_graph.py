#!/usr/bin/env python3
"""
On-Prem Agent with RAG Node powered by Qdrant
"""
import os
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
import json

class RAGNode:
    """RAG Node that retrieves relevant documents from Qdrant"""
    
    def __init__(self, collection_name: str = "DnD_Documents"):
        self.collection_name = collection_name
        self.vector_store = None
        self.embeddings = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector store and embeddings"""
        try:
            self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            # For existing collections, we need to connect to the client first
            from qdrant_client import QdrantClient
            client = QdrantClient("http://127.0.0.1:6333")
            
            # Check if collection exists
            if client.get_collection(self.collection_name):
                self.vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )
                print(f"✅ RAG Node initialized with existing collection: {self.collection_name}")
            else:
                print(f"❌ Collection {self.collection_name} not found. Please run the setup first.")
                raise Exception(f"Collection {self.collection_name} not found")
        except Exception as e:
            print(f"❌ Failed to initialize RAG Node: {e}")
            raise
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents for a query"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return retrieved context"""
        query = inputs.get("query", "")
        if not query:
            return {"context": [], "query": query}
        
        context = self.retrieve(query)
        return {
            "context": context,
            "query": query,
            "retrieved_docs": len(context)
        }

class LLMNode:
    """LLM Node that processes queries with context"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model_name = model_name
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Ollama LLM"""
        try:
            self.llm = OllamaLLM(model=self.model_name)
            print(f"✅ LLM Node initialized with model: {self.model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize LLM Node: {e}")
            raise
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and generate response"""
        query = inputs.get("query", "")
        context = inputs.get("context", [])
        
        if not context:
            # If no context, just answer the query
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Always provide direct, clear answers without showing your thinking process."),
                ("human", "Answer the following question directly and clearly: {query}")
            ])
        else:
            # If context available, use RAG approach
            context_text = "\n\n".join(context)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Always provide direct, clear answers without showing your thinking process. Never use <think> tags or explain your reasoning step by step."),
                ("human", """Use the following context to answer the question. If the context doesn't contain relevant information, say so and provide a general answer.

Context:
{context}

Question: {query}

Answer:""")
            ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            if context:
                response = chain.invoke({"context": context_text, "query": query})
            else:
                response = chain.invoke({"query": query})
            
            return {
                "response": response,
                "query": query,
                "context_used": len(context),
                "model": self.model_name
            }
        except Exception as e:
            return {
                "response": f"Error generating response: {e}",
                "query": query,
                "context_used": len(context),
                "model": self.model_name
            }

class AgentGraph:
    """Main Agent Graph that orchestrates the RAG and LLM nodes"""
    
    def __init__(self):
        self.rag_node = RAGNode()
        self.llm_node = LLMNode()
        self._build_graph()
    
    def _build_graph(self):
        """Build the processing graph"""
        # Create the graph: Query -> RAG -> LLM -> Response
        self.graph = (
            RunnablePassthrough.assign(
                context=self.rag_node
            )
            | self.llm_node
        )
        print("✅ Agent Graph built successfully!")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the entire graph"""
        try:
            result = self.graph.invoke({"query": query})
            return result
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "response": "Failed to process query"
            }
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure"""
        return {
            "nodes": [
                {
                    "name": "RAG Node",
                    "type": "Vector Store Retrieval",
                    "backend": "Qdrant",
                    "collection": self.rag_node.collection_name
                },
                {
                    "name": "LLM Node", 
                    "type": "Language Model",
                    "model": self.llm_node.model_name,
                    "backend": "Ollama"
                }
            ],
            "flow": "Query -> RAG Retrieval -> Context Enrichment -> LLM Generation -> Response",
            "total_documents": self._get_document_count()
        }
    
    def _get_document_count(self) -> int:
        """Get the total number of documents in the vector store"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient("http://localhost:6333")
            collection_info = client.get_collection(self.rag_node.collection_name)
            return collection_info.points_count
        except:
            return "Unknown"

def main():
    """Main function to demonstrate the agent graph"""
    print("🚀 Initializing On-Prem Agent with RAG...")
    
    try:
        # Initialize the agent graph
        agent = AgentGraph()
        
        # Display graph information
        print("\n📊 Agent Graph Information:")
        info = agent.get_graph_info()
        for node in info["nodes"]:
            print(f"  • {node['name']}: {node['type']} ({node['backend']})")
        print(f"  • Flow: {info['flow']}")
        print(f"  • Total Documents: {info['total_documents']}")
        
        # Interactive query loop
        print("\n💬 Interactive Query Mode (type 'quit' to exit):")
        while True:
            query = input("\n🔍 Enter your query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🔄 Processing: {query}")
            result = agent.process_query(query)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n📚 Context used: {result['context_used']} documents")
                print(f"🤖 Response: {result['response']}")
                print(f"🔧 Model: {result['model']}")
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure Qdrant is running and Ollama models are available")

if __name__ == "__main__":
    main()

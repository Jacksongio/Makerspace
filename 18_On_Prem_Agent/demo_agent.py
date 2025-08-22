#!/usr/bin/env python3
"""
Demo script for the On-Prem Agent with RAG
"""
from agent_graph import AgentGraph

def demo_agent():
    """Demonstrate the agent capabilities"""
    print("🎯 On-Prem Agent RAG Demo")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("Initializing agent...")
        agent = AgentGraph()
        
        # Show graph info
        info = agent.get_graph_info()
        print(f"\n📊 Agent Status:")
        print(f"  • RAG Node: {info['nodes'][0]['backend']} ({info['nodes'][0]['collection']})")
        print(f"  • LLM Node: {info['nodes'][1]['backend']} ({info['nodes'][1]['model']})")
        print(f"  • Documents: {info['total_documents']}")
        
        # Test queries
        test_queries = [
            "What is the Agent of Order?",
            "Tell me about D&D monsters",
            "What are some magical items?",
            "Explain character classes"
        ]
        
        print(f"\n🧪 Testing with {len(test_queries)} queries...")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            result = agent.process_query(query)
            
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   📚 Context: {result['context_used']} docs")
                print(f"   🤖 Response: {result['response'][:200]}...")
                print(f"   🔧 Model: {result['model']}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_agent()

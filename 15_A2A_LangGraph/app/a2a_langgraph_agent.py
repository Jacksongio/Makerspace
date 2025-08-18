"""Activity #1: LangGraph Graph to use A2A application through A2A protocol.

I built this to demonstrate how LangGraph can create agents that communicate with A2A servers through standardized protocols.
"""
import asyncio
import logging
from typing import Dict, Any, Annotated, TypedDict, List
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage


class AgentState(TypedDict):
    """State schema for tracking conversation context and A2A communication state."""
    messages: Annotated[List, add_messages]
    a2a_responses: List[str]
    current_task_id: str
    current_context_id: str
    a2a_query: str


class A2ATool:
    """Custom tool I built to handle A2A protocol communication and message formatting."""
    
    def __init__(self, base_url: str = "http://localhost:10000"):
        self.base_url = base_url
        self.client = None
        self.httpx_client = None
        
    async def initialize(self):
        """Discovers the A2A server's capabilities by fetching the agent card and initializing the client."""
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        resolver = A2ACardResolver(httpx_client=self.httpx_client, base_url=self.base_url)
        agent_card = await resolver.get_agent_card()
        self.client = A2AClient(httpx_client=self.httpx_client, agent_card=agent_card)
    
    async def send_message(self, message: str, task_id: str = None, context_id: str = None) -> Dict[str, Any]:
        """Formats and sends A2A protocol messages with proper task/context ID handling for conversation continuity."""
        if not self.client:
            await self.initialize()
            
        payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': message}],
                'message_id': uuid4().hex,
            },
        }
        
        if task_id:
            payload['message']['task_id'] = task_id
        if context_id:
            payload['message']['context_id'] = context_id
            
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        response = await self.client.send_message(request)
        return response.model_dump(mode='json', exclude_none=True)
    
    async def cleanup(self):
        """Properly closes HTTP connections to prevent resource leaks and hanging connections."""
        if self.httpx_client:
            await self.httpx_client.aclose()


def agent_node(state: AgentState, a2a_tool: A2ATool) -> Dict[str, Any]:
    """First workflow node that processes user input and prepares the query for A2A communication."""
    messages = state["messages"]
    user_input = messages[-1].content if messages else ""
    
    return {
        "messages": [AIMessage(content=f"Processing: {user_input}")],
        "a2a_query": user_input,
        "task_id": state.get("current_task_id"),
        "context_id": state.get("current_context_id"),
        "a2a_responses": state.get("a2a_responses", [])
    }


async def a2a_communication_node(state: AgentState, a2a_tool: A2ATool) -> Dict[str, Any]:
    """Second workflow node that handles the actual A2A server communication and response parsing."""
    a2a_query = state.get("a2a_query", "")
    task_id = state.get("task_id")
    context_id = state.get("current_context_id")
    
    print(f"DEBUG: a2a_query = {a2a_query}")
    print(f"DEBUG: task_id = {task_id}")
    print(f"DEBUG: context_id = {context_id}")
    
    if not a2a_query:
        return {
            "messages": [AIMessage(content="No query to send")],
            "a2a_responses": state.get("a2a_responses", [])
        }
    
    try:
        response = await a2a_tool.send_message(a2a_query, task_id, context_id)
        
        # Extract response content and IDs
        new_task_id = response.get("result", {}).get("id")
        new_context_id = response.get("result", {}).get("contextId")
        
        response_content = "No response content"
        if "result" in response and "artifacts" in response["result"]:
            artifacts = response["result"]["artifacts"]
            if artifacts and "parts" in artifacts[0]:
                parts = artifacts[0]["parts"]
                if parts and "text" in parts[0]:
                    response_content = parts[0]["text"]
        
        # Ensure all state fields are updated
        updated_state = {
            "messages": [AIMessage(content=f"A2A Response: {response_content}")],
            "a2a_responses": [response_content],
            "current_task_id": new_task_id,
            "current_context_id": new_context_id,
            "a2a_query": a2a_query  # Preserve the query
        }
        
        print(f"DEBUG: Returning state with a2a_responses: {updated_state['a2a_responses']}")
        return updated_state
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "a2a_responses": state.get("a2a_responses", []) + [error_msg]
        }


def build_a2a_agent_graph(a2a_tool: A2ATool):
    """Constructs the LangGraph workflow with two nodes for processing and A2A communication."""
    graph = StateGraph(AgentState)
    
    # Create proper node functions that capture the a2a_tool
    def agent_wrapper(state):
        return agent_node(state, a2a_tool)
    
    async def a2a_communication_wrapper(state):
        return await a2a_communication_node(state, a2a_tool)
    
    graph.add_node("agent", agent_wrapper)
    graph.add_node("a2a_communication", a2a_communication_wrapper)
    
    graph.set_entry_point("agent")
    graph.add_edge("agent", "a2a_communication")
    graph.add_edge("a2a_communication", END)
    
    return graph.compile()


async def main():
    """Main function that demonstrates the complete A2A LangGraph agent workflow."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize A2A tool
    a2a_tool = A2ATool()
    try:
        await a2a_tool.initialize()
        
        # Build and run graph
        graph = build_a2a_agent_graph(a2a_tool)
        
        initial_state = {
            "messages": [HumanMessage(content="What are the latest developments in artificial intelligence?")],
            "a2a_responses": [],
            "current_task_id": None,
            "current_context_id": None,
            "a2a_query": ""
        }
        
        result = await graph.ainvoke(initial_state)
        
        print("A2A LangGraph Agent Results:")
        print(f"Messages: {len(result['messages'])}")
        print(f"A2A Responses: {len(result['a2a_responses'])}")
        print(f"Task ID: {result.get('current_task_id')}")
        print(f"Context ID: {result.get('current_context_id')}")
        
        # Debug: Print the full result state
        print(f"\nDEBUG: Full result state keys: {list(result.keys())}")
        print(f"DEBUG: a2a_responses type: {type(result.get('a2a_responses'))}")
        print(f"DEBUG: a2a_responses content: {result.get('a2a_responses')}")
        
        for message in result['messages']:
            role = "User" if isinstance(message, HumanMessage) else "Agent"
            print(f"\n{role}: {message.content}")
    
    finally:
        # Clean up resources
        await a2a_tool.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Activity 2: Simple LangGraph Application with MCP Server Integration
Uses actual LangGraph StateGraph, nodes, edges, and state management
"""

import asyncio
from typing import Dict, List, Any, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """State for our LangGraph agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    tools_used: List[str]

class MCPLangGraphAgent:
    """Simple LangGraph agent that uses MCP server tools"""
    
    def __init__(self):
        self.mcp_client = None
        self.tools = []
        self.app = None
    
    async def initialize(self):
        """Initialize MCP client and build the LangGraph workflow"""
        print("Initializing MCP LangGraph Agent...")
        
        # Connect to MCP server
        self.mcp_client = MultiServerMCPClient({
            "local-server": {
                "command": "uv",
                "args": ["run", "server.py"],
                "cwd": "/Users/jacksongio/Makerspace/13_MCP/AIE7-MCP-Session",
                "transport": "stdio"
            }
        })
        
        # Load tools from MCP server
        self.tools = await self.mcp_client.get_tools()
        print(f"Loaded {len(self.tools)} MCP tools: {[tool.name for tool in self.tools]}")
        
        # Build LangGraph workflow
        self.build_graph()
        print("LangGraph workflow built!")
    
    def build_graph(self):
        """Build the LangGraph StateGraph with nodes and edges"""
        
        # Create StateGraph
        graph = StateGraph(AgentState)
        
        # Add nodes to the graph
        graph.add_node("tool_selector", self.select_tools)
        graph.add_node("tool_executor", ToolNode(self.tools))
        graph.add_node("response_formatter", self.format_response)
        
        # Define edges (workflow flow)
        graph.add_edge(START, "tool_selector")
        graph.add_edge("tool_selector", "tool_executor")
        graph.add_edge("tool_executor", "response_formatter")
        graph.add_edge("response_formatter", END)
        
        # Compile the graph
        self.app = graph.compile()
    
    async def select_tools(self, state: AgentState) -> AgentState:
        """LangGraph node: Analyze input and select appropriate tools"""
        
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            return state
        
        content = last_message.content.lower()
        tool_calls = []
        
        # Determine which tool to use based on user input
        if any(word in content for word in ["weather", "temperature", "forecast"]):
            # Extract location for weather
            location = self._extract_location(content) or "London"
            tool_calls.append({
                "name": "get_weather",
                "args": {"location": location, "include_forecast": "forecast" in content},
                "id": "weather_1"
            })
            state["tools_used"].append("get_weather")
        
        elif any(word in content for word in ["roll", "dice", "d6", "d20"]):
            # Extract dice notation
            notation = self._extract_dice_notation(content) or "2d6"
            tool_calls.append({
                "name": "roll_dice", 
                "args": {"notation": notation, "num_rolls": 1},
                "id": "dice_1"
            })
            state["tools_used"].append("roll_dice")
        
        elif any(word in content for word in ["search", "find", "look"]):
            # Extract search query
            query = self._extract_search_query(content)
            if query:
                tool_calls.append({
                    "name": "web_search",
                    "args": {"query": query},
                    "id": "search_1"
                })
                state["tools_used"].append("web_search")
        
        # Add tool calls to conversation if any were selected
        if tool_calls:
            ai_message = AIMessage(content="I'll help you with that!", tool_calls=tool_calls)
            state["messages"].append(ai_message)
        else:
            # No tools needed - provide help
            help_msg = AIMessage(content="""I can help you with:
Weather: "weather in Tokyo" or "forecast for Paris"
Dice: "roll 2d6" or "roll 1d20"
Search: "search for Python tutorials"

What would you like to do?""")
            state["messages"].append(help_msg)
        
        return state
    
    async def format_response(self, state: AgentState) -> AgentState:
        """LangGraph node: Format the final response based on tool results"""
        
        # Find the last tool message
        tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        
        if tool_messages and state["tools_used"]:
            last_result = tool_messages[-1].content
            last_tool = state["tools_used"][-1]
            
            # Format response based on tool type
            if last_tool == "get_weather":
                response = f"Weather Information:\n\n{last_result}"
            elif last_tool == "roll_dice":
                response = f"Dice Roll Result:\n{last_result}"
            elif last_tool == "web_search":
                response = f"Search Results:\n{last_result[:400]}..."
            else:
                response = f"Tool Result:\n{last_result}"
            
            # Add formatted response
            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)
        
        return state
    
    def _extract_location(self, text: str) -> str:
        """Helper: Extract location from text"""
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                return words[i + 1]
        return None
    
    def _extract_dice_notation(self, text: str) -> str:
        """Helper: Extract dice notation from text"""
        import re
        match = re.search(r'(\d+d\d+)', text.lower())
        return match.group(1) if match else None
    
    def _extract_search_query(self, text: str) -> str:
        """Helper: Extract search query from text"""
        query = text.lower()
        for word in ["search", "find", "look up"]:
            query = query.replace(word, "")
        return query.strip()
    
    async def process_message(self, user_input: str) -> str:
        """Process a user message through the LangGraph workflow"""
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "tools_used": []
        }
        
        # Run the LangGraph workflow
        final_state = await self.app.ainvoke(initial_state)
        
        # Return the last AI message
        ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        return ai_messages[-1].content if ai_messages else "No response generated."

async def run_interactive():
    """Run interactive mode"""
    
    print("Activity 2: Interactive LangGraph MCP Agent")
    print("=" * 50)
    
    agent = MCPLangGraphAgent()
    await agent.initialize()
    
    print("\nChat with the LangGraph agent (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("Agent: ", end="")
            response = await agent.process_message(user_input)
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

def main():
    """Main function"""
    asyncio.run(run_interactive())

if __name__ == "__main__":
    main()
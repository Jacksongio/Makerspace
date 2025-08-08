#!/usr/bin/env python3
"""
Proper test script to verify MCP server functionality.
This tests the server by connecting as an MCP client.
"""

from mcp import ClientSession, StdioServerParameters
import asyncio
import subprocess

async def test_mcp_server():
    """Test the MCP server by connecting as a client."""
    print("🧪 Testing MCP Server...")
    print("=" * 50)
    
    # Start the server process
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "server.py"],
        cwd="/Users/jacksongio/Makerspace/13_MCP/AIE7-MCP-Session"
    )
    
    async with ClientSession(server_params) as session:
        # Initialize the session
        await session.initialize()
        
        # List available tools
        tools_result = await session.list_tools()
        print(f"📋 Available tools: {len(tools_result.tools)}")
        for tool in tools_result.tools:
            print(f"  🔧 {tool.name}: {tool.description}")
        
        print("\n" + "="*50)
        
        # Test weather tool
        print("🌤️  Testing Weather Tool...")
        try:
            weather_result = await session.call_tool("get_weather", {
                "location": "New York",
                "include_forecast": False
            })
            print("✅ Weather tool result:")
            print(weather_result.content[0].text[:200] + "..." if len(weather_result.content[0].text) > 200 else weather_result.content[0].text)
        except Exception as e:
            print(f"❌ Weather tool error: {e}")
        
        print("\n" + "-"*30)
        
        # Test dice tool
        print("🎲 Testing Dice Tool...")
        try:
            dice_result = await session.call_tool("roll_dice", {
                "notation": "2d6",
                "num_rolls": 1
            })
            print("✅ Dice tool result:")
            print(dice_result.content[0].text)
        except Exception as e:
            print(f"❌ Dice tool error: {e}")
        
        print("\n" + "-"*30)
        
        # Test web search (if API key available)
        print("🌐 Testing Web Search Tool...")
        try:
            search_result = await session.call_tool("web_search", {
                "query": "MCP Model Context Protocol"
            })
            print("✅ Web search tool result:")
            print(search_result.content[0].text[:200] + "..." if len(search_result.content[0].text) > 200 else search_result.content[0].text)
        except Exception as e:
            print(f"⚠️  Web search tool (may need API key): {e}")
    
    print("\n" + "="*50)
    print("✅ MCP Server test completed!")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
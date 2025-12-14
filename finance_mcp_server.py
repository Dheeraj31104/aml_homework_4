#!/usr/bin/env python3
"""
MCP Server with Finance Tools
Run: python finance_mcp_server.py
"""

import json
import sys
import requests
from bs4 import BeautifulSoup
from typing import Any, Dict

class FinanceMCPServer:
    """MCP Server with 3 finance tools"""
    
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
    
    def register_tool(self, func, description: str, input_schema: Dict[str, Any]):
        """Register a tool"""
        self.tools[func.__name__] = {
            "function": func,
            "description": description,
            "input_schema": input_schema
        }
    
    def list_tools(self) -> list:
        """List available tools"""
        tools_list = []
        for name, info in self.tools.items():
            tools_list.append({
                "name": name,
                "description": info["description"],
                "inputSchema": info["input_schema"]
            })
        return tools_list
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            tool_func = self.tools[tool_name]["function"]
            result = tool_func(**arguments)
            return result
        except Exception as e:
            return {"error": f"Tool error: {str(e)}"}
    
    def handle_request(self, request_str: str) -> str:
        """Handle JSON-RPC request"""
        try:
            request = json.loads(request_str)
        except:
            return json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
        
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})
        
        response = {"jsonrpc": "2.0", "id": request_id}
        
        if method == "tools/list":
            response["result"] = {"tools": self.list_tools()}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            response["result"] = result
        elif method == "initialize":
            response["result"] = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": self.name, "version": "1.0.0"}}
        else:
            response["error"] = {"code": -32601, "message": "Method not found"}
        
        return json.dumps(response)
    
    def run(self):
        """Run server"""
        sys.stderr.write(f"\n{'='*70}\n")
        sys.stderr.write(f"  MCP SERVER: Finance Tools\n")
        sys.stderr.write(f"{'='*70}\n\n")
        sys.stderr.write(f"Available Tools:\n")
        sys.stderr.write(f"  1. get_stock_price(symbol) - Real stock prices\n")
        sys.stderr.write(f"  2. scrape_finance_news(topic) - Finance news scraping\n")
        sys.stderr.write(f"  3. calculate_investment(initial, rate, years) - Investment math\n\n")
        sys.stderr.write(f"Starting MCP server...\n")
        sys.stderr.write(f"{'-'*70}\n\n")
        sys.stderr.flush()
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                response = self.handle_request(line.strip())
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass

# ============================================================================
# TOOL 1: Get Stock Price
# ============================================================================

def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get real-time stock price"""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(symbol.upper())
        hist = stock.history(period="5d")
        
        if hist.empty:
            return {"error": f"Stock '{symbol}' not found"}
        
        current_price = hist["Close"].iloc[-1]
        previous_price = hist["Close"].iloc[-2]
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "52_week_high": round(stock.info.get("fiftyTwoWeekHigh", 0), 2),
            "52_week_low": round(stock.info.get("fiftyTwoWeekLow", 0), 2),
        }
    except ImportError:
        return {"error": "yfinance not installed"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# TOOL 2: Scrape Finance News (Web Scraping!)
# ============================================================================

def scrape_finance_news(topic: str) -> Dict[str, Any]:
    """Scrape finance news from Wikipedia"""
    try:
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"Topic '{topic}' not found"}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get title
        title = soup.find('h1', class_='firstHeading')
        title_text = title.get_text() if title else topic
        
        # Get content
        content_div = soup.find('div', id='mw-content-text')
        paragraphs = content_div.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs[:5]])
        content = ' '.join(content.split())
        
        return {
            "topic": title_text,
            "url": url,
            "content": content[:500],
            "length": len(content)
        }
        
    except Exception as e:
        return {"error": f"Scraping failed: {str(e)}"}

# ============================================================================
# TOOL 3: Calculate Investment Returns
# ============================================================================

def calculate_investment(initial: float, annual_rate: float, years: int) -> Dict[str, Any]:
    """Calculate compound investment returns"""
    try:
        if initial <= 0 or years <= 0:
            return {"error": "Invalid input"}
        
        rate_decimal = annual_rate / 100
        final_amount = initial * ((1 + rate_decimal) ** years)
        total_gain = final_amount - initial
        
        return {
            "initial": initial,
            "rate_percent": annual_rate,
            "years": years,
            "final_amount": round(final_amount, 2),
            "total_gain": round(total_gain, 2),
            "total_return_percent": round((total_gain / initial) * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    server = FinanceMCPServer("Finance MCP Server")
    
    # Register tools
    server.register_tool(
        get_stock_price,
        "Get real-time stock price",
        {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
    )
    
    server.register_tool(
        scrape_finance_news,
        "Scrape finance topic from Wikipedia",
        {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
    )
    
    server.register_tool(
        calculate_investment,
        "Calculate investment returns",
        {"type": "object", "properties": {
            "initial": {"type": "number"},
            "annual_rate": {"type": "number"},
            "years": {"type": "integer"}
        }, "required": ["initial", "annual_rate", "years"]}
    )
    
    # Run
    server.run()

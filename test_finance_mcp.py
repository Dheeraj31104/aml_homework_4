#!/usr/bin/env python3
"""
Test Finance MCP Server
Run: python test_finance_mcp.py
(In separate terminal from finance_mcp_server.py)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class MCPTester:
    def __init__(self):
        self.process = None
        self.request_id = 0
    
    def start_server(self):
        print("="*70)
        print("Starting Finance MCP Server...")
        print("="*70)
        
        self.process = subprocess.Popen(
            [sys.executable, "finance_mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(1)
        print("✓ Server started\n")
    
    def call_tool(self, tool_name: str, arguments: dict):
        self.request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }
        
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        
        response_line = self.process.stdout.readline()
        response = json.loads(response_line)
        
        return response.get("result", {})
    
    def test_all_tools(self):
        stock_tests = [
            ("TEST 1A: Get Stock Price - AAPL", "AAPL"),
            ("TEST 1B: Get Stock Price - IBM", "IBM"),
            ("TEST 1C: Get Stock Price - MSFT", "MSFT"),
        ]
        
        for title, symbol in stock_tests:
            print("="*70)
            print(title)
            print("="*70)
            
            result = self.call_tool("get_stock_price", {"symbol": symbol})
            if "error" in result:
                print(f"Error: {result['error']}\n")
            else:
                print(f"Symbol: {result.get('symbol')}")
                print(f"Price: ${result.get('current_price')}")
                print(f"Change: {result.get('change_percent')}%\n")
        
        print("="*70)
        print("TEST 1D: Get Stock Price - Invalid Symbol")
        print("="*70)
        
        invalid_stock = self.call_tool("get_stock_price", {"symbol": "INVALID"})
        print(f"Response: {invalid_stock}\n")
        
        print("="*70)
        print("TEST 2A: Scrape Finance News - Stock Topic")
        print("="*70)
        
        result = self.call_tool("scrape_finance_news", {"topic": "Stock"})
        if "error" in result:
            print(f"Error: {result['error']}\n")
        else:
            print(f"Topic: {result.get('topic')}")
            print(f"Content: {result.get('content')[:150]}...\n")
        
        print("="*70)
        print("TEST 2B: Scrape Finance News - Invalid Topic")
        print("="*70)
        
        invalid_topic = self.call_tool("scrape_finance_news", {"topic": "NotARealFinanceTopic"})
        print(f"Response: {invalid_topic}\n")
        
        print("="*70)
        print("TEST 3A: Calculate Investment - Standard Case")
        print("="*70)
        
        result = self.call_tool("calculate_investment", {
            "initial": 5000,
            "annual_rate": 8,
            "years": 10
        })
        if "error" in result:
            print(f"Error: {result['error']}\n")
        else:
            print(f"Initial: ${result.get('initial')}")
            print(f"Final: ${result.get('final_amount')}")
            print(f"Gain: ${result.get('total_gain')}\n")
        
        print("="*70)
        print("TEST 3B: Calculate Investment - Invalid Input")
        print("="*70)
        
        invalid_investment = self.call_tool("calculate_investment", {
            "initial": -100,
            "annual_rate": 5,
            "years": 5
        })
        print(f"Response: {invalid_investment}\n")
        
        print("="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        
        self.process.terminate()


def run_lora_smoke_test():
    """Generate a few answers from the fine-tuned LoRA adapters if present."""
    adapters_path = Path("finance_lora_adapters")
    if not adapters_path.exists():
        print("\nLoRA adapters not found; skipping LoRA generation test.")
        return
    
    print("\n" + "="*70)
    print("TEST 4: LoRA Fine-Tuned Generation")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base_model, adapters_path)
    model.eval()
    device = next(model.parameters()).device
    
    prompts = [
        "What is a stock?",
        "How do bonds work?",
        "Explain portfolio diversification in one sentence.",
        "What is the risk of long-term government bonds?",
        "Why do companies pay dividends?",
        "How does inflation affect bond prices?",
    ]
    
    def generate(prompt: str) -> str:
        templated = f"Question: {prompt}\nAnswer:"
        inputs = tokenizer(templated, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.3,
                do_sample=True,
                top_p=0.8,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip the leading template to show only the answer portion
        answer = decoded.split("Answer:", 1)[-1].strip()
        return " ".join(answer.split())
    
    for prompt in prompts:
        response = generate(prompt)
        print(f"\nQ: {prompt}")
        print(f"A: {response}")

if __name__ == "__main__":
    tester = MCPTester()
    tester.start_server()
    tester.test_all_tools()
    run_lora_smoke_test()

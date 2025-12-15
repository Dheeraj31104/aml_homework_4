# AML Homework 4 – Finance RAG, LoRA, MCP Tools

## Overview
- **RAG demo** (`finance_rag.py`): Uses a curated set of finance Q&A pairs, embeds the questions, and returns the nearest answer via FAISS.
- **LoRA fine-tuning** (`finance_lora.py`): Fine-tunes DistilGPT2 with LoRA on QA-style finance examples. Saves adapters in `finance_lora_adapters`.
- **Knowledge base builders**:
  - `finance_scraper.py`: Downloads the FinancialPhraseBank dataset zip from Hugging Face and writes a knowledge base JSON.
  - `finance_wiki_scraper.py`: Scrapes finance topics (Wikipedia + selected finance sites) into the knowledge base JSON.
- **MCP server** (`finance_mcp_server.py`): Exposes tools for stock prices (yfinance), finance news scrape (Wikipedia), and compound investment math over JSON-RPC.
- **MCP test harness** (`test_finance_mcp.py`): Starts the MCP server, exercises tools, and runs a LoRA generation smoke test if adapters exist.

## Quickstart
1) Clone and install deps (Python 3.10+ recommended):
```bash
git clone https://github.com/Dheeraj31104/aml_homework_4.git
cd aml_homework_4
pip install -r requirements.txt  # or pip install transformers datasets sentence-transformers faiss-cpu peft torch yfinance beautifulsoup4 requests
```

2) Build a knowledge base (choose one):
- Dataset-based (cleaner): `python finance_scraper.py`
- Web scrape: `python finance_wiki_scraper.py`

3) Fine-tune LoRA on curated QA (and KB-derived snippets):
```bash
python finance_lora.py
```
Adapters save to `finance_lora_adapters/`. Inference runs on CPU to avoid Apple MPS limits.

4) Run the RAG demo (curated QA lookup):
```bash
python finance_rag.py
```

5) Test MCP tools + LoRA smoke test:
```bash
python test_finance_mcp.py
```
Starts `finance_mcp_server.py` in a subprocess, calls tools (AAPL/IBM/MSFT, news, investment math), then runs LoRA prompts if adapters are present.

## Notes
- The repo currently includes large generated artifacts (LoRA checkpoints/adapters, knowledge base JSON). Clean up or add `.gitignore` for lighter pushes if desired.
- For better LoRA answers, keep using curated QA pairs or craft your own; avoid noisy scraped paragraphs.

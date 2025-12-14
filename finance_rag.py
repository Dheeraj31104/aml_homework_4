#!/usr/bin/env python3
"""
RAG Pipeline with Finance Data
Run: python task1_finance_rag.py
"""

import json
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # keep Transformers in PyTorch-only mode

from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

print("="*70)
print("TASK 1: RAG PIPELINE WITH FINANCE DATA")
print("="*70)

# STEP 1: Load or build curated QA pairs
print("\n1. Preparing curated finance QAs...")
curated_qas = [
    ("What is a stock?", "A stock is a share of ownership in a company, giving the holder a claim on assets and earnings."),
    ("What is a bond?", "A bond is a loan to an issuer that pays interest and returns principal at maturity."),
    ("What is a dividend?", "A dividend is a distribution of a company’s profits to shareholders."),
    ("What is portfolio diversification?", "Diversification spreads investments across assets to reduce risk from any single position."),
    ("What is risk management in investing?", "Risk management limits potential losses using controls like sizing, diversification, and hedging."),
    ("What is fixed income?", "Fixed income refers to investments that pay scheduled interest, such as government and corporate bonds."),
    ("What is capital appreciation?", "Capital appreciation is the increase in the value of an investment over time."),
    ("What is market volatility?", "Volatility measures how much prices fluctuate; higher volatility means larger swings up or down."),
    ("What is dividend investing?", "Dividend investing focuses on buying companies that pay regular dividends to shareholders."),
    ("What are government bonds?", "Government bonds are debt securities issued by a government to fund spending, backed by its taxing power."),
]
knowledge_questions = [q for q, _ in curated_qas]
knowledge_answers = [a for _, a in curated_qas]
print(f"   ✓ Loaded {len(knowledge_questions)} curated QAs")

# STEP 2: Create embeddings over questions (for nearest-neighbor QA)
print("\n2. Creating embeddings over questions...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(knowledge_questions, convert_to_numpy=True)
print(f"   ✓ Created {len(embeddings)} embeddings")

# STEP 3: Build FAISS index
print("\n3. Building FAISS index...")
import numpy as np
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"   ✓ Index ready with {index.ntotal} documents")

print("\n4. Answering via nearest curated QA...")

# STEP 4: Define RAG function (nearest QA match)
def answer_finance_question(query: str):
    """Answer finance question using nearest curated QA."""
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=1)
    top_idx = int(indices[0][0])
    
    matched_question = knowledge_questions[top_idx]
    matched_answer = knowledge_answers[top_idx]
    
    return {
        "query": query,
        "matched_question": matched_question,
        "response": matched_answer,
    }

# STEP 5: Test with finance questions
print("\n5. Testing RAG with finance questions...")
test_questions = [
    "What is a stock?",
    "What are bonds?",
    "What is dividend investing?",
]

for question in test_questions:
    print(f"\n   Q: {question}")
    result = answer_finance_question(question)
    print(f"   A: {result['response']}")

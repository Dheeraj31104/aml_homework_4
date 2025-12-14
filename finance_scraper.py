#!/usr/bin/env python3
"""
Finance Data Loader - pulls finance text from Hugging Face datasets
Run: python finance_scraper.py
"""

import io
import json
from datetime import datetime
from typing import Dict, Optional
import zipfile

import requests

class FinanceDatasetBuilder:
    DATA_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
    SUBSET_FILES: Dict[str, str] = {
        "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
        "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
        "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
        "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    }
    
    def __init__(
        self,
        dataset_name: str = "financial_phrasebank",
        subset: Optional[str] = "sentences_allagree",
        split: str = "train",
        limit: int = 500,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.limit = limit
        self.documents = []
    
    def fetch(self):
        """Load finance documents from Hugging Face"""
        print("="*70)
        print("LOADING HUGGING FACE FINANCE DATASET")
        print("="*70)
        print(f"Dataset : {self.dataset_name}")
        print(f"Subset  : {self.subset or 'default'}")
        print(f"Split   : {self.split}")
        print(f"Limit   : {self.limit if self.limit else 'all'}")
        print("-"*70)
        
        target_file = self.SUBSET_FILES.get(self.subset or "")
        if not target_file:
            available = ", ".join(self.SUBSET_FILES)
            raise ValueError(f"Unknown subset '{self.subset}'. Choose one of: {available}")
        
        response = requests.get(self.DATA_URL, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            if target_file not in zf.namelist():
                raise FileNotFoundError(f"{target_file} not found inside dataset zip")
            raw_text = zf.read(target_file).decode("latin-1")
        
        for idx, line in enumerate(raw_text.splitlines()):
            if self.limit and idx >= self.limit:
                break
            line = line.strip()
            if not line:
                continue
            if "@@" in line:  # clean stray markers
                line = line.replace("@@", "@")
            if "@" not in line:
                continue
            sentence, sentiment = line.rsplit("@", 1)
            doc = {
                "title": f"{self.dataset_name}:{idx}",
                "source": "takala/financial_phrasebank",
                "content": sentence.strip(),
                "label": sentiment.strip(),
                "retrieved_at": datetime.now().isoformat(),
            }
            self.documents.append(doc)
        
        print(f"\n✓ Loaded {len(self.documents)} documents from Hugging Face zip")
        return self.documents
    
    def save(self, filename="finance_knowledge_base.json"):
        """Save to file"""
        with open(filename, 'w') as f:
            json.dump(self.documents, f, indent=2)
        print(f"\n✓ Saved {len(self.documents)} documents to {filename}")


if __name__ == "__main__":
    builder = FinanceDatasetBuilder(
        dataset_name="financial_phrasebank",
        subset="sentences_allagree",
        split="train",
        limit=500,
    )
    builder.fetch()
    builder.save()

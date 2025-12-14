#!/usr/bin/env python3
"""
LoRA Fine-Tuning on Finance Data
Run: python task2_finance_lora.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset


# STEP 1: Prepare finance training data
print("\n1. Preparing finance training data...")

def build_qa_corpus(limit: int = 500):
    # Curated, stable QA pairs (preferred over scraped text)
    curated_qas = [
        ("What is a stock?", "A stock is a share of ownership in a company, giving the holder a claim on assets and earnings."),
        ("What is a bond?", "A bond is a loan to an issuer that pays interest and returns principal at maturity."),
        ("What is a dividend?", "A dividend is a distribution of a company’s profits to shareholders."),
        ("What is portfolio diversification?", "Diversification spreads investments across assets to reduce risk from any single position."),
        ("What is risk management in investing?", "Risk management limits potential losses using controls like sizing, diversification, and hedging."),
        ("What is fixed income?", "Fixed income refers to investments that pay scheduled interest, such as government and corporate bonds."),
        ("What is capital appreciation?", "Capital appreciation is the increase in the value of an investment over time."),
        ("What is market volatility?", "Volatility measures how much prices fluctuate; higher volatility means larger swings up or down."),
    ]

    kb_path = Path("finance_knowledge_base.json")
    qa_examples = []
    qa_examples.extend([f"Question: {q}\nAnswer: {a}" for q, a in curated_qas])

    def clean_snippet(text: str) -> str:
        # Trim and strip heavy numeric clutter for cleaner answers
        import re
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\b[\d]{4,}\b", "", text)  # remove long numbers/years
        return text.strip()

    if kb_path.exists():
        with kb_path.open("r") as f:
            docs = json.load(f)
        for doc in docs[:limit]:
            # Prefer dataset-originated entries; skip web-only noise if present
            source = (doc.get("source") or "").lower()
            if source and "wikipedia" not in source and "finance_site" not in source:
                title = (doc.get("title") or "this concept").strip()
                content = (doc.get("content") or "").strip()
                if not content:
                    continue
                first_sentence = content.split(".")[0].strip()
                if len(first_sentence) < 30:
                    first_sentence = content[:200].strip()
                snippet = clean_snippet(first_sentence)
                if len(snippet) < 30:
                    continue
                question = f"What is {title}?"
                qa_examples.append(f"Question: {question}\nAnswer: {snippet}")
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for qa in qa_examples:
        if qa not in seen:
            deduped.append(qa)
            seen.add(qa)
    return deduped[:limit]
    return qa_examples

corpus_texts = build_qa_corpus(limit=500)
finance_training_data = {"text": corpus_texts}

dataset = Dataset.from_dict(finance_training_data)
print(f"   ✓ Created {len(dataset)} QA-style training examples (from knowledge base if available)")

# STEP 2: Tokenize
print("\n2. Tokenizing data...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"   ✓ Tokenized")

# STEP 3: Load model
print("\n3. Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print(f"   ✓ Loaded distilgpt2")

# STEP 4: Configure LoRA
print("\n4. Configuring LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ LoRA configured")
print(f"   ✓ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# STEP 5: Train
print("\n5. Training (this takes ~2 minutes)...")
training_args = TrainingArguments(
    output_dir="./finance_lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    logging_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
print(f"   ✓ Training complete!")

# STEP 6: Save
print("\n6. Saving adapters...")
model.save_pretrained("./finance_lora_adapters")
print(f"   ✓ Saved to ./finance_lora_adapters")

# STEP 7: Test
print("\n7. Testing fine-tuned model...")
model.eval()
# Move to CPU for generation to avoid MPS temporary NDArray limits on some Apple GPUs
model_device = torch.device("cpu")
model.to(model_device)
test_prompts = [
    "What is a stock?",
    "How do bonds work?",
    "Explain portfolio diversification in one sentence.",
    "What is the risk of long-term government bonds?",
    "Why do companies pay dividends?",
    "How does inflation affect bond prices?",
]

def generate_answer(prompt: str) -> str:
    templated = f"Question: {prompt}\nAnswer:"
    inputs = tokenizer(templated, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
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
    answer = decoded.split("Answer:", 1)[-1].strip()
    return " ".join(answer.split())

for prompt in test_prompts:
    response = generate_answer(prompt)
    print(f"\n   Q: {prompt}")
    print(f"   A: {response}")

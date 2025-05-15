# LoRA fine-tuning script for Qwen3-14B with PEFT + evaluation + hyperparameter tuning
# Requirements: peft, transformers, accelerate, bitsandbytes

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
import torch
import json
import subprocess
import shutil
import pandas as pd
import re

# Paths
FILE_PATH: Path = Path(__file__).resolve()
MODEL_NAME = "Qwen/Qwen3-14B"
DATA_PATH = FILE_PATH.parent / "prepared" / "qwen3_multilang_sft.jsonl"
EVAL_SCRIPT = FILE_PATH.parent /"evaluation"/ "evaluate.py"
INPUT_PATH = FILE_PATH.parent / "input_data"
OUTPUT_PATH = FILE_PATH.parent / "output_data"

LANGUAGES = ["en", "zh", "de", "es"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load data
data = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        ex = json.loads(line)
        data.append({"text": ex["prompt"] + " " + ex["response"]})

# Convert to HF dataset
dataset = Dataset.from_list(data)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Grid search over learning rates and epochs
lr_values = [5e-6, 2e-6, 5e-7]
epoch_values = [1]
results_summary = []

for lr in lr_values:
    for epochs in epoch_values:
        run_name = f"lr{lr}_ep{epochs}"
        OUTPUT_DIR = FILE_PATH.parent/"train_output"/f"lora8_16_output_dropout0.1{run_name}"

        print(f"\nRunning config: learning_rate={lr}, epochs={epochs}")

        # Reload model each time
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            lr_scheduler_type="cosine",  
            warmup_ratio=0.1,  
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        model.save_pretrained(OUTPUT_DIR)
        print(" Model saved to:", OUTPUT_DIR)


        # Generate predictions and save submission file
        for lang in LANGUAGES:
            dev_file = INPUT_PATH / f"{lang}_dev.tsv"
            submission_file = OUTPUT_PATH / f"{lang}_qwen14b_dev.tsv"
            if not dev_file.exists():
                continue
            df = pd.read_csv(dev_file, sep="\t")
            preds = []
            for _, row in df.iterrows():
                prompt = (
                    f"<|im_start|>user\n"
                    f"Language: {row['lang']}\n"
                    "Please output only one neutral sentence in original sentence's language without any explanation.\n"
                    "Detoxify the following toxic sentence by rewriting it in a polite and respectful tone, while preserving its meaning.\n"
                    "Try to keep the structure as close as possible while ensuring the result is fluent, neutral, natural, and non-offensive.And don't provode any other sentences of thinking or doubting but just answer with the neutral sentence.\n\n\n"
                    f"Toxic: {row['toxic_sentence']}\nNeutral:<|im_end|>\n"
                )
                messages = [{"role": "user", "content": prompt}]
                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
                generated_ids = model.generate(**model_inputs, max_new_tokens=128)
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                final_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                final_response = re.sub(r"^(Neutral|中立|Neutre|Neutralität|Neutrale)?[:：\-\s]*", "", final_response, flags=re.IGNORECASE)
                preds.append(final_response)
            df["neutral_sentence"] = preds
            df.to_csv(submission_file, sep="\t", index=False)

        # Run evaluation for all languages
        avg_J = []
        for lang in LANGUAGES:
            submission_file = OUTPUT_PATH / f"{lang}_qwen14b_dev.tsv"
            reference_file = INPUT_PATH / f"{lang}_dev.tsv"

            if Path(submission_file).exists() and Path(reference_file).exists():
                try:
                    print(f"\nEvaluating {lang} on dev set...")
                    result = subprocess.run([
                        "python", str(EVAL_SCRIPT),
                        "--submission", str(submission_file),
                        "--reference", str(reference_file)
                    ], capture_output=True, text=True, check=True)
                    print(result.stdout)

                    for line in result.stdout.splitlines():
                        if line.startswith(f"| {lang}"):
                            j_score = float(line.split("|")[-1].strip())
                            avg_J.append(j_score)
                except subprocess.CalledProcessError as e:
                    print(f"Evaluation failed for {lang}:", e)
                    print(e.stderr)
            else:
                print(f"Missing files for {lang}, skipping evaluation.")

        if avg_J:
            mean_j = sum(avg_J) / len(avg_J)
            print(f"Avg J Score for config {run_name}: {mean_j:.4f}")
            results_summary.append({"config": run_name, "learning_rate": lr, "epochs": epochs, "J": mean_j})

# Final result summary
print("\nGrid Search Summary:")
for res in sorted(results_summary, key=lambda x: -x["J"]):
    print(f"{res['config']}: J={res['J']:.4f}")

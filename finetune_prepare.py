# Convert multilingual TSV training sets into SFT format
# Output: one merged JSONL file with prompt-response pairs

import pandas as pd
from pathlib import Path
import json

FILE_PATH: Path = Path(__file__).resolve()
INPUT_PATH = FILE_PATH.parent / "input_data"
OUTPUT_PATH = FILE_PATH.parent / "prepared"


LANGUAGES = ["en", "zh", "de", "es"]

all_data = []

for lang in LANGUAGES:
    tsv_file = INPUT_PATH / f"{lang}_train.tsv"
    if not tsv_file.exists():
        print(f"[!] Missing: {tsv_file}")
        continue

    df = pd.read_csv(tsv_file, sep="\t").dropna()

    for _, row in df.iterrows():
        toxic = row["toxic_sentence"].strip()
        neutral = row["neutral_sentence"].strip()
        prompt = (
            f"<|im_start|>user\n"
            f"Language: {lang}\n"
            "Identify the input language. Please output only one neutral sentence in that language without any explanation.\n"
            "Detoxify the following toxic sentence by rewriting it in a polite and respectful tone, while preserving its meaning.\n"
            "Try to keep the structure as close as possible while ensuring the result is fluent, neutral, natural, and non-offensive.\n\n"
            f"Toxic: {toxic}\nNeutral:<|im_end|>\n<|im_start|>assistant\n"
        )
        all_data.append({"prompt": prompt, "response": neutral})

# Save to JSONL format
jsonl_path = OUTPUT_PATH / "qwen3_multilang_sft.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as fout:
    for example in all_data:
        fout.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Saved {len(all_data)} prompt-response pairs to {jsonl_path}")

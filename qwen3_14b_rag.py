import pandas as pd
import faiss
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# === Path Config ===
FILE_PATH: Path = Path(__file__).resolve()
INPUT_PATH = FILE_PATH.parent / "input_data"
INDEX_PATH = FILE_PATH.parent / "rag_index"
OUTPUT_PATH = FILE_PATH.parent / "output_data"

# === Language → Embedding model mapping ===
LANG_MODEL_MAP = {
    "en": "paraphrase-MiniLM-L6-v2",
    "zh": "shibing624/text2vec-base-chinese",
    "de": "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "es": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}


embedding_models = {}

def get_embedding_model(lang: str):
    if lang in embedding_models:
        return embedding_models[lang]
    model_name = LANG_MODEL_MAP.get(lang, "sentence-transformers/distiluse-base-multilingual-cased-v2")  # fallback
    print(f"[Embedding] Loading model for {lang}: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_models[lang] = model
    return model

class Agent:
    def __init__(self, api_key, base_url="YOUR URL", model="Qwen/Qwen3-14B", temperature=0.7, max_tokens=1024):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, toxic_sentence: str, examples: list) -> str:
        prompt = (
            "<|im_start|>user\n"
            "Identify the input language. Please output only one neutral sentence in that language without any explanation.\n"
            "Detoxify the following toxic sentence by rewriting it in a polite and respectful tone, while preserving its meaning.\n"
            "Try to keep the structure as close as possible while ensuring the result is fluent, natural, and non-offensive.\n\n"
        )
        for ex in examples:
            prompt += f"Toxic: {ex['toxic_sentence']}\nNeutral: {ex['neutral_sentence']}\n\n"
        prompt += f"Toxic: {toxic_sentence}\nNeutral:<|im_end|>\n<|im_start|>assistant\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        output = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                if chunk.choices[0].delta.content:
                    output += chunk.choices[0].delta.content
        return output.strip()

# Retrieval logic by language
def retrieve_by_lang(toxic_sentence: str, lang: str, top_k=5):
    index_file = INDEX_PATH / f"{lang}_index.faiss"
    data_file = INDEX_PATH / f"{lang}_train_data.pkl"
    if not index_file.exists() or not data_file.exists():
        print(f"[WARNING] Missing index or data for {lang}, skipping retrieval.")
        return []

    model = get_embedding_model(lang)
    index = faiss.read_index(str(index_file))
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    vec = model.encode([toxic_sentence], normalize_embeddings=True)
    D, I = index.search(np.array(vec), top_k)
    return [data[i] for i in I[0] if i < len(data)]

# Detox function
def detoxify(input_file: Path, output_file: Path, api_key: str):
    df = pd.read_csv(input_file, sep="\t").dropna()
    agent = Agent(api_key=api_key)

    results = {"toxic_sentence": [], "neutral_sentence": [], "lang": []}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        toxic = row["toxic_sentence"]
        lang = row["lang"]
        examples = retrieve_by_lang(toxic, lang, top_k=5)

        try:
            neutral = agent.chat(toxic, examples)
        except Exception as e:
            print(f"[ERROR] Detox failed for: {toxic[:30]}... → {e}")
            neutral = ""

        results["toxic_sentence"].append(toxic)
        results["neutral_sentence"].append(neutral)
        results["lang"].append(lang)

    pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
    print(f"\nDetoxified results saved to: {output_file}")

# run
if __name__ == "__main__":
    detoxify(
        input_file=INPUT_PATH / "en_test.tsv",
        output_file=OUTPUT_PATH / "en_qwen14b_rag_sep_test.tsv",
        api_key="YOUR KEY"
    )
    detoxify(
        input_file=INPUT_PATH / "es_test.tsv",
        output_file=OUTPUT_PATH / "es_qwen14b_rag_sep_test.tsv",
        api_key="YOUR KEY"
    )

    detoxify(
        input_file=INPUT_PATH / "de_test.tsv",
        output_file=OUTPUT_PATH / "de_qwen14b_rag_sep_test.tsv",
        api_key="YOUR KEY"
    )

    detoxify(
        input_file=INPUT_PATH / "zh_test.tsv",
        output_file=OUTPUT_PATH / "zh_qwen14b_rag_sep_test.tsv",
        api_key="YOUR KEY"
    )

import pandas as pd
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# your file path
FILE_PATH: Path = Path(__file__).resolve()
TRAIN_DIR = FILE_PATH.parent / "input_data"
INDEX_DIR = FILE_PATH.parent / "rag_index"
SAMPLE_RATIO = 0.6

# Language-specific embedding models
LANG_MODEL_MAP = {
    "en": "paraphrase-MiniLM-L6-v2",
    "zh": "shibing624/text2vec-base-chinese",
    "de": "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "es": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"#spanish's own model is not open source
}

print(" Building RAG indices for each language...")

for lang, model_name in LANG_MODEL_MAP.items():
    file = TRAIN_DIR / f"{lang}_train.tsv"
    if not file.exists():
        print(f"[SKIPPED] File not found: {file}")
        continue

    print(f"\n[{lang}] Using embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    df = pd.read_csv(file, sep="\t").dropna()
    df = df.sample(frac=SAMPLE_RATIO, random_state=42).reset_index(drop=True)
    print(f" → Sampled {len(df)} examples from {file.name}")

    # Generate embeddings
    embeddings = model.encode(
        df["toxic_sentence"].tolist(),
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )

    # Create and save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    # Save FAISS index
    faiss.write_index(index, str(INDEX_DIR / f"{lang}_index.faiss"))
    with open(INDEX_DIR / f"{lang}_train_data.pkl", "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)


    print(f" Saved: {lang}_index.faiss and {lang}_data.pkl")

print("\nAll language-specific RAG indices have been built.")

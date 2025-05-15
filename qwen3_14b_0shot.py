import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
from pathlib import Path


FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH = FILE_PATH.parent / "input_data"
OUTPUT_DATA_PATH = FILE_PATH.parent / "output_data"

class Agent:
    def __init__(self, api_key, base_url="YOUR URL", model="Qwen/Qwen3-14B", temperature=0.7, max_tokens=1024):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, toxic_sentence: str) -> str:
        """
        Construct the prompt and get neutral sentence from the LLM.
        """
        prompt = (
            "<|im_start|>user\n"
            "Judge the input language. Please output only one neutral sentence in its original language without other explaining phrases.\n"
            "Detoxify the following toxic sentence by rewriting it in a neutral and polite way, while preserving the original meaning as much as possible.\n"
            "Try to minimize changes and retain the original structure of sentences, making the result fluent, natural, and respectful.\n\n"
            "Toxic: MF and I have a mutual enemy in sanctimonious bullshit.\n"
            "Neutral: My friend and I share a mutual enemy who engages in self-righteous and pretentious behavior.\n\n"
            f"Toxic: {toxic_sentence}\nNeutral:<|im_end|>\n<|im_start|>assistant\n"
        )

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


def detoxify_tsv(input_path: str, output_path: str, api_key="Your Key"):
    df = pd.read_csv(input_path, sep="\t")
    
    agent = Agent(api_key=api_key)
    
    # new generated neutral sentences
    new_data = {"toxic_sentence": [], "neutral_sentence": [], "lang": []}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        toxic = row["toxic_sentence"]
        lang = row["lang"]
        try:
            neutral = agent.chat(toxic)
        except Exception as e:
            print(f"Error on sentence: {toxic[:50]}..., Error: {e}")
            neutral = ""
        new_data["toxic_sentence"].append(toxic)
        new_data["neutral_sentence"].append(neutral)
        new_data["lang"].append(lang)

    output_df = pd.DataFrame(new_data)
    output_df.to_csv(output_path, sep="\t", index=False)
    print(f"Detoxified TSV saved to: {output_path}")



detoxify_tsv(
    input_path= Path(INPUT_DATA_PATH, "zh_test.tsv"),
    output_path= Path(OUTPUT_DATA_PATH, "zh_qwen14b_0shot_test.tsv")
)

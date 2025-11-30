import os
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = [
    # TODO: repo id가 정확한지 Hugging Face에서 확인 필요
    ("Llama3", "meta-llama/Llama-3-8b-instruct", {}),
    ("Qwen", "Qwen/Qwen1.5-1.8B-Chat", {"trust_remote_code": True}),
]


def test_model(label: str, model_id: str, extra_kwargs: dict) -> None:
    print(f"=== Testing {label}: {model_id}")
    print(
        f"HUGGINGFACE_HUB_TOKEN set: {bool(os.environ.get('HUGGINGFACE_HUB_TOKEN'))}"
    )
    try:
        print("  -> Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **extra_kwargs)

        print("  -> Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **extra_kwargs)
        model.eval()

        print("  -> Running a tiny generation...")
        inputs = tokenizer("Test", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("  SUCCESS, sample output:", repr(text))
    except Exception as e:
        print("  ERROR while loading or generating:")
        print(" ", repr(e))
        traceback.print_exc()
    print()


if __name__ == "__main__":
    print("Running one-shot HF model load test\n")
    for label, mid, kwargs in MODELS:
        test_model(label, mid, kwargs)


from datasets import load_dataset

# 1) 코드 데이터셋 (The Stack v2)
#  - 언어를 꼭 Python으로 제한하고 싶으면 config 자리에 "Python"을 넣어봅니다.
#  - 혹시 "Python" config 에러가 나면 두 번째 줄 대신
#      stack = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
#    로 전체를 스트리밍하세요.
ds = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",   # 언어 폴더 지정 가능
    streaming=True,
    split="train"
)

print(next(iter(ds)))  # 코드 content 직접 확인 가능


# 2) JSON 인스트럭션 데이터셋 (tatsu-lab/alpaca)
alpaca = load_dataset(
    "tatsu-lab/alpaca",
    split="train",
    streaming=True,
)
print("=== Alpaca Example ===")
print(next(iter(alpaca)))  # 첫 샘플 출력
print("\n")


# 3) 독성/혐오 텍스트 데이터셋 (RealToxicityPrompts)
tox = load_dataset(
    "allenai/real-toxicity-prompts",
    split="train",
    streaming=True,
)
print("=== RealToxicityPrompts Example ===")
print(next(iter(tox)))     # 첫 샘플 출력

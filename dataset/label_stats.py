import argparse
import json
import os
from collections import Counter, defaultdict


def compute_stats(path: str) -> None:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return

    combo_counts: Counter = Counter()
    code_counts: Counter = Counter()
    harm_counts: Counter = Counter()
    struct_counts: Counter = Counter()
    source_counts: Counter = Counter()
    per_source_counts = defaultdict(
        lambda: {
            "code": Counter(),
            "harm": Counter(),
            "struct": Counter(),
        }
    )

    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            code = int(row.get("label_code", 0))
            harm = int(row.get("label_harm", 0))
            struct = int(row.get("label_struct", 0))
            source = row.get("source", "unknown")

            combo_counts[(code, harm, struct)] += 1
            code_counts[code] += 1
            harm_counts[harm] += 1
            struct_counts[struct] += 1
            source_counts[source] += 1
            per_source_counts[source]["code"][code] += 1
            per_source_counts[source]["harm"][harm] += 1
            per_source_counts[source]["struct"][struct] += 1
            total += 1

    print(f"File: {path}")
    print(f"Total samples: {total}")
    print()

    print("Label combo counts (label_code, label_harm, label_struct):")
    for combo, cnt in sorted(combo_counts.items()):
        print(f"  {combo}: {cnt}")
    print()

    print("Marginal counts:")
    print("  label_code:")
    for v, cnt in sorted(code_counts.items()):
        print(f"    {v}: {cnt}")
    print("  label_harm:")
    for v, cnt in sorted(harm_counts.items()):
        print(f"    {v}: {cnt}")
    print("  label_struct:")
    for v, cnt in sorted(struct_counts.items()):
        print(f"    {v}: {cnt}")
    print()

    print("Source counts:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")
    print()

    print("Per-source label breakdown:")
    for src in sorted(per_source_counts.keys()):
        stats = per_source_counts[src]
        print(f"  Source: {src}")
        print("    label_code:")
        for v, cnt in sorted(stats["code"].items()):
            print(f"      {v}: {cnt}")
        print("    label_harm:")
        for v, cnt in sorted(stats["harm"].items()):
            print(f"      {v}: {cnt}")
        print("    label_struct:")
        for v, cnt in sorted(stats["struct"].items()):
            print(f"      {v}: {cnt}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="간단한 multi_contrast JSONL 라벨 분포 확인 스크립트"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="dataset/multi_contrast_train_final.jsonl",
        help=(
            "라벨 분포를 분석할 JSONL 파일 경로 "
            "(기본: dataset/multi_contrast_train_final.jsonl; "
            "final 라벨은 dataset/multi_contrast_train_final.jsonl 사용 권장)"
        ),
    )
    args = parser.parse_args()
    compute_stats(args.path)


if __name__ == "__main__":
    main()

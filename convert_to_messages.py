#!/usr/bin/env python3
"""
将 csau_en_ch_*.jsonl（text + label）转为 LoRA SFT 用的 messages 格式。

任务：根据说话人语言切换习惯、句法、风格等判断是哪个用户，回复 user ID。
每条: {"text": "...", "label": "24"}
-> user: 指令 + 文本; assistant: "user24"

若存在 data/label_names.json（格式 {"0": "user0", ...}），则 assistant 按映射输出；否则输出 user{label}。
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "sft"
LABEL_NAMES_PATH = DATA_DIR / "label_names.json"

# 用户指定的提示词（说话人识别 + 只回复 user ID）
USER_INSTRUCTION = (
    "Classify which user wrote the following sentence.\n"
    "There are 50 users: user0 to user49.\n"
    "Answer with EXACTLY one ID in the format userXX (0-49).\n"
    "Do NOT output anything else.\n\n"
    "Text:\n"
)


def load_label_names():
    if not LABEL_NAMES_PATH.exists():
        return None
    with open(LABEL_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_line(obj: dict, label_names: dict | None) -> dict:
    text = obj["text"]
    label = obj["label"]
    # 默认输出 user0, user12, user24 等；若有映射则用映射
    answer = (label_names or {}).get(label, f"user{label}")

    user_content = USER_INSTRUCTION + text
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(answer)},
        ]
    }


def convert_file(in_path: Path, out_path: Path, label_names: dict | None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out_obj = convert_line(obj, label_names)
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    label_names = load_label_names()
    if label_names:
        print(f"使用类别映射: {LABEL_NAMES_PATH} ({len(label_names)} 类)")
    else:
        print("未找到 label_names.json，assistant 将输出 user0, user1, ... user49。")

    for name in ("csau_en_ch_train", "csau_en_ch_test"):
        in_path = DATA_DIR / f"{name}.jsonl"
        out_path = OUT_DIR / f"{name}_messages.jsonl"
        if not in_path.exists():
            print(f"跳过（不存在）: {in_path}")
            continue
        n = convert_file(in_path, out_path, label_names)
        print(f"已转换: {in_path} -> {out_path} ({n} 条)")

    print("完成。SFT 数据目录:", OUT_DIR)


if __name__ == "__main__":
    main()

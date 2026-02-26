#!/usr/bin/env python3
"""
在测试集上评估说话人识别：跑 Qwen2.5-7B-Instruct（或其它本地模型），
输出准确率、精确率、召回率、F1、混淆矩阵等。

依赖: pip install transformers torch scikit-learn tqdm peft

用法:
  # 仅基础模型（或合并后的完整模型）
  python eval_speaker_id.py --model_path models/Qwen2.5-7B-Instruct
  # 基础模型 + LoRA 适配器（如 ms-swift 微调后的 checkpoint）
  python eval_speaker_id.py --model_path models/Qwen2.5-7B-Instruct --adapter_path output/csau_lora/.../checkpoint-xxx
  python eval_speaker_id.py --model_path models/Qwen2.5-7B-Instruct --max_samples 500 --output_dir eval_results
  python eval_speaker_id.py --model_path models/Qwen2.5-7B-Instruct --load_in_8bit   # 显存不足时
"""

import argparse
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 与 data/convert_to_messages.py 中一致的提示词
USER_INSTRUCTION = (
    "Classify which user wrote the following sentence.\n"
    "There are 50 users: user0 to user49.\n"
    "Answer with EXACTLY one ID in the format userXX (0-49).\n"
    "Do NOT output anything else.\n\n"
    "Text:\n"
)

ROOT = Path(__file__).resolve().parent
DEFAULT_TEST = ROOT / "data" / "csau_en_ch_test.jsonl"
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-7B-Instruct"
LABELS = [str(i) for i in range(50)]


def load_test_data(path: Path, max_samples: int | None):
    import json
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label"])
    return texts, labels


def parse_user_id(raw: str) -> str | None:
    """从模型输出中解析 user ID，返回 "0"～"49" 或 None。

    解析策略（尽量提高可解析率，降低“未解析”比例）：
    1）优先匹配包含 'user' 的形式：user12, User 12, user_12 等；
    2）再匹配中文“用户12”之类；
    3）最后兜底：提取所有 1~2 位数字，只要存在 0~49 的数字，就取第一个。
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    # 1. 带 user 前缀的形式：user12, User 12, user_12 等
    m = re.search(r"user\D*?(\d{1,2})", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 0 <= num < 50:
            return str(num)

    # 2. 中文“用户12”之类
    m = re.search(r"用户\D*?(\d{1,2})", text)
    if m:
        num = int(m.group(1))
        if 0 <= num < 50:
            return str(num)

    # 3. 兜底：提取所有 1~2 位数字，只要有 0~49 就用第一个
    nums = re.findall(r"\d{1,2}", text)
    for token in nums:
        # 处理类似 '07' 这种前导 0 的情况
        num = int(token)
        if 0 <= num < 50:
            return str(num)

    return None


def build_input_ids(tokenizer, text: str, max_length: int = 2048):
    """根据 tokenizer 是否支持 chat_template 自动构建输入 token ids。

    - 若 tokenizer.chat_template 存在且非空：使用 apply_chat_template（适合 Qwen、Llama 等聊天模型）；
    - 否则：退化为普通文本拼接提示词，使用 tokenizer(...) 编码（适合 Gemma 等无 chat_template 的模型）。
    """
    user_content = USER_INSTRUCTION + text
    messages = [{"role": "user", "content": user_content}]

    # 优先尝试 chat template（与使用 chat 模式训练的模型保持一致）
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            return ids.squeeze(0)
        except Exception:
            # 个别模型可能 chat_template 存在但不兼容，遇到异常则自动退回到纯文本提示
            pass

    # 退化为普通指令 + 文本提示
    prompt = user_content
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return encoded["input_ids"].squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker ID on test set")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL), help="本地基础模型目录（或合并后的完整模型）")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA 适配器目录（HF 格式）；与 model_path 同时用时加载 base+adapter")
    parser.add_argument("--test_path", type=str, default=str(DEFAULT_TEST), help="测试集 jsonl (text, label)")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "eval_results"), help="结果输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估样本数，默认全量")
    parser.add_argument("--batch_size", type=int, default=1, help="推理 batch 大小，显存小用 1")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--load_in_8bit", action="store_true", help="8bit 量化省显存")
    parser.add_argument("--dump_raw_path", type=str, default=None, help="可选：将每条样本的原始输出与解析结果写入该 jsonl 文件（便于排查解析问题）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载测试集...")
    texts, y_true = load_test_data(Path(args.test_path), args.max_samples)
    n = len(texts)
    print(f"测试样本数: {n}")

    print("加载模型与 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # decoder-only 架构下推理应使用左侧 padding，避免警告并保证生成对齐
    tokenizer.padding_side = "left"
    model_kwargs = {"trust_remote_code": True}
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"已加载 LoRA 适配器: {args.adapter_path}")
    if not args.load_in_8bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    y_pred = []

    dump_f = None
    if args.dump_raw_path:
        dump_path = Path(args.dump_raw_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_f = open(dump_path, "w", encoding="utf-8")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in tqdm(range(0, n, args.batch_size), desc="推理"):
        batch_texts = texts[i : i + args.batch_size]
        batch_input_ids = []
        for t in batch_texts:
            # 自动根据是否存在 chat_template 构建输入 ids，兼容不同模型（Qwen、Llama、Gemma 等）
            ids = build_input_ids(tokenizer, t, max_length=2048)
            batch_input_ids.append(ids)
        input_lengths = [x.size(0) for x in batch_input_ids]
        max_len = max(input_lengths)
        # decoder-only 必须左 padding，否则会报警告且生成错位；pad_sequence 默认右 pad，这里手动左 pad
        device = next(model.parameters()).device
        input_ids = torch.full((len(batch_input_ids), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(batch_input_ids), max_len, dtype=torch.long, device=device)
        for j, (ids, L) in enumerate(zip(batch_input_ids, input_lengths)):
            input_ids[j, -L:] = ids.to(device)
            attention_mask[j, -L:] = 1
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        # 左 pad 时 prompt 占满前 max_len 位，新生成的是 out[:, max_len:]
        for j in range(len(batch_texts)):
            new_tokens = out[j, max_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = parse_user_id(raw)
            pred_id = pred if pred is not None else "-1"
            y_pred.append(pred_id)

            if dump_f is not None:
                import json

                idx = i + j
                dump_obj = {
                    "index": idx,
                    "text": texts[idx],
                    "gold_label": y_true[idx],
                    "raw_output": raw,
                    "parsed_id": pred,
                }
                dump_f.write(json.dumps(dump_obj, ensure_ascii=False) + "\n")

    # 未解析到的当作错误类，用 "-1" 标记；算指标时可选择是否剔除
    y_pred = [p if p in LABELS else "-1" for p in y_pred]

    # 指标（这里只对“已成功解析出 userID 的样本”计算）
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    # 只保留解析成功的样本
    mask = [p != "-1" for p in y_pred]
    y_true_eval = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_eval = [yp for yp, m in zip(y_pred, mask) if m]
    n_eval = len(y_true_eval)

    if n_eval > 0:
        acc = accuracy_score(y_true_eval, y_pred_eval)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_eval, y_pred_eval, labels=LABELS, average="macro", zero_division=0
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true_eval, y_pred_eval, labels=LABELS, average="weighted", zero_division=0
        )
    else:
        acc = precision = recall = f1 = precision_w = recall_w = f1_w = 0.0

    print("\n============ 评估结果（仅统计已解析样本） ============")
    print(f"参与统计的样本数 (parsed): {n_eval} / {n}")
    print(f"准确率 (Accuracy):        {acc:.4f}")
    print(f"精确率 (Precision macro): {precision:.4f}")
    print(f"召回率 (Recall macro):    {recall:.4f}")
    print(f"F1 (macro):               {f1:.4f}")
    print(f"F1 (weighted):            {f1_w:.4f}")
    print(f"Precision (weighted):     {precision_w:.4f}")
    print(f"Recall (weighted):        {recall_w:.4f}")

    report = classification_report(y_true_eval, y_pred_eval, labels=LABELS, zero_division=0)
    print("\n---------- 分类报告 (classification_report) ----------")
    print(report)

    # 混淆矩阵也只看解析成功的样本（不包含 -1 列）
    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=LABELS)
    cm_path = out_dir / "confusion_matrix.csv"
    import csv
    # newline 必须是空字符串，编码用 encoding 指定
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + LABELS)
        for i, row in enumerate(cm):
            w.writerow([LABELS[i]] + list(row))
    print(f"\n混淆矩阵已保存: {cm_path} (行=真实, 列=预测；仅包含已解析样本)")

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Num parsed samples: {n_eval} / {n}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision (macro): {precision:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write(f"F1 (weighted): {f1_w:.4f}\n\n")
        f.write(report)
    print(f"分类报告已保存: {report_path}")

    # 未解析比例（不参与指标统计，仅提示）
    unparsed = sum(1 for p in y_pred if p == "-1")
    if unparsed > 0:
        print(f"\n未解析到 user ID 的样本数: {unparsed} / {n} ({100 * unparsed / n:.2f}%)")

    if dump_f is not None:
        dump_f.close()


if __name__ == "__main__":
    main()

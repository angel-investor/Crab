import json
import re
import os

import sys

def eval_avqa(fp=None, filter_json=None):
    if fp is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('fp', nargs='?', default='/root/autodl-tmp/Crab/ckpt/inference_avqa/infer_results.jsonl')
        parser.add_argument('--filter', type=str, default=None, help='Path to subset json (e.g. head_samples.json)')
        args = parser.parse_args()
        fp = args.fp
        filter_json = args.filter

    # 如果有过滤器，先读取允许的 qid 集合
    allowed_qids = None
    if filter_json and os.path.exists(filter_json):
        print(f"Filtering by: {filter_json}")
        with open(filter_json, 'r') as f:
            subset_data = json.load(f)
            allowed_qids = {str(s['question_id']) for s in subset_data}
            print(f"Subset size: {len(allowed_qids)}")

    if not os.path.exists(fp):
        print(f"Error: {fp} not found.")
        print("Usage: python scripts/eval_results.py [path_to_jsonl] [--filter subset.json]")
        return

    correct = 0
    total = 0
    
    print(f"Reading results from {fp}...")
    
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except:
                continue
            
            # 过滤逻辑
            qid = str(item.get('qid', ''))
            if allowed_qids is not None and qid not in allowed_qids:
                continue
                
            predict_text = item.get('predict', '')
            gt_output = item.get('output', '')
            
            # 提取 GT 答案（去掉 </s> 和空格）
            gt_answer = gt_output.replace('</s>', '').strip().lower()
            if not gt_answer:
                continue
            
            # 提取 Pred 答案
            # 优先匹配 <answer>...</answer>
            pred_match = re.search(r'<answer>\s*(.+?)\s*</answer>', predict_text, re.IGNORECASE)
            if pred_match:
                pred_answer = pred_match.group(1).strip().lower()
            else:
                # 备选匹配：匹配 "the answer is XXX" 或 "answer: XXX"
                alt_match = re.search(r'(?:the answer is|answer:)\s*(\S+)', predict_text, re.IGNORECASE)
                if alt_match:
                    pred_answer = alt_match.group(1).strip().lower()
                else:
                    total += 1
                    continue
            
            # 清洗特殊 token 和标点（如 </s>、<pad>、. , ; 等）
            pred_answer = re.sub(r'<[^>]+>', '', pred_answer)  # 去掉所有 <...> 标签
            pred_answer = pred_answer.strip().rstrip('.,;:?!').strip()
            
            if pred_answer == gt_answer:
                correct += 1
            total += 1
            
    if total > 0:
        print("-" * 30)
        print(f"Total Samples: {total}")
        print(f"Correct:       {correct}")
        print(f"Accuracy:      {correct/total*100:.2f}%")
        print("-" * 30)
    else:
        print("No valid samples evaluated.")

if __name__ == "__main__":
    eval_avqa()

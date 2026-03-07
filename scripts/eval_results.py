import json
import re
import os

import sys

def eval_avqa(fp=None):
    if fp is None:
        if len(sys.argv) > 1:
            fp = sys.argv[1]
        else:
            # 默认指向刚才配置的消融实验结果路径（带 _no_audio 后缀）
            fp = '/root/autodl-tmp/Crab/ckpt/inference_avqa_no_audio/infer_results.jsonl'
            
    if not os.path.exists(fp):
        print(f"Error: {fp} not found.")
        print("Usage: python scripts/eval_results.py [path_to_jsonl]")
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
                    pred_answer = alt_match.group(1).strip().lower().rstrip('.,;')
                else:
                    total += 1
                    continue
            
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

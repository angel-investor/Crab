import json
import re
import os

def eval_avqa():
    fp = '/root/autodl-tmp/Crab/ckpt/inference_avqa/infer_results.jsonl'
    if not os.path.exists(fp):
        print(f"Error: {fp} not found.")
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
            
            # Extract GT answer
            gt_match = re.search(r'the answer is\s+(\S+?)\.?$', gt_output, re.IGNORECASE)
            if gt_match is None:
                continue
            gt_answer = gt_match.group(1).strip().lower()
            
            # Extract Pred answer
            pred_match = re.search(r'<answer>\s*(.+?)\s*</answer>', predict_text, re.IGNORECASE)
            if pred_match is None:
                pred_match = re.search(r'(?:the answer is|answer:)\s*(\S+)', predict_text, re.IGNORECASE)
            
            if pred_match is None:
                total += 1
                continue
                
            pred_answer = (pred_match.group(1) or pred_match.group(2)).strip().lower().rstrip('.,;')
            
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

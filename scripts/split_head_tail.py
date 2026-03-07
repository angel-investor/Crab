import json
import os
from collections import Counter
import numpy as np

def split_avqa_head_tail():
    # 路径定义
    annotation_path = 'data/music_avqa_data/test_samples.json'
    output_dir = 'data/music_avqa_data/robust_split'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(annotation_path):
        print(f"Error: {annotation_path} not found.")
        return

    print(f"Loading annotations from {annotation_path}...")
    with open(annotation_path, 'r') as f:
        samples = json.load(f)

    # 1. 统计答案分布
    answers = [s['answer'].strip().lower() for s in samples]
    answer_counts = Counter(answers)
    
    # 2. 计算阈值 (参考论文: 1.2 * Mean)
    counts = list(answer_counts.values())
    mean_count = np.mean(counts)
    threshold = 1.2 * mean_count
    
    print(f"Unique Answers: {len(answer_counts)}")
    print(f"Mean Count: {mean_count:.2f}")
    print(f"Head Threshold: {threshold:.2f}")

    # 3. 划分集合
    head_answers = {ans for ans, count in answer_counts.items() if count > threshold}
    
    head_samples = []
    tail_samples = []
    
    for s in samples:
        ans = s['answer'].strip().lower()
        if ans in head_answers:
            head_samples.append(s)
        else:
            tail_samples.append(s)

    # 4. 保存结果
    head_path = os.path.join(output_dir, 'head_samples.json')
    tail_path = os.path.join(output_dir, 'tail_samples.json')
    
    with open(head_path, 'w') as f:
        json.dump(head_samples, f, indent=4)
    with open(tail_path, 'w') as f:
        json.dump(tail_samples, f, indent=4)

    print("-" * 30)
    print(f"Head Samples: {len(head_samples)} ({len(head_samples)/len(samples)*100:.2f}%)")
    print(f"Tail Samples: {len(tail_samples)} ({len(tail_samples)/len(samples)*100:.2f}%)")
    print(f"Files saved in {output_dir}")
    print("-" * 30)

if __name__ == "__main__":
    split_avqa_head_tail()

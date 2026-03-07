import json
import os
import argparse

def cleanup_results(fp, output_fp=None):
    if not os.path.exists(fp):
        print(f\"Error: {fp} not found.\")
        return
    
    if output_fp is None:
        output_fp = fp + \".clean\"
        
    seen = set()
    unique_items = []
    duplicate_count = 0
    
    print(f\"Reading from {fp}...\")
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 组合 vid 和 qid 作为唯一键
                key = (item.get('vid'), item.get('qid'))
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)
                else:
                    duplicate_count += 1
            except:
                continue
    
    print(f\"Found {len(unique_items)} unique samples and {duplicate_count} duplicates.\")
    
    with open(output_fp, 'w', encoding='utf-8') as f:
        for item in unique_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\\n')
    
    print(f\"Cleaned results saved to: {output_fp}\")
    print(f\"You can now run: python scripts/eval_results.py {output_fp}\")

if __name__ == \"__main__\":
    parser = argparse.ArgumentParser()
    parser.add_argument(\"file\", help=\"Path to the jsonl file to cleanup\")
    args = parser.parse_args()
    cleanup_results(args.file)

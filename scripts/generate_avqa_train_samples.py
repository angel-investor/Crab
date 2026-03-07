"""
生成 MUSIC-AVQA 训练集的标注文件 valid_train_samples.json
格式与 test_samples.json 相同，供 UnifiedDataset.add_avqa_task_samples() 使用

用法：
    cd /root/autodl-tmp/Crab
    python scripts/generate_avqa_train_samples.py
"""
import json
import os

# ===== 路径配置 =====
AVQA_ROOT = '/root/autodl-tmp/Crab/data/music-avqa'
QA_TRAIN_JSON = os.path.join(AVQA_ROOT, 'avqa-train.json')  # MUSIC-AVQA 原始训练集标注
VIDEO_DIR = os.path.join(AVQA_ROOT, 'avqa-videos/MUSIC-AVQA-videos-Real')
AUDIO_DIR = os.path.join(AVQA_ROOT, 'audio')
OUTPUT_PATH = 'data/music_avqa_data/valid_train_samples.json'

def main():
    if not os.path.exists(QA_TRAIN_JSON):
        # 也尝试其他常见文件名
        alt_paths = [
            os.path.join(AVQA_ROOT, 'train_qa.json'),
            os.path.join(AVQA_ROOT, 'music_avqa_train.json'),
            os.path.join(AVQA_ROOT, 'train.json'),
        ]
        found = None
        for p in alt_paths:
            if os.path.exists(p):
                found = p
                break
        if found is None:
            print(f"ERROR: 找不到 MUSIC-AVQA 训练标注文件，已尝试以下路径：")
            print(f"  {QA_TRAIN_JSON}")
            for p in alt_paths:
                print(f"  {p}")
            print(f"\n请先运行：ls {AVQA_ROOT}/ 确认标注文件名称，然后修改本脚本的 QA_TRAIN_JSON 变量")
            return
        qa_path = found
    else:
        qa_path = QA_TRAIN_JSON

    print(f"读取标注文件：{qa_path}")
    with open(qa_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    samples = []
    skipped = 0

    for item in raw:
        # 兼容不同字段名
        video_id = item.get('video_id') or item.get('vid')
        question_id = item.get('question_id') or item.get('qid') or item.get('id')
        question = item.get('question_content') or item.get('question')
        answer = item.get('anser') or item.get('answer')  # 注意 MUSIC-AVQA 原始 typo: 'anser'
        q_type = item.get('type') or item.get('question_type') or ['Unknown']

        if not video_id or not question or not answer:
            skipped += 1
            continue

        video_path = os.path.join(VIDEO_DIR, f'{video_id}.mp4')
        audio_path = os.path.join(AUDIO_DIR, f'{video_id}.mp3')

        samples.append({
            'video_id': str(video_id),
            'question_id': str(question_id),
            'type': q_type,
            'video_path': video_path,
            'audio_path': audio_path,
            'question': question,
            'answer': answer,
        })

    print(f"生成样本数：{len(samples)}，跳过：{skipped}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"已保存至：{OUTPUT_PATH}")

    # 验证第一条
    if samples:
        print(f"\n第一条样本预览：")
        s = samples[0]
        print(f"  video_id: {s['video_id']}, question: {s['question'][:40]}..., answer: {s['answer']}")

if __name__ == '__main__':
    main()

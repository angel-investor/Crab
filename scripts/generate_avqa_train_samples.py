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
QA_TRAIN_JSON = os.path.join(AVQA_ROOT, 'avqa-train.json')
VIDEO_DIR_REAL = os.path.join(AVQA_ROOT, 'avqa-videos/MUSIC-AVQA-videos-Real')
VIDEO_DIR_SYNTH = os.path.join(AVQA_ROOT, 'avqa-videos/MUCIS-AVQA-videos-Synthetic')  # 注意拼写 typo
AUDIO_DIR = os.path.join(AVQA_ROOT, 'audio')
CONVERTED_LABEL_DIR = os.path.join(AVQA_ROOT, 'converted_label')
OUTPUT_PATH = 'data/music_avqa_data/valid_train_samples.json'

def get_video_path(video_id: str) -> str:
    """根据 video_id 前缀自动判断是合成视频还是真实视频，返回正确路径。"""
    # 合成视频：以字母前缀开头（esa, eva, evv, sa 等）
    if video_id[:2].isalpha():
        return os.path.join(VIDEO_DIR_SYNTH, f'{video_id}.mp4')
    else:
        return os.path.join(VIDEO_DIR_REAL, f'{video_id}.mp4')

def make_label_text(answer: str) -> str:
    """生成训练目标文本，与推理输出格式匹配。"""
    return f"According to the video and audio, the answer is {answer}."

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

        video_path = get_video_path(str(video_id))
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

    # 同时生成 converted_label/{question_id}.txt 训练目标文件
    os.makedirs(CONVERTED_LABEL_DIR, exist_ok=True)
    label_generated = 0
    for s in samples:
        label_path = os.path.join(CONVERTED_LABEL_DIR, f"{s['question_id']}.txt")
        if not os.path.exists(label_path):
            with open(label_path, 'w', encoding='utf-8') as lf:
                lf.write(make_label_text(s['answer']))
            label_generated += 1
    print(f"已生成 {label_generated} 个 converted_label txt 文件 (路径: {CONVERTED_LABEL_DIR})")

    # 验证第一条
    if samples:
        print(f"\n第一条样本预览：")
        s = samples[0]
        print(f"  video_id: {s['video_id']}, question: {s['question'][:40]}..., answer: {s['answer']}")

if __name__ == '__main__':
    main()

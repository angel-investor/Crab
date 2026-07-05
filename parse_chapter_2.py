from docx import Document
import re

doc = Document("zzz-paper/第6版.docx")
text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract(start_pattern, end_pattern):
    starts = [m.start() for m in re.finditer(start_pattern, text)]
    ends = [m.start() for m in re.finditer(end_pattern, text)]
    
    if len(starts) >= 1 and len(ends) >= 1:
        # Assuming second match is the actual chapter content if TOC has the first match
        s = starts[1] if len(starts) > 1 else starts[0]
        # Find the first end that is AFTER s
        valid_ends = [e for e in ends if e > s]
        if valid_ends:
            return text[s:valid_ends[0]][:2000] # Return first 2000 chars for overview
    return "Not found"

chap2 = extract(r"第二章\s+相关工作与理论基础", r"第三章\s+研究方法")

if chap2 != "Not found":
    print("Found Chapter 2. Checking for keywords...")
    print("SAM/分割:", "SAM" in chap2 or "分割" in chap2)
    print("LoRA/Hyper-LoRA/PEFT:", "LoRA" in chap2 or "PEFT" in chap2 or "高效微调" in chap2)
    print("Q-Former:", "Q-Former" in chap2)
else:
    print("Chapter 2 not found")


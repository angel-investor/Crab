from docx import Document
import re
doc = Document("zzz-paper/第6版.docx")
text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# using regex to find chapters
def extract(start_pattern, end_pattern):
    # Find all matches for start
    starts = [m.start() for m in re.finditer(start_pattern, text)]
    ends = [m.start() for m in re.finditer(end_pattern, text)]
    if len(starts) >= 2 and len(ends) >= 2:
        return text[starts[1]:ends[1]]
    elif len(starts) == 1 and len(ends) >= 1:
        return text[starts[0]:ends[0]]
    else:
        return "Not found"

print("==== 1.1 ====")
print(extract(r"1\.1\s+研究背景", r"1\.2\s+研究目标"))
print("\n==== 1.3 ====")
print(extract(r"1\.3\s+研究意义", r"1\.4\s+论文结构"))

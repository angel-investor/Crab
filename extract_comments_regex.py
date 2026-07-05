import zipfile
import re
import sys

def get_word_xml_content(docx_file):
    try:
        with zipfile.ZipFile(docx_file) as docx:
            comments_xml = docx.read('word/comments.xml').decode('utf-8')
            
            # Find all comments
            comment_blocks = re.findall(r'<w:comment\b[^>]*w:author="([^"]*)"[^>]*>(.*?)</w:comment>', comments_xml, re.DOTALL)
            
            comments = []
            for author, block in comment_blocks:
                # Extract text from w:t tags
                texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', block, re.DOTALL)
                text = "".join(texts)
                comments.append(f"批注人 [{author}]: {text}")
                
            return "\n".join(comments) if comments else "No comments extracted."
    except Exception as e:
        return f"Error: {str(e)}"

print(get_word_xml_content(sys.argv[1]))

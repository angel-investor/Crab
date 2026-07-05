import zipfile
import sys
from bs4 import BeautifulSoup

def get_word_comments(docx_file):
    try:
        with zipfile.ZipFile(docx_file) as docx:
            if 'word/comments.xml' not in docx.namelist():
                return "No comments found."
            
            xml_content = docx.read('word/comments.xml').decode('utf-8')
            soup = BeautifulSoup(xml_content, 'xml')
            
            comments = soup.find_all('comment')
            if not comments:
                return "No <comment> tags found."
            
            result = []
            for comment in comments:
                author = comment.get('w:author', 'Unknown')
                texts = [t.text for t in comment.find_all('t')]
                result.append(f"批注人 [{author}]: {''.join(texts)}")
            
            return "\n".join(result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(get_word_comments(sys.argv[1]))

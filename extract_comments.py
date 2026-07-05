import zipfile
import xml.etree.ElementTree as ET
import sys

def get_word_xml_content(docx_file):
    try:
        with zipfile.ZipFile(docx_file) as docx:
            if 'word/comments.xml' not in docx.namelist():
                return "No comments found in the document."
            
            comments_xml = docx.read('word/comments.xml')
            
            # Word XML namespaces
            ns = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            }
            
            comments_tree = ET.fromstring(comments_xml)
            
            comments = []
            for comment in comments_tree.findall('.//w:comment', ns):
                author = comment.get(f"{{{ns['w']}}}author")
                
                # extract text
                texts = []
                for t in comment.findall('.//w:t', ns):
                    if t.text:
                        texts.append(t.text)
                text = "".join(texts)
                comments.append(f"批注人 [{author}]: {text}")
                
            return "\n".join(comments)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(get_word_xml_content(sys.argv[1]))
    else:
        print("Please provide a docx file path.")

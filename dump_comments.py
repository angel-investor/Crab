import zipfile
import sys

def dump_comments(docx_file):
    try:
        with zipfile.ZipFile(docx_file) as docx:
            if 'word/comments.xml' in docx.namelist():
                xml_content = docx.read('word/comments.xml').decode('utf-8')
                print(xml_content[:2000] + "..." if len(xml_content) > 2000 else xml_content)
            else:
                print("No comments.xml found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dump_comments(sys.argv[1])

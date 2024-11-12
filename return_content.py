import json
from pathlib import Path


DATA_PATH = Path("~/Desktop/project").expanduser() 
INDEX_FILE = DATA_PATH / "index.jsonl"

def load_index():
    index = {}
    with INDEX_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            index[entry["docid"]] = entry["filename"]
    return index

# Use a cached index to avoid reloading for each request
file_index = load_index()

# Retrieve file by docid
def get_file_by_docid(docid: str):
    # Check if the docid exists in the index
    filename = file_index.get(docid)
    print(filename)
    if not filename:
        return None
    

    # Construct the file path based on the filename in the index
    file_path = DATA_PATH / filename
    print(f'file_path', file_path)
    if not file_path.exists():
        return None

    # Read and return file content
    with file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    return content
print(get_file_by_docid(100))

def get_item(docid: str):
    content = get_file_by_docid(docid)

    return {"docid": docid, "content": content}
print(get_item(100))
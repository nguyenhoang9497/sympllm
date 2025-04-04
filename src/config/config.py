import os
from pathlib import Path

def get_absolute_paths():
    return  Path(__file__).parent.parent.parent.absolute()

localOllamaClientHost = "http://127.0.0.1:11434"
CHROMA_PATH = os.path.join(get_absolute_paths(), 'chroma')
DOCUMENT_PATH = os.path.join(get_absolute_paths(), 'data')
PROCESSED_FILES_PATH = os.path.join(CHROMA_PATH, "processed_files.json")
PROMPT_TEMPLATE = """
You are a helpful assistant. You can only make answers based on the provided context.
If information isn't available in context to answer, politely say you don't have knowledge about that topic.

Answer this question: {question} based on the context: {context}

"""
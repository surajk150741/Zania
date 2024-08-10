import sys
import os
sys.path.append(os.getcwd())
from langchain_community.document_loaders import PyPDFLoader
from core.llm import embeddings
from core.config import setting
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
import os
text_splitter = SemanticChunker(embeddings)

def load_or_create_chroma_vector_store(handbook_file_path: str):
    """
    Load or create a chroma vector store for the given handbook.

    Args:
        handbook_file_path (str): Path to the handbook file.

    Returns:
        chroma: The chroma vector store.
    """
    file_name = os.path.splitext(os.path.basename(handbook_file_path))[0]
    db_directory = "db/" + file_name +"v1"
    print('db',db_directory)

    if os.path.exists(db_directory) and os.path.isdir(db_directory):
        # If the directory exists, load the existing Chroma vector store
        doc_search = Chroma(persist_directory=db_directory, embedding_function=embeddings)
        print('vs')
    else:
        # If the directory does not exist, load the document, split it, embed each chunk, and create a new Chroma vector store
        loader = PyPDFLoader(handbook_file_path)
        document_chunks = loader.load_and_split(text_splitter=text_splitter)
        print('vs',document_chunks[0].page_content)
        doc_search = Chroma.from_documents(document_chunks, embeddings, persist_directory=db_directory)
        # doc_search.persist()

    return doc_search

if __name__ == "__main__":
    handbook_file_path = setting.HANDBOOK_FILE
    
    try:
        chroma_vector_store = load_or_create_chroma_vector_store(handbook_file_path)
        print('vsc',chroma_vector_store)
        # Perform operations with chroma_vector_store if needed
    except Exception as e:
        print(f"An error occurred: {e}")

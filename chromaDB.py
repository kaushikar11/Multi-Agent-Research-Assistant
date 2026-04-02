from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os, shutil
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "./papers"
def load_documents():
    print("loading documents......")
    loader = PyPDFDirectoryLoader(DATA_PATH, glob = "*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    print("splitting text......")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500, 
        length_function = len,
        add_start_index = True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def save_to_chroma(chunks : List[Document]):

    print("saving to chroma......")
    CHROMA_PATH="./chroma1"
    print(f"Using path: {CHROMA_PATH}")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


def generate_data_store():
        
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def main():
    generate_data_store()

if __name__ ==  "__main__":
    main()

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


PDF_FOLDER = "./papers"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
 
 

def ingest():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
 
    if not pdf_files:
        print("No PDFs found in ./papers")
        return
 
    print(f"Found {len(pdf_files)} PDF(s)\n")
 
    docs = []
    for i, filename in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Reading: {filename}")
            path = os.path.join(PDF_FOLDER, filename)
            loader = PyMuPDFLoader(path)
            loaded = loader.load()
            docs.extend(loaded)
            print(f"         → {len(loaded)} pages loaded")
    
    print(f"\nSplitting {len(docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"→ {len(chunks)} chunks created")

    print(f"\nEmbedding chunks (this may take a moment)...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    batch_size = 50
    all_chunks_done = 0
    vector_db = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            vector_db.add_documents(batch)
        all_chunks_done += len(batch)
        pct = int((all_chunks_done / len(chunks)) * 100)
        bar = ("█" * (pct // 5)).ljust(20)
        print(f"\r  [{bar}] {pct}% ({all_chunks_done}/{len(chunks)})", end="", flush=True)
 
    print(f"\n\nSaving index to {FAISS_INDEX_PATH}...")
    vector_db.save_local(FAISS_INDEX_PATH)
 
    print(f"\nDone! {len(chunks)} chunks indexed from {len(pdf_files)} PDF(s)")
 
 
if __name__ == "__main__":
    ingest()
 
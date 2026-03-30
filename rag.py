from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from chromadb.utils import embedding_functions
import os

load_dotenv()

DOCS_FOLDER = "docs"
COLLECTION_NAME = "miei_documenti"

def init_vector_store():
    client = chromadb.Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    documents, ids = [], []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = [line.strip() for line in content.split("\n") if line.strip()]
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                documents.append(chunk)
                ids.append(doc_id)
                print(f"[INGESTION] Caricato chunk: {doc_id}")
    if documents:
        collection.add(documents=documents, ids=ids)
        print(f"\n[INGESTION] Totale chunks caricati: {len(documents)}\n")
    return collection

def retrieve(collection, query: str, n_results: int = 3) -> list:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]

def ask(collection, llm, question: str) -> str:
    print(f"\n[RAG] Domanda: {question}")
    relevant_chunks = retrieve(collection, question)
    context = "\n".join(relevant_chunks)
    print(f"[RAG] Chunks recuperati:\n{context}\n")
    messages = [
        SystemMessage(content="""Sei un assistente utile. Rispondi in italiano.
Rispondi SOLO basandoti sul contesto fornito.
Se la risposta non è nel contesto, dì 'Non ho informazioni su questo argomento.'"""),
        HumanMessage(content=f"Contesto:\n{context}\n\nDomanda: {question}")
    ]
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    print("=== RAG Minimale ===\n")
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    print("[SETUP] Inizializzazione vector store...")
    collection = init_vector_store()
    print("RAG pronto! Scrivi 'quit' per uscire.\n")
    while True:
        question = input("Tu: ")
        if question.lower() == "quit":
            break
        answer = ask(collection, llm, question)
        print(f"\nRAG: {answer}\n")
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import chromadb
from chromadb.utils import embedding_functions
import os, re

load_dotenv()

DOCS_FOLDER = "docs2"
COLLECTION_NAME = "novatech_docs"
CHUNK_SIZE = 3
CHUNK_OVERLAP = 1
TOP_K = 5

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    text = re.sub(r'#+\s', '', text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    chunks = []
    i = 0
    while i < len(lines):
        chunk = lines[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def load_documents(folder: str):
    documents, ids, metadatas = [], [], []
    for filename in os.listdir(folder):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                documents.append(chunk)
                ids.append(doc_id)
                metadatas.append({"source": filename, "chunk_index": i})
                print(f"[INGESTION] {doc_id}: {chunk[:60]}...")
    return documents, ids, metadatas

def init_vector_store():
    client = chromadb.Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    documents, ids, metadatas = load_documents(DOCS_FOLDER)
    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"\n[INGESTION] Totale chunks: {len(documents)}\n")
    return collection

def retrieve(collection, query: str):
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]
    return [(c, s, d) for c, s, d in zip(chunks, sources, distances) if d < 1.5]

class RAGWithMemory:
    def __init__(self, collection):
        self.collection = collection
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")
        self.history = []

    def ask(self, question: str) -> str:
        retrieved = retrieve(self.collection, question)
        if not retrieved:
            return "Non ho trovato informazioni rilevanti nei documenti."
        context = "\n\n".join([f"[{s}]: {c}" for c, s, d in retrieved])
        print(f"\n[RAG] Chunks recuperati ({len(retrieved)}):")
        for c, s, d in retrieved:
            print(f"  - [{s}] (distanza: {d:.2f}): {c[:60]}...")
        self.history.append(HumanMessage(content=f"""Contesto:
{context}

Domanda: {question}

Rispondi basandoti sul contesto. Cita sempre la fonte usando [nome_file].
Se la risposta non è nel contesto, dì 'Non ho informazioni su questo.'"""))
        messages = [
            SystemMessage(content="""Sei un assistente per NovaTech.
Rispondi in italiano, in modo preciso e conciso.
Cita sempre la fonte delle informazioni usando [nome_file].
Non inventare informazioni non presenti nel contesto.""")
        ] + self.history
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        return response.content

if __name__ == "__main__":
    print("=== RAG v2 — NovaTech Q&A ===\n")
    print("[SETUP] Caricamento documenti...")
    collection = init_vector_store()
    rag = RAGWithMemory(collection)
    print("RAG pronto! Scrivi 'quit' per uscire.\n")
    while True:
        question = input("Tu: ")
        if question.lower() == "quit":
            break
        answer = rag.ask(question)
        print(f"\nRAG: {answer}\n")
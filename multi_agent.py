from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from chromadb.utils import embedding_functions
import os

load_dotenv()

def init_vector_store():
    client = chromadb.Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="novatech_multi",
        embedding_function=embedding_fn
    )
    docs_folder = "docs2"
    documents, ids, metadatas = [], [], []
    for filename in os.listdir(docs_folder):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(docs_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for i, line in enumerate(lines):
                documents.append(line)
                ids.append(f"{filename}_{i}")
                metadatas.append({"source": filename})
    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return collection

class RAGWorker:
    def __init__(self, collection):
        self.collection = collection
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")

    def run(self, query: str) -> str:
        print(f"  [RAGWorker] Cerco nei documenti: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        chunks = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]
        distances = results["distances"][0]
        filtered = [(c, s) for c, s, d in zip(chunks, sources, distances) if d < 1.5]
        if not filtered:
            return "Non ho trovato informazioni rilevanti nei documenti."
        context = "\n".join([f"[{s}]: {c}" for c, s in filtered])
        response = self.llm.invoke([
            SystemMessage(content="Sei un assistente NovaTech. Rispondi in italiano citando sempre la fonte [nome_file]. Se non trovi l'informazione dì 'Non ho informazioni su questo.'"),
            HumanMessage(content=f"Contesto:\n{context}\n\nDomanda: {query}")
        ])
        return response.content

class CalcWorker:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")

    def run(self, query: str) -> str:
        print(f"  [CalcWorker] Calcolo: '{query}'")
        response = self.llm.invoke([
            SystemMessage(content="Sei una calcolatrice precisa. Estrai l'espressione matematica dalla richiesta, calcola il risultato e rispondi in italiano mostrando i passaggi."),
            HumanMessage(content=query)
        ])
        return response.content

class ChatWorker:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")

    def run(self, query: str) -> str:
        print(f"  [ChatWorker] Rispondo genericamente: '{query}'")
        response = self.llm.invoke([
            SystemMessage(content="Sei un assistente utile e conciso. Rispondi in italiano."),
            HumanMessage(content=query)
        ])
        return response.content

class PlannerAgent:
    def __init__(self, collection):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")
        self.rag_worker = RAGWorker(collection)
        self.calc_worker = CalcWorker()
        self.chat_worker = ChatWorker()

    def decide(self, query: str) -> str:
        response = self.llm.invoke([
            SystemMessage(content="""Sei un orchestratore. Analizza la richiesta e rispondi con UNA SOLA parola:
- RAG → se la domanda riguarda NovaTech, i suoi prodotti, prezzi, funzionalità, roadmap, problemi
- CALC → se la domanda richiede un calcolo matematico
- CHAT → per qualsiasi altra domanda generica

Rispondi SOLO con RAG, CALC o CHAT. Nient'altro."""),
            HumanMessage(content=query)
        ])
        decision = response.content.strip().upper()
        return decision if decision in ["RAG", "CALC", "CHAT"] else "CHAT"

    def run(self, query: str) -> str:
        print(f"\n[Planner] Analizzo: '{query}'")
        decision = self.decide(query)
        print(f"[Planner] Decisione: → {decision}Worker")
        if decision == "RAG":
            return self.rag_worker.run(query)
        elif decision == "CALC":
            return self.calc_worker.run(query)
        else:
            return self.chat_worker.run(query)

if __name__ == "__main__":
    print("=== Multi-Agent System ===\n")
    print("[SETUP] Caricamento documenti...")
    collection = init_vector_store()
    planner = PlannerAgent(collection)
    print("\nSistema pronto! Scrivi 'quit' per uscire.")
    print("\nProva a chiedere:")
    print("  - 'Quanto costa il piano Pro?'        → RAGWorker")
    print("  - 'Quanto fa 347 * 12?'               → CalcWorker")
    print("  - 'Cos'è il project management?'      → ChatWorker\n")
    while True:
        query = input("Tu: ")
        if query.lower() == "quit":
            break
        response = planner.run(query)
        print(f"\nRisposta: {response}\n")
# multi_agent.py — Sistema Multi-Agente con RAG

## Architettura

Il sistema è composto da quattro componenti principali:

**Planner** — Riceve la query in input e decide quale worker
attivare, usando Claude come LLM per il routing.

**RAGWorker** — Recupera informazioni da ChromaDB (vector store),
concatena i chunk rilevanti al prompt e genera la risposta.

**CalcWorker** — Valuta espressioni matematiche in sicurezza
usando Python eval() con un namespace ristretto.

**ChatWorker** — Fallback generico: risponde con Claude senza
contesto documentale aggiuntivo.

## Stack tecnico

| Componente       | Tecnologia                |
|------------------|---------------------------|
| LLM              | Claude (claude-sonnet-4-6)|
| Vector DB        | ChromaDB                  |
| Embedding model  | sentence-transformers     |
| Framework        | Python 3.14 + langchain   |
| Env management   | python-dotenv             |

## Limiti noti

- Il routing del Planner è deterministico ma fragile:
  dipende dalla risposta testuale dell'LLM.
- ChromaDB restituisce sempre n risultati anche se
  irrilevanti (nessuna soglia di similarity).
- Nessuna memoria conversazionale tra sessioni.
- CalcWorker non gestisce errori di sintassi.

## Idee future

- [ ] Aggiungere similarity threshold nel RAGWorker
- [ ] Memoria persistente (vedi agent_with_memory.py)
- [ ] Streaming delle risposte
- [ ] API REST con FastAPI
- [ ] Test automatici per ogni worker
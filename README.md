# python-ai-agent-lab

RAG pipeline, multi-agent system and LLM tools built from scratch in Python.

---

## Overview

This repository documents a progressive build of AI agent systems using the Anthropic API and open-source tooling. Each module is self-contained and introduces one new concept on top of the previous.

---

## Modules

| File | Description |
|------|-------------|
| `hello_llm.py` | Basic API call to Claude — prompt in, response out |
| `agent.py` | Stateful agent class with conversation history |
| `agent_with_tools.py` | Tool-calling agent with calculator and note search |
| `agent_with_memory.py` | Persistent memory via JSON file + structured logging |
| `rag.py` | Minimal RAG pipeline with ChromaDB and sentence-transformers |
| `rag_v2.py` | Improved RAG with better chunking and similarity filtering |
| `multi_agent.py` | Multi-agent system with Planner routing to specialized workers |

---

## Architecture — multi_agent.py

```
User query
    │
    ▼
 Planner  ──── Claude API
    │
    ├──► RAGWorker   → ChromaDB → docs2/
    ├──► CalcWorker  → Python eval()
    └──► ChatWorker  → Claude API
```

The **Planner** reads the incoming query and routes it to the appropriate worker:
- `RAGWorker` — retrieves relevant chunks from the vector store and generates a grounded answer
- `CalcWorker` — evaluates mathematical expressions
- `ChatWorker` — fallback for general conversation via LLM

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Claude (claude-sonnet-4-6) via Anthropic API |
| Vector DB | ChromaDB |
| Embedding model | sentence-transformers |
| Framework | Python 3.14 + LangChain |
| Environment | python-dotenv |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/python-ai-agent-lab.git
cd python-ai-agent-lab

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install anthropic langchain langchain-anthropic chromadb sentence-transformers python-dotenv

# Add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run any module
python multi_agent.py
```

---

## Known Limitations

- Planner routing depends on LLM text output — brittle for edge cases
- ChromaDB returns `n` results regardless of relevance (no similarity threshold in v1)
- No conversational memory across sessions in the multi-agent system
- `CalcWorker` does not handle malformed expressions gracefully

---

## Roadmap

- [ ] Add similarity threshold filtering to RAGWorker
- [ ] Persistent session memory across conversations
- [ ] Streaming responses
- [ ] REST API layer with FastAPI
- [ ] Unit tests for each worker

---

## License

MIT

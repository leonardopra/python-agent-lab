from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import math, json, os
from datetime import datetime

load_dotenv()

LOG_FILE = "agent_log.txt"
MEMORY_FILE = "memory.json"

def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Risultato: {result}"
    except Exception as e:
        return f"Errore: {e}"

def search_notes(query: str) -> str:
    try:
        with open("notes.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        results = [line.strip() for line in lines if query.lower() in line.lower()]
        return "Trovato:\n" + "\n".join(results) if results else f"Nessun risultato per '{query}'"
    except FileNotFoundError:
        return "File notes.txt non trovato."

def save_memory(history: list):
    serialized = []
    for msg in history:
        if isinstance(msg, SystemMessage):
            serialized.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            serialized.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "ai", "content": msg.content})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    log(f"[MEMORY] History salvata ({len(serialized)} messaggi)")

def load_memory() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        serialized = json.load(f)
    history = []
    for msg in serialized:
        if msg["role"] == "system":
            history.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            history.append(AIMessage(content=msg["content"]))
    log(f"[MEMORY] History caricata ({len(history)} messaggi)")
    return history

class AgentWithMemory:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")
        loaded = load_memory()
        if loaded:
            self.history = loaded
            log("[AGENT] Sessione ripresa dalla memoria esistente")
        else:
            self.history = [
                SystemMessage(content="""Sei un assistente utile. Rispondi in italiano.
Hai accesso a questi tool:
- calculator(expression): per calcoli, es. calculator('2+2')
- search_notes(query): per cercare nelle note, es. search_notes('LangChain')

Quando serve un tool rispondi SOLO così:
TOOL: nome_tool
INPUT: input_del_tool

Altrimenti rispondi normalmente.""")
            ]
            log("[AGENT] Nuova sessione avviata")

    def run_tool(self, tool_name: str, tool_input: str) -> str:
        log(f"[TOOL] Chiamato: {tool_name} | Input: {tool_input}")
        if tool_name == "calculator":
            result = calculator(tool_input)
        elif tool_name == "search_notes":
            result = search_notes(tool_input)
        else:
            result = f"Tool '{tool_name}' non riconosciuto."
        log(f"[TOOL] Risultato: {result}")
        return result

    def chat(self, user_input: str) -> str:
        log(f"[USER] {user_input}")
        self.history.append(HumanMessage(content=user_input))
        response = self.llm.invoke(self.history)
        response_text = response.content

        if "TOOL:" in response_text and "INPUT:" in response_text:
            lines = response_text.strip().split("\n")
            tool_name = lines[0].replace("TOOL:", "").strip()
            tool_input = lines[1].replace("INPUT:", "").strip()
            tool_result = self.run_tool(tool_name, tool_input)
            self.history.append(AIMessage(content=response_text))
            self.history.append(HumanMessage(content=f"Risultato del tool: {tool_result}"))
            final_response = self.llm.invoke(self.history)
            response_text = final_response.content

        self.history.append(AIMessage(content=response_text))
        log(f"[AGENT] {response_text}")
        save_memory(self.history)
        return response_text

if __name__ == "__main__":
    agent = AgentWithMemory()
    print("\nAgente con memoria pronto! Scrivi 'quit' per uscire.\n")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"\nAgente: {response}\n")
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import math

load_dotenv()

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Risultato: {result}"
    except Exception as e:
        return f"Errore nel calcolo: {e}"

def search_notes(query: str) -> str:
    try:
        with open("notes.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        results = [line.strip() for line in lines if query.lower() in line.lower()]
        return "Trovato:\n" + "\n".join(results) if results else f"Nessun risultato per '{query}'"
    except FileNotFoundError:
        return "File notes.txt non trovato."

class AgentWithTools:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")
        self.history = [
            SystemMessage(content="""Sei un assistente utile. Rispondi in italiano.
Hai accesso a questi tool:
- calculator(expression): per fare calcoli, es. calculator('2+2')
- search_notes(query): per cercare nelle note, es. search_notes('LangChain')

Quando l'utente chiede un calcolo o di cercare qualcosa nelle note,
rispondi SOLO con il tool da usare in questo formato esatto:
TOOL: nome_tool
INPUT: input_del_tool

Altrimenti rispondi normalmente.""")
        ]

    def run_tool(self, tool_name: str, tool_input: str) -> str:
        print(f"\n[LOG] Tool chiamato: {tool_name} | Input: {tool_input}")
        if tool_name == "calculator":
            result = calculator(tool_input)
        elif tool_name == "search_notes":
            result = search_notes(tool_input)
        else:
            result = f"Tool '{tool_name}' non riconosciuto."
        print(f"[LOG] Risultato tool: {result}\n")
        return result

    def chat(self, user_input: str) -> str:
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
        return response_text

if __name__ == "__main__":
    agent = AgentWithTools()
    print("Agente con tools pronto! Scrivi 'quit' per uscire.\n")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"Agente: {response}\n")
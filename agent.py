from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

class Agent:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-6")
        self.history = [
            SystemMessage(content="Sei un assistente utile e conciso. Rispondi sempre in italiano.")
        ]

    def chat(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        response = self.llm.invoke(self.history)
        self.history.append(AIMessage(content=response.content))
        return response.content

if __name__ == "__main__":
    agent = Agent()
    print("Agente pronto! Scrivi 'quit' per uscire.\n")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"Agente: {response}\n")
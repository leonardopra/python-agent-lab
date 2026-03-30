from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("Spiegami cos'è un agente LLM in 2 frasi.")
print("Risposta Claude:")
print(response.content)
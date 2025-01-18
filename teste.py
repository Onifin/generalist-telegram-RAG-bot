from langchain_google_genai import ChatGoogleGenerativeAI
from rag import RAG

# Inicializar o modelo
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Criar instância do RAG
rag = RAG(llm)

# Carregar documento
rag.load_document('pje.txt')

# Gerar resposta
response = rag.generate_response("do que trata o manual do advogado?")

print(response["answer"])
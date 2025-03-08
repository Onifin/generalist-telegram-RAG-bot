from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader
from typing import Dict
import yaml

# Carregar o arquivo YAML
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class RAG:
    def __init__(self, llm, persist_directory="./db", max_history=5):
        """
        Initialize RAG with a language model and conversation history support.
        
        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI)
            persist_directory (str): Directory to persist the vector database
            max_history (int): Maximum number of conversation turns to maintain
        """

        print("Iniciando modelo")
        self.llm = llm
        self.embedding = OllamaEmbeddings(model="all-minilm:33m")
        print("Modelo iniciado")

        self.persist_directory = persist_directory
        self.max_history = max_history
        self.conversation_history = []
        self.history_handler = history_handler()
        
        print("Criando banco de dados de veetores")

        self.chunk_size = system_prompt = config['retrieval_settings']['chunk_size']
        self.chunk_overlap = system_prompt = config['retrieval_settings']['chunk_overlap']
        self.files_dir = config['documents']['path']
        self.vectordb = self.create_vector_db(files_dir = self.files_dir, chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)
       
        print("Banco de dados criado")

        system_prompt = config['prompt_template']['system']

        # Initialize prompt template with conversation history
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human","""
                **Histórico de Mensagens:**
                {conversation_history}

                **Pergunta:**
                {input}

                **Contexto:**
                {context}"""
            )
        ])
        
        print("Iniciando RAG")
        # Initialize retriever
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity"
        )
        
        # Initialize chains
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        print("RAG iniciada")

    def create_vector_db(self, files_dir, chunk_size=1000, chunk_overlap=200):
        """
        Load and process a text document into the vector database.
        
        Args:
            file_path (str): Path to the text file
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """

        loader = DirectoryLoader(files_dir, glob="./*.txt", loader_cls=TextLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        texts = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(documents = texts,
                                        persist_directory = self.persist_directory,
                                        embedding = self.embedding)
        
        return vectordb

    def _format_conversation_history(self, user_id: str) -> str:
        """
        Format the conversation history into a string.
        
        Returns:
            str: Formatted conversation history
        """

        self.conversation_history = self.history_handler.get_user_history(user_id)

        if not self.conversation_history:
            return "Nenhum histórico de conversa anterior."
            
        formatted_history = []

        for i, (query, response) in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Pergunta {i}: {query}")
            formatted_history.append(f"Resposta {i}: {response}\n")
            
        return "\n".join(formatted_history)

    def generate_response(self, query: str, user_id: str) -> Dict:
        """
        Generate a response based on the query using RAG and conversation history.
        
        Args:
            query (str): The input query
            
        Returns:
            dict: The response from the retrieval chain
        """
        # Prepare input with conversation history
        input_dict = {
            "input": query,
            "conversation_history": self._format_conversation_history(user_id)
        }
        
        # Generate response
        result = self.retrieval_chain.invoke(input_dict)
        
        # Update conversation history
        if "answer" in result:
            self.conversation_history.append((query, result["answer"]))

            # Maintain only the last max_history turns
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)

            self.history_handler.update_history(user_id, self.conversation_history)
            
        return result

class history_handler:
    def __init__(self):
        self.history_list = dict()

    def get_user_history(self, user_id):
        try:
            return self.history_list[user_id]
        except:
            self.history_list[user_id] = []
            return self.history_list[user_id]

    def update_history(self, user_id, history):
        self.history_list[user_id] = history
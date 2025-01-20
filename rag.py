from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from typing import List, Dict

class RAG:
    def __init__(self, llm, persist_directory="./db", max_history=5):
        """
        Initialize RAG with a language model and conversation history support.
        
        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI)
            persist_directory (str): Directory to persist the vector database
            max_history (int): Maximum number of conversation turns to maintain
        """
        self.llm = llm
        self.embedding = OllamaEmbeddings(model="mxbai-embed-large")

        self.persist_directory = persist_directory
        self.max_history = max_history
        self.conversation_history = []
        
        self.vectordb = Chroma(
            collection_name="documents",
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )
        
        # Initialize prompt template with conversation history
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente virtual do que tem conhecimento sobre os chapéu de palha, e seu nome é Luffy da Silva. Responda as perguntas baseando-se no contexto fornecido e no histórico da conversa."),
            ("human", """Histórico da conversa:
            {conversation_history}
            
            Caso a mensagem seja uma pergunta, responda usando o contexto fornecido, se ele for relevante (eu não possuo acesso ao contexto, apenas você possui. caso ele não seja relevante, não precisa ser mencionado) NÃO FALE SOBRE O CONTEXTO NA SUA RESPOSTA. Pergunta: {input}. Contexto: {context}.""")
        ])
        
        # Initialize retriever
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2}
        )
        
        # Initialize chains
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)

    def load_document(self, file_path, chunk_size=1000, chunk_overlap=0):
        """
        Load and process a text document into the vector database.
        
        Args:
            file_path (str): Path to the text file
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        loader = TextLoader(file_path = file_path, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        text_contents = [doc.page_content for doc in texts]
        self.vectordb.add_texts(texts=text_contents)

    def _format_conversation_history(self) -> str:
        """
        Format the conversation history into a string.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.conversation_history:
            return "Nenhum histórico de conversa anterior."
            
        formatted_history = []
        for i, (query, response) in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Pergunta {i}: {query}")
            formatted_history.append(f"Resposta {i}: {response}\n")
            
        return "\n".join(formatted_history)

    def generate_response(self, query: str) -> Dict:
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
            "conversation_history": self._format_conversation_history()
        }
        
        # Generate response
        result = self.retrieval_chain.invoke(input_dict)
        
        # Update conversation history
        if "answer" in result:
            self.conversation_history.append((query, result["answer"]))
            # Maintain only the last max_history turns
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
        
        return result

    def clear_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []

    def get_conversation_history(self) -> List[tuple]:
        """
        Get the current conversation history.
        
        Returns:
            List[tuple]: List of (query, response) pairs
        """
        return self.conversation_history.copy()
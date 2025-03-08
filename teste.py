from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persist_db = 'db'

embeddings = OllamaEmbeddings(model="all-minilm:33m")

loader = TextLoader(file_path = "extracted_text.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents = texts,
                                 persist_directory = persist_db,
                                 embedding = embeddings)

retriever = vectordb.as_retriever(
            search_type="similarity"
        )
        
        # Initialize chains
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        print("RAG iniciada")
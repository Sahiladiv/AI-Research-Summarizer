from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from PyPDF2 import PdfReader
import os


class CreateChromaDatabase:

    def __init__(self, key):
        self.chroma_path = 'chroma_db/'
        self.directory_path = 'research_papers/'
        self.key = key
    
    def generate_data_stores(self):
        documents = self.load_document()
        extracted_document = self.extract_text_from_pdfs(documents)
        chunks = self.split_text(documents)
        self.save_to_chroma(chunks)

    def load_document(self):
        loader = DirectoryLoader(self.directory_path, glob="**/*.pdf")
        documents = loader.load()
        return documents

    def extract_text_from_pdfs(self, documents):
        extracted_texts = {}
        for doc in documents:
            document_path = doc.metadata["source"]
            raw_text = ""
            pdf_reader = PdfReader(document_path)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content + "\n"
            extracted_texts[os.path.basename(document_path)] = raw_text
        return extracted_texts

    def split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        chunks = text_splitter.split_documents(documents)
        return chunks

    def save_to_chroma(self, chunks):
        embeddings = OpenAIEmbeddings(openai_api_key=self.key)
        db = Chroma.from_documents(
            chunks, embeddings, persist_directory=self.chroma_path 
        )
        db.persist()
        print("Chunks have been saved in Chroma Database!")


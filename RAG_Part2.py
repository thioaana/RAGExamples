import os, time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._migration")


from langchain.schema import Document



# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader


# 1. DOWNLOAD TEXT FROM SPECIFIC URLS AND SAVE THEM LOCALLY
from langchain_community.document_loaders import WebBaseLoader, TextLoader

# List of blog URLs
urls = [
    "https://blog.langchain.dev/announcing-langsmith",
    "https://blog.langchain.com/announcing-langsmith-is-now-a-transactable-offering-in-the-azure-marketplace/",
    "https://blog.langchain.com/benchmarking-question-answering-over-csv-data/"
]

# Load and clean with WebBaseLoader
loader = WebBaseLoader(urls)
docs = loader.load()

import os
save_path = "data/blog_texts"
os.makedirs(save_path, exist_ok=True)

for i, doc in enumerate(docs, start=1):
    filename = f"post_{i}.txt"
    with open(os.path.join(save_path, filename), "w", encoding="utf-8") as f:
        f.write(doc.page_content)


# 2. UPLOAD LLM AND AMBEDDINGS MODELS
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")


# 3. LOAD THE SAVED TEXTS
loaders = [
    TextLoader('data/blog_texts/post_1.txt'),
    TextLoader('data/blog_texts/post_2.txt'),
    TextLoader('data/blog_texts/post_3.txt')
]
docs = []
for l in loaders:
    docs.extend(l.load())

# print(f"Number of documents : {len(docs)}")
# print(docs[0].page_content[:500])


# **Retrieving Full Docments rathere than Chunks**
# Here, We will the return Original documents aren't too big in length

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

child_spitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="Full_Documents",
    embedding_function=embeddings
)

store = InMemoryStore()

full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_spitter,
)

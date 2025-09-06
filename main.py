import os, time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._migration")

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain.chains import RetrievalQA
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader # new library
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import DocArrayInMemorySearch # new library
from langchain.indexes import VectorstoreIndexCreator


# IMPORT LLM & EMBEDDING MODELS
# llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.1)
# llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

pathToFile = 'data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=pathToFile, csv_args={"delimiter": ","})


# index = VectorstoreIndexCreator(
#     vectorstore_cls=DocArrayInMemorySearch,
#     embedding=embeddings).from_loaders([loader])


# Check for GPU usage (Ollama-specific, best effort)
if os.environ.get("OLLAMA_NUM_GPU", "0") != "0":
    print("Ollama is configured to use GPU.")
else:
    print("Ollama is likely running on CPU (default).")


docs = loader.load()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

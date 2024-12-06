import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import gradio as gr
import asyncio
from groq import AsyncGroq


load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("API key not found. Please set up your `.env` file or visit https://www.groq.com/demo for a demo.")
    exit()

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
rag_llm = ChatGroq(model="llama3-8b-8192", max_tokens=4096)  # Used for create initial runbook and final output
rag_llm2 = ChatGroq(model="llama3-70b-8192", temperature=0.1, max_tokens=4096)  # Used to create initial runbook

# For Data Ingestion
class CustomDirectoryLoader(DirectoryLoader):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def _load(self, path):
        if path.endswith(".txt"):
            return TextLoader(path).load()
        elif path.endswith(".pdf"):
            return PyPDFLoader(path).load()
        else:
            return []

# Check if documents are already indexed, and avoid re-indexing
DOCUMENTS_INDEXED_PATH = "./data/indexed_documents.pkl"

if os.path.exists(DOCUMENTS_INDEXED_PATH):
    print("Documents are already indexed.")
    vectorstore = Chroma(persist_directory=DOCUMENTS_INDEXED_PATH, embedding_function=embed_model)
    retriever = vectorstore.as_retriever()
else:
    # Create the loader for the directory
    loader = CustomDirectoryLoader("./data/", use_multithreading=True)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    documents = loader.load_and_split(text_splitter=text_splitter)  # Load text
    vectorstore = Chroma.from_documents(documents, embedding=embed_model, collection_name="groq_rag",persist_directory=DOCUMENTS_INDEXED_PATH)
    retriever = vectorstore.as_retriever()

    print(f"Documents indexed: {len(documents)}")




#RAG pipeline

RAG_SYSTEM_PROMPT = """
You are a solution architect at a well-known company in charge of deploying Databricks on AWS for POC purposes.
Write a runbook for your task."
```
{context}
```
Generate a structured runbook in markdown format that includes:
1. Prerequisites
2. Step-by-step instructions
3. Links to relevant documentation
4. Expected outcomes at each step
5. Troubleshooting tips for common issues

If you don't have enough information to answer, please state that clearly.
"""

RAG_HUMAN_PROMPT = "{input}"

RAG_OUTPUT="Here are relevant outputs {output1} and {output2} create the final runbook"


FINAL_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT + """
Here are relevant outputs:
```
{output1}
````
```
{output2}
```
Refer these outputs to create the final runbook.
"""

# Unified Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages([ ("system", RAG_SYSTEM_PROMPT),    ("human", RAG_HUMAN_PROMPT)])

FINAL_RAG_PROMPT = ChatPromptTemplate.from_messages([ ("system", FINAL_SYSTEM_PROMPT),    ("human", RAG_HUMAN_PROMPT)])


# Helper Functions
def format_docs(docs: List[Document]) -> str:
    """Format the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Combined Pipeline
rag_pipeline = (
    {
        # Step 1: Retrieve and format documents
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough()  # User query
    }
    | {
        # Step 2: Generate two intermediate outputs (simulating two LLMs)
        "output1": RAG_PROMPT | rag_llm | StrOutputParser(),
        "output2": RAG_PROMPT | rag_llm2 | StrOutputParser()
    }
    | {
        # Step 3: Combine outputs and generate the final result
        "context": RunnablePassthrough(),  # Pass formatted document context
        "input": RunnablePassthrough(),  # Pass user query
        "output1": RunnablePassthrough(),  # Pass intermediate output1
        "output2": RunnablePassthrough(),  # Pass intermediate output2
    }
    | FINAL_RAG_PROMPT  # Combine all into the final structured runbook
    | rag_llm  # Run final LLM step
    | StrOutputParser()  # Parse output
)

async def main(query):
    client = AsyncGroq()
    stream = await  rag_pipeline.ainvoke(query) 
    return stream

#Synchronous wrapper function
def generate_runbook(query):
    output = rag_pipeline.invoke(query)
    yield output


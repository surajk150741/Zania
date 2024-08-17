import sys
import os
sys.path.append(os.getcwd())

from core.config import setting
from services.utils.document_to_vector_store import load_or_create_chroma_vector_store

import langchain
from core.llm import llm
import time
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
# to print chain retrive context info
langchain.verbose=True
llm = llm
vectorstore = load_or_create_chroma_vector_store(setting.HANDBOOK_FILE)
retriever = vectorstore.as_retriever()
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm, include_original = True
    )

##### Using contextual compression for more relevant document############
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=multi_query_retriever
)
##### Using contextual compression for more relevant document############  ## This I am only testing now, will deploy if there will be huge documents#########


# class CustomRetriever(BaseRetriever):
#     def __init__(self, vectorstore):
#         self.vectorstore = vectorstore

#     def _get_relevant_documents(self, query: str) -> List[Document]:
#         # Custom retrieval logic here
#         documents = self.vectorstore.get_relevant_documents(query)
#         # Example: sort documents or apply additional filtering
#         return documents
# documents = vectorstore.get_relevant_documents(query)
def docs_qa(query: str):
    # docs = multi_query_retriever.invoke(query)
    docs = compression_retriever.invoke(query)

    # # Step 1: Retrieve documents using the custom retriever
    # documents = vectorstore.get_relevant_documents(query)
    # retriever = documents
    # retrieved_docs = retriever._get_relevant_documents(query)
    
    # # Step 2: Format the retrieved documents as context
    # context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # # Step 3: Generate an answer using the LLM
    # prompt = f"""
    # Use the following pieces of context to answer the user's question.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # ----------------------------------------------------------------
    # {context}
    # ----------------------------------------------------------------

    # The user's question that you have to answer from the above pieces of context is:
    # {query}
    # """
    # response = llm(prompt)
    
    return docs
docs = docs_qa('What is the name of the company?')
# print(docs)

# Helper function for printing docs. Going to use it to see the documents my retriever is fetching
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
pretty_print_docs(docs)

import sys
import os
sys.path.append(os.getcwd())

from core.config import setting
from services.utils.document_to_vector_store import load_or_create_chroma_vector_store

import langchain
from core.llm import llm,embeddings
import time
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAI
# to print chain retrive context info
langchain.verbose=True
llm = llm
vectorstore = load_or_create_chroma_vector_store(setting.HANDBOOK_FILE)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm, include_original = True
    )

##### Using contextual compression for more relevant document############
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=multi_query_retriever
# )
##### Using contextual compression for more relevant document############  ## This I am only testing now, will deploy if there will be huge documents#########

##### Using Embedding_filter is cheaper and faster than compression_retriever############
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)   
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=multi_query_retriever
)
##### Using Embedding_filter is cheaper and faster than compression_retriever############  ## This I am only testing now, will deploy if there will be huge documents#########
########## YES, this is better


def docs_qa(query: str):
    # docs = multi_query_retriever.invoke(query)
    docs = compression_retriever.invoke(query)  ### need to decide whether to pass this docs or the docs only

    
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
d = pretty_print_docs(docs)


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

# to print chain retrive context info
langchain.verbose=True
llm = llm
vectorstore = load_or_create_chroma_vector_store(setting.HANDBOOK_FILE)
retriever = vectorstore.as_retriever()
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
    docs = retriever.invoke(query)

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


# retriever=vecstore.as_retriever()

# ############### Multi Query Retriever ###############
# from langchain.retrievers.multi_query import MultiQueryRetriever

# multi_query_retriever = MultiQueryRetriever.from_llm(
#     retriever=retriever, llm=llm, include_original = True
#     )
# def docs_qa_sql(question:str,llm=llm):
#     time_s=time.time()
    
#     #initialize RetrievalQA chain
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=multi_query_retriever ,verbose=True)#, callbacks=[handler])
#     #configure the system prompt for retrive context info
#     qa.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template="""
# You are a mysql expert. Given an input question, create a syntactically correct mysql query.
# Use the following format:

# SQLQuery: "SQL Query to run"
# Use the following pieces of context to genarate the sql query as answer to the user's question.
# ----------------------------------------------------------------
# {context}
# ----------------------------------------------------------------
# If you unable to generate the sql query, just say that you don't know, don't send any random sql query as an answer.
# The user's question that you have to convert to SQL query from the above pieces of context is:
# """

#     with get_openai_callback() as cb:
#         out=qa.invoke(question)
        
#     sql=out['result'].replace("```sql\n",'')
#     sql = sql.replace("\n```",'')
#     sql = sql.replace("SQLQuery: ",'')
#     out = sql.replace("\n"," ")
#     return out, cb.__dict__, time.time()-time_s


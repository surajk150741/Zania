import sys
import os
sys.path.append(os.getcwd())

from core.config import setting
from services.utils.document_to_chromadb import load_or_create_chroma_vector_store

import langchain
from core.llm import llm
import time

# to print chain retrive context info
langchain.verbose=True

vecstore = load_or_create_chroma_vector_store(setting.HANDBOOK_FILE)
retriever=vecstore.as_retriever()

############### Multi Query Retriever ###############
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm, include_original = True
    )
def docs_qa_sql(question:str,llm=llm):
    time_s=time.time()
    
    #initialize RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=multi_query_retriever ,verbose=True)#, callbacks=[handler])
    #configure the system prompt for retrive context info
    qa.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template="""
You are a mysql expert. Given an input question, create a syntactically correct mysql query.
Use the following format:

SQLQuery: "SQL Query to run"
Use the following pieces of context to genarate the sql query as answer to the user's question.
----------------------------------------------------------------
{context}
----------------------------------------------------------------
If you unable to generate the sql query, just say that you don't know, don't send any random sql query as an answer.
The user's question that you have to convert to SQL query from the above pieces of context is:
"""

    with get_openai_callback() as cb:
        out=qa.invoke(question)
        
    sql=out['result'].replace("```sql\n",'')
    sql = sql.replace("\n```",'')
    sql = sql.replace("SQLQuery: ",'')
    out = sql.replace("\n"," ")
    return out, cb.__dict__, time.time()-time_s


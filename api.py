from fastapi import FastAPI, Request, Path, HTTPException
app=FastAPI()
import uvicorn
import os
import time
from langchain_openai import ChatOpenAI
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import PromptTemplate
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd

import http.server
import socketserver
# from limebot import search_core
from main import search_core
from QA import docs_qa

ans = docs_qa("What is their vacation policy?")

# data={
# "query": "What is their vacation policy?",
# "rid": "2",
# "email":"surajk150741@gmail.com"
# }
# search_core = search_core(query=data["query"],rid=data["rid"],email=data["email"])
# ans = search_core.handbook_tool(data["query"])
# # out=search_core.execute(data["query"])
# print('uio',ans)

# @app.get('/agent_report_generation')
# def report_generation(query: str):
#     time_s=time.time()
#     output = search_core.execute(query)
#     answer = output['handbook_output']
#     elapsed_time = time.time() - time_s
#     print(f"Execution time: {elapsed_time} seconds")
#     return answer

# templates = Jinja2Templates(directory="C:/Users/suraj/OneDrive/Desktop/Personal/bhole/gen-AI/Zania_Assignment")
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     # Render the HTML template
#     return templates.TemplateResponse("index.html", {"request": request})
# if __name__=='__main__':
#     uvicorn.run("api_main:app", host='localhost', port=7000, reload=False, workers=1)


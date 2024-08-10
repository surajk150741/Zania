
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY_ZANIA")
import sys
from langchain_openai import ChatOpenAI
import openai
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.getcwd())

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embeddings=OpenAIEmbeddings
embeddings=embeddings(model="text-embedding-ada-002")

if __name__=="__main__":
    from langchain.schema import HumanMessage
    message = HumanMessage(
        content="who is building you?"
    )
    llm=llm
    print(llm.invoke([message]))
    
    embd=embeddings
    emb=embd.embed_query("Who is the CEO of the company?")
    print(len(emb))
    print(emb[:5])
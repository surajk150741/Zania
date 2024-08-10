import os
import sys
sys.path.append(os.getcwd())

import json
import time
from core.llm import llm
from memory import CustomMessageConverter
from core.config import setting
# from doc_QA import docs_qa
from QA import docs_qa
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool, Tool, BaseTool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.callbacks import FileCallbackHandler, get_openai_callback
import os

system_prompt="""
You are a helpful AI agent for the company Zania, which is an AI security startup based in San Francisco that develops autonomous AI agents for enterprise security. You are capable of doing following task:
- For inquiries regarding a handbook of Zania, which includes information about company's basic information and policies.
"""

class handbook_query(BaseModel):
    query: str = Field(...,description="The user query related to handbook")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")])


class search_core:
    def __init__(self,query:str,rid:str,email:str):
        self.query=query
        self.rid = rid
        self.email = email
        self.handbook_output='null'
        
    def handbook_tool(self,query: str)->str:
        """
        The handbook_tool provides basic information and policies of the company Zania. 
        It includes the core Policies, California Policies, closing statement and Acknowledgment of Receipt and Review.
        """
        self.handbook_output = docs_qa(query=query)
        return self.handbook_output

    def execute(self,query:str):
        
        chat_message_history = SQLChatMessageHistory(
            session_id=self.rid,
            connection_string=setting.CHAT_MEMORY_DATABASE,
            custom_message_converter=CustomMessageConverter(author_email=self.email),
        )
        
        memory = ConversationBufferWindowMemory(
            k=6,
            memory_key="chat_history",
            chat_memory=chat_message_history,
            return_messages=True
        )
        
        tools = [
            Tool(
                name="handbook_tool",
                args_schema=handbook_query,
                description="""
        The handbook_tool provides basic information and policies of the company Zania. 
        It includes the core Policies, California Policies, closing statement and Acknowledgment of Receipt and Review.
        """,
                func=self.handbook_tool
            )
        ]
        
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)

        with get_openai_callback() as cb:
            output=agent_executor.invoke({'input':query,'chat_history':memory.load_memory_variables({})['chat_history']})
        
        print(output['chat_history'],'\n\n')
        
        # print('handbook_test',self.handbook_output)
        if self.handbook_output != 'null':
            print('handbook_test2')
            memo_content= self.handbook_output
            memory.save_context({"input": query}, {"output": memo_content})

        final_response={'handbook_output':self.handbook_output,
                        'llm_generate_output': output['output']}
        

        return final_response

def execute(query: str,rid:str,email:str):
    """
    Execute the search operation with error handling.

    Args:
        query (str): The user query.

    Returns:
        dict: The search output.
    """
    search=search_core(query,rid,email)
    # ans2 = search.handbook_tool(query)
    ans = search.execute(query)
    return ans
    # except Exception as e:
    #     print({'function_name': 'execute', 'error_message': f"An error occurred during execute: {e}"})
    #     return {'error': str(e)}

if __name__=="__main__":
    data={
    "query": "What is the name of ceo?",
    "rid": "1",
    "email":"surajk150741@gmail.com"
}

    out=execute(**data)
    #print(json.dumps(out, indent=4))
    print(out)
import os
import sys
sys.path.append(os.getcwd())


from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from core.config import setting
from services.utils.document_to_vector_store import load_or_create_chroma_vector_store
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field

vectorstore = load_or_create_chroma_vector_store(setting.HANDBOOK_FILE)
query = 'What is the name of the company?'
class Semantic_Doc_Retriever(BaseModel):
    """A Semantic Doc retriever that returns the top k documents that are most relevant to the user query.

    This retriever is provided with a vector store db that is created by following step:
    1. Loading a pdf handbook
    2. Making document chunks of the pdf document.
    3. Passing document chunks through Chroma Vector Store along with OpenAI Embeddings.

    This retriever is also provided a query which is the user query 
    """
    vectorstore: vectorstore
    k: int = Field(default=4, description="Number of documents to retrieve")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve the most relevant documents for the given query."""
        return self.vectorstore.similarity_search(query, k=self.k)

retriever = Semantic_Doc_Retriever(vectorstore=vectorstore, k=5)
relevant_docs = retriever.get_relevant_documents("example query")

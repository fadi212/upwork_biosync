from llama_index.llms.openai import OpenAI
from llama_index.core import get_response_synthesizer, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever

from utils.config import (MAX_TOKENS, TEMPERATURE, SIMILARITY_TOP_K, OPENAI_API_KEY, logger)
from src.pinecone_service import create_or_load_vector_store_index

###################################################
# Prompt
###################################################
template = '''
Role: You are an expert medical analyst specializing in analyzing patient data for diagnosing different tests based on the input. 
You have been provided with patient information, including metrics such as pregnancies, glucose levels, 
blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age, and many more depending upon the nature of the test.

Task: You are asked to analyze the patient data and take the reference while diagnosing from the vectors retrieved. The vectors contain the guidelines for you to make a possible decision. 

Guidelines:
- Summarize Clearly: Provide a clear, concise analysis of the patient data.
- Data Insights: Extract key metrics and explain their significance.
- Diagnosis: Use the provided metrics to determine presence/absence of disease.
- Explain Decision: Explain the reasoning.
- Use Relevant Data: Don't add info outside given data.
- Accuracy: Keep it accurate to the provided vector data.

---
Extra Instructions from User: 
{extra_instructions}
---

Vectors Retrieved: "{context_str}"

Patient Report: {query_str}

Diagnosis:
'''
qa_prompt = PromptTemplate(template)

###################################################
# Custom RAG QueryEngine
###################################################
class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine with Pinecone-based retrieval."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str, extra_instructions: str = ""):
        # 1) Retrieve top-k nodes
        nodes = self.retriever.retrieve(query_str)
        # 2) Build context
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        logger.info(f"Context String: {context_str}")

        # 3) Format final prompt
        formatted_prompt = self.qa_prompt.format(
            context_str=context_str,
            query_str=query_str,
            extra_instructions=extra_instructions
        )

        # 4) Query LLM
        response = self.llm.complete(formatted_prompt)
        return str(response)


def query_index(index_name: str, query_text: str, extra_instructions: str = "") -> str:
    """
    Loads or creates the vector store index from Pinecone,
    sets up the RAG engine, and runs a query against it.
    """
    # 1) Create or load index
    index: VectorStoreIndex = create_or_load_vector_store_index(index_name)

    # 2) Build retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=SIMILARITY_TOP_K)

    # 3) Build response_synthesizer
    response_synthesizer = get_response_synthesizer(response_mode="refine")

    # 4) Create custom query engine
    query_engine = RAGStringQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        llm=OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            model="gpt-4o-mini",  # or "gpt-3.5-turbo", etc.
            max_tokens=MAX_TOKENS
        ),
        qa_prompt=qa_prompt
    )

    # 5) Query
    answer = query_engine.custom_query(query_text, extra_instructions)
    return answer

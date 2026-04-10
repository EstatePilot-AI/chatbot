from langchain_core.output_parsers import StrOutputParser
from prompt_templates import NEGOTIATE_PROMPT

class NegotiationChain:
    def __init__(self, llm):
        self.chain = NEGOTIATE_PROMPT | llm | StrOutputParser()
        
    def execute(self, query: str, property_details: dict, history: str) -> str:
        return self.chain.invoke({
            "query": query, 
            "property_details": str(property_details),
            "history": history
        })

from langchain_core.output_parsers import StrOutputParser
from prompt_templates import SEARCH_PROMPT

class SearchChain:
    def __init__(self, llm):
        self.chain = SEARCH_PROMPT | llm | StrOutputParser()
        
    def execute(self, query: str, properties: list, history: str) -> str:
        return self.chain.invoke({
            "query": query, 
            "properties": str(properties),
            "history": history
        })

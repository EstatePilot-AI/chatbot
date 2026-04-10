from langchain_core.output_parsers import StrOutputParser
from prompt_templates import COMPARE_PROMPT

class CompareChain:
    def __init__(self, llm):
        self.chain = COMPARE_PROMPT | llm | StrOutputParser()
        
    def execute(self, query: str, comparison_data: list, history: str) -> str:
        return self.chain.invoke({
            "query": query, 
            "comparison_data": str(comparison_data),
            "history": history
        })

import re
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.output_parsers import StrOutputParser
from prompt_templates import UNIFIED_ANALYZER_PROMPT

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self, llm):
        self.chain = UNIFIED_ANALYZER_PROMPT | llm | StrOutputParser()

    def analyze(self, query: str, history: str, last_properties: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calls the LLM once to get intent, filters, and property selection.
        """
        # Create a lightweight string representation of last properties for context
        props_context = "No recently shown properties."
        if last_properties:
            short_props = []
            for p in last_properties[:10]: # Limit to top 10 for context
                short_props.append(
                    f"ID: {p.get('propertyId')}, Type: {p.get('propertyType')}, "
                    f"District: {p.get('district')}, Price: {p.get('price'):,}"
                )
            props_context = "\n".join(short_props)

        try:
            raw_response = self.chain.invoke({
                "query": query, 
                "history": history, 
                "last_properties": props_context
            })
            clean_json = self._clean_json_text(raw_response)
            data = json.loads(clean_json)
            
            # Normalize and ensure schema
            return self._normalize_analysis(data)
        except Exception as e:
            logger.error(f"Error in QueryAnalyzer: {e}")
            return {
                "intent": "chat", 
                "filters": {}, 
                "sort_by": "relevance", 
                "extra_preferences": ""
            }

    def _clean_json_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text.strip()

    def _normalize_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures intent exists and filters are present.
        """
        intent = data.get("intent", "search").lower()
        filters = data.get("filters", {})
        
        # Canonical keys
        canonical_filters = {
            "propertyType": filters.get("propertyType"),
            "finishingType": filters.get("finishingType"),
            "min_price": filters.get("min_price"),
            "max_price": filters.get("max_price"),
            "min_area": filters.get("min_area"),
            "max_area": filters.get("max_area"),
            "rooms": filters.get("rooms"),
            "bathrooms": filters.get("bathrooms"),
            "governorate": filters.get("governorate"),
            "city": filters.get("city"),
            "district": filters.get("district"),
        }
        
        return {
            "intent": intent,
            "filters": canonical_filters,
            "selected_property_id": data.get("selected_property_id"),
            "sort_by": data.get("sort_by", "relevance"),
            "extra_preferences": data.get("extra_preferences", "")
        }

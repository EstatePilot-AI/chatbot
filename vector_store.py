import logging
import json
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Number of candidates to retrieve from FAISS before filtering/scoring.
# Larger pool gives the FilterEngine more to work with.
DEFAULT_TOP_K = 20


def _property_to_text(prop: Dict[str, Any]) -> str:
    """
    Convert a property dict into a rich, searchable natural-language string.
    Skips zero / None / unknown values to keep the text clean.
    All available fields are included so the embedding captures location,
    type, price, area, and room details simultaneously.
    """
    parts: List[str] = []

    def add(label: str, val: Any, suffix: str = ""):
        if val and val not in (0, 0.0, "Unknown", "unknown", ""):
            parts.append(f"{label}: {val}{suffix}")

    # Property identity
    add("Type",       prop.get("propertyType"))
    add("Finishing",  prop.get("finishingType"))
    add("Status",     prop.get("propertyStatus"))

    # Location — include every level for richer semantic matching
    loc_parts = []
    for field in ("district", "city", "governorate", "country"):
        v = prop.get(field)
        if v and v not in (0, "Unknown", "unknown", ""):
            loc_parts.append(str(v))
    if loc_parts:
        parts.append(f"Location: {', '.join(loc_parts)}")

    # Physical details
    add("Price",     prop.get("price"),     " EGP")
    add("Area",      prop.get("area"),      " sqm")
    add("Rooms",     prop.get("rooms"))
    add("Bathrooms", prop.get("bathrooms"))

    # Building detail (useful for tiebreaking / specific lookups)
    add("Floor", prop.get("floorNumber"))
    add("Street", prop.get("street"))

    return " | ".join(parts) if parts else "Property details unavailable"


class VectorStoreManager:
    def __init__(
        self,
        embedding_model_name: str = "models/gemini-embedding-001",
    ):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        self.vector_store: FAISS | None = None

    def build_vector_store(self, properties: List[Dict[str, Any]]):
        if not properties:
            logger.warning("No properties to build vector store.")
            return

        documents = []
        for prop in properties:
            content = _property_to_text(prop)
            doc = Document(
                page_content=content,
                metadata={"property": json.dumps(prop, ensure_ascii=False)},
            )
            documents.append(doc)

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info(
            f"FAISS vector store rebuilt: {len(documents)} properties indexed."
        )

    def search_properties(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        if not self.vector_store:
            logger.warning("Vector store is not initialised yet.")
            return []

        results = self.vector_store.similarity_search(query, k=top_k)
        return [
            json.loads(doc.metadata["property"]) for doc in results
        ]

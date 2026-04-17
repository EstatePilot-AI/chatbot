import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from cache_manager import CacheManager
from decision_engine import DecisionEngine
from filter_engine import FilterEngine
from prompt_templates import CLARIFICATION_PROMPT, RECOMMEND_PROMPT
from query_analyzer import QueryAnalyzer
# from query_parser import QueryParser
# from router import IntentRouter
from search_chain import SearchChain
from compare_chain import CompareChain
from negotiation_chain import NegotiationChain
from session_manager import SessionManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_URL = "https://estatepilot.runasp.net/api/Property/GetAllPropertiesWithDetails"

# How many FAISS candidates to retrieve before filtering / scoring.
# Larger pool → FilterEngine has more candidates to work with.
FAISS_CANDIDATE_POOL = 100

# How many results to surface to the user after filtering + scoring.
TOP_N_RESULTS = 5

# Arabic fallback note injected when no property matched the hard filters.
FALLBACK_NOTE_AR = (
    "⚠️ مفيش عقار بالمواصفات دي بالظبط في قاعدة البيانات، "
    "بس دي أقرب النتايج المتاحة ليك:\n\n"
)

# ---------------------------------------------------------------------------
# Module-level singletons (created once at import time, reused across requests)
# ---------------------------------------------------------------------------
load_dotenv()

cache_mgr   = CacheManager(api_url=API_URL, refresh_interval_sec=600)
session_mgr = SessionManager()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.warning(
        "GOOGLE_API_KEY is not set. Google Gemini requests may fail or be rejected."
    )

logger.info("Loading Google Gemini 2.0 Flash model ...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.7,
    max_tokens=4096,
)


query_analyzer           = QueryAnalyzer(llm=llm)
search_chain_instance     = SearchChain(llm=llm)
compare_chain_instance    = CompareChain(llm=llm)
negotiation_chain_instance = NegotiationChain(llm=llm)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: fetching properties and building FAISS index …")
    await cache_mgr.refresh_cache()
    yield
    # Teardown (extend here if needed)


app = FastAPI(
    title="AI Real Estate Advisor",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Configuration - Allow requests from any origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: int
    query: str


class SmartSearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ---------------------------------------------------------------------------
# Shared pipeline helper
# ---------------------------------------------------------------------------
# Thresholds for clarification and relaxation
MIN_CONFIDENCE_THRESHOLD = 0.30  # Clarify if top result is below this
RELAXATION_CONFIDENCE    = 0.60  # If below this, try relaxation

import re

# (Dumb regex selection logic removed)

def _run_rag_pipeline(
    query: str,
    accumulated_filters: Dict[str, Any],
    sort_by: str = "relevance",
    faiss_top_k: int = FAISS_CANDIDATE_POOL,
    result_limit: int = TOP_N_RESULTS,
) -> tuple[List[Dict[str, Any]], bool, float]:
    """
    Core RAG pipeline with multi-level fallback:
    1. Level 1: Standard (Strict penalties).
    2. Level 2: Relaxation (Looser price/area/matching).
    """
    vector_store_mgr = cache_mgr.vector_store_manager
    raw_candidates   = vector_store_mgr.search_properties(query, top_k=faiss_top_k)

    if not raw_candidates:
        return [], False, 0.0

    # ── Level 1: Standard Search ──
    filtered, used_fallback = FilterEngine.apply_filters(raw_candidates, accumulated_filters)
    
    def get_ranked_with_conf(props, filters, sort, relax):
        if not props: return [], 0.0
        # Re-score with relaxation
        scored = []
        for p in props:
            # We must inject the score into the object to sort by it as secondary
            # We also pass the query for keyword fallback
            p["_conf"] = DecisionEngine._score(p, filters, relaxation=relax, raw_query=query)
            scored.append(p)
        
        # In DecisionEngine.rank_properties, it calls _score again. 
        # We need to make sure it uses the relaxation factor.
        # But DecisionEngine is static, so we'll just sort here manually to be safe or update DecisionEngine.
        
        # Primary Sort: sort_by (price_desc, price_asc, area_desc, area_asc)
        def sort_key(p):
            primary = 0
            if sort == "price_desc": primary = -(p.get("price") or 0)
            elif sort == "price_asc": primary = (p.get("price") or 0)
            elif sort == "area_desc": primary = -(p.get("area") or 0)
            elif sort == "area_asc": primary = (p.get("area") or 0)
            return (primary, -p["_conf"])
            
        scored.sort(key=sort_key)
        max_c = max([p["_conf"] for p in scored]) if scored else 0.0
        return scored, max_c

    # Attempt Level 1
    ranked_l1, conf_l1 = get_ranked_with_conf(filtered, accumulated_filters, sort_by, relax=1.0)
    
    # Attempt Level 2 Relaxation if Level 1 is weak
    if conf_l1 < RELAXATION_CONFIDENCE and accumulated_filters:
        logger.info(f"Low L1 confidence ({conf_l1:.2f}), attempting L2 Relaxation...")
        ranked_l2, conf_l2 = get_ranked_with_conf(filtered, accumulated_filters, sort_by, relax=2.0)
        
        # If relaxation significantly helped or L1 was essentially empty, use L2
        if conf_l2 > conf_l1:
            return ranked_l2[:result_limit], used_fallback, conf_l2

    return ranked_l1[:result_limit], used_fallback, conf_l1


# ---------------------------------------------------------------------------
# POST /ai-advisor
# ---------------------------------------------------------------------------
@app.post("/ai-advisor")
async def ai_advisor_endpoint(req: ChatRequest):
    user_id = req.user_id
    query   = req.query
    logger.info(f"[{user_id}] Query received: {query}")

    # ── 1. Pull conversation history ─────────────────────────────────────
    history_str = session_mgr.get_formatted_history(user_id)

    # ── 2. Unified Analysis (Intent + Filters + Selection in ONE call) ────
    last_props = session_mgr.get_last_shown_properties(user_id)
    analysis = query_analyzer.analyze(query, history_str, last_properties=last_props)
    intent = analysis.get("intent", "search")
    
    # ── 2a. Check for Smart Property Selection ────────────────────────────
    if intent == "select_property":
        selected_id = analysis.get("selected_property_id")
        
        if selected_id:
            # Match Found
            egy_reply = f"تمام 👌\nhttps://estate-pilot-shop.vercel.app/properties/{selected_id}"
            logger.info(f"[{user_id}] AI Selection matched → ID {selected_id}")
            # Identify the property object for the return payload (optional but good)
            matched_prop = next((p for p in last_props if p.get("propertyId") == selected_id), None)
            top_props = [matched_prop] if matched_prop else []
        else:
            # Fallback (Ambiguous or Null)
            egy_reply = "تؤمرني يا فندم، تقدر تشوف كل العقارات المتاحة من هنا:\nhttps://estate-pilot-shop.vercel.app/properties"
            logger.info(f"[{user_id}] Selection ambiguous → returned general link")
            top_props = []

        session_mgr.add_interaction(user_id, query, egy_reply)
        return {
            "module": "Selection",
            "filters_extracted": session_mgr.get_accumulated_filters(user_id),
            "top_properties": top_props,
            "reply_in_egyptian_arabic": egy_reply,
        }
    
    # ── 3. Merge into session ─────────────────────────────────────────────
    # Backwards compatible mapping
    session_mgr.merge_filters(user_id, analysis)
    
    accumulated_filters = session_mgr.get_accumulated_filters(user_id)
    extra_preferences   = session_mgr.get_extra_preferences(user_id)
    
    sort_by = analysis.get("sort_by", "relevance")
    limit   = analysis.get("limit", TOP_N_RESULTS)

    logger.info(f"[{user_id}] Detected intent: {intent}")
    logger.info(f"[{user_id}] Accumulated filters: {accumulated_filters}")
    logger.info(f"[{user_id}] Extra preferences: {extra_preferences}")

    # ── 5. RAG pipeline ───────────────────────────────────────────────────
    augmented_query = f"{query} | Preferences: {extra_preferences}" if extra_preferences else query
    
    try:
        top_properties, used_fallback, confidence = _run_rag_pipeline(
            augmented_query, 
            accumulated_filters,
            sort_by=sort_by,
            result_limit=limit
        )
    except Exception as e:
        logger.error(f"[{user_id}] RAG pipeline error: {e}")
        top_properties = []
        used_fallback  = False
        confidence     = 0.0

    logger.info(f"[{user_id}] Pipeline → {len(top_properties)} results (conf={confidence:.2f})")

    # ── 6. Clarification gate ─────────────────────────────────────────────
    # Clarify if:
    # A) Vague query on first turn
    # B) OR confidence of best result is below threshold
    is_vague = session_mgr.all_filters_none(user_id) and not extra_preferences
    
    if (is_vague and session_mgr.is_first_turn(user_id)) or (confidence < MIN_CONFIDENCE_THRESHOLD and len(top_properties) > 0):
        reason = "Vague query" if is_vague else f"Low confidence ({confidence:.2f})"
        logger.info(f"[{user_id}] {reason} → clarification.")
        try:
            clarification_chain = CLARIFICATION_PROMPT | llm | StrOutputParser()
            egy_reply = clarification_chain.invoke({"query": query, "history": history_str}).strip()
        except Exception as e:
            logger.error(f"[{user_id}] Clarification chain error: {e}")
            egy_reply = "ممكن تقولي أكتر عن اللي بتدور عليه؟ زي الميزانية والمنطقة وعدد الأوض."

        session_mgr.add_interaction(user_id, query, egy_reply)
        return {
            "module": "Guided Conversation",
            "filters_extracted": accumulated_filters,
            "top_properties": [],
            "confidence_score": confidence,
            "explanation": f"Clarification: {reason}",
            "reply_in_egyptian_arabic": egy_reply,
        }

    # ── 7. Build base payload ─────────────────────────────────────────────
    output_payload: Dict[str, Any] = {
        "module":                  intent.title(),
        "filters_extracted":       accumulated_filters,
        "top_properties":          top_properties,
        "recommendation":          None,
        "comparison":              None,
        "negotiation":             None,
        "fallback_used":           used_fallback,
        "explanation":             "",
        "reply_in_egyptian_arabic": "",
    }

    egy_reply  = ""
    explanation = ""

    # ── 8. Intent dispatch ────────────────────────────────────────────────

    # ─ Search ─────────────────────────────────────────────────────────────
    if intent == "search":
        if top_properties:
            try:
                egy_reply = search_chain_instance.execute(
                    query=query,
                    properties=top_properties,
                    history=history_str,
                )
            except Exception as e:
                logger.error(f"[{user_id}] SearchChain failed, using hard fallback: {e}")
                # Construct a basic text fallback if LLM fails
                egy_reply = "آسف يا فندم، السيستم مهنج شوية في الكلام بس دي الشقق اللي لقتها لك وتناسبك:\n\n"
                for i, p in enumerate(top_properties[:3], 1):
                    egy_reply += f"{i}. {p.get('propertyType')} في {p.get('district')}، مساحة {p.get('area')}م، بسعر {p.get('price'):,} جنيه.\n"
                egy_reply += "\nممكن تسألني عن أي واحدة فيهم بالتفصيل لما السيستم يرجع طبيعي."
            explanation = "FAISS → filter → score → Hard Fallback (LLM failed)."
        else:
            egy_reply   = "مش لاقي عقارات تناسب البحث ده دلوقتي للأسف."
            explanation = "No properties found after filtering."
        
        # Save results for future selection
        if top_properties:
            session_mgr.set_last_shown_properties(user_id, top_properties)
            egy_reply += "\n\nحضرتك مهتم بأنهي شقة؟"

    # ─ Recommend ──────────────────────────────────────────────────────────
    elif intent == "recommend":
        best_prop: Optional[Dict] = top_properties[0] if top_properties else None
        output_payload["recommendation"] = best_prop
        if best_prop:
            try:
                recommend_chain = RECOMMEND_PROMPT | llm | StrOutputParser()
                egy_reply = recommend_chain.invoke({
                    "query":         query,
                    "best_property": str(best_prop),
                    "history":       history_str,
                }).strip()
            except Exception as e:
                logger.error(f"[{user_id}] Recommend failed, falling back to Search: {e}")
                egy_reply = search_chain_instance.execute(query, [best_prop], history_str)
            explanation = "Scored all candidates; top-ranked property recommended."
        else:
            egy_reply   = "مش لاقي عقار مناسب ليك دلوقتي للأسف."
            explanation = "No properties found to recommend."

    # ─ Compare ────────────────────────────────────────────────────────────
    elif intent == "compare":
        top_2 = top_properties[:2]
        output_payload["comparison"] = top_2
        if len(top_2) >= 2:
            try:
                egy_reply = compare_chain_instance.execute(
                    query=query,
                    comparison_data=top_2,
                    history=history_str,
                )
            except Exception as e:
                logger.error(f"[{user_id}] Compare failed, falling back to Search: {e}")
                egy_reply = search_chain_instance.execute(query, top_2, history_str)
            explanation = "Top-2 scored properties compared."
        else:
            egy_reply   = "محتاج على الأقل عقارين عشان أقارن بينهم — مش لاقي كفاية."
            explanation = "Not enough properties to compare."

    # ─ Negotiate ──────────────────────────────────────────────────────────
    elif intent == "negotiate":
        best_prop = top_properties[0] if top_properties else None
        output_payload["negotiation"] = best_prop
        if best_prop:
            try:
                egy_reply = negotiation_chain_instance.execute(
                    query=query,
                    property_details=best_prop,
                    history=history_str,
                )
            except Exception as e:
                logger.error(f"[{user_id}] Negotiate failed, falling back to Search: {e}")
                egy_reply = search_chain_instance.execute(query, [best_prop], history_str)
            explanation = "Best-scored property identified; negotiation advice generated."
        else:
            egy_reply   = "مش لاقي عقار محدد عشان نتفاوض عليه."
            explanation = "No property identified for negotiation."

    # ── 9. Prepend fallback note when soft scoring was used ───────────────
    if used_fallback and egy_reply:
        egy_reply = FALLBACK_NOTE_AR + egy_reply

    # ── 10. Finalise payload + save interaction ───────────────────────────
    output_payload["reply_in_egyptian_arabic"] = egy_reply.strip()
    output_payload["explanation"]              = explanation

    session_mgr.add_interaction(user_id, query, egy_reply.strip())
    logger.info(f"[{user_id}] Response dispatched successfully.")
    return output_payload


# ---------------------------------------------------------------------------
# POST /smartsearch
# ---------------------------------------------------------------------------
@app.post("/smartsearch")
async def smart_search_endpoint(req: SmartSearchRequest):
    """
    Lightweight endpoint: returns ordered property IDs only (no LLM call).
    Applies the full RAG pipeline (FAISS → filter → score) using filters
    extracted from the query via QueryParser.
    """
    query  = req.query
    top_k  = req.top_k
    logger.info(f"SmartSearch query: {query}")

    # Parse extraction directly from query
    try:
        analysis = query_analyzer.analyze(query, "No previous history.")
        extraction_filters = analysis.get("filters", {})
        sort_by = analysis.get("sort_by", "relevance")
        limit   = analysis.get("limit", top_k)
    except Exception as e:
        logger.error(f"SmartSearch QueryParser error: {e}")
        extraction_filters = {}
        sort_by = "relevance"
        limit = top_k

    try:
        top_properties, _, _ = _run_rag_pipeline(
            query,
            extraction_filters,
            sort_by=sort_by,
            faiss_top_k=max(limit * 4, FAISS_CANDIDATE_POOL),
            result_limit=limit,
        )
    except Exception as e:
        logger.error(f"SmartSearch pipeline error: {e}")
        top_properties = []

    property_ids = [
        p.get("propertyId") for p in top_properties if p.get("propertyId")
    ]
    return {"property_ids": property_ids}

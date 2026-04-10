import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Pure-Python scoring engine.  No LLM calls — fast and deterministic.

    Both public methods now accept a `filters` dict (the structured output
    of QueryParser) instead of a raw query string, so scoring is consistent
    with the extracted constraints rather than ad-hoc text parsing.
    """

    # ── Public API ────────────────────────────────────────────────────────

    @staticmethod
    def rank_properties(
        properties: List[Dict[str, Any]],
        filters: Dict[str, Any],
        sort_by: str = "relevance",
    ) -> List[Dict[str, Any]]:
        """
        Score properties and sort them.
        Primary Sort: sort_by (price_desc, price_asc, area_desc, area_asc)
        Secondary Sort: Soft score (relevance)
        """
        if not properties:
            return []

        filters = filters or {}
        
        # Calculate scores for all properties
        scored_data = []
        for p in properties:
            score = DecisionEngine._score(p, filters)
            scored_data.append({"prop": p, "score": score})

        # Multi-level sorting
        def sort_key(item):
            p = item["prop"]
            s = item["score"]
            
            # Primary sort values (default to 0/neutral for relevance)
            primary = 0
            if sort_by == "price_desc":
                primary = -(p.get("price") or 0)
            elif sort_by == "price_asc":
                primary = (p.get("price") or 0)
            elif sort_by == "area_desc":
                primary = -(p.get("area") or 0)
            elif sort_by == "area_asc":
                primary = (p.get("area") or 0)
            
            # Note: we use -s to sort by score descending (high score first)
            return (primary, -s)

        scored_data.sort(key=sort_key)
        
        return [item["prop"] for item in scored_data]

    @staticmethod
    def score_properties(
        properties: List[Dict[str, Any]],
        filters: Dict[str, Any],
        sort_by: str = "relevance",
    ) -> Optional[Dict[str, Any]]:
        """Return only the single best-ranked property (or None)."""
        ranked = DecisionEngine.rank_properties(properties, filters, sort_by)
        return ranked[0] if ranked else None

    # ── Weights for normalized scoring ──
    WEIGHTS = {
        "location": 0.35,
        "price": 0.25,
        "area": 0.15,
        "rooms_baths": 0.10,
        "type_finishing": 0.10,
        "preferences": 0.05
    }

    # ── Property Categories ──
    CATEGORIES = {
        "residential": ["apartment", "villa", "studio", "duplex", "penthouse", "twinhouse", "townhouse"],
        "commercial": ["office", "clinic", "shop", "pharmacy", "medical"]
    }

    @staticmethod
    def _strip_article(s: str) -> str:
        s = s.lower().strip()
        for prefix in ("ال", "el ", "al ", "el-", "al-"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        return s.strip()

    @staticmethod
    def _grade_location(filter_val: str, prop_val: str) -> float:
        """Graded location matching (0.0 to 1.0)."""
        if not filter_val or not prop_val:
            return 0.0
        
        f = DecisionEngine._strip_article(filter_val)
        p = DecisionEngine._strip_article(prop_val)
        
        if f == p:
            return 1.0
        if f in p or p in f:
            return 0.7
        return 0.0

    @staticmethod
    def _grade_property_type(filter_type: str, prop_type: str) -> float:
        """Graded property type matching (0.0 to 1.0)."""
        if not filter_type or not prop_type:
            return 0.0
            
        ft = filter_type.lower()
        pt = prop_type.lower()
        
        if ft == pt or ft in pt:
            return 1.0
            
        # Check if they share a category
        for cat, types in DecisionEngine.CATEGORIES.items():
            if any(t in ft for t in types) and any(t in pt for t in types):
                return 0.5
                
        return 0.0

    @staticmethod
    def _score(prop: Dict[str, Any], filters: Dict[str, Any], relaxation: float = 1.0, raw_query: str = "") -> float:
        """
        Calculates a normalized confidence score (0.0 to 1.0).
        relaxation: multiplier for tolerances / divisor for penalties.
                    1.0 = strict, > 1.0 = relaxed.
        raw_query: used as a fallback keyword check if structured filters are empty.
        """
        # Safe field reads
        price     = prop.get("price", 0) or 0
        area      = prop.get("area", 0) or 0
        rooms     = prop.get("rooms", 0) or 0
        bathrooms = prop.get("bathrooms", 0) or 0

        district    = str(prop.get("district", "")).lower()
        city        = str(prop.get("city", "")).lower()
        governorate = str(prop.get("governorate", "")).lower()
        prop_type   = str(prop.get("propertyType", "")).lower()
        finishing   = str(prop.get("finishingType", "")).lower()

        # ── 1. Location (0.0 - 1.0) ──
        loc_score = 1.0
        loc_components = []
        if filters.get("district"):
            loc_components.append(DecisionEngine._grade_location(filters["district"], district))
        if filters.get("city"):
            loc_components.append(DecisionEngine._grade_location(filters["city"], city))
        if filters.get("governorate"):
            loc_components.append(DecisionEngine._grade_location(filters["governorate"], governorate))
            
        if loc_components:
            # When relaxed, we give more credit to partial matches
            loc_score = sum(loc_components) / len(loc_components)
            if relaxation > 1.0 and loc_score > 0:
                loc_score = min(1.0, loc_score * 1.2)
        elif raw_query:
            # SAFETY NET: If no structured location, check if any property location field appears in raw query
            # Handles "Ain Shams" if LLM failed to extract it.
            q = raw_query.lower()
            if district in q or city in q or governorate in q:
                loc_score = 0.8
            else:
                loc_score = 0.5 # Neutral fallback instead of 1.0 to prioritize keyword matches

        # ── 2. Price (0.0 - 1.0) ──
        price_score = 1.0
        max_price = filters.get("max_price")
        min_price = filters.get("min_price")
        
        # Effective tolerance for expansion
        tol = 1.0 + (0.2 * (relaxation - 1.0)) # 1.0, 1.2, 1.4...

        if max_price and price > 0:
            if price <= max_price * tol:
                price_score = 1.0
            else:
                overshoot_ratio = (price - (max_price * tol)) / (max_price * tol)
                # Penalty is reduced by relaxation factor
                price_score = max(0.0, 1.0 - (overshoot_ratio * 2 / relaxation))
                
        elif min_price and price > 0:
            if price >= min_price / tol:
                price_score = 1.0
            else:
                shortfall_ratio = ((min_price / tol) - price) / (min_price / tol)
                price_score = max(0.0, 1.0 - (shortfall_ratio * 2 / relaxation))

        # ── 3. Area (0.0 - 1.0) ──
        area_score = 1.0
        min_area = filters.get("min_area")
        max_area = filters.get("max_area")
        
        if min_area and area > 0:
            if area >= min_area / tol:
                area_score = 1.0
            else:
                shortfall_ratio = ((min_area / tol) - area) / (min_area / tol)
                area_score = max(0.0, 1.0 - (shortfall_ratio * 2.5 / relaxation))
        elif max_area and area > 0:
            if area <= max_area * tol:
                area_score = 1.0

        # Small default boost for larger areas
        if not min_area and not max_area:
            area_score = min(1.0, 0.8 + (area / 1000))

        # ── 4. Rooms and Bathrooms (0.0 - 1.0) ──
        rb_score = 1.0
        rb_comps = []
        
        target_rooms = filters.get("rooms")
        if target_rooms and rooms > 0:
            diff = abs(int(rooms) - int(target_rooms))
            if diff == 0: rb_comps.append(1.0)
            elif diff == 1: rb_comps.append(0.5 * relaxation)
            else: rb_comps.append(max(0.0, 1.0 - (diff * 0.3 / relaxation)))
            
        target_baths = filters.get("bathrooms")
        if target_baths and bathrooms > 0:
            diff = abs(int(bathrooms) - int(target_baths))
            if diff == 0: rb_comps.append(1.0)
            elif diff == 1: rb_comps.append(0.5 * relaxation)
            else: rb_comps.append(max(0.0, 1.0 - (diff * 0.3 / relaxation)))
            
        if rb_comps:
            rb_score = min(1.0, sum(rb_comps) / len(rb_comps))

        # ── 5. Type and Finishing (0.0 - 1.0) ──
        type_score = 1.0
        type_comps = []
        
        if filters.get("propertyType"):
            grade = DecisionEngine._grade_property_type(filters["propertyType"], prop_type)
            if relaxation > 1.0 and grade > 0:
                grade = min(1.0, grade * 1.5)
            type_comps.append(grade)
            
        if filters.get("finishingType"):
            ft = filters["finishingType"].lower()
            if ft in finishing or finishing in ft:
                type_comps.append(1.0)
            else:
                type_comps.append(0.3 * (relaxation - 1.0)) # relaxed partial match
                
        if type_comps:
            type_score = min(1.0, sum(type_comps) / len(type_comps))

        # ── 6. Extra Preferences (0.0 - 1.0) ──
        pref_score = 1.0
        prefs = filters.get("extra_preferences", "").lower()
        if prefs:
            pref_score = 0.5
            if any(k in prefs for k in ["فخم", "luxury", "راقي"]):
                if "fully" in finishing or "متشطب" in finishing: pref_score += 0.25
                if price > 2000000: pref_score += 0.25
            if any(k in prefs for k in ["عيلة", "family", "كبير", "واسع"]):
                if area > 120: pref_score += 0.25
                if rooms > 1: pref_score += 0.25
            pref_score = min(1.0, pref_score)

        # ── Weighted Final Score ──
        w = DecisionEngine.WEIGHTS
        final_score = (
            (loc_score * w["location"]) +
            (price_score * w["price"]) +
            (area_score * w["area"]) +
            (rb_score * w["rooms_baths"]) +
            (type_score * w["type_finishing"]) +
            (pref_score * w["preferences"])
        )
        
        return round(final_score, 4)

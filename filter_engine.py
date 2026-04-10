import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)



def _strip_article(s: str) -> str:
    """
    Remove the Arabic definite article 'ال' and common English equivalents
    ('el ', 'al ') for fuzzy location matching.
    """
    s = s.lower().strip()
    for prefix in ("ال", "el ", "al ", "el-", "al-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip()


def _passes_hard_filters(prop: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Returns True only if the property satisfies extreme data validity.
    ALL numeric and location constraints are now handled via soft scoring.
    """
    price     = prop.get("price", 0)   or 0
    area      = prop.get("area", 0)    or 0
    status    = str(prop.get("propertyStatus", "")).lower()
    
    # QUALITY GATE: Reject invalid data
    if price <= 0 or area <= 0:
        return False
        
    # AVAILABILITY GATE: Reject sold items
    if "sold" in status or "تباع" in status:
        return False

    return True


class FilterEngine:
    """
    Two-phase retrieval filter applied after FAISS semantic search.

    Phase 1 — Hard filter: every active constraint must be satisfied.
    Phase 2 — Fallback: if nothing passes hard filter, return all candidates
               unfiltered so the DecisionEngine can soft-rank them, and flag
               `used_fallback=True` so the caller can inform the user.
    """

    @staticmethod
    def apply_filters(
        properties: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Returns (result_list, used_fallback).

        * used_fallback=False  → at least one property passed hard filters.
        * used_fallback=True   → no property passed; all candidates returned
                                 for soft scoring with fallback note to user.
        """
        if not properties:
            return [], False

        # If all filters are None there's nothing to filter — pass everything
        active = {k: v for k, v in filters.items() if v is not None}
        if not active:
            return properties, False

        hard_filtered = []
        for p in properties:
            if _passes_hard_filters(p, filters):
                hard_filtered.append(p)
            else:
                # Debug logging to find out why properties are being rejected
                p_id = p.get("propertyId", "??")
                district = p.get("district", "N/A")
                city = p.get("city", "N/A")
                logger.info(f"FilterEngine: Property {p_id} ({district}/{city}) rejected by hard filters for request {filters}")

        if hard_filtered:
            logger.info(
                f"FilterEngine: {len(hard_filtered)}/{len(properties)} "
                "properties passed hard filters."
            )
            return hard_filtered, False

        # ── Fallback ───────────────────────────────────────────────────────
        logger.warning(
            "FilterEngine: No properties passed hard filters. "
            "Falling back to full candidate list for soft scoring."
        )
        return properties, True

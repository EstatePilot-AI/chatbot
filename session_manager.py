from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Canonical filter schema — all keys always present, all default to None.
# Matches QueryParser output exactly.
# ---------------------------------------------------------------------------
# Canonical filter schema
EMPTY_FILTERS: Dict[str, Any] = {
    "propertyType":  None,
    "finishingType": None,
    "min_price":     None,
    "max_price":     None,
    "min_area":      None,
    "max_area":      None,
    "rooms":         None,
    "bathrooms":     None,
    "governorate":   None,
    "city":          None,
    "district":      None,
}

class SessionManager:
    """
    In-memory per-user session store.
    """

    def __init__(self):
        self.sessions: Dict[int, Dict[str, Any]] = {}

    def get_session(self, user_id: int) -> Dict[str, Any]:
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "history": [],
                "filters": dict(EMPTY_FILTERS),
                "extra_preferences": "",
                "last_shown_properties": [],
            }
        return self.sessions[user_id]

    def add_interaction(self, user_id: int, user_query: str, bot_response: str):
        session = self.get_session(user_id)
        session["history"].append({"user": user_query, "bot": bot_response})
        if len(session["history"]) > 5:
            session["history"] = session["history"][-5:]

    def set_last_shown_properties(self, user_id: int, properties: List[Dict[str, Any]]):
        self.get_session(user_id)["last_shown_properties"] = properties

    def get_last_shown_properties(self, user_id: int) -> List[Dict[str, Any]]:
        return self.get_session(user_id).get("last_shown_properties", [])

    def get_formatted_history(self, user_id: int) -> str:
        session = self.get_session(user_id)
        lines: List[str] = []
        for turn in session["history"]:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot:  {turn['bot']}")
        return "\n".join(lines) if lines else "No previous history."

    def merge_filters(self, user_id: int, extraction: Dict[str, Any]):
        """
        Merge results from QueryParser into the session.
        - filters: persistent merge (only overwrite if not None)
        - extra_preferences: semi-persistent (overwrite if extraction has value)
        """
        session = self.get_session(user_id)
        
        # Merge nested filters
        new_filters = extraction.get("filters", {})
        for key in EMPTY_FILTERS:
            val = new_filters.get(key)
            if val is not None:
                session["filters"][key] = val
        
        # Carry forward / overwrite extra_preferences
        new_prefs = extraction.get("extra_preferences", "")
        if new_prefs:
            session["extra_preferences"] = new_prefs

    def get_accumulated_filters(self, user_id: int) -> Dict[str, Any]:
        return self.get_session(user_id)["filters"]

    def get_extra_preferences(self, user_id: int) -> str:
        return self.get_session(user_id).get("extra_preferences", "")

    def reset_filters(self, user_id: int):
        self.get_session(user_id)["filters"] = dict(EMPTY_FILTERS)
        self.get_session(user_id)["extra_preferences"] = ""

    def is_first_turn(self, user_id: int) -> bool:
        return len(self.get_session(user_id)["history"]) == 0

    def all_filters_none(self, user_id: int) -> bool:
        filters = self.get_accumulated_filters(user_id)
        return all(v is None for v in filters.values())

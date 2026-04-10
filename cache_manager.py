import asyncio
import time
import requests
import logging
from typing import List, Dict, Any
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, api_url: str, refresh_interval_sec: int = 600):
        self.api_url = api_url
        self.refresh_interval_sec = refresh_interval_sec
        self.cached_properties: List[Dict[str, Any]] = []
        self.last_fetched = 0
        self.vector_store_manager = VectorStoreManager()
        self.is_running = False

    async def fetch_properties(self) -> List[Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.api_url, timeout=10),
            )
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Error fetching properties: {e}")
            return []

    async def refresh_cache(self):
        logger.info("Refreshing properties cache from API...")
        new_data = await self.fetch_properties()
        if new_data:
            self.cached_properties = new_data
            self.last_fetched = time.time()
            self.vector_store_manager.build_vector_store(self.cached_properties)
            logger.info(f"Cache refreshed with {len(self.cached_properties)} properties.")
        else:
            logger.warning("Failed to retrieve new data, keeping old cache if available.")

    async def auto_refresh_loop(self):
        self.is_running = True
        while self.is_running:
            await self.refresh_cache()
            await asyncio.sleep(self.refresh_interval_sec)

    def start_background_refresh(self):
        asyncio.create_task(self.auto_refresh_loop())

    def get_properties(self) -> List[Dict[str, Any]]:
        return self.cached_properties
    
    def get_property_by_id(self, property_id: int) -> Dict[str, Any]:
        for p in self.cached_properties:
            if p.get("propertyId") == property_id:
                return p
        return None

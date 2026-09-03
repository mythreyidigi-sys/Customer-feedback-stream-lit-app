"""
digital_footprint.py
=====================
Personal/individual use case: surfaces what's publicly discoverable about a
*consenting* individual (the app's own user, auditing themselves) across
search engines, social profiles and data-broker sites, and manages
takedown/removal-request workflows for outdated or harmful listings.

Intended flow: the user requests an audit of their own name/profile inside
the app (explicit self-service action) -- this module does not scrape or
aggregate data about third parties.

`DiscoverySourceClient` stubs a real lookup (search API, data-broker opt-out
API, social-platform API). Swap in the real integration per source; the
scoring/workflow logic here stays the same.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional


class ListingCategory(str, Enum):
    SEARCH_RESULT = "search_result"
    SOCIAL_PROFILE = "social_profile"
    DATA_BROKER = "data_broker"
    NEWS_ARTICLE = "news_article"
    OTHER = "other"


class TakedownStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    IN_PROGRESS = "in_progress"
    REMOVED = "removed"
    DENIED = "denied"


@dataclass
class DiscoveredListing:
    id: str
    category: ListingCategory
    source_name: str
    url: str
    snippet: str
    discovered_at: datetime
    contains_sensitive_pii: bool = False   # e.g. address, phone, DOB
    is_outdated: bool = False
    takedown_status: TakedownStatus = TakedownStatus.NOT_REQUESTED
    takedown_requested_at: Optional[datetime] = None


class DiscoverySourceClient:
    """Stub -- replace `search` with a real search/data-broker/social API call."""
    def __init__(self, source_name: str, category: ListingCategory,
                 search_fn: Optional[Callable[[str], List[dict]]] = None):
        self.source_name = source_name
        self.category = category
        self._search_fn = search_fn or (lambda query: [])

    def search(self, query: str) -> List[dict]:
        return self._search_fn(query)


class DigitalFootprintAuditor:
    """
    Runs a self-service audit for a consenting user, scores each listing for
    sensitivity/staleness, and tracks a takedown-request workflow to
    resolution.
    """

    _SENSITIVE_MARKERS = ["phone", "address", "date of birth", "dob", "ssn", "email:"]

    def __init__(self, subject_name: str, consent_confirmed: bool):
        if not consent_confirmed:
            raise ValueError(
                "DigitalFootprintAuditor requires explicit self-audit consent "
                "from the subject before any lookup runs."
            )
        self.subject_name = subject_name
        self.sources: List[DiscoverySourceClient] = []
        self.listings: List[DiscoveredListing] = []

    def register_source(self, client: DiscoverySourceClient) -> None:
        self.sources.append(client)

    def run_audit(self) -> List[DiscoveredListing]:
        found = []
        counter = len(self.listings)
        for client in self.sources:
            for raw in client.search(self.subject_name):
                counter += 1
                listing = DiscoveredListing(
                    id=f"listing-{counter}",
                    category=client.category,
                    source_name=client.source_name,
                    url=raw.get("url", ""),
                    snippet=raw.get("snippet", ""),
                    discovered_at=datetime.utcnow(),
                    contains_sensitive_pii=self._has_sensitive_pii(raw.get("snippet", "")),
                    is_outdated=raw.get("is_outdated", False),
                )
                self.listings.append(listing)
                found.append(listing)
        return found

    def _has_sensitive_pii(self, snippet: str) -> bool:
        lowered = snippet.lower()
        return any(marker in lowered for marker in self._SENSITIVE_MARKERS)

    # -- prioritization -----------------------------------------------------------
    def priority_for_removal(self) -> List[DiscoveredListing]:
        """Sensitive-PII data-broker listings and outdated harmful content first."""
        def priority_key(listing: DiscoveredListing):
            score = 0
            if listing.category == ListingCategory.DATA_BROKER:
                score += 2
            if listing.contains_sensitive_pii:
                score += 2
            if listing.is_outdated:
                score += 1
            return -score
        return sorted(self.listings, key=priority_key)

    # -- takedown workflow ----------------------------------------------------------
    def request_takedown(self, listing_id: str) -> DiscoveredListing:
        listing = self._find(listing_id)
        listing.takedown_status = TakedownStatus.REQUESTED
        listing.takedown_requested_at = datetime.utcnow()
        # Real integration point: fire off the source's opt-out/removal-request
        # API or generate a pre-filled legal removal-request form here.
        return listing

    def update_takedown_status(self, listing_id: str, status: TakedownStatus) -> DiscoveredListing:
        listing = self._find(listing_id)
        listing.takedown_status = status
        return listing

    def takedown_dashboard(self) -> List[dict]:
        return [{
            "id": l.id, "source": l.source_name, "category": l.category.value,
            "status": l.takedown_status.value, "sensitive_pii": l.contains_sensitive_pii,
        } for l in self.listings]

    def _find(self, listing_id: str) -> DiscoveredListing:
        for l in self.listings:
            if l.id == listing_id:
                return l
        raise KeyError(f"No listing found for id={listing_id}")

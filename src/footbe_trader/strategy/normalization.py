"""Name normalization utilities for teams and leagues.

Provides canonical name generation and alias-based matching for
cross-platform entity resolution between API-Football and Kalshi.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from footbe_trader.common.logging import get_logger

logger = get_logger(__name__)

# Default paths for alias files
# Path: src/footbe_trader/strategy/normalization.py -> configs/
DEFAULT_TEAM_ALIASES_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "team_aliases.yaml"
DEFAULT_LEAGUE_ALIASES_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "league_aliases.yaml"


# Common suffixes/prefixes to remove from team names
TEAM_TOKENS_TO_STRIP = frozenset([
    "fc", "cf", "sc", "afc", "fk", "ac", "ssc", "rcd", "cd", "vfb", "vfl",
    "sv", "fsv", "as", "ss", "us", "og", "rc", "aj", "ca", "ud", "losc",
    "tsg", "rb", "bsc", "cfc", "lfc", "mufc", "thfc", "nufc", "wwfc",
    "avfc", "cpfc", "fcc", "ocsc", "nycfc", "lafc", "skc", "rsl",
    "1", "04", "05", "98", "1846", "1899", "1907", "1919", "1907",
    "29", "63",
])


@dataclass
class NormalizationResult:
    """Result of name normalization."""
    
    original: str
    normalized: str
    canonical: str | None = None
    match_source: str | None = None  # "exact", "alias", "fuzzy"
    confidence: float = 1.0


@dataclass
class TeamAliasRegistry:
    """Registry for team name aliases."""
    
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    global_aliases: dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "TeamAliasRegistry":
        """Load alias registry from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("team_aliases_not_found", path=str(path))
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        registry = cls()
        
        for league_key, aliases in data.items():
            if league_key == "global":
                registry.global_aliases = aliases or {}
            elif isinstance(aliases, dict):
                registry.aliases[league_key] = aliases
        
        logger.info(
            "team_aliases_loaded",
            leagues=len(registry.aliases),
            global_count=len(registry.global_aliases),
        )
        return registry
    
    def get_canonical(self, name: str, league_hint: str | None = None) -> str | None:
        """Look up canonical name for a team variant.
        
        Args:
            name: Team name variant to look up.
            league_hint: Optional league to check first.
            
        Returns:
            Canonical name if found, None otherwise.
        """
        name_lower = name.lower().strip()
        
        # Check league-specific aliases first
        if league_hint and league_hint in self.aliases:
            if name_lower in self.aliases[league_hint]:
                return self.aliases[league_hint][name_lower]
        
        # Check all league aliases
        for aliases in self.aliases.values():
            if name_lower in aliases:
                return aliases[name_lower]
        
        # Check global aliases
        if name_lower in self.global_aliases:
            result = self.global_aliases[name_lower]
            # Empty string in global means "strip this token"
            return result if result else None
        
        return None


@dataclass
class LeagueAliasRegistry:
    """Registry for league name aliases."""
    
    # canonical_key -> list of aliases
    aliases: dict[str, list[str]] = field(default_factory=dict)
    # alias -> canonical_key (reverse lookup)
    reverse_lookup: dict[str, str] = field(default_factory=dict)
    # API-Football league_id -> canonical_key
    api_football_ids: dict[int, str] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "LeagueAliasRegistry":
        """Load league alias registry from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("league_aliases_not_found", path=str(path))
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        registry = cls()
        
        # Load API-Football ID mappings
        if "api_football_league_ids" in data:
            registry.api_football_ids = data["api_football_league_ids"]
        
        # Load league aliases
        for league_name, league_data in data.items():
            if league_name == "api_football_league_ids":
                continue
            
            if not isinstance(league_data, dict):
                continue
            
            canonical_key = league_data.get("canonical_key", league_name)
            aliases = league_data.get("aliases", [])
            
            registry.aliases[canonical_key] = aliases
            
            # Build reverse lookup
            for alias in aliases:
                registry.reverse_lookup[alias.lower()] = canonical_key
            
            # Also add the canonical key itself
            registry.reverse_lookup[canonical_key.lower()] = canonical_key
        
        logger.info(
            "league_aliases_loaded",
            leagues=len(registry.aliases),
            api_ids=len(registry.api_football_ids),
        )
        return registry
    
    def get_canonical_key(self, name: str) -> str | None:
        """Look up canonical key for a league name.
        
        Args:
            name: League name to look up.
            
        Returns:
            Canonical key if found, None otherwise.
        """
        return self.reverse_lookup.get(name.lower().strip())
    
    def get_key_for_api_football_id(self, league_id: int) -> str | None:
        """Get canonical key for an API-Football league ID."""
        return self.api_football_ids.get(league_id)


class TeamNameNormalizer:
    """Normalizes team names to canonical form."""
    
    def __init__(self, alias_registry: TeamAliasRegistry | None = None):
        """Initialize normalizer.
        
        Args:
            alias_registry: Optional pre-loaded alias registry.
        """
        self.aliases = alias_registry or TeamAliasRegistry.from_yaml(DEFAULT_TEAM_ALIASES_PATH)
    
    def normalize(self, name: str, league_hint: str | None = None) -> NormalizationResult:
        """Normalize a team name.
        
        Args:
            name: Raw team name.
            league_hint: Optional league context for alias lookup.
            
        Returns:
            NormalizationResult with normalized and canonical names.
        """
        original = name
        
        # Step 1: Basic normalization
        normalized = self._basic_normalize(name)
        
        # Step 2: Check for exact canonical match
        canonical = self.aliases.get_canonical(normalized, league_hint)
        if canonical:
            return NormalizationResult(
                original=original,
                normalized=normalized,
                canonical=canonical,
                match_source="alias",
                confidence=1.0,
            )
        
        # Step 3: Try with original name
        canonical = self.aliases.get_canonical(original, league_hint)
        if canonical:
            return NormalizationResult(
                original=original,
                normalized=normalized,
                canonical=canonical,
                match_source="alias",
                confidence=1.0,
            )
        
        # Step 4: Use normalized form as canonical
        return NormalizationResult(
            original=original,
            normalized=normalized,
            canonical=normalized,
            match_source="normalized",
            confidence=0.8,
        )
    
    def _basic_normalize(self, name: str) -> str:
        """Apply basic normalization rules.
        
        - Lowercase
        - Remove diacritics
        - Remove punctuation
        - Collapse whitespace
        - Strip common tokens (FC, SC, etc.)
        """
        # Lowercase
        result = name.lower()
        
        # Remove diacritics (Atlético -> Atletico)
        result = self._remove_diacritics(result)
        
        # Remove punctuation except hyphens and apostrophes in words
        result = re.sub(r"[^\w\s'-]", " ", result)
        
        # Replace hyphens with spaces
        result = result.replace("-", " ")
        
        # Collapse whitespace
        result = " ".join(result.split())
        
        # Strip common tokens
        words = result.split()
        filtered_words = [w for w in words if w not in TEAM_TOKENS_TO_STRIP]
        
        # Keep at least one word
        if filtered_words:
            result = " ".join(filtered_words)
        else:
            result = " ".join(words)
        
        return result.strip()
    
    @staticmethod
    def _remove_diacritics(text: str) -> str:
        """Remove diacritical marks from text."""
        # Normalize to NFD (decomposed form)
        normalized = unicodedata.normalize("NFD", text)
        # Remove combining diacritical marks
        return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


class LeagueNameNormalizer:
    """Normalizes league names to canonical keys."""
    
    def __init__(self, alias_registry: LeagueAliasRegistry | None = None):
        """Initialize normalizer.
        
        Args:
            alias_registry: Optional pre-loaded alias registry.
        """
        self.aliases = alias_registry or LeagueAliasRegistry.from_yaml(DEFAULT_LEAGUE_ALIASES_PATH)
    
    def normalize(self, name: str) -> NormalizationResult:
        """Normalize a league name.
        
        Args:
            name: Raw league name.
            
        Returns:
            NormalizationResult with canonical key.
        """
        original = name
        
        # Basic normalization
        normalized = self._basic_normalize(name)
        
        # Look up canonical key
        canonical = self.aliases.get_canonical_key(normalized)
        if canonical:
            return NormalizationResult(
                original=original,
                normalized=normalized,
                canonical=canonical,
                match_source="alias",
                confidence=1.0,
            )
        
        # Try original
        canonical = self.aliases.get_canonical_key(original)
        if canonical:
            return NormalizationResult(
                original=original,
                normalized=normalized,
                canonical=canonical,
                match_source="alias",
                confidence=1.0,
            )
        
        # Fall back to normalized form
        return NormalizationResult(
            original=original,
            normalized=normalized,
            canonical=normalized.replace(" ", "_"),
            match_source="normalized",
            confidence=0.5,
        )
    
    def get_key_for_league_id(self, league_id: int) -> str | None:
        """Get canonical key for API-Football league ID."""
        return self.aliases.get_key_for_api_football_id(league_id)
    
    def _basic_normalize(self, name: str) -> str:
        """Apply basic normalization rules."""
        # Lowercase
        result = name.lower()
        
        # Remove diacritics
        result = unicodedata.normalize("NFD", result)
        result = "".join(c for c in result if unicodedata.category(c) != "Mn")
        
        # Remove punctuation
        result = re.sub(r"[^\w\s]", " ", result)
        
        # Collapse whitespace
        result = " ".join(result.split())
        
        return result.strip()


# Module-level singleton instances
_team_normalizer: TeamNameNormalizer | None = None
_league_normalizer: LeagueNameNormalizer | None = None


def get_team_normalizer() -> TeamNameNormalizer:
    """Get or create the singleton team normalizer."""
    global _team_normalizer
    if _team_normalizer is None:
        _team_normalizer = TeamNameNormalizer()
    return _team_normalizer


def get_league_normalizer() -> LeagueNameNormalizer:
    """Get or create the singleton league normalizer."""
    global _league_normalizer
    if _league_normalizer is None:
        _league_normalizer = LeagueNameNormalizer()
    return _league_normalizer


def normalize_team_name(name: str, league_hint: str | None = None) -> str:
    """Convenience function to get canonical team name.
    
    Args:
        name: Raw team name.
        league_hint: Optional league context.
        
    Returns:
        Canonical team name.
    """
    result = get_team_normalizer().normalize(name, league_hint)
    return result.canonical or result.normalized


def normalize_league_name(name: str) -> str:
    """Convenience function to get canonical league key.
    
    Args:
        name: Raw league name.
        
    Returns:
        Canonical league key.
    """
    result = get_league_normalizer().normalize(name)
    return result.canonical or result.normalized


@lru_cache(maxsize=1000)
def fuzzy_match_ratio(s1: str, s2: str) -> int:
    """Compute fuzzy match ratio between two strings.
    
    Uses a simple token-based similarity metric.
    Returns value from 0-100.
    """
    # Normalize both strings
    s1_norm = normalize_team_name(s1).lower()
    s2_norm = normalize_team_name(s2).lower()
    
    # Exact match
    if s1_norm == s2_norm:
        return 100
    
    # Token-based similarity
    tokens1 = set(s1_norm.split())
    tokens2 = set(s2_norm.split())
    
    if not tokens1 or not tokens2:
        return 0
    
    # Jaccard similarity * 100
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    if union == 0:
        return 0
    
    # Weight by overlap
    jaccard = intersection / union
    
    # Bonus for substring containment
    bonus = 0
    if s1_norm in s2_norm or s2_norm in s1_norm:
        bonus = 20
    
    return min(100, int(jaccard * 80 + bonus))

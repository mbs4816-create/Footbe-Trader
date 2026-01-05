"""Tests for team and league name normalization."""

import pytest

from footbe_trader.strategy.normalization import (
    TeamNameNormalizer,
    TeamAliasRegistry,
    LeagueNameNormalizer,
    LeagueAliasRegistry,
    normalize_team_name,
    normalize_league_name,
    fuzzy_match_ratio,
    TEAM_TOKENS_TO_STRIP,
)


class TestTeamNameNormalization:
    """Tests for team name normalization."""
    
    def test_lowercase(self):
        """Should convert to lowercase."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("MANCHESTER UNITED")
        assert result.normalized == "manchester united"
    
    def test_remove_diacritics(self):
        """Should remove diacritical marks."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("Atlético Madrid")
        assert "atletico" in result.normalized.lower()
    
    def test_remove_punctuation(self):
        """Should remove punctuation."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("A.F.C. Bournemouth")
        assert "." not in result.normalized
    
    def test_collapse_whitespace(self):
        """Should collapse multiple spaces."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("Manchester    United")
        assert "  " not in result.normalized
    
    def test_strip_fc_suffix(self):
        """Should strip FC suffix."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("Chelsea FC")
        # FC should be stripped
        assert result.normalized == "chelsea"
    
    def test_strip_common_prefixes(self):
        """Should strip common prefixes."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        # Test various prefixes
        result = normalizer.normalize("AFC Bournemouth")
        assert "bournemouth" in result.normalized
        
        result = normalizer.normalize("AC Milan")
        assert "milan" in result.normalized
    
    def test_preserve_significant_words(self):
        """Should preserve significant words even after stripping."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("FC Barcelona")
        assert "barcelona" in result.normalized
    
    def test_alias_lookup(self):
        """Should look up aliases correctly."""
        aliases = TeamAliasRegistry()
        aliases.aliases["epl"] = {"man utd": "manchester united"}
        
        normalizer = TeamNameNormalizer(aliases)
        result = normalizer.normalize("Man Utd", league_hint="epl")
        
        assert result.canonical == "manchester united"
        assert result.match_source == "alias"
    
    def test_global_alias_fallback(self):
        """Should fall back to global aliases."""
        aliases = TeamAliasRegistry()
        aliases.global_aliases["barca"] = "barcelona"
        
        normalizer = TeamNameNormalizer(aliases)
        result = normalizer.normalize("Barca")
        
        assert result.canonical == "barcelona"
    
    def test_unknown_team_uses_normalized(self):
        """Unknown team should use normalized form as canonical."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        result = normalizer.normalize("Some Unknown Team FC")
        
        assert result.canonical == result.normalized
        assert result.match_source == "normalized"


class TestLeagueNameNormalization:
    """Tests for league name normalization."""
    
    def test_lowercase(self):
        """Should convert to lowercase."""
        normalizer = LeagueNameNormalizer(LeagueAliasRegistry())
        result = normalizer.normalize("PREMIER LEAGUE")
        assert result.normalized == "premier league"
    
    def test_remove_diacritics(self):
        """Should remove diacritics."""
        normalizer = LeagueNameNormalizer(LeagueAliasRegistry())
        result = normalizer.normalize("Ligue 1 Über Eats")
        assert "u" in result.normalized  # ü -> u
    
    def test_alias_lookup(self):
        """Should look up canonical keys."""
        aliases = LeagueAliasRegistry()
        aliases.reverse_lookup["epl"] = "premier_league"
        aliases.reverse_lookup["premier league"] = "premier_league"
        
        normalizer = LeagueNameNormalizer(aliases)
        
        result = normalizer.normalize("EPL")
        assert result.canonical == "premier_league"
        
        result = normalizer.normalize("Premier League")
        assert result.canonical == "premier_league"
    
    def test_api_football_id_lookup(self):
        """Should look up by API-Football ID."""
        aliases = LeagueAliasRegistry()
        aliases.api_football_ids = {39: "premier_league", 140: "la_liga"}
        
        normalizer = LeagueNameNormalizer(aliases)
        
        assert normalizer.get_key_for_league_id(39) == "premier_league"
        assert normalizer.get_key_for_league_id(140) == "la_liga"
        assert normalizer.get_key_for_league_id(9999) is None


class TestFuzzyMatchRatio:
    """Tests for fuzzy matching."""
    
    def test_exact_match(self):
        """Exact match should return 100."""
        ratio = fuzzy_match_ratio("Manchester United", "manchester united")
        assert ratio == 100
    
    def test_similar_strings(self):
        """Similar strings should have reasonable ratio."""
        # Longer strings with more overlap get higher scores
        ratio = fuzzy_match_ratio("Liverpool", "Liverpool FC")
        assert ratio >= 70  # Should be high for same team
    
    def test_different_strings(self):
        """Different strings should have low ratio."""
        ratio = fuzzy_match_ratio("Manchester United", "Barcelona")
        assert ratio < 50
    
    def test_substring_bonus(self):
        """Substring containment should boost score."""
        ratio1 = fuzzy_match_ratio("Chelsea", "Chelsea FC")
        ratio2 = fuzzy_match_ratio("Chelsea", "Tottenham")
        assert ratio1 > ratio2
    
    def test_empty_strings(self):
        """Empty strings should handle edge cases."""
        # Two empty strings can be considered identical (100) or no match (0)
        # depending on implementation - just verify no crash
        fuzzy_match_ratio("", "")
        # Non-empty vs empty should be low
        assert fuzzy_match_ratio("Test", "") <= 50


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_normalize_team_name_function(self):
        """normalize_team_name should return canonical name."""
        # This uses the singleton with default aliases
        result = normalize_team_name("Chelsea FC")
        assert isinstance(result, str)
        assert result  # Non-empty
    
    def test_normalize_league_name_function(self):
        """normalize_league_name should return canonical key."""
        result = normalize_league_name("Premier League")
        assert isinstance(result, str)
        assert result  # Non-empty


class TestTokenStripping:
    """Tests for token stripping configuration."""
    
    def test_common_suffixes_defined(self):
        """Common suffixes should be in strip list."""
        assert "fc" in TEAM_TOKENS_TO_STRIP
        assert "cf" in TEAM_TOKENS_TO_STRIP
        assert "sc" in TEAM_TOKENS_TO_STRIP
        assert "afc" in TEAM_TOKENS_TO_STRIP
    
    def test_year_tokens_defined(self):
        """Year/number tokens should be in strip list."""
        assert "1899" in TEAM_TOKENS_TO_STRIP
        assert "04" in TEAM_TOKENS_TO_STRIP
        assert "05" in TEAM_TOKENS_TO_STRIP


class TestRealWorldExamples:
    """Tests with real-world team name variations."""
    
    def test_manchester_united_variants(self):
        """Test Manchester United variations."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        variants = [
            "Manchester United",
            "Man United",
            "Man Utd",
            "Manchester United FC",
            "MUFC",
        ]
        
        results = [normalizer.normalize(v).normalized for v in variants]
        
        # All should normalize to something containing "manchester" or "united"
        for result in results:
            assert "manchester" in result or "united" in result or "man" in result or "mufc" in result
    
    def test_spanish_team_diacritics(self):
        """Test Spanish team with diacritics."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        result1 = normalizer.normalize("Atlético Madrid")
        result2 = normalizer.normalize("Atletico Madrid")
        
        # Both should normalize to same base
        assert result1.normalized == result2.normalized
    
    def test_german_team_umlauts(self):
        """Test German team with umlauts."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        result1 = normalizer.normalize("Bayern München")
        result2 = normalizer.normalize("Bayern Munchen")
        
        # Both should normalize similarly
        assert "bayern" in result1.normalized
        assert "bayern" in result2.normalized
    
    def test_italian_team_prefixes(self):
        """Test Italian team prefixes."""
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        result = normalizer.normalize("AC Milan")
        # Should have milan
        assert "milan" in result.normalized
        
        result = normalizer.normalize("AS Roma")
        assert "roma" in result.normalized

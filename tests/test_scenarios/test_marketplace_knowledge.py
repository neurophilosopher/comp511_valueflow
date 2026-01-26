"""Tests for marketplace knowledge builder."""

from __future__ import annotations

from scenarios.marketplace.knowledge import (
    _get_default_knowledge,
    build_market_knowledge,
)


class TestMarketKnowledge:
    """Tests for marketplace knowledge building."""

    def test_default_knowledge_structure(self):
        """Test that default knowledge has expected structure."""
        knowledge = _get_default_knowledge()

        assert "general" in knowledge
        assert "buyer" in knowledge
        assert "seller" in knowledge
        assert "auctioneer" in knowledge

        # Each category should have knowledge facts
        for _category, facts in knowledge.items():
            assert isinstance(facts, list)
            assert len(facts) > 0

    def test_build_buyer_knowledge(self):
        """Test building knowledge for a buyer."""
        params = {
            "budget": 500,
            "strategy": "value_seeker",
            "preferred_categories": ["electronics", "books"],
        }

        knowledge = build_market_knowledge("Alice", "buyer", params)

        assert isinstance(knowledge, list)
        assert len(knowledge) > 0

        # Should include general knowledge
        knowledge_text = " ".join(knowledge)
        assert "marketplace" in knowledge_text.lower() or "trading" in knowledge_text.lower()

    def test_build_seller_knowledge(self):
        """Test building knowledge for a seller."""
        params = {
            "inventory": [
                {"item": "Widget", "category": "electronics", "base_price": 100, "quantity": 5}
            ],
            "pricing_strategy": "competitive",
        }

        knowledge = build_market_knowledge("Bob", "seller", params)

        assert isinstance(knowledge, list)
        assert len(knowledge) > 0

    def test_build_auctioneer_knowledge(self):
        """Test building knowledge for an auctioneer."""
        params = {
            "auction_style": "english",
            "commission_rate": 0.05,
        }

        knowledge = build_market_knowledge("Max", "auctioneer", params)

        assert isinstance(knowledge, list)
        assert len(knowledge) > 0

        # Should mention commission or auction
        knowledge_text = " ".join(knowledge)
        assert "commission" in knowledge_text.lower() or "auction" in knowledge_text.lower()

    def test_buyer_budget_in_knowledge(self):
        """Test that buyer's budget appears in knowledge."""
        params = {"budget": 750}

        knowledge = build_market_knowledge("TestBuyer", "buyer", params)
        knowledge_text = " ".join(knowledge)

        assert "750" in knowledge_text

    def test_seller_inventory_in_knowledge(self):
        """Test that seller's inventory appears in knowledge."""
        params = {
            "inventory": [
                {"item": "Rare Book", "category": "books", "base_price": 200, "quantity": 1}
            ]
        }

        knowledge = build_market_knowledge("TestSeller", "seller", params)
        knowledge_text = " ".join(knowledge)

        assert "Rare Book" in knowledge_text or "1" in knowledge_text

    def test_different_strategies_produce_different_knowledge(self):
        """Test that different buyer strategies produce different knowledge."""
        params1 = {"strategy": "value_seeker"}
        params2 = {"strategy": "collector"}

        knowledge1 = build_market_knowledge("Buyer1", "buyer", params1)
        knowledge2 = build_market_knowledge("Buyer2", "buyer", params2)

        # Convert to strings for comparison
        text1 = " ".join(knowledge1)
        text2 = " ".join(knowledge2)

        # They should be different (at least in strategy-specific parts)
        assert text1 != text2

    def test_omegaconf_conversion(self):
        """Test that OmegaConf containers are properly converted."""
        try:
            from omegaconf import OmegaConf

            # Create an OmegaConf dict like what would come from Hydra config
            params = OmegaConf.create(
                {
                    "budget": 1000,
                    "strategy": "collector",
                    "preferred_categories": ["art", "antiques"],
                }
            )

            knowledge = build_market_knowledge("OmegaAgent", "buyer", params)

            assert isinstance(knowledge, list)
            assert len(knowledge) > 0

            # Check that the knowledge includes budget info
            knowledge_text = " ".join(knowledge)
            assert "1000" in knowledge_text
        except ImportError:
            # Skip test if OmegaConf not installed
            pass

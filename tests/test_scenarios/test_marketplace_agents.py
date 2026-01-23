"""Tests for marketplace agent prefabs."""

from __future__ import annotations

from scenarios.marketplace.agents import AuctioneerAgent, BuyerAgent, SellerAgent


class TestBuyerAgent:
    """Tests for BuyerAgent prefab."""

    def test_default_params(self):
        """Test default parameter values."""
        buyer = BuyerAgent()

        assert buyer.params.get("name") == "Buyer"
        assert buyer.params.get("budget") == 1000
        assert buyer.params.get("strategy") == "value_seeker"

    def test_build_creates_agent(self, mock_model, mock_memory_bank):
        """Test that build creates a valid buyer agent."""
        buyer = BuyerAgent()
        buyer.params = {
            "name": "TestBuyer",
            "goal": "Find good deals",
            "budget": 500,
            "strategy": "bargain_hunter",
            "preferred_categories": ["electronics"],
        }

        agent = buyer.build(mock_model, mock_memory_bank)

        assert agent.name == "TestBuyer"

    def test_buyer_strategies(self, mock_model, mock_memory_bank):
        """Test different buyer strategies."""
        strategies = ["value_seeker", "collector", "bargain_hunter"]

        for strategy in strategies:
            buyer = BuyerAgent()
            buyer.params = {"name": f"{strategy}_buyer", "strategy": strategy}

            agent = buyer.build(mock_model, mock_memory_bank)
            assert agent is not None

    def test_buyer_can_act(self, mock_model, mock_memory_bank):
        """Test that buyer agent can act."""
        buyer = BuyerAgent()
        buyer.params = {"name": "ActingBuyer", "budget": 100}

        agent = buyer.build(mock_model, mock_memory_bank)
        action = agent.act()

        assert isinstance(action, str)


class TestSellerAgent:
    """Tests for SellerAgent prefab."""

    def test_default_params(self):
        """Test default parameter values."""
        seller = SellerAgent()

        assert seller.params.get("name") == "Seller"
        assert seller.params.get("pricing_strategy") == "competitive"
        assert seller.params.get("inventory") == []

    def test_build_creates_agent(self, mock_model, mock_memory_bank):
        """Test that build creates a valid seller agent."""
        seller = SellerAgent()
        seller.params = {
            "name": "TestSeller",
            "goal": "Maximize sales",
            "inventory": [
                {"item": "Widget", "category": "electronics", "base_price": 50, "quantity": 10}
            ],
            "pricing_strategy": "premium",
        }

        agent = seller.build(mock_model, mock_memory_bank)

        assert agent.name == "TestSeller"

    def test_seller_pricing_strategies(self, mock_model, mock_memory_bank):
        """Test different seller pricing strategies."""
        strategies = ["premium", "competitive", "clearance"]

        for strategy in strategies:
            seller = SellerAgent()
            seller.params = {
                "name": f"{strategy}_seller",
                "pricing_strategy": strategy,
            }

            agent = seller.build(mock_model, mock_memory_bank)
            assert agent is not None

    def test_seller_with_inventory(self, mock_model, mock_memory_bank):
        """Test seller with multiple inventory items."""
        seller = SellerAgent()
        seller.params = {
            "name": "InventorySeller",
            "inventory": [
                {"item": "Item1", "category": "cat1", "base_price": 10, "quantity": 5},
                {"item": "Item2", "category": "cat2", "base_price": 20, "quantity": 3},
            ],
        }

        agent = seller.build(mock_model, mock_memory_bank)
        assert agent.name == "InventorySeller"


class TestAuctioneerAgent:
    """Tests for AuctioneerAgent prefab."""

    def test_default_params(self):
        """Test default parameter values."""
        auctioneer = AuctioneerAgent()

        assert auctioneer.params.get("name") == "Auctioneer"
        assert auctioneer.params.get("auction_style") == "english"
        assert auctioneer.params.get("commission_rate") == 0.05

    def test_build_creates_agent(self, mock_model, mock_memory_bank):
        """Test that build creates a valid auctioneer agent."""
        auctioneer = AuctioneerAgent()
        auctioneer.params = {
            "name": "TestAuctioneer",
            "goal": "Facilitate trades",
            "auction_style": "dutch",
            "commission_rate": 0.1,
        }

        agent = auctioneer.build(mock_model, mock_memory_bank)

        assert agent.name == "TestAuctioneer"

    def test_auction_styles(self, mock_model, mock_memory_bank):
        """Test different auction styles."""
        styles = ["english", "dutch", "sealed"]

        for style in styles:
            auctioneer = AuctioneerAgent()
            auctioneer.params = {
                "name": f"{style}_auctioneer",
                "auction_style": style,
            }

            agent = auctioneer.build(mock_model, mock_memory_bank)
            assert agent is not None

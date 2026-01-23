"""Tests for BasicEntity prefab."""

from __future__ import annotations

from src.entities.agents.basic_entity import BasicEntity


class TestBasicEntity:
    """Tests for BasicEntity prefab."""

    def test_default_params(self):
        """Test default parameter values."""
        entity = BasicEntity()

        assert entity.params.get("name") == "Agent"
        assert entity.params.get("goal") == ""
        assert entity.description != ""

    def test_build_creates_agent(self, mock_model, mock_memory_bank):
        """Test that build creates a valid agent."""
        entity = BasicEntity()
        entity.params = {"name": "TestAgent", "goal": "Test goal"}

        agent = entity.build(mock_model, mock_memory_bank)

        assert agent.name == "TestAgent"

    def test_build_with_goal(self, mock_model, mock_memory_bank):
        """Test building agent with a goal."""
        entity = BasicEntity()
        entity.params = {
            "name": "GoalAgent",
            "goal": "Achieve something important",
        }

        agent = entity.build(mock_model, mock_memory_bank)

        # Agent should have Goal component
        assert agent.name == "GoalAgent"

    def test_build_without_goal(self, mock_model, mock_memory_bank):
        """Test building agent without a goal."""
        entity = BasicEntity()
        entity.params = {"name": "NoGoalAgent", "goal": ""}

        agent = entity.build(mock_model, mock_memory_bank)
        assert agent.name == "NoGoalAgent"

    def test_agent_can_act(self, mock_model, mock_memory_bank):
        """Test that built agent can perform actions."""
        entity = BasicEntity()
        entity.params = {"name": "ActingAgent", "goal": "Do something"}

        agent = entity.build(mock_model, mock_memory_bank)

        # Agent should be able to act
        action = agent.act()
        assert isinstance(action, str)

    def test_agent_can_observe(self, mock_model, mock_memory_bank):
        """Test that built agent can receive observations."""
        entity = BasicEntity()
        entity.params = {"name": "ObservingAgent"}

        agent = entity.build(mock_model, mock_memory_bank)

        # Should not raise
        agent.observe("Something happened in the environment.")

    def test_build_with_personality(self, mock_model, mock_memory_bank):
        """Test building agent with personality."""
        entity = BasicEntity()
        entity.params = {
            "name": "PersonalityAgent",
            "personality": "Friendly and helpful",
            "background": "Grew up in a small village",
        }

        agent = entity.build(mock_model, mock_memory_bank)
        assert agent.name == "PersonalityAgent"

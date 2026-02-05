"""Tests for SocialMediaGameMaster."""

import pytest

from src.environments.social_media.game_master import (
    SocialMediaGameMaster,
    _MinimalGameMasterEntity,
)


class TestSocialMediaGameMaster:
    """Tests for SocialMediaGameMaster prefab."""

    def test_build_creates_app(self) -> None:
        """Test that build creates a SocialMediaApp."""
        gm = SocialMediaGameMaster(
            params={
                "name": "test_gm",
                "timeline_limit": 10,
            }
        )

        # Build (model and memory_bank not used)
        entity = gm.build(model=None, memory_bank=None)  # type: ignore[arg-type]

        assert gm.app is not None
        assert entity.name == "test_gm"

    def test_build_initializes_follower_graph(self) -> None:
        """Test that build sets up initial follower graph."""
        gm = SocialMediaGameMaster(
            params={
                "name": "test_gm",
                "initial_graph": {
                    "Alice": ["Bob", "Charlie"],
                    "Bob": ["Alice"],
                },
            }
        )

        gm.build(model=None, memory_bank=None)  # type: ignore[arg-type]

        assert "Bob" in gm.app.get_following("Alice")
        assert "Charlie" in gm.app.get_following("Alice")
        assert "Alice" in gm.app.get_following("Bob")

    def test_build_creates_seed_posts(self) -> None:
        """Test that build creates seed posts."""
        gm = SocialMediaGameMaster(
            params={
                "name": "test_gm",
                "seed_posts": [
                    {"author": "Alice", "content": "First post!", "tags": ["seed"]},
                    {"author": "Bob", "content": "Second post!"},
                ],
            }
        )

        gm.build(model=None, memory_bank=None)  # type: ignore[arg-type]

        posts = gm.app.get_all_posts()
        assert len(posts) == 2
        assert posts[0].author == "Alice"
        assert posts[0].content == "First post!"
        assert "seed" in posts[0].tags
        assert posts[1].author == "Bob"

    def test_app_property_raises_before_build(self) -> None:
        """Test that accessing app before build raises error."""
        gm = SocialMediaGameMaster()

        with pytest.raises(RuntimeError, match="not yet built"):
            _ = gm.app

    def test_entities_initialized_as_users(self) -> None:
        """Test that entity names become users in the app."""

        class MockEntity:
            def __init__(self, name: str) -> None:
                self.name = name

        entities = [MockEntity("Alice"), MockEntity("Bob")]
        gm = SocialMediaGameMaster(entities=entities)  # type: ignore[arg-type]

        gm.build(model=None, memory_bank=None)  # type: ignore[arg-type]

        # Users should exist (have empty follower/following sets)
        assert gm.app.get_following("Alice") == set()
        assert gm.app.get_following("Bob") == set()


class TestMinimalGameMasterEntity:
    """Tests for _MinimalGameMasterEntity."""

    def test_entity_name(self) -> None:
        """Test entity name property."""
        from src.environments.social_media.app import SocialMediaApp

        app = SocialMediaApp()
        entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        assert entity.name == "test_gm"

    def test_get_state(self) -> None:
        """Test getting entity state."""
        from src.environments.social_media.app import SocialMediaApp

        app = SocialMediaApp()
        app.post("Alice", "Test post")
        entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        state = entity.get_state()

        assert state["name"] == "test_gm"
        assert "app_state" in state
        assert len(state["app_state"]["posts"]) == 1

    def test_from_state(self) -> None:
        """Test restoring entity from state."""
        from src.environments.social_media.app import SocialMediaApp

        app = SocialMediaApp()
        app.post("Alice", "Test post")
        original = _MinimalGameMasterEntity(name="test_gm", app=app)
        state = original.get_state()

        restored = _MinimalGameMasterEntity.from_state(state)

        assert restored.name == "test_gm"
        # Verify app state was restored
        posts = restored._app.get_all_posts()
        assert len(posts) == 1
        assert posts[0].content == "Test post"

    def test_act_is_noop(self) -> None:
        """Test that act returns empty string."""
        from src.environments.social_media.app import SocialMediaApp

        app = SocialMediaApp()
        entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        result = entity.act(None)

        assert result == ""

    def test_observe_is_noop(self) -> None:
        """Test that observe does nothing."""
        from src.environments.social_media.app import SocialMediaApp

        app = SocialMediaApp()
        entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        # Should not raise
        entity.observe("test observation")

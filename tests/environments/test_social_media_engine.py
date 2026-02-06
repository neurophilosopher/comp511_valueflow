"""Tests for SocialMediaEngine and action parsing."""

from typing import Any

import pytest
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib

from src.environments.social_media.app import SocialMediaApp
from src.environments.social_media.engine import (
    ActionResult,
    SocialMediaEngine,
    execute_action,
    parse_action,
)


class TestSocialMediaEngine:
    """Tests for SocialMediaEngine type conformance."""

    def test_engine_inherits_from_abc(self) -> None:
        """Test that SocialMediaEngine is an instance of Engine ABC."""
        engine = SocialMediaEngine()
        assert isinstance(engine, engine_lib.Engine)


class TestParseAction:
    """Tests for action parsing."""

    def test_parse_post_action(self) -> None:
        """Test parsing post action."""
        raw = "ACTION: post | TARGET: none | CONTENT: Hello world!"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "post"
        assert parsed["target"] == "none"
        assert parsed["content"] == "Hello world!"

    def test_parse_reply_action(self) -> None:
        """Test parsing reply action."""
        raw = "ACTION: reply | TARGET: 42 | CONTENT: I agree with this"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "reply"
        assert parsed["target"] == "42"
        assert parsed["content"] == "I agree with this"

    def test_parse_like_action(self) -> None:
        """Test parsing like action."""
        raw = "ACTION: like | TARGET: 42 | CONTENT: none"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "like"
        assert parsed["target"] == "42"

    def test_parse_boost_action(self) -> None:
        """Test parsing boost action."""
        raw = "ACTION: boost | TARGET: 42 | CONTENT: none"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "boost"
        assert parsed["target"] == "42"

    def test_parse_follow_action(self) -> None:
        """Test parsing follow action."""
        raw = "ACTION: follow | TARGET: Alice | CONTENT: none"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "follow"
        assert parsed["target"] == "Alice"

    def test_parse_unfollow_action(self) -> None:
        """Test parsing unfollow action."""
        raw = "ACTION: unfollow | TARGET: Bob | CONTENT: none"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "unfollow"
        assert parsed["target"] == "Bob"

    def test_parse_skip_action(self) -> None:
        """Test parsing skip action."""
        raw = "ACTION: skip | TARGET: none | CONTENT: none"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "skip"

    def test_parse_case_insensitive(self) -> None:
        """Test that parsing is case insensitive."""
        raw = "action: POST | target: NONE | content: Hello"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "post"
        assert parsed["content"] == "Hello"

    def test_parse_with_extra_text(self) -> None:
        """Test parsing with extra text around the action."""
        raw = "I think I'll ACTION: post | TARGET: none | CONTENT: My thoughts"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "post"
        assert parsed["content"] == "My thoughts"

    def test_parse_invalid_returns_skip(self) -> None:
        """Test that invalid input defaults to skip."""
        raw = "I don't know what to do"
        parsed = parse_action(raw)

        assert parsed["action_type"] == "skip"


class TestExecuteAction:
    """Tests for action execution."""

    def test_execute_post(self) -> None:
        """Test executing post action."""
        app = SocialMediaApp()
        parsed = {"action_type": "post", "target": "none", "content": "Hello!"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert result.action_type == "post"
        assert result.post_id == 1
        assert app.get_post(1) is not None

    def test_execute_post_no_content_fails(self) -> None:
        """Test that post without content fails."""
        app = SocialMediaApp()
        parsed = {"action_type": "post", "target": "none", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False

    def test_execute_reply(self) -> None:
        """Test executing reply action."""
        app = SocialMediaApp()
        original_id = app.post("Bob", "Original post")
        parsed = {"action_type": "reply", "target": str(original_id), "content": "My reply"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert result.action_type == "reply"
        reply = app.get_post(result.post_id)
        assert reply is not None
        assert reply.reply_to == original_id

    def test_execute_reply_invalid_target(self) -> None:
        """Test reply to non-existent post fails."""
        app = SocialMediaApp()
        parsed = {"action_type": "reply", "target": "999", "content": "Reply"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False

    def test_execute_like(self) -> None:
        """Test executing like action."""
        app = SocialMediaApp()
        post_id = app.post("Bob", "Like me!")
        parsed = {"action_type": "like", "target": str(post_id), "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert app.get_like_count(post_id) == 1

    def test_execute_like_already_liked(self) -> None:
        """Test liking already liked post still succeeds."""
        app = SocialMediaApp()
        post_id = app.post("Bob", "Like me!")
        app.like("Alice", post_id)
        parsed = {"action_type": "like", "target": str(post_id), "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True  # Idempotent
        assert app.get_like_count(post_id) == 1

    def test_execute_boost(self) -> None:
        """Test executing boost action."""
        app = SocialMediaApp()
        post_id = app.post("Bob", "Boost me!")
        parsed = {"action_type": "boost", "target": str(post_id), "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert result.post_id is not None
        boost = app.get_post(result.post_id)
        assert boost is not None
        assert boost.boost_of == post_id

    def test_execute_follow(self) -> None:
        """Test executing follow action."""
        app = SocialMediaApp()
        app._ensure_user("Bob")
        parsed = {"action_type": "follow", "target": "Bob", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert "Bob" in app.get_following("Alice")

    def test_execute_unfollow(self) -> None:
        """Test executing unfollow action."""
        app = SocialMediaApp()
        app.follow("Alice", "Bob")
        parsed = {"action_type": "unfollow", "target": "Bob", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert "Bob" not in app.get_following("Alice")

    def test_execute_skip(self) -> None:
        """Test executing skip action."""
        app = SocialMediaApp()
        parsed = {"action_type": "skip", "target": "none", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is True
        assert result.action_type == "skip"

    def test_execute_unknown_action(self) -> None:
        """Test that unknown action type fails."""
        app = SocialMediaApp()
        parsed = {"action_type": "dance", "target": "none", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False


class TestExecuteActionEdgeCases:
    """Edge-case tests for execute_action."""

    def test_reply_non_integer_target(self) -> None:
        """Test reply with non-integer target string returns failure."""
        app = SocialMediaApp()
        parsed = {"action_type": "reply", "target": "abc", "content": "My reply"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False
        assert "Invalid post ID" in result.message

    def test_reply_empty_content(self) -> None:
        """Test reply with empty content ('none') fails."""
        app = SocialMediaApp()
        post_id = app.post("Bob", "Original")
        parsed = {"action_type": "reply", "target": str(post_id), "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False
        assert "requires content" in result.message.lower()

    def test_boost_nonexistent_post(self) -> None:
        """Test boost of non-existent post fails gracefully."""
        app = SocialMediaApp()
        parsed = {"action_type": "boost", "target": "999", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False
        assert "non-existent" in result.message.lower()

    def test_like_nonexistent_post(self) -> None:
        """Test like of non-existent post fails gracefully."""
        app = SocialMediaApp()
        parsed = {"action_type": "like", "target": "999", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False
        assert "non-existent" in result.message.lower()

    def test_like_non_integer_target(self) -> None:
        """Test like with non-integer target returns failure."""
        app = SocialMediaApp()
        parsed = {"action_type": "like", "target": "abc", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False
        assert "Invalid post ID" in result.message

    def test_boost_non_integer_target(self) -> None:
        """Test boost with non-integer target returns failure."""
        app = SocialMediaApp()
        parsed = {"action_type": "boost", "target": "abc", "content": "none"}

        result = execute_action(app, "Alice", parsed)

        assert result.success is False
        assert "Invalid post ID" in result.message


class TestRunLoop:
    """Integration tests for SocialMediaEngine.run_loop()."""

    def test_run_loop_basic(self) -> None:
        """Test run_loop executes steps with mock entities."""
        from unittest.mock import MagicMock

        from src.environments.social_media.game_master import _MinimalGameMasterEntity

        app = SocialMediaApp()
        app._ensure_user("Alice")
        app._ensure_user("Bob")

        gm_entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        # Create mock entities that return structured actions
        alice = MagicMock(spec=entity_lib.Entity)
        alice.name = "Alice"
        alice.act.return_value = "ACTION: post | TARGET: none | CONTENT: Hello from Alice!"

        bob = MagicMock(spec=entity_lib.Entity)
        bob.name = "Bob"
        bob.act.return_value = "ACTION: post | TARGET: none | CONTENT: Hello from Bob!"

        engine = SocialMediaEngine()
        log: list[dict[str, Any]] = []

        engine.run_loop(
            game_masters=[gm_entity],
            entities=[alice, bob],
            max_steps=2,
            log=log,
        )

        # Both agents posted each step => 4 posts total
        assert len(app.get_all_posts()) == 4
        assert len(log) == 2

    def test_run_loop_with_premise(self) -> None:
        """Test run_loop delivers premise to all entities."""
        from unittest.mock import MagicMock

        from src.environments.social_media.game_master import _MinimalGameMasterEntity

        app = SocialMediaApp()
        gm_entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        agent = MagicMock(spec=entity_lib.Entity)
        agent.name = "Alice"
        agent.act.return_value = "ACTION: skip | TARGET: none | CONTENT: none"

        engine = SocialMediaEngine()

        engine.run_loop(
            game_masters=[gm_entity],
            entities=[agent],
            premise="Welcome to the simulation!",
            max_steps=1,
        )

        # Premise should be the first observe call
        first_observe_call = agent.observe.call_args_list[0]
        assert "Welcome to the simulation!" in first_observe_call[0][0]

    def test_run_loop_no_entities_raises(self) -> None:
        """Test run_loop raises ValueError with no entities."""
        from unittest.mock import MagicMock

        gm = MagicMock(spec=entity_lib.Entity)
        gm.name = "gm"

        engine = SocialMediaEngine()

        with pytest.raises(ValueError, match="No entities"):
            engine.run_loop(game_masters=[gm], entities=[], max_steps=1)

    def test_run_loop_no_game_masters_raises(self) -> None:
        """Test run_loop raises ValueError with no game masters."""
        from unittest.mock import MagicMock

        agent = MagicMock(spec=entity_lib.Entity)
        agent.name = "Alice"

        engine = SocialMediaEngine()

        with pytest.raises(ValueError, match="No game masters"):
            engine.run_loop(game_masters=[], entities=[agent], max_steps=1)

    def test_run_loop_checkpoint_callback(self) -> None:
        """Test run_loop calls checkpoint callback after each step."""
        from unittest.mock import MagicMock

        from src.environments.social_media.game_master import _MinimalGameMasterEntity

        app = SocialMediaApp()
        gm_entity = _MinimalGameMasterEntity(name="test_gm", app=app)

        agent = MagicMock(spec=entity_lib.Entity)
        agent.name = "Alice"
        agent.act.return_value = "ACTION: skip | TARGET: none | CONTENT: none"

        callback = MagicMock()
        engine = SocialMediaEngine()

        engine.run_loop(
            game_masters=[gm_entity],
            entities=[agent],
            max_steps=3,
            checkpoint_callback=callback,
        )

        assert callback.call_count == 3
        callback.assert_any_call(1)
        callback.assert_any_call(2)
        callback.assert_any_call(3)


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_creation(self) -> None:
        """Test creating ActionResult."""
        result = ActionResult(
            success=True,
            action_type="post",
            message="Posted #1",
            post_id=1,
        )

        assert result.success is True
        assert result.action_type == "post"
        assert result.message == "Posted #1"
        assert result.post_id == 1

    def test_action_result_without_post_id(self) -> None:
        """Test ActionResult without post_id."""
        result = ActionResult(success=True, action_type="like", message="Liked #1")

        assert result.post_id is None

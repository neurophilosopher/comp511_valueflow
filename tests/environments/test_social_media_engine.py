"""Tests for SocialMediaEngine and action parsing."""

from src.environments.social_media.app import SocialMediaApp
from src.environments.social_media.engine import (
    ActionResult,
    execute_action,
    parse_action,
)


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

"""Tests for SocialMediaApp."""

import pytest

from src.environments.social_media.app import Post, SocialMediaApp


class TestPost:
    """Tests for Post dataclass."""

    def test_post_to_dict_and_back(self) -> None:
        """Test serialization roundtrip."""
        post = Post(
            id=1,
            author="Alice",
            content="Hello world",
            step=5,
            reply_to=None,
            boost_of=None,
            tags=["test", "greeting"],
        )
        data = post.to_dict()
        restored = Post.from_dict(data)

        assert restored.id == post.id
        assert restored.author == post.author
        assert restored.content == post.content
        assert restored.step == post.step
        assert restored.tags == post.tags

    def test_post_with_reply_to(self) -> None:
        """Test post with reply_to field."""
        post = Post(id=2, author="Bob", content="Reply", step=6, reply_to=1)
        data = post.to_dict()
        restored = Post.from_dict(data)

        assert restored.reply_to == 1


class TestSocialMediaApp:
    """Tests for SocialMediaApp."""

    def test_post_creates_entry(self) -> None:
        """Test basic post creation."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "Hello world")

        assert post_id == 1
        post = app.get_post(post_id)
        assert post is not None
        assert post.author == "Alice"
        assert post.content == "Hello world"

    def test_post_with_tags(self) -> None:
        """Test post with metadata tags."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "Breaking news!", tags=["misinfo"])

        post = app.get_post(post_id)
        assert post is not None
        assert "misinfo" in post.tags

    def test_reply(self) -> None:
        """Test reply to existing post."""
        app = SocialMediaApp()
        original_id = app.post("Alice", "Original post")
        reply_id = app.post("Bob", "I agree!", reply_to=original_id)

        reply = app.get_post(reply_id)
        assert reply is not None
        assert reply.reply_to == original_id

    def test_reply_to_nonexistent_fails(self) -> None:
        """Test that reply to non-existent post fails."""
        app = SocialMediaApp()
        with pytest.raises(ValueError, match="non-existent"):
            app.post("Alice", "Reply", reply_to=999)

    def test_boost(self) -> None:
        """Test boost creates new post referencing original."""
        app = SocialMediaApp()
        original_id = app.post("Alice", "Original content")
        boost_id = app.boost("Bob", original_id)

        boost = app.get_post(boost_id)
        assert boost is not None
        assert boost.author == "Bob"
        assert boost.content == "Original content"
        assert boost.boost_of == original_id

    def test_boost_of_boost_references_original(self) -> None:
        """Test that boosting a boost references the root post."""
        app = SocialMediaApp()
        original_id = app.post("Alice", "Original")
        boost1_id = app.boost("Bob", original_id)
        boost2_id = app.boost("Charlie", boost1_id)

        boost2 = app.get_post(boost2_id)
        assert boost2 is not None
        assert boost2.boost_of == original_id  # References root, not intermediate

    def test_like(self) -> None:
        """Test liking a post."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "Like me!")

        assert app.like("Bob", post_id) is True
        assert app.get_like_count(post_id) == 1

        # Liking again returns False
        assert app.like("Bob", post_id) is False
        assert app.get_like_count(post_id) == 1

    def test_unlike(self) -> None:
        """Test removing a like."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "Like me!")
        app.like("Bob", post_id)

        assert app.unlike("Bob", post_id) is True
        assert app.get_like_count(post_id) == 0

        # Unlike when not liked returns False
        assert app.unlike("Bob", post_id) is False

    def test_follow(self) -> None:
        """Test follow functionality."""
        app = SocialMediaApp()

        assert app.follow("Alice", "Bob") is True
        assert "Bob" in app.get_following("Alice")
        assert "Alice" in app.get_followers("Bob")

        # Following again returns False
        assert app.follow("Alice", "Bob") is False

    def test_cannot_follow_self(self) -> None:
        """Test that users cannot follow themselves."""
        app = SocialMediaApp()
        assert app.follow("Alice", "Alice") is False

    def test_unfollow(self) -> None:
        """Test unfollow functionality."""
        app = SocialMediaApp()
        app.follow("Alice", "Bob")

        assert app.unfollow("Alice", "Bob") is True
        assert "Bob" not in app.get_following("Alice")
        assert "Alice" not in app.get_followers("Bob")

        # Unfollowing when not following returns False
        assert app.unfollow("Alice", "Bob") is False

    def test_timeline_shows_followed_users(self) -> None:
        """Test timeline includes posts from followed users."""
        app = SocialMediaApp()
        app.follow("Alice", "Bob")
        app.follow("Alice", "Charlie")

        app.post("Bob", "Bob's post")
        app.post("Charlie", "Charlie's post")
        app.post("Diana", "Diana's post")  # Not followed

        timeline = app.get_timeline("Alice")

        authors = {p.author for p in timeline}
        assert "Bob" in authors
        assert "Charlie" in authors
        assert "Diana" not in authors

    def test_timeline_includes_own_posts(self) -> None:
        """Test timeline includes user's own posts."""
        app = SocialMediaApp()
        app.post("Alice", "My post")

        timeline = app.get_timeline("Alice")
        assert len(timeline) == 1
        assert timeline[0].author == "Alice"

    def test_timeline_chronological_order(self) -> None:
        """Test timeline is ordered newest first."""
        app = SocialMediaApp()
        app.follow("Alice", "Bob")

        app.current_step = 1
        app.post("Bob", "First")
        app.current_step = 2
        app.post("Bob", "Second")
        app.current_step = 3
        app.post("Bob", "Third")

        timeline = app.get_timeline("Alice")

        assert timeline[0].content == "Third"
        assert timeline[1].content == "Second"
        assert timeline[2].content == "First"

    def test_timeline_limit(self) -> None:
        """Test timeline respects limit parameter."""
        app = SocialMediaApp()
        for i in range(10):
            app.post("Alice", f"Post {i}")

        timeline = app.get_timeline("Alice", limit=5)
        assert len(timeline) == 5

    def test_get_reply_count(self) -> None:
        """Test reply count calculation."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "Original")
        app.post("Bob", "Reply 1", reply_to=post_id)
        app.post("Charlie", "Reply 2", reply_to=post_id)

        assert app.get_reply_count(post_id) == 2

    def test_get_boost_count(self) -> None:
        """Test boost count calculation."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "Original")
        app.boost("Bob", post_id)
        app.boost("Charlie", post_id)

        assert app.get_boost_count(post_id) == 2

    def test_format_timeline(self) -> None:
        """Test timeline formatting for observations."""
        app = SocialMediaApp()
        app.follow("Alice", "Bob")
        app.current_step = 1
        app.post("Bob", "Hello everyone!")

        formatted = app.format_timeline("Alice")

        assert "Your Feed" in formatted
        assert "Bob" in formatted
        assert "Hello everyone!" in formatted
        assert "step 1" in formatted

    def test_serialization_roundtrip(self) -> None:
        """Test full app state serialization."""
        app = SocialMediaApp()
        app.current_step = 5

        # Create some state
        app.follow("Alice", "Bob")
        post_id = app.post("Alice", "Test post", tags=["test"])
        app.like("Bob", post_id)
        app.boost("Bob", post_id)

        # Serialize and restore
        data = app.to_dict()
        restored = SocialMediaApp.from_dict(data)

        # Verify state
        assert restored.current_step == 5
        assert "Bob" in restored.get_following("Alice")
        assert restored.get_like_count(post_id) == 1
        assert restored.get_boost_count(post_id) == 1

        post = restored.get_post(post_id)
        assert post is not None
        assert post.tags == ["test"]

    def test_get_replies(self) -> None:
        """Test getting replies to a post, sorted chronologically."""
        app = SocialMediaApp()
        app.current_step = 0
        original_id = app.post("Alice", "Original post")

        app.current_step = 1
        app.post("Bob", "Reply 1", reply_to=original_id)
        app.current_step = 2
        app.post("Charlie", "Reply 2", reply_to=original_id)
        # Unrelated post
        app.post("Diana", "Not a reply")

        replies = app.get_replies(original_id)
        assert len(replies) == 2
        assert replies[0].author == "Bob"
        assert replies[1].author == "Charlie"

    def test_get_replies_empty(self) -> None:
        """Test get_replies returns empty list when no replies exist."""
        app = SocialMediaApp()
        post_id = app.post("Alice", "No replies here")

        assert app.get_replies(post_id) == []

    def test_boost_nonexistent_post_raises(self) -> None:
        """Test that boosting a non-existent post raises ValueError."""
        app = SocialMediaApp()
        with pytest.raises(ValueError, match="non-existent"):
            app.boost("Alice", 999)

    def test_like_nonexistent_post_raises(self) -> None:
        """Test that liking a non-existent post raises ValueError."""
        app = SocialMediaApp()
        with pytest.raises(ValueError, match="non-existent"):
            app.like("Alice", 999)

    def test_unlike_nonexistent_post_raises(self) -> None:
        """Test that unliking a non-existent post raises ValueError."""
        app = SocialMediaApp()
        with pytest.raises(ValueError, match="non-existent"):
            app.unlike("Alice", 999)

    def test_get_all_posts(self) -> None:
        """Test getting all posts sorted."""
        app = SocialMediaApp()
        app.current_step = 1
        app.post("Alice", "First")
        app.current_step = 2
        app.post("Bob", "Second")

        posts = app.get_all_posts()
        assert len(posts) == 2
        assert posts[0].content == "First"
        assert posts[1].content == "Second"

"""Social media app with local content storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Post:
    """A social media post."""

    id: int
    author: str
    content: str
    step: int
    reply_to: int | None = None
    boost_of: int | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize post to dictionary."""
        return {
            "id": self.id,
            "author": self.author,
            "content": self.content,
            "step": self.step,
            "reply_to": self.reply_to,
            "boost_of": self.boost_of,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Post:
        """Deserialize post from dictionary."""
        return cls(
            id=data["id"],
            author=data["author"],
            content=data["content"],
            step=data["step"],
            reply_to=data.get("reply_to"),
            boost_of=data.get("boost_of"),
            tags=data.get("tags", []),
        )


class SocialMediaApp:
    """Local social media content store.

    Manages posts, follower relationships, and likes without external server.
    """

    def __init__(self) -> None:
        """Initialize empty social media state."""
        self._posts: dict[int, Post] = {}
        self._next_id: int = 1
        self._following: dict[str, set[str]] = {}  # user -> users they follow
        self._followers: dict[str, set[str]] = {}  # user -> users who follow them
        self._likes: dict[int, set[str]] = {}  # post_id -> users who liked
        self._current_step: int = 0

    @property
    def current_step(self) -> int:
        """Current simulation step."""
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        """Set current simulation step."""
        self._current_step = value

    def _ensure_user(self, user: str) -> None:
        """Ensure user exists in follower/following dicts."""
        if user not in self._following:
            self._following[user] = set()
        if user not in self._followers:
            self._followers[user] = set()

    def post(
        self,
        user: str,
        content: str,
        reply_to: int | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Create a new post.

        Args:
            user: Author of the post.
            content: Text content.
            reply_to: Post ID this is replying to (optional).
            tags: Metadata tags for tracking (optional).

        Returns:
            The new post's ID.
        """
        self._ensure_user(user)

        if reply_to is not None and reply_to not in self._posts:
            raise ValueError(f"Cannot reply to non-existent post {reply_to}")

        post = Post(
            id=self._next_id,
            author=user,
            content=content,
            step=self._current_step,
            reply_to=reply_to,
            tags=tags or [],
        )
        self._posts[post.id] = post
        self._likes[post.id] = set()
        self._next_id += 1
        return post.id

    def boost(self, user: str, post_id: int) -> int:
        """Boost (repost) an existing post.

        Creates a new post entry that references the original.

        Args:
            user: User doing the boost.
            post_id: ID of post to boost.

        Returns:
            The new boost post's ID.
        """
        self._ensure_user(user)

        if post_id not in self._posts:
            raise ValueError(f"Cannot boost non-existent post {post_id}")

        original = self._posts[post_id]

        # If boosting a boost, reference the original
        root_id = original.boost_of if original.boost_of else post_id

        boost_post = Post(
            id=self._next_id,
            author=user,
            content=original.content,
            step=self._current_step,
            boost_of=root_id,
            tags=original.tags.copy(),
        )
        self._posts[boost_post.id] = boost_post
        self._likes[boost_post.id] = set()
        self._next_id += 1
        return boost_post.id

    def like(self, user: str, post_id: int) -> bool:
        """Like a post.

        Args:
            user: User liking the post.
            post_id: ID of post to like.

        Returns:
            True if newly liked, False if already liked.
        """
        self._ensure_user(user)

        if post_id not in self._posts:
            raise ValueError(f"Cannot like non-existent post {post_id}")

        if user in self._likes[post_id]:
            return False

        self._likes[post_id].add(user)
        return True

    def unlike(self, user: str, post_id: int) -> bool:
        """Remove like from a post.

        Args:
            user: User removing like.
            post_id: ID of post to unlike.

        Returns:
            True if like removed, False if wasn't liked.
        """
        if post_id not in self._posts:
            raise ValueError(f"Cannot unlike non-existent post {post_id}")

        if user not in self._likes[post_id]:
            return False

        self._likes[post_id].discard(user)
        return True

    def follow(self, user: str, target: str) -> bool:
        """Follow another user.

        Args:
            user: User who wants to follow.
            target: User to follow.

        Returns:
            True if newly followed, False if already following.
        """
        if user == target:
            return False

        self._ensure_user(user)
        self._ensure_user(target)

        if target in self._following[user]:
            return False

        self._following[user].add(target)
        self._followers[target].add(user)
        return True

    def unfollow(self, user: str, target: str) -> bool:
        """Unfollow a user.

        Args:
            user: User who wants to unfollow.
            target: User to unfollow.

        Returns:
            True if unfollowed, False if wasn't following.
        """
        self._ensure_user(user)
        self._ensure_user(target)

        if target not in self._following[user]:
            return False

        self._following[user].discard(target)
        self._followers[target].discard(user)
        return True

    def get_following(self, user: str) -> set[str]:
        """Get users that this user follows."""
        self._ensure_user(user)
        return self._following[user].copy()

    def get_followers(self, user: str) -> set[str]:
        """Get users that follow this user."""
        self._ensure_user(user)
        return self._followers[user].copy()

    def get_like_count(self, post_id: int) -> int:
        """Get number of likes on a post."""
        return len(self._likes.get(post_id, set()))

    def get_boost_count(self, post_id: int) -> int:
        """Get number of boosts of a post."""
        return sum(1 for p in self._posts.values() if p.boost_of == post_id)

    def get_reply_count(self, post_id: int) -> int:
        """Get number of direct replies to a post."""
        return sum(1 for p in self._posts.values() if p.reply_to == post_id)

    def get_post(self, post_id: int) -> Post | None:
        """Get a post by ID."""
        return self._posts.get(post_id)

    def get_timeline(self, user: str, limit: int = 20) -> list[Post]:
        """Get chronological timeline for a user.

        Returns posts from users they follow, plus their own posts,
        sorted newest first.

        Args:
            user: User whose timeline to generate.
            limit: Maximum number of posts to return.

        Returns:
            List of posts, newest first.
        """
        self._ensure_user(user)

        # Get posts from followed users and self
        visible_authors = self._following[user] | {user}

        # Filter to posts by visible authors (includes boosts)
        visible_posts = [p for p in self._posts.values() if p.author in visible_authors]

        # Sort by step (newest first), then by ID for stable ordering
        visible_posts.sort(key=lambda p: (-p.step, -p.id))

        return visible_posts[:limit]

    def get_replies(self, post_id: int) -> list[Post]:
        """Get all replies to a post, sorted chronologically."""
        replies = [p for p in self._posts.values() if p.reply_to == post_id]
        replies.sort(key=lambda p: (p.step, p.id))
        return replies

    def format_timeline(self, user: str, limit: int = 20) -> str:
        """Format timeline as readable string for agent observation.

        Args:
            user: User whose timeline to format.
            limit: Maximum posts to include.

        Returns:
            Formatted string representation of timeline.
        """
        posts = self.get_timeline(user, limit)

        if not posts:
            following_count = len(self._following.get(user, set()))
            follower_count = len(self._followers.get(user, set()))
            return (
                f"=== Your Feed (0 posts) ===\n\n"
                f"No posts yet.\n\n"
                f"---\n"
                f"Following: {following_count} users | Followers: {follower_count}"
            )

        lines = [f"=== Your Feed ({len(posts)} posts) ===\n"]

        for post in posts:
            lines.append(self._format_post(post))

        following_count = len(self._following.get(user, set()))
        follower_count = len(self._followers.get(user, set()))
        lines.append("---")
        lines.append(f"Following: {following_count} users | Followers: {follower_count}")

        return "\n".join(lines)

    def _format_post(self, post: Post, indent: int = 0) -> str:
        """Format a single post as string."""
        prefix = "  " * indent
        likes = self.get_like_count(post.id)
        boosts = self.get_boost_count(post.id)
        replies = self.get_reply_count(post.id)

        # Header line
        if post.boost_of:
            original = self._posts.get(post.boost_of)
            original_author = original.author if original else "unknown"
            header = f"{prefix}[#{post.id}] {post.author} boosted from @{original_author} (step {post.step}):"
        elif post.reply_to:
            header = f"{prefix}[#{post.id}] {post.author} (step {post.step}) replying to #{post.reply_to}:"
        else:
            header = f"{prefix}[#{post.id}] {post.author} (step {post.step}):"

        # Content
        content = f'{prefix}"{post.content}"'

        # Stats
        stats = f"{prefix}Likes: {likes} | Boosts: {boosts} | Replies: {replies}"

        return f"{header}\n{content}\n{stats}\n"

    def get_all_posts(self) -> list[Post]:
        """Get all posts, sorted by step then ID."""
        posts = list(self._posts.values())
        posts.sort(key=lambda p: (p.step, p.id))
        return posts

    def to_dict(self) -> dict[str, Any]:
        """Serialize app state to dictionary for checkpoints."""
        return {
            "posts": [p.to_dict() for p in self._posts.values()],
            "next_id": self._next_id,
            "following": {u: list(f) for u, f in self._following.items()},
            "followers": {u: list(f) for u, f in self._followers.items()},
            "likes": {str(pid): list(users) for pid, users in self._likes.items()},
            "current_step": self._current_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialMediaApp:
        """Deserialize app state from dictionary."""
        app = cls()
        app._next_id = data["next_id"]
        app._current_step = data.get("current_step", 0)

        # Restore posts
        for post_data in data["posts"]:
            post = Post.from_dict(post_data)
            app._posts[post.id] = post

        # Restore following/followers
        app._following = {u: set(f) for u, f in data["following"].items()}
        app._followers = {u: set(f) for u, f in data["followers"].items()}

        # Restore likes (keys are strings in JSON)
        app._likes = {int(pid): set(users) for pid, users in data["likes"].items()}

        return app

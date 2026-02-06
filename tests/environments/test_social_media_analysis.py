"""Tests for social media analysis tools."""

from src.environments.social_media.analysis import (
    TransmissionChain,
    TransmissionEvent,
    analyze_simulation,
    calculate_keyword_overlap,
    chains_to_edge_list,
    chains_to_summary,
    extract_keywords,
    find_transmission_chains,
)
from src.environments.social_media.app import SocialMediaApp


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_extracts_hashtags(self) -> None:
        """Test hashtag extraction."""
        content = "This is #amazing and #cool"
        keywords = extract_keywords(content)

        assert "amazing" in keywords
        assert "cool" in keywords

    def test_extracts_significant_words(self) -> None:
        """Test significant word extraction."""
        content = "Scientists discovered breakthrough research"
        keywords = extract_keywords(content)

        assert "scientists" in keywords
        assert "discovered" in keywords
        assert "breakthrough" in keywords
        assert "research" in keywords

    def test_filters_common_words(self) -> None:
        """Test that common words are filtered."""
        content = "This is what they have been doing"
        keywords = extract_keywords(content)

        assert "this" not in keywords
        assert "what" not in keywords
        assert "they" not in keywords
        assert "have" not in keywords
        assert "been" not in keywords
        assert "doing" in keywords  # 5 chars, not in common list

    def test_lowercase_normalization(self) -> None:
        """Test that keywords are lowercased."""
        content = "BREAKING NEWS about SCIENCE"
        keywords = extract_keywords(content)

        assert "breaking" in keywords
        assert "news" in keywords
        assert "science" in keywords
        assert "BREAKING" not in keywords


class TestCalculateKeywordOverlap:
    """Tests for keyword overlap calculation."""

    def test_identical_sets(self) -> None:
        """Test overlap of identical sets."""
        keywords = {"science", "research", "discovery"}
        overlap = calculate_keyword_overlap(keywords, keywords)

        assert overlap == 1.0

    def test_disjoint_sets(self) -> None:
        """Test overlap of completely different sets."""
        set1 = {"science", "research"}
        set2 = {"sports", "games"}
        overlap = calculate_keyword_overlap(set1, set2)

        assert overlap == 0.0

    def test_partial_overlap(self) -> None:
        """Test partial overlap."""
        set1 = {"science", "research", "discovery"}
        set2 = {"science", "research", "news"}
        overlap = calculate_keyword_overlap(set1, set2)

        # Intersection: 2, Union: 4
        assert overlap == 0.5

    def test_empty_sets(self) -> None:
        """Test with empty sets."""
        assert calculate_keyword_overlap(set(), {"a", "b"}) == 0.0
        assert calculate_keyword_overlap({"a", "b"}, set()) == 0.0
        assert calculate_keyword_overlap(set(), set()) == 0.0


class TestFindTransmissionChains:
    """Tests for transmission chain detection."""

    def test_detects_direct_boosts(self) -> None:
        """Test detection of direct boosts."""
        app = SocialMediaApp()
        app.current_step = 0
        seed_id = app.post("Alice", "Breaking news! #misinfo", tags=["misinfo_seed"])

        app.current_step = 1
        app.boost("Bob", seed_id)

        chains = find_transmission_chains(app, seed_tags=["misinfo_seed"])

        assert len(chains) == 1
        assert len(chains[0].events) == 1
        assert chains[0].events[0].transmission_type == "boost"
        assert chains[0].events[0].from_user == "Alice"
        assert chains[0].events[0].to_user == "Bob"

    def test_detects_replies(self) -> None:
        """Test detection of replies to seed posts."""
        app = SocialMediaApp()
        app.current_step = 0
        seed_id = app.post("Alice", "Controversial claim", tags=["misinfo_seed"])

        app.current_step = 1
        app.post("Bob", "I agree with this!", reply_to=seed_id)

        chains = find_transmission_chains(app, seed_tags=["misinfo_seed"])

        assert len(chains) == 1
        assert len(chains[0].events) == 1
        assert chains[0].events[0].transmission_type == "reply"

    def test_detects_keyword_matches(self) -> None:
        """Test detection of keyword-based transmission."""
        app = SocialMediaApp()
        app.current_step = 0
        app.post(
            "Alice",
            "Scientists discovered dangerous chemicals in water supply!",
            tags=["misinfo_seed"],
        )

        app.current_step = 1
        # Bob posts similar content without explicit connection
        app.post(
            "Bob",
            "I heard scientists found dangerous chemicals everywhere!",
        )

        chains = find_transmission_chains(app, seed_tags=["misinfo_seed"], keyword_threshold=0.3)

        assert len(chains) == 1
        # Should detect keyword match
        keyword_events = [e for e in chains[0].events if e.transmission_type == "keyword_match"]
        assert len(keyword_events) >= 1

    def test_no_false_positives_below_threshold(self) -> None:
        """Test that unrelated posts are not matched."""
        app = SocialMediaApp()
        app.current_step = 0
        app.post("Alice", "Breaking news about science!", tags=["misinfo_seed"])

        app.current_step = 1
        app.post("Bob", "Beautiful sunset today, love nature")

        chains = find_transmission_chains(app, seed_tags=["misinfo_seed"], keyword_threshold=0.3)

        assert len(chains) == 1
        # Bob's unrelated post should not be in the chain
        assert all(e.to_user != "Bob" for e in chains[0].events)

    def test_chain_reach_calculation(self) -> None:
        """Test reach calculation."""
        app = SocialMediaApp()
        app.current_step = 0
        seed_id = app.post("Alice", "Original post", tags=["seed"])

        app.current_step = 1
        app.boost("Bob", seed_id)
        app.boost("Charlie", seed_id)

        chains = find_transmission_chains(app, seed_tags=["seed"])

        assert len(chains) == 1
        assert chains[0].reach == 3  # Alice, Bob, Charlie

    def test_chain_depth_calculation(self) -> None:
        """Test depth calculation with replies (boosts reference original so depth=1)."""
        app = SocialMediaApp()
        app.current_step = 0
        seed_id = app.post("Alice", "Original", tags=["seed"])

        app.current_step = 1
        reply1_id = app.post("Bob", "Interesting!", reply_to=seed_id)

        app.current_step = 2
        app.post("Charlie", "Agreed!", reply_to=reply1_id)

        chains = find_transmission_chains(app, seed_tags=["seed"])

        assert len(chains) == 1
        assert chains[0].depth == 2  # Two hops: seed -> reply -> reply

    def test_chain_breadth_calculation(self) -> None:
        """Test breadth (leaf count) calculation."""
        app = SocialMediaApp()
        app.current_step = 0
        seed_id = app.post("Alice", "Original", tags=["seed"])

        # Three direct replies = 3 leaves
        app.current_step = 1
        app.post("Bob", "Reply 1", reply_to=seed_id)
        app.post("Charlie", "Reply 2", reply_to=seed_id)
        app.post("Diana", "Reply 3", reply_to=seed_id)

        chains = find_transmission_chains(app, seed_tags=["seed"])

        assert len(chains) == 1
        assert chains[0].breadth == 3  # Three leaf nodes
        assert chains[0].depth == 1  # All direct replies

    def test_chain_size_calculation(self) -> None:
        """Test size (total post count) calculation."""
        app = SocialMediaApp()
        app.current_step = 0
        seed_id = app.post("Alice", "Original", tags=["seed"])

        app.current_step = 1
        reply_id = app.post("Bob", "Reply", reply_to=seed_id)
        app.boost("Charlie", seed_id)

        app.current_step = 2
        app.post("Diana", "Reply to reply", reply_to=reply_id)

        chains = find_transmission_chains(app, seed_tags=["seed"])

        assert len(chains) == 1
        assert chains[0].size == 4  # seed + boost + reply + reply-to-reply

    def test_empty_chain_breadth_and_size(self) -> None:
        """Test breadth and size for a chain with no events."""
        chain = TransmissionChain(
            seed_post_id=1,
            seed_author="Alice",
            seed_content="Lonely post",
            seed_tags=["seed"],
        )

        assert chain.size == 1
        assert chain.breadth == 1  # Seed itself is the only leaf
        assert chain.depth == 0


class TestChainsToEdgeList:
    """Tests for edge list conversion."""

    def test_creates_edges(self) -> None:
        """Test edge list creation."""
        chain = TransmissionChain(
            seed_post_id=1,
            seed_author="Alice",
            seed_content="Test",
            seed_tags=["test"],
            events=[
                TransmissionEvent(
                    from_post_id=1,
                    to_post_id=2,
                    from_user="Alice",
                    to_user="Bob",
                    step=1,
                    transmission_type="boost",
                )
            ],
        )

        edges = chains_to_edge_list([chain])

        assert len(edges) == 1
        assert edges[0]["source"] == "Alice"
        assert edges[0]["target"] == "Bob"
        assert edges[0]["type"] == "boost"


class TestChainsToSummary:
    """Tests for summary generation."""

    def test_generates_summary(self) -> None:
        """Test summary generation."""
        chain = TransmissionChain(
            seed_post_id=1,
            seed_author="Alice",
            seed_content="Test",
            seed_tags=["test"],
            events=[
                TransmissionEvent(
                    from_post_id=1,
                    to_post_id=2,
                    from_user="Alice",
                    to_user="Bob",
                    step=1,
                    transmission_type="boost",
                ),
                TransmissionEvent(
                    from_post_id=1,
                    to_post_id=3,
                    from_user="Alice",
                    to_user="Charlie",
                    step=1,
                    transmission_type="reply",
                ),
            ],
        )

        summary = chains_to_summary([chain])

        assert summary["total_chains"] == 1
        assert summary["total_events"] == 2
        assert summary["total_reach"] == 3  # Alice, Bob, Charlie
        assert summary["max_breadth"] == 2  # Bob and Charlie are leaves
        assert summary["max_size"] == 3  # seed + 2 events
        assert summary["chains"][0]["breadth"] == 2
        assert summary["chains"][0]["size"] == 3

    def test_empty_chains(self) -> None:
        """Test summary with no chains."""
        summary = chains_to_summary([])

        assert summary["total_chains"] == 0
        assert summary["total_events"] == 0
        assert summary["max_breadth"] == 0
        assert summary["max_size"] == 0


class TestAnalyzeSimulation:
    """Tests for the main analysis function."""

    def test_analyze_app_directly(self) -> None:
        """Test analyzing a SocialMediaApp directly."""
        app = SocialMediaApp()
        app.current_step = 0
        app.post("Alice", "Test #misinfo content", tags=["misinfo_seed"])

        app.current_step = 1
        app.post("Bob", "Normal post")

        analysis = analyze_simulation(app, seed_tags=["misinfo_seed"])

        assert "summary" in analysis
        assert "edges" in analysis
        assert "posts" in analysis
        assert analysis["summary"]["total_chains"] == 1

    def test_analyze_from_dict(self) -> None:
        """Test analyzing from app state dict."""
        app = SocialMediaApp()
        app.current_step = 0
        app.post("Alice", "Test content", tags=["seed"])

        state = app.to_dict()
        analysis = analyze_simulation(state, seed_tags=["seed"])

        assert analysis["summary"]["total_chains"] == 1

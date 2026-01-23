"""Tests for marketplace event generator."""

from __future__ import annotations

import random

from scenarios.marketplace.events import (
    create_closing_events,
    create_opening_events,
    generate_market_events,
)


class TestMarketEvents:
    """Tests for marketplace event generation."""

    def test_generate_events_returns_list(self):
        """Test that generate_market_events returns a list."""
        params = {"event_frequency": 1.0}  # Always generate
        events = generate_market_events(params)

        assert isinstance(events, list)

    def test_generate_events_with_high_frequency(self):
        """Test event generation with high frequency."""
        params = {"event_frequency": 1.0}
        rng = random.Random(42)

        events = generate_market_events(params, step=0, rng=rng)

        # With frequency 1.0, should always generate an event
        assert len(events) == 1
        assert isinstance(events[0], str)

    def test_generate_events_with_zero_frequency(self):
        """Test event generation with zero frequency."""
        params = {"event_frequency": 0.0}
        rng = random.Random(42)

        events = generate_market_events(params, step=0, rng=rng)

        # With frequency 0.0, should never generate events
        assert len(events) == 0

    def test_generate_events_deterministic_with_seed(self):
        """Test that events are deterministic with same seed."""
        params = {"event_frequency": 1.0}

        rng1 = random.Random(42)
        events1 = generate_market_events(params, rng=rng1)

        rng2 = random.Random(42)
        events2 = generate_market_events(params, rng=rng2)

        assert events1 == events2

    def test_generate_different_events_with_different_seeds(self):
        """Test that different seeds produce different events."""
        params = {"event_frequency": 1.0}

        # Generate many events to increase chance of difference
        events1 = []
        events2 = []

        for i in range(10):
            rng1 = random.Random(i)
            rng2 = random.Random(i + 100)

            e1 = generate_market_events(params, rng=rng1)
            e2 = generate_market_events(params, rng=rng2)

            events1.extend(e1)
            events2.extend(e2)

        # Lists should be different
        assert events1 != events2

    def test_opening_events(self):
        """Test opening events generation."""
        events = create_opening_events({})

        assert isinstance(events, list)
        assert len(events) > 0

        # Should mention opening or beginning
        events_text = " ".join(events).lower()
        assert "open" in events_text or "begin" in events_text or "morning" in events_text

    def test_closing_events(self):
        """Test closing events generation."""
        events = create_closing_events({})

        assert isinstance(events, list)
        assert len(events) > 0

        # Should mention closing or ending
        events_text = " ".join(events).lower()
        assert "close" in events_text or "end" in events_text or "final" in events_text

    def test_event_content_is_meaningful(self):
        """Test that generated events have meaningful content."""
        params = {"event_frequency": 1.0}

        for _ in range(20):
            rng = random.Random()
            events = generate_market_events(params, rng=rng)

            for event in events:
                # Events should be non-empty strings
                assert len(event) > 10
                # Events should contain words (not just punctuation)
                assert any(c.isalpha() for c in event)

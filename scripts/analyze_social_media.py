#!/usr/bin/env python3
"""Analyze social media simulation results for transmission chains.

This script takes simulation output and extracts transmission chains
to understand how content (especially misinformation) spreads.

Usage:
    python scripts/analyze_social_media.py results.json
    python scripts/analyze_social_media.py results.json --output analysis.json
    python scripts/analyze_social_media.py results.json --threshold 0.4 --tags misinfo_seed,health
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path before importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.environments.social_media.analysis import (
    analyze_simulation,
    print_analysis_report,
)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze social media simulation for transmission chains"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file (JSON with app_state or direct app state)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for analysis results (JSON)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.3,
        help="Keyword overlap threshold for transmission detection (default: 0.3)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="misinfo_seed",
        help="Comma-separated seed tags to track (default: misinfo_seed)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress printed report",
    )
    parser.add_argument(
        "--edges-only",
        action="store_true",
        help="Output only edge list (for network tools)",
    )

    args = parser.parse_args()

    # Parse seed tags
    seed_tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # Run analysis
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Analyzing: {input_path}")
    print(f"Seed tags: {seed_tags}")
    print(f"Keyword threshold: {args.threshold}")

    analysis = analyze_simulation(
        input_path,
        seed_tags=seed_tags,
        keyword_threshold=args.threshold,
    )

    # Print report
    if not args.quiet:
        print_analysis_report(analysis)

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = analysis["edges"] if args.edges_only else analysis

        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

"""Structured event logging for simulation output.

This module provides clean, structured logging of simulation events
suitable for LLM training pipelines and analysis.

Approach:
1. Use TeeStdout (from logging_setup.py) to capture raw stdout to a file
2. Post-process the raw log to extract structured events
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SimulationEvent:
    """A structured simulation event."""

    step: int
    event_type: str  # observation, action, resolution, termination_check, game_master
    agent: str | None = None
    content: str = ""
    metadata: dict = field(default_factory=dict)


# Patterns to match Concordia's verbose output
EVENT_PATTERNS = {
    "termination": re.compile(r"^Terminate\? (.+)$", re.MULTILINE),
    "game_master": re.compile(r"^Game master: (.+)$", re.MULTILINE),
    "observation": re.compile(
        r"^Entity (.+?) observed: (.*?)(?=^Entity |^The suggested|^The resolved|^Terminate|^Game master:|^Skipping|^Step \d|^Calling checkpoint|\Z)",
        re.MULTILINE | re.DOTALL,
    ),
    "action": re.compile(r"^Entity (.+?) chose action: (.+?)$", re.MULTILINE),
    "putative_event": re.compile(
        r"^The suggested action or event to resolve was: (.+?)(?=^Entity |^The resolved|^Terminate|\Z)",
        re.MULTILINE | re.DOTALL,
    ),
    "resolution": re.compile(
        r"^The resolved event was: (.+?)(?=^Entity |^Terminate|^Step \d|^Calling checkpoint|\Z)",
        re.MULTILINE | re.DOTALL,
    ),
    "skip_step": re.compile(r"^Skipping the action phase", re.MULTILINE),
    "checkpoint": re.compile(r"^(?:Calling checkpoint callback at step|Step) (\d+)", re.MULTILINE),
}


def parse_simulation_log(raw_log: str) -> list[SimulationEvent]:
    """Parse raw simulation log into structured events.

    Args:
        raw_log: Raw stdout capture from simulation.

    Returns:
        List of structured events.
    """
    # Strip ANSI color codes
    clean_log = re.sub(r"\x1b\[[0-9;]*m", "", raw_log)

    # Remove Python logging lines (timestamp | LEVEL | logger | message)
    clean_log = re.sub(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|.*$", "", clean_log, flags=re.MULTILINE
    )

    events: list[SimulationEvent] = []
    current_step = 0

    # Find all termination checks to determine step boundaries
    term_matches = list(EVENT_PATTERNS["termination"].finditer(clean_log))

    # Process the log in order by finding all events and sorting by position
    all_matches: list[tuple[int, str, re.Match]] = []  # (position, event_type, match)

    for event_type, pattern in EVENT_PATTERNS.items():
        for match in pattern.finditer(clean_log):
            all_matches.append((match.start(), event_type, match))

    # Sort by position in the log
    all_matches.sort(key=lambda x: x[0])

    # Track which step we're in based on termination checks
    term_positions = [m.start() for m in term_matches]
    step_idx = 0

    for pos, event_type, match in all_matches:
        # Update step based on termination check positions
        while step_idx < len(term_positions) and pos >= term_positions[step_idx]:
            step_idx += 1
        current_step = max(0, step_idx - 1) if step_idx > 0 else 0

        if event_type == "termination":
            result = match.group(1).strip()
            events.append(
                SimulationEvent(step=current_step, event_type="termination_check", content=result)
            )

        elif event_type == "game_master":
            gm_name = match.group(1).strip()
            events.append(
                SimulationEvent(step=current_step, event_type="game_master", content=gm_name)
            )

        elif event_type == "observation":
            agent = match.group(1).strip()
            content = match.group(2).strip()
            # Clean up content
            content = re.sub(r"\n{3,}", "\n\n", content)
            events.append(
                SimulationEvent(
                    step=current_step, event_type="observation", agent=agent, content=content
                )
            )

        elif event_type == "action":
            agent = match.group(1).strip()
            content = match.group(2).strip()
            events.append(
                SimulationEvent(
                    step=current_step, event_type="action", agent=agent, content=content
                )
            )

        elif event_type == "putative_event":
            content = match.group(1).strip()
            events.append(SimulationEvent(step=current_step, event_type="attempt", content=content))

        elif event_type == "resolution":
            content = match.group(1).strip()
            events.append(
                SimulationEvent(step=current_step, event_type="resolution", content=content)
            )

        elif event_type == "skip_step":
            events.append(
                SimulationEvent(
                    step=current_step, event_type="skip_step", content="Action phase skipped"
                )
            )

        elif event_type == "checkpoint":
            step_num = int(match.group(1))
            events.append(
                SimulationEvent(
                    step=current_step,
                    event_type="checkpoint",
                    content=f"Step {step_num}",
                    metadata={"step": step_num},
                )
            )

    return events


def format_events_text(events: list[SimulationEvent], max_observation_length: int = 500) -> str:
    """Format events as human-readable text.

    Args:
        events: List of simulation events.
        max_observation_length: Maximum length for observation content.

    Returns:
        Formatted text string.
    """
    lines = ["# Simulation Event Log", "# " + "=" * 50, ""]

    current_step = -1

    for event in events:
        # Add step header when step changes
        if event.event_type == "termination_check" and event.step != current_step:
            current_step = event.step
            lines.append(f"\n[Step {event.step}]")
            continue

        if event.event_type == "game_master":
            lines.append(f"  Game Master: {event.content}")

        elif event.event_type == "observation":
            content = event.content
            if len(content) > max_observation_length:
                first_para = content.split("\n\n")[0]
                if len(first_para) > max_observation_length:
                    content = first_para[:max_observation_length] + "..."
                else:
                    content = first_para + " [...]"
            # Replace newlines in content for single-line output
            content = content.replace("\n", " ").replace("  ", " ")
            lines.append(f"  {event.agent} observed: {content}")

        elif event.event_type == "action":
            content = event.content.replace("\n", " ").replace("  ", " ")
            lines.append(f"  {event.agent} chose: {content}")

        elif event.event_type == "attempt":
            content = event.content.replace("\n", " ").replace("  ", " ")[:200]
            lines.append(f"  Attempting: {content}...")

        elif event.event_type == "resolution":
            content = event.content.replace("\n", " ").replace("  ", " ")[:200]
            lines.append(f"  Result: {content}...")

        elif event.event_type == "skip_step":
            lines.append("  (Step skipped)")

        elif event.event_type == "checkpoint":
            lines.append("  [Checkpoint saved]")

    lines.append("\n# End of simulation")
    return "\n".join(lines)


def format_events_jsonl(events: list[SimulationEvent]) -> str:
    """Format events as JSONL.

    Args:
        events: List of simulation events.

    Returns:
        JSONL string (one JSON object per line).
    """
    return "\n".join(json.dumps(asdict(e)) for e in events)


def process_raw_log(
    raw_log_path: Path | str,
    output_path: Path | str | None = None,
    format: str = "text",
) -> list[SimulationEvent]:
    """Process a raw log file into structured events.

    Args:
        raw_log_path: Path to the raw stdout log file.
        output_path: Optional path to write the structured output.
            If None, writes to same directory as raw log.
        format: Output format ("text" or "jsonl").

    Returns:
        List of parsed events.
    """
    raw_log_path = Path(raw_log_path)
    raw_log = raw_log_path.read_text(encoding="utf-8")

    events = parse_simulation_log(raw_log)

    if output_path is None:
        suffix = ".txt" if format == "text" else ".jsonl"
        output_path = raw_log_path.parent / f"simulation_events{suffix}"
    else:
        output_path = Path(output_path)

    if format == "text":
        output_path.write_text(format_events_text(events), encoding="utf-8")
    else:
        output_path.write_text(format_events_jsonl(events), encoding="utf-8")

    return events

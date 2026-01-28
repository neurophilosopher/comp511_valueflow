#!/usr/bin/env python3
"""Main entry point for running Concordia simulations with Hydra configuration.

This script provides the Hydra-decorated main function that:
1. Loads and validates configuration
2. Sets up logging
3. Creates the appropriate simulator
4. Runs the simulation
5. Saves results

Usage:
    # Run with default configuration
    python run_experiment.py

    # Run marketplace scenario
    python run_experiment.py scenario=marketplace

    # Override parameters
    python run_experiment.py simulation.execution.max_steps=50 model=claude

    # Multi-model simulation
    python run_experiment.py model=multi_model

    # View configuration without running
    python run_experiment.py --cfg job

    # Multirun with different seeds
    python run_experiment.py --multirun experiment.seed=1,2,3,4,5
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.simulators.multi_model import MultiModelSimulator
from src.utils.config_helpers import get_output_paths
from src.utils.event_logger import process_raw_log
from src.utils.logging_setup import TeeStdout, setup_logging
from src.utils.validation import ConfigValidationError, validate_config

# Load environment variables from .env file
load_dotenv()


def save_results(
    config: DictConfig,
    result: str | list[Mapping[str, Any]],
) -> None:
    """Save simulation results to configured output paths.

    Args:
        config: Hydra configuration.
        result: Simulation result (HTML string or raw log).
    """
    output_paths = get_output_paths(config)
    sim_config = config.simulation

    # Save HTML log
    if sim_config.logging.save_html and isinstance(result, str):
        html_path = output_paths.get("html")
        if html_path:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            with html_path.open("w", encoding="utf-8") as f:
                f.write(result)
            print(f"HTML log saved to: {html_path}")

    # Save raw log
    if sim_config.logging.save_raw and isinstance(result, list):
        raw_path = output_paths.get("raw_log")
        if raw_path:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Raw log saved to: {raw_path}")


def print_config_summary(config: DictConfig) -> None:
    """Print a summary of the configuration.

    Args:
        config: Hydra configuration.
    """
    print("\n" + "=" * 60)
    print("Concordia Simulation Framework")
    print("=" * 60)
    print(f"Experiment: {config.experiment.name}")
    print(f"Scenario: {config.scenario.name}")
    print(f"Simulation: {config.simulation.name}")
    print(f"Model: {config.model.name}")
    print(f"Max Steps: {config.simulation.execution.max_steps}")
    print(f"Output Dir: {config.experiment.output_dir}")
    print("=" * 60 + "\n")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="experiment",
)
def main(config: DictConfig) -> float | None:
    """Main entry point for running simulations.

    Args:
        config: Hydra configuration (automatically populated).

    Returns:
        Optional metric value for Hydra sweeps/optimization.
    """
    # Setup logging
    logger = setup_logging(config)

    # Print configuration summary
    print_config_summary(config)

    # Validate configuration
    try:
        warnings = validate_config(config)
        for warning in warnings:
            logger.warning(warning)
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        return None

    # Log full configuration at debug level
    logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(config)}")

    # Create simulator
    logger.info("Initializing simulator...")
    simulator = MultiModelSimulator(config)

    # Get logging config
    sim_config = config.simulation
    event_log_format = sim_config.logging.get("event_log_format")
    output_dir = Path(config.experiment.output_dir)

    # Set up raw stdout capture for event logging
    raw_log_path = output_dir / "run_experiment.log"

    try:
        # Setup simulation (creates models, embedder, simulation instance)
        logger.info("Setting up simulation...")
        simulator.setup()

        # Run simulation with stdout capture
        logger.info("Starting simulation...")
        with TeeStdout(raw_log_path):
            result = simulator.run()

        # Save results
        save_results(config, result)

        # Post-process raw log into structured events if configured
        if event_log_format and raw_log_path.exists():
            try:
                if event_log_format == "both":
                    # Generate both text and jsonl formats
                    process_raw_log(raw_log_path, format="text")
                    process_raw_log(raw_log_path, format="jsonl")
                    logger.info(
                        f"Structured event logs saved to: {output_dir}/simulation_events.txt and .jsonl"
                    )
                else:
                    process_raw_log(raw_log_path, format=event_log_format)
                    suffix = "jsonl" if event_log_format == "jsonl" else "txt"
                    logger.info(
                        f"Structured event log saved to: {output_dir}/simulation_events.{suffix}"
                    )
            except Exception as e:
                logger.warning(f"Failed to generate structured event log: {e}")

        logger.info("Simulation completed successfully!")

        # Return a metric for Hydra sweeps (e.g., could be a score)
        return 0.0

    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
        return None

    except Exception as e:
        logger.exception(f"Simulation failed with error: {e}")
        raise


def run_quick_test() -> None:
    """Run a quick test with mock models (no API calls).

    This function can be called directly for testing without Hydra.
    """
    from omegaconf import OmegaConf

    from src.utils.testing import MockLanguageModel, create_test_config, mock_embedder

    print("\n" + "=" * 60)
    print("Running Quick Test (Mock Mode)")
    print("=" * 60 + "\n")

    # Create test configuration
    test_config = create_test_config()
    _config = OmegaConf.create(test_config)

    # Create mock models
    mock_model = MockLanguageModel()
    models = {"mock": mock_model}

    # Build simulation manually for testing
    from concordia.typing import prefab as prefab_lib

    from src.entities.agents.basic_entity import BasicEntity
    from src.entities.game_masters.basic_gm import BasicGameMaster
    from src.simulation.simulation import Simulation

    # Create prefabs
    prefabs = {
        "basic_entity": BasicEntity(),
        "basic_game_master": BasicGameMaster(),
    }

    # Create instances
    instances = [
        prefab_lib.InstanceConfig(
            prefab="basic_entity",
            role=prefab_lib.Role.ENTITY,
            params={"name": "TestAgent", "goal": "Complete the test scenario"},
        ),
        prefab_lib.InstanceConfig(
            prefab="basic_game_master",
            role=prefab_lib.Role.GAME_MASTER,
            params={"name": "test_narrator"},
        ),
    ]

    # Create config
    concordia_config = prefab_lib.Config(
        prefabs=prefabs,
        instances=instances,
        default_premise="This is a test simulation.",
        default_max_steps=3,
    )

    # Create and run simulation
    simulation = Simulation(
        config=concordia_config,
        models=models,
        entity_to_model={"_default_": "mock"},
        embedder=mock_embedder,
    )

    print("Running simulation...")
    result = simulation.play(max_steps=3, return_html_log=False)

    print(f"\nSimulation completed with {len(result)} log entries")
    print(f"Mock model was called {mock_model.call_count} times")
    print("\nTest passed!")


if __name__ == "__main__":
    # Check for quick test mode
    if "--quick-test" in sys.argv:
        sys.argv.remove("--quick-test")
        run_quick_test()
    else:
        main()

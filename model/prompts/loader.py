"""
Prompt Loader Module

Loads prompt configurations from YAML files with support for project-specific overrides.
Default prompts are loaded from the 'defaults' directory within this package.
Project-specific overrides can be placed in the directory specified by PROMPTS_DIR env variable.
"""

import os
from pathlib import Path
from typing import Any

import yaml

# Directory containing default prompts (shipped with the package)
DEFAULTS_DIR = Path(__file__).parent / "defaults"


def get_prompts_override_dir() -> Path | None:
    """Get the prompts override directory from environment variable."""
    prompts_dir = os.getenv("PROMPTS_DIR")
    if prompts_dir:
        path = Path(prompts_dir)
        if path.exists() and path.is_dir():
            return path
    return None


def load_prompt_config(prompt_name: str) -> dict[str, Any]:
    """
    Load a prompt configuration by name.

    First checks for an override file in PROMPTS_DIR, then falls back to defaults.

    Args:
        prompt_name: Name of the prompt file (without .yaml extension)

    Returns:
        Dictionary containing the prompt configuration

    Raises:
        FileNotFoundError: If the prompt file doesn't exist in either location
    """
    filename = f"{prompt_name}.yaml"

    # Check for override first
    override_dir = get_prompts_override_dir()
    if override_dir:
        override_path = override_dir / filename
        if override_path.exists():
            with open(override_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    # Fall back to defaults
    default_path = DEFAULTS_DIR / filename
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Prompt configuration '{prompt_name}' not found in "
        f"override dir ({override_dir}) or defaults ({DEFAULTS_DIR})"
    )


def get_prompt_file_path(prompt_name: str) -> Path:
    """
    Get the path to a prompt file (override or default).

    Args:
        prompt_name: Name of the prompt file (without .yaml extension)

    Returns:
        Path to the prompt file that would be loaded
    """
    filename = f"{prompt_name}.yaml"

    override_dir = get_prompts_override_dir()
    if override_dir:
        override_path = override_dir / filename
        if override_path.exists():
            return override_path

    return DEFAULTS_DIR / filename

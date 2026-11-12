from pathlib import Path

import yaml

from data.experiment import ExperimentSpec


def load_experiment_spec(
    path: str | Path,
) -> ExperimentSpec:
    # Convert the supplied path to Path. Raise FileNotFoundError if it doesn't exist.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment spec file not found: {path}")

    # Load the YAML content from the file.
    with open(path, "r") as f:
        content = yaml.safe_load(f)
        # Ensure the content is a dictionary.
        if not isinstance(content, dict):
            raise ValueError(f"Invalid format: {path}")

    # Parse the content into an ExperimentSpec instance.
    return ExperimentSpec.model_validate(content)
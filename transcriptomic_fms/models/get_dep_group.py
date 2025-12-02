"""Helper script to get optional dependency group for a model."""

from pathlib import Path
import sys
import tomllib


def get_dep_group(model_name: str) -> str:
    """Get optional dependency group for a model."""
    # First try to get from model class if it's registered and can be instantiated
    try:
        from transcriptomic_fms.models import get_model

        m = get_model(model_name)
        dep_group = m.get_optional_dependency_group()
        if dep_group:
            return dep_group
    except Exception:
        pass

    # If model can't be imported/instantiated, check pyproject.toml directly
    try:
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        optional_deps = config.get("project", {}).get("optional-dependencies", {})
        if model_name in optional_deps:
            return model_name
    except Exception:
        pass

    return ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    model_name = sys.argv[1]
    dep_group = get_dep_group(model_name)
    if dep_group:
        print(dep_group)

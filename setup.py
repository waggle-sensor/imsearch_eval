"""
Setup configuration for imsearch_eval package.

This package provides an abstract framework for benchmarking vector databases and models.
Adapters are available as optional dependencies.
"""
from setuptools import setup, find_packages
from pathlib import Path

VERSION = "0.1.32"

def parse_requirements(requirements_path: Path) -> list[str]:
    """Helper function to parse requirements file, filtering out empty lines and comments."""
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

# Get requirements from files
core_requirements_path = Path(__file__).parent / "imsearch_eval/requirements.txt"
adapters_dir = Path(__file__).parent / "imsearch_eval/adapters"

# Core dependencies
CORE_DEPS = parse_requirements(core_requirements_path)

# Automatically discover adapters and their dependencies
EXTRAS = {}
all_adapter_deps = []
if adapters_dir.exists():
    for adapter_dir in adapters_dir.iterdir():
        # Skip non-directories and special files
        if not adapter_dir.is_dir() or adapter_dir.name.startswith("_") or adapter_dir.name == "__pycache__":
            continue
        
        adapter_name = adapter_dir.name
        adapter_requirements_path = adapter_dir / "requirements.txt"
        
        # Only add adapter if it has a requirements.txt file
        if adapter_requirements_path.exists():
            adapter_deps = parse_requirements(adapter_requirements_path)
            if adapter_deps:  # Only add if there are dependencies
                EXTRAS[adapter_name] = adapter_deps
                all_adapter_deps.extend(adapter_deps)
# Add "all" extra that includes all adapter dependencies (deduplicated)
# Use dict.fromkeys() to preserve order while removing duplicates
EXTRAS["all"] = list(dict.fromkeys(all_adapter_deps))

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else "Abstract benchmarking framework for vector databases and models."

setup(
    name="imsearch_eval",
    version=VERSION,
    description="Abstract benchmarking framework for vector databases and models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Francisco Lozano",
    author_email="francisco.lozano@northwestern.edu",
    url="https://github.com/waggle-sensor/imsearch_eval",
    packages=find_packages(),
    install_requires=CORE_DEPS,
    extras_require=EXTRAS,
    python_requires=">=3.8",
    keywords="benchmarking, vector-database, evaluation, metrics, ndcg",
)


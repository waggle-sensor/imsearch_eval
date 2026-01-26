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
triton_requirements_path = Path(__file__).parent / "imsearch_eval/adapters/triton/requirements.txt"
weaviate_requirements_path = Path(__file__).parent / "imsearch_eval/adapters/weaviate/requirements.txt"
milvus_requirements_path = Path(__file__).parent / "imsearch_eval/adapters/milvus/requirements.txt"
huggingface_requirements_path = Path(__file__).parent / "imsearch_eval/adapters/huggingface/requirements.txt"

# Core dependencies
CORE_DEPS = parse_requirements(core_requirements_path)

# Optional dependencies for different adapters
triton_deps = parse_requirements(triton_requirements_path)
weaviate_deps = parse_requirements(weaviate_requirements_path)
milvus_deps = parse_requirements(milvus_requirements_path)
huggingface_deps = parse_requirements(huggingface_requirements_path)
EXTRAS = {
    "triton": triton_deps,
    "weaviate": weaviate_deps,
    "milvus": milvus_deps,
    "huggingface": huggingface_deps,
    "all": triton_deps + weaviate_deps + milvus_deps + huggingface_deps,
}

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


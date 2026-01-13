"""
Setup configuration for imsearch_eval package.

This package provides an abstract framework for benchmarking vector databases and models.
Adapters are available as optional dependencies.
"""

from setuptools import setup, find_packages

VERSION = "0.1.22"

# Core dependencies (always required)
CORE_DEPS = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "Pillow>=9.0.0",
]

# Optional adapter dependencies
EXTRAS = {
    "triton": [
        "tritonclient[grpc]>=2.0.0",
    ],
    "weaviate": [
        "weaviate-client>=4.0.0",
    ],
    "milvus": [
        "pymilvus>=2.6.6",
    ],
    "huggingface": [
        "datasets>=4.4.1",
        "huggingface-hub>=0.16.0",
    ],
    # Convenience extra that includes everything
    "all": [
        "tritonclient[grpc]>=2.0.0",
        "weaviate-client>=4.0.0",
        "pymilvus>=2.6.6",
        "datasets>=4.4.1",
        "huggingface-hub>=0.16.0",
    ],
    # For development
    "dev": [
        "pytest>=7.0.0",
    ],
}

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Abstract benchmarking framework for vector databases and models."

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


"""Setup script for RoboCandyWrapper."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="robocandywrapper",
    version="0.2.7",
    description="Sweet wrappers for extending and remixing LeRobot Datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RoboCandyWrapper Contributors",
    python_requires=">=3.10",
    packages=find_packages(include=["robocandywrapper", "robocandywrapper.*"]),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "lerobot>=0.4,<0.5",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


from setuptools import setup, find_packages
from pathlib import Path
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    """
    Read requirements from requirements.txt (relative to setup.py location)
    """
    requirements_path = Path(__file__).parent / "requirements.txt"

    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements.txt not found at: {requirements_path}")

    with requirements_path.open(encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]

    # Remove editable install marker if accidentally present
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="medical_cost_prediction",
    version="0.1.0",
    description="Machine Learning project for predicting medical insurance costs",
    author="Georginho",
    author_email="George.sam@live.co.uk",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    python_requires=">=3.10",
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

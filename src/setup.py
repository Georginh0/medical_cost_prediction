from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements.
    """
    with open(file_path, "r") as file_obj:
        requirements = [
            line.strip() for line in file_obj if line.strip()
        ]  # Read and strip newlines in one step

    # Remove '-e .' if it's present in the list
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="MLproject",
    version="0.1",
    author="Georginho",
    author_email="George.sam@live.co.uk",
    packages=find_packages(),
    install_requires=open("project-template/requirements.txt").read().splitlines(),
)

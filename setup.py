import ast
import os

from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """Get version inside __init__ file"""
    with open(os.path.join(HERE, "targetviz", "__init__.py"), "r") as f:
        data = f.read()
    lines = data.split("\n")
    for line in lines:
        if line.startswith("__version__"):
            version_tuple = ast.literal_eval(line.split("=")[-1].strip())
            version = "".join(map(str, version_tuple))
            break
    return version


def get_dependencies():
    """Get package dependencies from requirements.txt file"""
    with open(os.path.join(HERE, "requirements.txt"), "r") as f:
        return [line.strip() for line in f if line.strip()]


setup(
    version=get_version(),
    install_requires=get_dependencies(),
    include_package_data=True,
    package_data={
        "targetviz": ["templates/*"],
    },
)

"""
Setup script for pySHP package
"""

from setuptools import setup, find_packages

with open("pySHP/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("pySHP/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pySHP",
    version="0.1.0",
    author="Translated from MATLAB shp_toolbox",
    description="Python Spherical Harmonics Parameterization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pySHP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)

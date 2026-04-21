from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "A Python package for hydrological-geophysical model integration and inversion."

# Read requirements from file if it exists
def read_requirements(filename):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

long_description = read_readme()

setup(
    name="PyHydroGeophysX",
    version="0.3.0",
    author="Hang Chen",
    author_email="hangchen.work@gmail.com",
    description="A Python package for hydrological-geophysical model integration and inversion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geohang/PyHydroGeophysX",
    project_urls={
        "Documentation": "https://geohang.github.io/PyHydroGeophysX/",
        "Source": "https://github.com/geohang/PyHydroGeophysX",
        "Tracker": "https://github.com/geohang/PyHydroGeophysX/issues",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21,<3.0",
        "scipy>=1.8,<3.0",
        "matplotlib>=3.5,<4.0",
        "pygimli>=1.5.5,<2.0",
        "simpeg>=0.24,<1.0",
        "flopy>=3.5,<4.0",
        "pftools>=1.3",
        "joblib>=1.2",
        "tqdm>=4.62",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x"],
        "agents": [
            "openai",
            "google-generativeai",
            "anthropic",
        ],
        "climate": [
            "pydaymet>=0.16.0",  # For meteorological data retrieval
            "pandas>=1.3.0",  # For data manipulation
            "xarray>=0.19.0",  # For gridded data handling
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-gallery",
            "sphinx_rtd_theme",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "flake8",
            "black",
        ],
        "all": [
            # Combines all optional dependencies
            "cupy-cuda11x",
            "openai", "google-generativeai", "anthropic",
            "pydaymet>=0.16.0", "pandas>=1.3.0", "xarray>=0.19.0",
            "sphinx>=5.0", "sphinx-gallery", "sphinx_rtd_theme",
            "pytest>=7.0", "pytest-cov", "flake8", "black",
        ]
    },
    include_package_data=True,
    package_data={
        "PyHydroGeophysX": [
            "data/*", 
            "examples/*",
            "docs/*",
            "*.md",
            "*.txt",
            "*.yml",
            "*.yaml"
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "geophysics", 
        "hydrology", 
        "ERT", 
        "electrical resistivity tomography",
        "seismic", 
        "tomography", 
        "inversion", 
        "MODFLOW", 
        "ParFlow",
        "watershed monitoring",
        "time-lapse",
        "petrophysics"
    ],
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here if needed
            # "pyhydrogeo=PyHydroGeophysX.cli:main",
        ],
    },
    zip_safe=False,  # Required for some data file access
)

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
    version="0.1.0",
    author="Hang Chen",
    author_email="your_email@example.com",
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
        "numpy>=1.19",
        "scipy>=1.5",
        "matplotlib>=3.2",
        "tqdm>=4.0",
        # Core dependencies only - heavy deps are optional
    ],
    extras_require={
        "geophysics": [
            "pygimli>=1.5",   # Optional, heavy dependencies for real geophysical usage
            "flopy",
            "cupy",
            "parflow",
            "joblib",
            "meshop",
        ],
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
            "nbsphinx>=0.9.1",
            "sphinx-copybutton>=0.5.2",
            "sphinx-gallery>=0.14.0",
            "palettable",
            "Pillow",  # For image processing
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "all": [
            # Combines all optional dependencies
            "pygimli>=1.5", "flopy", "cupy", "parflow", "joblib", "meshop",
            "sphinx>=7.1.2", "sphinx-rtd-theme>=1.3.0", "myst-parser>=2.0.0",
            "nbsphinx>=0.9.1", "sphinx-copybutton>=0.5.2", "sphinx-gallery>=0.14.0",
            "palettable", "Pillow", "pytest>=6.0", "pytest-cov", "black", "flake8", "mypy",
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
        "License :: OSI Approved :: MIT License",
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
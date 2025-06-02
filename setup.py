from setuptools import setup, find_packages

# Only core scientific dependencies that are always available
core_requirements = [
    "numpy>=1.19.0",
    "scipy>=1.6.0", 
    "matplotlib>=3.3.0",
    "tqdm>=4.50.0",
]

# All geophysical packages are optional
geophysics_requirements = [
    "pygimli",
    "joblib", 
    "cupy",
]

# Documentation requirements
docs_requirements = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.17.0",
    "nbsphinx>=0.8.0",
    "sphinx-copybutton>=0.5.0",
    "sphinx-gallery>=0.10.0",
]

setup(
    name="PyHydroGeophysX",
    version="0.1.0",
    packages=find_packages(),
    install_requires=core_requirements,  # Only core deps by default
    extras_require={
        'full': geophysics_requirements,
        'geophysics': ['pygimli'],
        'gpu': ['cupy'],
        'parallel': ['joblib'],
        'docs': docs_requirements,
        'all': geophysics_requirements + docs_requirements,
    },
    author="Hang Chen",
    description="A comprehensive package for geophysical modeling and inversion in watershed monitoring",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
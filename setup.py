from setuptools import setup, find_packages

# Core dependencies that are always needed
core_requirements = [
    "numpy",
    "scipy", 
    "matplotlib",
    "tqdm",
]

# Optional heavy dependencies
optional_requirements = [
    "pygimli",  # Only needed for actual geophysical modeling
    "joblib",
    "cupy",     # Only needed for GPU acceleration
]

setup(
    name="PyHydroGeophysX",
    version="0.1.0",
    packages=find_packages(),
    install_requires=core_requirements,
    extras_require={
        'full': optional_requirements,
        'geophysics': ['pygimli'],
        'gpu': ['cupy'],
        'parallel': ['joblib'],
    },
    author="Hang Chen",
    description="A comprehensive package for geophysical modeling and inversion in watershed monitoring",
    python_requires=">=3.8",
)
from setuptools import setup, find_packages

setup(
    name="PyHydroGeophysX",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy", 
        "matplotlib",
        "pygimli",
        "joblib",
        "tqdm",
    ],
    author="Hang Chen",
    description="A comprehensive package for geophysical modeling and inversion in watershed monitoring",
    python_requires=">=3.8",
)
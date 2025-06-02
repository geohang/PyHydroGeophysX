from setuptools import setup, find_packages

setup(
    name="PyHydroGeophysX",
    version="0.1.0",
    author="Hang Chen",
    author_email="your_email@example.com",
    description="A Python package for hydrological-geophysical model integration and inversion.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/geohang/PyHydroGeophysX",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.5",
        "matplotlib>=3.2",
        "tqdm>=4.0",
        # Do NOT list pygimli, flopy, cupy, parflow, joblib, meshop, etc. here unless *absolutely required* at import time!
        # If your package needs these for *optional* features, use 'extras_require' below.
    ],
    extras_require={
        "geophysics": [
            "pygimli>=1.5",   # Optional, heavy dependencies for real geophysical usage
            "flopy",
            "cupy",
            "parflow",
            "joblib",
            "meshop",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

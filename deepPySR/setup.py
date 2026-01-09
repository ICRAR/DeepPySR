from setuptools import setup, find_packages

setup(
    name="deepPySR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pysr",
        "numpy",
        "sympy",
        "matplotlib",
        "networkx",
        "pandas"
    ],
    author="FulingChen",
    description="A deep symbolic regression package based on PySR",
)

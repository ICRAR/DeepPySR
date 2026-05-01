from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).resolve().parent
long_description = ""
readme_path = here / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="deepPySR",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pysr",
        "numpy",
        "sympy",
        "matplotlib",
        "networkx",
        "pandas",
        "pyCirclize",
        "torch",
        "pykan",
        "scikit-learn",
        "tqdm",
        "scipy",
        "imblearn",
        "PyYAML",
        "xgboost",
    ],
    author="FulingChen",
    description="A deep symbolic regression package based on PySR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DeepPySR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

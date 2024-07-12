from setuptools import setup, find_packages

setup(
    name="Steerers",
    packages=find_packages(include=["Steerers*"]),
    install_requires=[
        "DeDoDe",
    ],
    python_requires=">=3.9.0",
    version="0.0.1",
    author="Atsuki Sato",
)

#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="huygens",
    version="0.1",
    author="Simon Burton",
    author_email="simon@arrowtheory.com",
    description="huygens graphic design package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/punkdit/huygens",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)


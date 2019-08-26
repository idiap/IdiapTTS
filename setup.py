#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Setup idiaptts"""

from itertools import dropwhile
from os import path
from setuptools import find_packages, setup


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("idiaptts", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    return []


def get_install_requirements():
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()

    return requirements


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="idiaptts",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            # "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python :: 3.6",
            # "Programming Language :: Python :: 3.7",
        ],
        packages=find_packages(exclude=["egs", "tools", "tests"]),
        install_requires=get_install_requirements(),
        ext_modules=get_extensions()
    )


if __name__ == "__main__":
    setup_package()

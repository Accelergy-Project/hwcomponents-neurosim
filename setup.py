from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import sys
import subprocess
import os


def readme():
    """Returns readme contents"""
    with open("README.md") as f:
        return f.read()


class Build(build_ext):
    """Calls makefile"""

    def run(self):
        print("Building NeuroSim")
        if subprocess.call(["make", "make"]) != 0:
            sys.exit(-1)
        build_ext.run(self)


setup(
    name="hwcomponents-neurosim",
    version="0.1",
    description="A package for estimating the energy and area of NeuroSim components",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    keywords="hardware energy estimation analog adc neurosim pim processing-in-memory cim",
    author="Tanner Andrulis",
    author_email="andrulis@mit.edu",
    license="MIT",
    install_requires=[],
    python_requires=">=3.12",
    packages=find_packages(),
    package_data={
        "hwcomponents_neurosim": [
            "cells/*",
            "NeuroSim/main",
            "default_config.cfg",
            "neurosim.estimator.yaml",
        ],
    },
    include_package_data=True,
    py_modules=["hwcomponents_neurosim"],
    entry_points={},
    cmdclass={
        "build_ext": Build,
    },
)

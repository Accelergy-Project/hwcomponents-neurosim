from setuptools import setup, find_packages
from setuptools.command.build import build
import sys
import subprocess
import os


def readme():
    """Returns readme contents"""
    with open("README.md") as f:
        return f.read()


class CustomBuildCommand(build):
    """Custom build command that runs make build before the normal build process."""

    def run(self):
        try:
            print("Running 'make build'...")
            subprocess.check_call(["make", "build"], cwd=os.path.dirname(os.path.realpath(__file__)))
            print("'make build' completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'make build': {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("Warning: 'make' command not found. Skipping build step.")
        except Exception as e:
            print(f"Unexpected error running 'make build': {e}")
            sys.exit(1)
        
        # Call the parent build command
        super().run()


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
            "neurosim.model.yaml",
        ],
    },
    include_package_data=True,
    py_modules=["hwcomponents_neurosim"],
    entry_points={},
    cmdclass={
        "build": CustomBuildCommand,
    },
)

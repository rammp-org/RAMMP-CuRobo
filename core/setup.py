"""Shim for editable installs on older pip (Ubuntu 22.04 / Jetson ships pip
without full PEP 660 support). All metadata lives in pyproject.toml."""

from setuptools import setup

setup()

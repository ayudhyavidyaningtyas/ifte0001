"""
Setup configuration for Vivi's Equity Research Agent.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vivi-equity-research",
    version="1.0.0",
    author="Vivi",
    description="AI-powered equity research agent with DCF valuation and PDF report generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vivi-equity-research",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "yfinance>=0.2.36",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "python-dotenv>=1.0.1",
        "reportlab>=4.0.9",
        "matplotlib>=3.8.2",
    ],
    entry_points={
        "console_scripts": [
            "vivi-research=main:main",
        ],
    },
)

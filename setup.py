from setuptools import setup, find_packages

setup(
    name="llm-agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "python-dotenv",
        "mangum",
        "openai",
        "yfinance",
        "google-search-results==2.4.2",
        "motor",
    ],
)
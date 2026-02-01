"""
Vivi's Agentic AI - Equity Research Agent
"""

from .agent import run_agent
from .data_fetcher import fetch_financial_data, get_price_history
from .valuation import calculate_dcf, calculate_dcf_sensitivity
from .pdf_generator import generate_equity_report_pdf

__version__ = "1.0.0"

__all__ = [
    "run_agent",
    "fetch_financial_data",
    "get_price_history",
    "calculate_dcf",
    "calculate_dcf_sensitivity",
    "generate_equity_report_pdf",
]

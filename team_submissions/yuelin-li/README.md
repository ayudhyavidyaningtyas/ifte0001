Project: Fundamental Financial Analysis and Valuation.

Project Overview
This project implements an end-to-end fundamental analysis and valuation pipeline using Python.
It demonstrates how publicly available financial data can be collected, processed, analysed, and translated into valuation outputs in a reproducible manner.

The project is designed as a coursework-style demonstration of financial data engineering and valuation logic, rather than a production-grade trading or investment system.

1.Repository Structure

`run_demo1.py`  
  Main analytical pipeline implementing data ingestion, financial metrics computation, valuation modelling, visualisation, and investment memo generation.

`check_yfinance_fields.py`  
  Utility script used to inspect which data fields are available from Yahoo Finance before running the main analysis.

`generate_readme.py`  
  Helper script that automatically generates the README file based on the repository structure.  
  This script is not part of the analytical workflow.

`README.md`  
  Project documentation.

2.Main Script: `run_demo1.py`

`run_demo1.py` implements a complete fundamental analysis and valuation workflow consisting of the following stages:

（1）. Market Data Ingestion
- Downloads historical OHLCV (Open, High, Low, Close, Volume) price data using the `yfinance` library.
- Performs basic integrity checks and standardises date indices for subsequent analysis.

（2）. Financial Statement Collection
- Retrieves annual income statements, balance sheets, and cash flow statements via `yfinance`.
- Converts all financial values into a consistent unit (USD millions).
- Computes free cash flow (FCF) manually when it is not directly available from the source data.

（3）. Financial Metrics Computation
- Organises multi-year financial statement data into a structured time-series DataFrame.
- Computes key financial indicators, including:
  - Profitability: net margin, operating margin, ROE, ROA  
  - Growth: revenue year-over-year growth  
  - Leverage and liquidity: debt-to-equity ratio, current ratio  
  - Efficiency: asset turnover  
- Applies sanity checks to detect abnormal ratios caused by unit inconsistencies or missing data.

（4）. Valuation Models
- **Discounted Cash Flow (DCF) Valuation**
  - Five-year explicit FCF forecast with declining growth assumptions.
  - Terminal value estimated using the Gordon Growth method.
  - Net cash adjustments applied when balance sheet data is available.
  - Sensitivity analysis conducted on WACC and terminal growth assumptions.
- **Relative Valuation**
  - Simple EV/Revenue multiple used as a market-based cross-check against the DCF results.

（5）. Visualisation
- Generates time-series plots for revenue, free cash flow, and selected financial ratios.
- Saves all charts as PNG files for reporting and inspection.

（6）. Investment Memo Generation
- Produces a structured investment memo summarising:
  - Financial performance
  - Valuation outcomes
  - Key risks and catalysts
- Supports two modes:
  - Local template-based generation (default)
  - Optional LLM-assisted generation using the OpenAI API when an API key is provided

（7）. Output Management
- Stores all intermediate and final outputs (CSV, JSON, PNG, TXT) in a structured output directory.
- Prints a concise summary of key valuation results at the end of each run.

3.How to Run

（1）. Inspect available Yahoo Finance fields:
python check_yfinance_fields.py
（2）. Run the main analysis pipeline:
python run_demo1.py


4.Output

The script generates:

Historical market data (CSV)

Financial metrics and summaries (CSV / JSON)

Valuation results (DCF and relative valuation, JSON)

Time-series charts (PNG)

Investment memo (TXT)

A concise run summary printed to the console

All outputs are saved under the specified output directory.


5.Requirements

(1)Python 3.8+

(2)Required packages:
pip install yfinance pandas numpy matplotlib
pip install openai



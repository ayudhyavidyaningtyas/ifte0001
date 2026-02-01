# Project: Fundamental Financial Analysis and Valuation.

## Project Overview
This project implements an end-to-end fundamental analysis and valuation pipeline using Python.
It demonstrates how publicly available financial data can be collected, processed, analysed, and

---


## Repository Structure

- `run_demo1.py`  
  Main analytical pipeline implementing data ingestion, financial metrics computation, valuation modelling, visualisation, and investment memo generation.

- `check_yfinance_fields.py`  
  Utility script used to inspect which data fields are available from Yahoo Finance before running the main analysis.


- `README.md`  
  Project documentation.

---

## Main Script: `run_demo1.py`

`run_demo1.py` implements a complete fundamental analysis and valuation workflow consisting of the following stages:


### 1. Market Data Ingestion
- Downloads historical OHLCV (Open, High, Low, Close, Volume) data using `yfinance`
- Performs integrity checks and standardises date indices

### 2. Financial Statement Collection
- Retrieves income statements, balance sheets, and cash flow statements via `yfinance`
- Converts all values to USD millions
- Computes free cash flow (FCF) manually when necessary

### 3. Financial Metrics Computation
- Constructs multi-year time-series financial data
- Computes profitability, growth, leverage, liquidity, and efficiency metrics
- Applies sanity checks to detect abnormal ratios


### 4. Valuation Models
- **Discounted Cash Flow (DCF)** with five-year projections and terminal value
- **Relative valuation** using EV/Revenue as a cross-check

### 5. Visualisation
- Generates time-series plots for revenue, FCF, and key ratios

### 6. Investment Memo Generation
- Produces a structured investment memo
- Supports local template or optional LLM-based generation (OpenAI API)

### 7. Output Management
- Saves all outputs (CSV, JSON, PNG, TXT) to a structured directory
- Prints a concise valuation summary to the console

---


## How to Run

### 1. Inspect Yahoo Finance fields
```bash
python check_yfinance_fields.py
```
2. Run the main analysis
```bash
python run_demo1.py
```

---


## Output

The script generates the following outputs:

- **Historical market data** (CSV)
- **Financial metrics and summaries** (CSV / JSON)
- **Valuation results**  
  - Discounted Cash Flow (DCF) valuation  
  - Relative valuation  
  (JSON)
- **Time-series charts** (PNG)
- **Investment memo** (TXT)
- **Run summary** printed to the console

All outputs are saved under the specified output directory.

---

## Requirements

### Python Version
- Python **3.8+**

### Required Packages
```bash
pip install yfinance pandas numpy matplotlib
pip install openai



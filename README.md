# VERA: Valuation & Equity Research Assistant

## Overview
VERA is an automated equity research workflow that combines data retrieval, financial statement analysis, and multiple valuation approaches into a single, reproducible pipeline. It is designed to help fundamental analysis where transparency of assumptions and consistency of outputs matter as much as speed.

At a high level, VERA pulls accounting data and market data, computes standard ratio diagnostics, runs several valuation methods (intrinsic and relative), optionally produces a research-note style report that synthesises outputs into an investment view.

## System Architecture

VERA is structured as a sequential pipeline of six modules. Each module is designed to ingest shared inputs and produce structured outputs.

### Core Modules

1. **Financial Data Downloader** (`downloader.py`)
   - Retrieves financial statements from Alpha Vantage API
   - Enhanced with market data from Yahoo Finance (e.g., price and valuation fields where available).
   - Computes trailing twelve months (TTM) figures to support near-term ratio and valuation work.
   - Single API call architecture minimises data retrieval and "over-using" the API calls to Alpha Vantage, making it more efficient.


2. **Ratio Analysis** (`ratios.py`)
   - Calculates 40+ financial ratios across six categories
   - Categories: Profitability, Liquidity, Leverage, Efficiency, Cash Flow, Valuation
   - Supports multi-period comparative analysis
   - Exports results to Excel workbooks

3. **DCF Valuation** (`dcf.py`)
   - Implements 10-year discounted cash flow model
   - Three scenario framework: Bear, Base, Bullish
   - Synthetic credit rating methodology for cost of debt estimation 
   - Downloads Beta and adjust it to Blume-adjusted Beta
   - Downloads US 10Y Treasury yield (spot) as risk free rate
   - Sensitivity analysis across WACC and terminal growth rate dimensions

4. **Relative Valuation** (`relative.py`)
   - Peer-based multiple analysis (P/E, EV/EBITDA, P/S)
   - Applies outlier control using an interquartile-range filter to reduce extreme observations
   - Implements dividend discount model as supplementary methodology

5. **Target Price Analysis** (`target.py`)
   - Forward P/E methodology with 12-month horizon
   - Dynamic scenario generation based on current market multiples (forward P/E)
   - Five-scenario framework from Very Bear to Very Bull
   - Adjusts earnings estimates and P/E premiums independently

6. **Investment Memo Generator** (`memo_generator.py`) - OpenAI API Key needed.
   - Automated research note generation using GPT-4 (OpenAI)
   - Automated qualitative analysis using GPT-4
   - Converts outputs into a structured narrative such as business context, thesis and key drivers, valuation, risks and catalysts. 
   - Produces professional HTML reports with embedded visualisations
   - Implements blended valuation (DCF, target price, and qualitative analysis). 

### Orchestration

- **Main Analysis** (`vera_main.py`): Runs the core pipeline. 
- **Complete System** (`vera_complete_with_memo.py`): Runs the core pipeline and memo generation. 

## Methodology

### Data Architecture

VERA follows a single-ingestion architecture: financial statement data is downloaded once and then passed through the pipeline. This reduces the chance of internal inconsistencies (e.g., ratios computed from one dataset and valuations computed from another) and keeps API usage minimal. 

### Valuation Framework

The system implements four independent valuation methodologies, each contributing distinct analytical perspectives:

**Discounted Cash Flow (DCF)**: The model projects free cash flow over a decade and discounts using WACC (Beta and Risk free rate downloaded; Cost of debt using synthetic rating based on interest coverage ratio). Terminal value is estimated via a perpetuity growth model (Gordon-growth Model). 

**Relative Valuation**: Compares trailing multiples (P/E, EV/EBITDA, P/S) against peer median. Applies statistical filters to exclude outliers beyond 1.5 times interquartile range. Weights equally across three multiples to derive consensus relative value.

**Forward Target Price**: Uses analyst consensus earnings estimates to project next-twelve-month EPS. Calculates target P/E as function of peer median adjusted for company-specific premium. Bear case is set to current forward P/E. 

**Qualitative Assessment**: Used to incorporate non-financial drivers that are typically discussed in equity research (moat, execution quality, market position). In VERA, the qualitative output is implemented as a scoring rubric that can be mapped into a valuation adjustment. This analysis is done by LLM. 

### Investment Memo Synthesis

The memo generator implements a weighted blended valuation:
- Forward P/E (70%): Primary method given reliance on consensus forecasts
- Qualitative Assessment (20%): To capture market sentiments and competitiveness
- DCF Base Case (10%): Long-term intrinsic value
This weighting prioritises near-term consensus.

## Installation

### Prerequisites

- Python 3.8 or higher
- Alpha Vantage API key (free tier sufficient, already hard-coded to the python files)
- OpenAI API key (optional, required only for memo generation)

### Dependencies

```bash
pip install pandas numpy yfinance openpyxl openai matplotlib requests
```

### Configuration

Update API key in `vera_main.py` line 4:
```python
API_KEY = "YOUR_ALPHA_VANTAGE_KEY"
```

## Usage

### Basic Analysis

Execute complete financial analysis without memo generation:

```bash
python vera_main.py
```

Prompts for ticker symbol and runs all four valuation methods. Generates Excel workbook with comprehensive ratio analysis. Total API consumption: 1 Alpha Vantage call.

### Complete Analysis with Memo

Execute analysis and generate professional research note:

```bash
python run_demo.py
```

Extends basic analysis with automated investment memo generation. Prompts for OpenAI API key at runtime. Total API consumption: 1 Alpha Vantage + 1-2 OpenAI calls.

### Output Files

- `{TICKER}_financialdata.xlsx`
- `{TICKER}_ratio_analysis.xlsx`
- `{TICKER}_Research_Note.html`

### Memo Generation Process

1. Output collection and aggregation from all analytical modules
2. Fundamental scoring via GPT-4 with constrained rubric (10-point scale)
3. Blended valuation calculation through pre-determined percentages
4. LLM-powered thesis generation (1,300+ words) with embedded quantitative metrics
5. Chart generation comparing performance to benchmarks and peers

### Computational Requirements

- Memo generation requires 30-60 seconds due to GPT-4 API latency
- Chart rendering requires matplotlib backend compatibility
- Excel export requires openpyxl write permissions

## File Structure

```
vera_system/
├── vera_main.py                    # Main orchestrator
├── vera_complete_with_memo.py      # Extended system with memo
├── downloader.py                   # Data retrieval module
├── ratios.py                       # Financial ratio analysis
├── dcf.py                          # DCF valuation model
├── relative.py                     # Relative valuation & DDM
├── target.py                       # Forward target price
├── memo_generator.py               # Automated report generation
└── README.md                       # Documentation
```

## References

### Data Sources

- Alpha Vantage API: Financial statement data
- Yahoo Finance: Market data and analyst estimates
- OpenAI GPT-4: Natural language generation for investment thesis

## Notes

### API Rate Limits

Alpha Vantage free tier: 5 calls per minute, 25 calls per day. System architecture enables 25 complete analyses daily through single-call efficiency.

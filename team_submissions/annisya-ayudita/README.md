# Equity Research AI Agent — Fundamental Analysis (Annisya Putri Ayudita)

**Author:** Annisya Putri Ayudita  
**Module:** IFTE0001 - Introduction to Financial Markets 25/26  
**Agent Type:** Fundamental Analysis Agent  
**Target Company:** Alphabet Inc. (GOOGL)

---

## Project Overview

This model represents an AI agent, that is designed to conduct fundamental equity analysis in support of investment decision making process. The agent automates equity research report generation by combining:

1. **Intrinsic Valuation**: FCFF-based Discounted Cash Flow (DCF) model
2. **Relative Valuation**: Forward P/E multiples analysis
3. **LLM-Generated Research Report**: Generating equity research memo using Google Gemini API

The agent ingests 5 years of historical financial data, computes comprehensive financial ratios, performs valuation analysis with Bear/Base/Bull scenarios, and generates a 2-page equity research report with investment recommendations.

**Key Features**:

1. Historical Financial Data (Income Statement, Balance Sheet, Cash Flow)
2. TTM (Trailing Twelve Months) as Base Year
3. Ratio Analysis
4. FCFF Calculation
5. WACC via CAPM
6. DCF and Sensitivity Analysis
7. Market multiples Valuation using Forward P/E
8. Blended Price Target (DCF + multiples weighting)
9. Outputs & Reporting

---

## Repository Structure

```
├── README.md                                               # --> This file
├── requirements.txt                                        # --> Python dependencies
├── run_demo.py                                             # --> Demo script (py)
├── Equity_Research_AI_Agent (Annisya Putri Ayudita).ipynb  # --> Main notebook (ipynb)
└── outputs/
    └── Equity_Research_Report_GOOGL_YYYYMMDD.html  # Generated report
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Google Gemini API Key
- Optional: Jupyter Notebook or JupyterLab 


### Step 1: Clone or Download the Repository
```bash
# If using Git
git clone <repository-url>
cd <repository-folder>

# Or extract the zipped folder
unzip Group_IFTE0001_Fundamental_Annisya.zip
cd Group_IFTE0001_Fundamental_Annisya
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Obtain Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Keep the key ready for input when running the demo

---

## Running the Demo

### Option A: Demo Script 
```bash
python run_demo.py
```
The script will:
1. Prompt user for the Gemini API key
2. Fetch financial data for GOOGL from Yahoo Finance
3. Perform DCF and Multiples valuation
4. Generate an HTML equity research report
5. Save the report to `outputs/` folder

### Option B: Full Edit Notebook
```bash
Equity_Research_AI_Agent.ipynb
```
Run all cells sequentially. The notebook provides detailed explanations and intermediate outputs for each analysis step.

---

## Key Features

### 1. Data Ingestion Pipeline
- Fetches 5 years of annual financial statements (Income Statement, Balance Sheet, Cash Flow)
- Retrieves quarterly data for TTM (Trailing Twelve Months) calculations
- Sources: Yahoo Finance API via `yfinance` library

### 2. Financial Ratio Analysis
Computes financial ratios across categories:
- **Profitability:** Gross Margin, Operating Margin, Net Margin, ROA, ROE, ROIC
- **Liquidity:** Current Ratio, Quick Ratio, Cash Ratio
- **Leverage:** Debt/Equity, Interest Coverage, Net Debt/EBITDA
- **Efficiency:** Asset Turnover, Receivables Turnover, DSO
- **Valuation:** P/E, P/B, P/S, EV/EBITDA, FCF Yield

### 3. DCF Valuation Model
- **FCFF-based:** Free Cash Flow to Firm methodology
- **WACC via CAPM:** Risk-free rate, Market Risk Premium, Beta-adjusted Cost of Equity
- **Scenario Analysis:** Bear (10%), Base (80%), Bull (10%) probability-weighted
- **Sensitivity Analysis:** Terminal Growth vs WACC matrix

### 4. Multiples Valuation
- Industry average Forward P/E as anchor
- Scenario adjustments for Bear/Base/Bull cases
- Probability-weighted target price

### 5. LLM Report Generation
- Uses Google Gemini (gemini-2.5-flash or later)
- Generates equity research memo
- Includes: Business Overview, Investment Thesis, Valuation Summary, Risks
- Output: 2-page HTML report with charts and tables

---

## Output Files

| File | Description |
|------|-------------|
| `Equity_Research_Report_{TICKER}_YYYYMMDD.html` | Professional 2-page Equity Research Report |
| `Equity_Research_AI_Agent_{TICKER}_Valuation_Model.xlsx` | Excel workbook with financial ratios, DCF projections, Multiples, and valuation summary |

---

## Configuration

Key parameters can be adjusted in the notebook or `run_demo.py`:

```python
TICKER = "GOOGL"                    # Stock ticker symbol
FORECAST_YEARS = 5                  # DCF projection period
DCF_WEIGHT = 0.25                   # Weight for DCF in blended target
MULTIPLES_WEIGHT = 0.75             # Weight for Multiples in blended target
UPSIDE_BUY_THRESHOLD = 0.15         # +15% for BUY recommendation
DOWNSIDE_SELL_THRESHOLD = -0.15     # -15% for SELL recommendation
```

---

## Sample Output

**Recommendation Logic:**
- **BUY:** If upside > +15%
- **HOLD:** If upside between -15% and +15%
- **SELL:** If downside > -15%

**Example Output:**
```
Recommendation: HOLD
Current Price: $338.00
Blended Target: $318.83
Upside/Downside: -5.7%
```

---

## Limitations & Disclaimers

1. **Data Dependency:** Relies on Yahoo Finance data availability and accuracy
2. **Assumption Sensitivity:** DCF results are highly sensitive to growth and discount rate assumptions
3. **LLM Limitations:** AI-generated narratives should be reviewed for accuracy
4. **Educational Purpose:** This tool is for educational purposes only and does not constitute investment advice

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Yahoo Finance timeout | Wait and retry; check internet connection |
| Gemini API error | Verify API key is valid and has quota remaining |
| Missing data for ticker | Some tickers may have incomplete financials; try another |

---

## References

- [Yahoo Finance API (yfinance)](https://github.com/ranaroussi/yfinance)
- [Google Gemini API](https://ai.google.dev/)
- Damodaran, A. (2012). *Investment Valuation*
- CFA Institute. (2020). *Equity Asset Valuation*

---

*Last Updated: January 2026*

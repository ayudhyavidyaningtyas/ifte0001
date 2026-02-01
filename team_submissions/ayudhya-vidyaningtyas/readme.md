# VERA — Valuation & Equity Research Assistant

An AI-powered equity research system that automates fundamental analysis, DCF valuation, relative valuation, and institutional-quality report generation for publicly traded companies.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [How to Run](#how-to-run)
6. [Module Breakdown](#module-breakdown)
7. [API Keys](#api-keys)
8. [Output Files](#output-files)
9. [Contact](#contact)

---

## Overview

VERA performs end-to-end equity research in a single run:

1. Downloads financial statements (Income Statement, Balance Sheet, Cash Flow) from **Alpha Vantage**
2. Pulls live market data (price, market cap, shares, beta) from **Yahoo Finance**
3. Calculates financial ratios (profitability, liquidity, leverage, efficiency)
4. Runs a **Discounted Cash Flow (DCF)** valuation with WACC and sensitivity analysis
5. Performs **relative valuation** using peer multiples (P/E, EV/EBITDA, P/S)
6. Generates **12-month and 18-month target prices** using forward P/E peer-relative methodology
7. Produces a professional **HTML research note** with an AI-generated investment memo (via OpenAI GPT-4o)

---

## Architecture

```
vera_main.py          ← Entry point; orchestrates the full pipeline
│
├── downloader.py     ← Alpha Vantage API client + Yahoo Finance market data
├── ratios.py         ← Financial ratio calculations & Excel export
├── dcf.py            ← DCF model (WACC via CAPM, sensitivity tables)
├── relative.py       ← Peer multiples valuation + DDM
├── target.py         ← 12M / 18M target price engine (forward P/E)
├── memo_generator.py ← HTML report builder + OpenAI investment memo
└── run_demo.py       ← Wrapper that calls vera_main then generates the memo
```

**Execution flow:**

```
run_demo.py
  └→ vera_main.main()          # Steps 1–6 (no OpenAI needed)
       └→ downloader            # Alpha Vantage + yfinance
       └→ ratios                # Ratio analysis
       └→ dcf                   # DCF valuation
       └→ relative              # Peer multiples
       └→ target                # Target prices
  └→ memo_generator             # Step 7 (requires OpenAI key, entered at runtime)
```

---

## Prerequisites

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical calculations |
| `requests` | Alpha Vantage API calls |
| `yfinance` | Yahoo Finance market data & peer data |
| `openpyxl` | Excel export |
| `matplotlib` | Performance chart in the report |
| `openai` | GPT-4o memo generation (Step 7 only) |
| `python-dateutil` | Date arithmetic in target price module |

Install all at once:

```bash
pip install pandas numpy requests yfinance openpyxl matplotlib openai python-dateutil
```

---

## Setup

1. **Clone or copy** all `.py` files into the same directory.
2. **Set your Alpha Vantage API key** in `vera_main.py`:
   ```python
   # vera_main.py, line 6
   API_KEY = "YOUR_ALPHA_VANTAGE_KEY"
   ```
   Free key → [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

3. *(Optional)* Have an **OpenAI API key** ready — it is entered interactively at runtime when you choose to generate the memo.

---

## How to Run

### Basic Analysis (no memo)

Runs all four valuation methods and exports Excel outputs. No OpenAI key required.

```bash
python vera_main.py
```

### Complete Analysis with Memo

Runs the full pipeline and generates the professional HTML research note. Requires an OpenAI API key at runtime.

```bash
python run_demo.py
```

You will be prompted to:
1. **Enter a ticker symbol** (e.g., `GOOGL`, `MSFT`, `AAPL`)
2. After analysis completes, **choose whether to generate the memo** (`Y/N`)
3. If yes, **paste your OpenAI API key** when prompted

> **Rate limit note:** Alpha Vantage free-tier allows 5 API calls/minute and 25/day. The downloader includes a 12-second delay between calls to stay within limits.

---

## Module Breakdown

### `downloader.py` — `AlphaVantageDownloader`

Handles all data ingestion.

- Calls Alpha Vantage for Income Statement, Balance Sheet, and Cash Flow (annual + quarterly)
- Calculates **TTM (Trailing Twelve Months)** by summing the four most recent quarters
- Fetches live market data from Yahoo Finance (`impliedSharesOutstanding` is used to handle dual-class share structures like Alphabet's GOOG/GOOGL)
- Exports raw financial data to an Excel workbook

### `ratios.py` — `RatioAnalyzer`

Computes and exports financial ratios across all available periods:

- **Profitability:** Gross Margin, Operating Margin, Net Margin, EBITDA Margin, ROA, ROE
- **Liquidity:** Current Ratio, Quick Ratio
- **Leverage:** Debt-to-Equity, Debt-to-Assets, Interest Coverage
- **Efficiency:** Asset Turnover
- **Market:** P/E, EV/EBITDA, P/S, Dividend Yield

Results are saved to `{TICKER}_ratios.xlsx`.

### `dcf.py` — `DCFModel`

Performs a full Discounted Cash Flow valuation:

- **WACC calculation** using CAPM (risk-free rate fetched live from ^TNX, equity risk premium from Damodaran benchmarks)
- Projects Free Cash Flow for 5 years using historical FCF margins and revenue growth
- Applies a terminal value with the Gordon Growth Model
- Generates a **sensitivity table** varying WACC and terminal growth rate
- Returns the base-case intrinsic value per share

### `relative.py` — Peer Multiples & DDM

Runs a market-relative valuation:

- Fetches trailing P/E, EV/EBITDA, and P/S for **core peers** (MSFT, AAPL, META) and an extended set
- Applies outlier filters before computing medians
- Derives implied share prices from each multiple
- Includes a **Dividend Discount Model (DDM)** section (noted as non-primary for low-payout companies like Alphabet)

### `target.py` — Target Price Engine

Produces **12-month and 18-month target prices**:

- Fetches forward EPS estimates from Yahoo Finance (with fallbacks to growth-rate projections)
- Uses **peer-relative forward P/E** methodology: the target company's forward P/E is benchmarked against peer medians
- Outputs a summary table and sensitivity analysis for both horizons

### `memo_generator.py` — HTML Report Builder

Assembles the final institutional-style research note:

- Generates a **1-year performance chart** (normalised returns vs. S&P 500, NASDAQ, and peers)
- Calls **GPT-4o** to write the qualitative investment thesis, risk factors, and rating rationale
- Combines all valuation outputs (DCF, multiples, target prices) into a professionally formatted **HTML document**
- The output file is saved as `{TICKER}_Research_Note.html`

---

## API Keys

| Key | Where to set | Notes |
|---|---|---|
| **Alpha Vantage** | `vera_main.py` line 6 (`API_KEY`) | Free tier: 25 calls/day. Sufficient for one full analysis. |
| **OpenAI** | Entered at runtime when prompted | Only needed for the memo/report. GPT-4o is used. |

> ⚠️ Do not commit API keys to version control. Consider moving them to environment variables for production use.

---

## Output Files

After a full run you will find:

| File | Generated by | Contents |
|---|---|---|
| `{TICKER}_financialdata.xlsx` | `downloader.py` | Raw Income Statement, Balance Sheet, Cash Flow + market data |
| `{TICKER}_ratios.xlsx` | `ratios.py` | All calculated financial ratios by period |
| `{TICKER}_Research_Note.html` | `memo_generator.py` | The final institutional research report (open in any browser) |

---

## Contact

**Ayudhya Vidyaningtyas**
ayudhya.vidyaningtyas.25@ucl.ac.uk

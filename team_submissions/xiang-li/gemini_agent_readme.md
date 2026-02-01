# Gemini Investment Agent Demo

## Overview

This project demonstrates an **AI-powered investment recommendation agent** using Google's Gemini API. The agent:

- Fetches stock data via `yfinance`.
- Computes **DCF and peer-analysis valuations**.
- Integrates valuations using **adaptive weighting** to generate a final target price.
- Generates a **professional investment memo** using the Gemini model.
- Outputs results in **Markdown, CSV, and JSON** formats.

This demo allows the assessor to **run the agent end-to-end** and verify outputs.

---

## Environment Details

| Component                     | Version / Info                                      |
|--------------------------------|---------------------------------------------------|
| Python                          | 3.12.4 (Anaconda)                                 |
| Operating System                | Windows 11 (64-bit)                                |
| Processor                        | Intel64 Family 6 Model 142 Stepping 12           |
| pandas                          | 2.3.3                                             |
| yfinance                        | 0.2.66                                            |
| google-generativeai             | 0.8.6                                             |

**Current Working Directory:** Where the script is run from (all outputs are saved here).

---

## Setup Instructions

1. **Clone or download** this repository to your local machine.

2. **Create and activate a Python virtual environment** (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. **Install required packages**:

```bash
pip install pandas==2.3.3 yfinance==0.2.66 google-generativeai==0.8.6
```

4. **Set your Gemini API key** in the script:

```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

> Note: Make sure your API key has permissions to call `generate_content` on `models/gemini-2.5-flash`.

---

## Running the Demo

To run the investment agent for a stock (e.g., GOOG):

```bash
python your_agent_script.py
```

### What happens:

1. The script fetches **current stock data**.  
2. Calculates **DCF and peer analysis valuations**.  
3. Integrates these into a **final predicted price**.  
4. Generates an **investment recommendation** and expected return.  
5. Produces a **professional investment memo** using Gemini.  
6. Saves outputs:

- Markdown memo: `GOOG_investment_memo_<timestamp>.md`
- CSV summary: `GOOG_integrated_valuation.csv`
- CSV weighting comparison: `GOOG_weighting_comparison.csv`
- JSON detailed results: `GOOG_analysis_results_<timestamp>.json`

---

## File Structure

```
.
├── your_agent_script.py        # Main agent script
├── README.md                   # This documentation
├── GOOG_investment_memo_*.md   # Generated investment memo
├── GOOG_integrated_valuation.csv
├── GOOG_weighting_comparison.csv
└── GOOG_analysis_results_*.json
```

---

## Notes

- The **final target price** in the memo and JSON output corresponds to the **integrated adaptive valuation**.  
- Ensure your Gemini API key is **valid** and your system can access the internet.  
- You can test **other stocks** by changing the `symbol` variable in the script.  
- All outputs are saved in the **current working directory** unless you modify the paths.

---

## Contact / Support

For
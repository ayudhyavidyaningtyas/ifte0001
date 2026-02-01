# 📊 Vivi's Agentic AI - Equity Research Agent

An end-to-end AI-powered equity research agent that combines real-time financial data with custom DCF valuation and GPT-4 powered narrative generation to produce institutional-quality research reports.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Real-time Financial Data** - Fetches live data from Yahoo Finance API
- **DCF Valuation** - Multi-method discounted cash flow analysis with sensitivity analysis
- **AI-Generated Narrative** - GPT-4 powered investment thesis and risk assessment
- **Professional PDF Output** - Institutional-quality research reports
- **Agentic Workflow** - LangChain-powered autonomous analysis

## 📁 Project Structure

```
vivi-equity-research/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── agent.py              # LangChain agent setup
│   ├── data_fetcher.py       # Financial data fetching
│   ├── valuation.py          # DCF and WACC calculations
│   ├── pdf_generator.py      # PDF report generation
│   ├── visualization.py      # Chart generation
│   └── utils.py              # Helper functions
├── tests/                    # Unit tests (to be added)
├── data/                     # Data storage
├── outputs/                  # Generated PDF reports
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vivi-equity-research.git
   cd vivi-equity-research
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## 🎯 Usage

### Command Line

Run the main script:

```bash
python main.py
```

You'll be prompted to enter a stock ticker (e.g., GOOGL, AAPL, MSFT).

### Python Script

```python
from src import run_agent, generate_equity_report_pdf

# Analyze a stock
ticker = "GOOGL"
query = f"Analyze {ticker} and produce a comprehensive equity research report."
report_text = run_agent(query, verbose=True)

# Generate PDF
pdf_path = generate_equity_report_pdf(
    ticker,
    report_text=report_text,
    output_path=f"outputs/{ticker}_report.pdf"
)
```

### Individual Components

```python
from src import fetch_financial_data, calculate_dcf, calculate_dcf_sensitivity

# Fetch financial data
data = fetch_financial_data("GOOGL")
print(data['current_price'])

# Calculate DCF valuation
dcf = calculate_dcf("GOOGL")
print(f"Fair Value: ${dcf['fair_value_per_share']}")

# Sensitivity analysis
sensitivity = calculate_dcf_sensitivity("GOOGL")
```

## 📊 Output

The agent produces a comprehensive PDF report including:

1. **Executive Summary**
   - Investment rating (BUY/HOLD/SELL)
   - Target price
   - Key metrics

2. **Financial Snapshot**
   - Latest financial metrics
   - 5-year financial history
   - Revenue trend chart

3. **DCF Valuation**
   - Fair value calculation
   - Sensitivity analysis table
   - Key assumptions

4. **AI-Generated Analysis**
   - Business overview
   - Industry positioning
   - Valuation analysis
   - Risk assessment
   - Investment recommendation

## 🔧 Configuration

### Valuation Parameters

Modify in `src/valuation.py`:

```python
def calculate_dcf(
    ticker: str,
    projection_years: int = 7,        # Projection period
    terminal_growth: float = 0.035,   # Terminal growth rate
    discount_rate: Optional[float] = None  # WACC override
)
```

### Agent Parameters

Modify in `src/agent.py`:

```python
def run_agent(
    query: str,
    verbose: bool = True,
    max_iterations: int = 10  # Maximum agent iterations
)
```

## 🧪 Testing

```bash
# Run tests (to be implemented)
pytest tests/
```

## 📝 API Reference

### Core Functions

#### `fetch_financial_data(ticker: str)`
Fetches comprehensive financial data for a stock.

**Returns:** Dictionary containing financial metrics, statements, and ratios.

#### `calculate_dcf(ticker: str, **kwargs)`
Performs DCF valuation analysis.

**Returns:** Dictionary with fair value, upside potential, and rating.

#### `run_agent(query: str, verbose: bool = True)`
Executes the equity research agent.

**Returns:** Markdown-formatted research report.

#### `generate_equity_report_pdf(ticker: str, report_text: str, output_path: str)`
Generates professional PDF report.

**Returns:** Path to generated PDF file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This tool is for **educational purposes only**. The analysis and reports generated are not investment advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for financial data
- [LangChain](https://github.com/langchain-ai/langchain) for agent framework
- [OpenAI](https://openai.com/) for GPT-4 API
- [ReportLab](https://www.reportlab.com/) for PDF generation

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ by Vivi**

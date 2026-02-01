# AI Trading Agent - GOOGL Strategy

## Overview

A quantitative trading system that combines technical indicators with LLM-generated analysis. 
- Automatically downloads market data and computes trading signals
- Backtests strategy with comprehensive performance metrics
- Generates professional trade notes using Google Gemini

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API
Create `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Run
```bash
python run_demo.py
```

## Strategy

**Asset:** GOOGL | **Frequency:** Daily | **Period:** 2015 - Present

**Indicators:** 
- MA(10/30) for trend
- RSI(14) for momentum  
- MACD(12,26,9) for confirmation

**Entry/Exit:**
- **Buy:** MA10 > MA30 AND MACD > Signal
- **Sell:** MA10 < MA30 AND RSI > 55

**Position Sizing:**
- Dynamic allocation based on sigmoid mapping of RSI
- Max exposure: 70%
- Transaction cost: 0.1%

## Output

The pipeline generates:
- `metrics.json` - CAGR, Sharpe Ratio, Max Drawdown, Win Rate
- `trade_note.md` - Professional AI-generated analysis

## File Structure

```
├── agent_strategy.py      # Data ingestion, indicators, signals, backtest, recommendation
├── llm_report.py          # LLM trade note generator
├── run_demo.py            # Main pipeline
├── agent_strategy.ipynb   # Full research notebook
└── requirements.txt       # Python dependencies
```

## Module Functions

**agent_strategy.py:**
- `download_data()` - Fetch market data via yfinance
- `compute_indicators()` - Calculate MA, RSI, MACD
- `generate_dynamic_signals()` - Generate signals + position sizing
- `backtest()` - Run backtest and compute metrics
- `generate_trade_recommendation()` - Current market recommendation

**llm_report.py:**
- `generate_trade_note()` - Create AI-powered analysis

**run_demo.py:**
- Orchestrates: download → indicators → signals → backtest → LLM analysis

## Requirements

```
numpy, pandas, matplotlib, yfinance, ta, python-dotenv, google-generativeai
```

## Disclaimer

Educational and research purposes only. Past performance does not guarantee future results. 
Trading carries substantial risk of loss.


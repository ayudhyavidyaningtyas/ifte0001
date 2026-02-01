# 📊 AI Analyst Agents in Asset Management
---
***Introduction to Financial Markets (IFTE0001)***

Group A

Msc Banking & Digital Finance 

University College London 

February 2026

---

## 📁 Project Overview

This repository contains the work from a group project where each team member developed an AI agent prototype using different fundamental and technical analysis approaches for financial markets. After internal benchmarking, one agent was selected for the final report and demonstration based on alignment with coursework deliverables in terms of data collection and ingestion, decision logic, and reproducibility.

## 📠 Assignment Context

The project involved creating AI-powered agents that leverage Large Language Models (LLMs) and various data sources to provide financial analysis and trading recommendations. Each agent demonstrates different methodologies for analyzing equities and generating investment decisions.

## 🏆 Team Members and Agent Descriptions

### A. Fundamental Analysis

#### Ayudhya V.
**Agent Description:** Model chosen for final demonstration. An AI agent that ingests financial data from AlphaVantage and Yahoo Finance, then generates an equity research report powered by OpenAI LLM with a Buy/Hold/Sell recommendation based on DCF, target price based on forward multiples, and quantified qualitative analysis and explanation of risks, catalyst, and outlook.

#### Annisya Putri Ayudita
**Agent Description:** An AI agent that computes financial ratios, multiples, and DCF valuation (CAPM based, with expected market return proxied by S&P 500). Recommendations are derived from a weighted blend of DCF and forward P/E implied target prices, using Yahoo Finance and Gemini LLM generated report text.

#### Vivi S.
**Agent Description:** An end-to-end AI-powered equity research agent, combining real-time financial data (Yahoo Finance), custom DCF valuation with sensitivity analysis leveraging Langchain using GPT-4 agent-driven narrative generation.

#### Xiang L.
**Agent Description:** Built an AI agent that generates investment recommendations (Buy/Hold/Sell) based on a forecasted stock price. The recommendations are supported by sensitivity analysis, and the system is powered by the Gemini LLM model.

#### Yuelin L.
**Agent Description:** An AI agent that retrieves financial and market data, computes valuation multiples (P/E, P/B, P/S, EV/EBITDA), and generates structured investment memos. The agent separates deterministic valuation calculations from optional LLM-based narrative generation, ensuring transparency, reproducibility.

### B. Technical Analysis

#### Pingfan Q.
**Agent Description:** A technical analyst agent that ingests market data, computes technical indicators, generates trading signals, back-tests performance, and produces LLM-generated timing and risk-overlay trade recommendations supported by risk and performance metrics.

#### Tsz Fung H.
**Agent Description:** A two-compartment trading system, considering market regimes, using Google (OHLC daily data 2015-2025) with the Composite RSI (CRSI) and trend state segmentation. Combined a short-term compartment, focused on volatility and mean reversion with a long-term compartment, trend transitions (60/40), and position sizing based on a sigmoid function. Analysed returns, maximum losses, the Sharpe ratio, and the win rate.

#### Xinyi L.
**Agent Description:** Developed a technical agent that fetches market data, generates trading signals with MA, RSI, and MACD, dynamically sizes positions via a sigmoid function, back-tests performance, and produces LLM-based trade report recommendations.

## ⚙️ Key Features

- **Data Integration:** Agents utilize various data sources including AlphaVantage and Yahoo Finance
- **Valuation Methods:** DCF analysis, multiples-based valuation (P/E, P/B, P/S, EV/EBITDA), technical indicators
- **LLM Integration:** OpenAI GPT-4, Google Gemini, Langchain for narrative generation and analysis
- **Analysis Approaches:** Both fundamental (financial ratios, DCF, qualitative analysis) and technical (RSI, MACD, MA, trend analysis)
- **Output:** Investment recommendations (Buy/Hold/Sell), research reports, trading signals, and risk assessments

## 🖥️ Technologies Used

- **Data Sources:** AlphaVantage, Yahoo Finance
- **LLM Models:** OpenAI GPT-4, Google Gemini
- **Frameworks:** Langchain
- **Analysis Methods:** DCF valuation, Technical indicators, CAPM, Sensitivity analysis

## 🏛️ Repository Structure

Each team member's agent implementation can be found in their respective directories or branches. The final selected agent (Ayudhya V.'s fundamental analysis agent) serves as the primary demonstration model.



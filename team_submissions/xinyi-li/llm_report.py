import os
import json
from datetime import date
from dotenv import load_dotenv
import google.generativeai as genai


def generate_trade_note(backtest_results: dict, strategy_config: dict, 
                        recommendation: dict):
    """
    Generate a professional trade note using an LLM based on
    strategy description, backtest metrics, and current recommendation.

    Parameters
    ----------
    backtest_results : dict
        Dictionary containing 'CAGR', 'Sharpe', 'Max_Drawdown', 'WinRate_Daily'
    strategy_config : dict
        Dictionary containing strategy parameters (ticker, ma_short, ma_long, etc.)
    recommendation : dict
        Dictionary containing current market state and agent recommendation
    """

    # =========================
    # 1. Load API key
    # =========================
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

    genai.configure(api_key=api_key)


    # =========================
    # 2. Construct prompt
    # =========================
    prompt = f"""
You are a professional quantitative trading analyst specializing in technical strategy documentation.

Generate a comprehensive, beautifully formatted 1-2 page professional trade note in Markdown based on the following data:

- Asset: {strategy_config.get('ticker', 'GOOGL')}
- Frequency: Daily
- Indicators: MA({strategy_config.get('ma_short', 10)}/{strategy_config.get('ma_long', 30)}), RSI({strategy_config.get('rsi_window', 14)}), MACD(12,26,9)
- Entry rule: MA{strategy_config.get('ma_short', 10)} > MA{strategy_config.get('ma_long', 30)} and MACD > Signal (trend confirmation)
- Exit rule: MA{strategy_config.get('ma_short', 10)} < MA{strategy_config.get('ma_long', 30)} and RSI cross above 55 (uptrend confirmation)
- Position sizing: Dynamic continuous allocation based on inverse sigmoid RSI mapping (midpoint: RSI 75, steepness: 0.04)
  * Exposure gradually reduced as RSI exceeds 75, reflecting risk management in overbought conditions
- Maximum exposure constraint: Position weights capped at {strategy_config.get('position_cap', 0.7):.0%}
- Position smoothing: EMA(span=2) applied to position weights
- Transaction cost: {strategy_config.get('transaction_cost', 0.001):.1%} per effective trade

Backtest period: January 2015–most recent available trading date  
Report generation date: {date.today().isoformat()}

## BACKTEST RESULTS
- **CAGR (2015–Present):** {backtest_results.get('CAGR', 0):.2%}
- **Sharpe Ratio:** {backtest_results.get('Sharpe', 0):.2f}
- **Maximum Drawdown:** {backtest_results.get('Max_Drawdown', 0):.2%}
- **Daily Win Rate:** {backtest_results.get('WinRate_Daily', 0):.2%}

## CURRENT MARKET STATE
- Market Regime: {recommendation.get("Market Regime", "N/A")}
- Momentum Signal: {recommendation.get("Momentum", "N/A")}
- RSI (14): {recommendation.get("RSI", "N/A")}
- Recommended Action: {recommendation.get("Recommended Action", "N/A")}
- Target Position Weight: {recommendation.get("Target Position Weight", 0):.1%}

## REQUIREMENTS FOR THE REPORT
Please structure the report with the following sections, using clear Markdown formatting:

1. **Executive Summary** (1 paragraph)
   - High-level overview of strategy performance, approach, and current recommendation

2. **Strategy Overview** (2-3 paragraphs)
   - Economic and technical intuition behind the approach
   - Explanation of trend-following + momentum combination
   - Why RSI-based dynamic sizing matters

3. **Strategy Mechanics** (with subsections)
   - **Entry & Exit Rules:** Detail the dual confirmation approach
   - **Dynamic Position Sizing:** Explain sigmoid function and risk adaptation
   - **Risk Management:** Discuss position cap and smoothing benefits

4. **Performance Analysis** 
   - Create a clean table showing Metric | Value | Interpretation including:
     * Initial → Final Capital and Total Return %
     * CAGR, Sharpe Ratio, Max Drawdown, Daily Win Rate
   - Discuss absolute returns (total return %, cumulative gain) vs risk metrics
   - Analyze risk-adjusted returns (Sharpe ratio significance)
   - Examine maximum drawdown in context of position sizing effectiveness
   - Comment on win rate consistency and return distribution
   - Provide interpretation of whether returns justify the risk taken

5. **Key Performance Insights** (with callout boxes using blockquotes)
   - Risk-adjusted return quality
   - Downside protection effectiveness
   - Consistency across market regimes

6. **Current Market Position & Recommendation**
   - Table: Indicator | Reading | Implication
   - Current regime and positioning recommendation

7. **Limitations & Risk Factors**
   - Parameter sensitivity and regime dependence
   - Slippage and execution risks
   - Potential failure modes and improvements

8. **Enhancements & Future Development**
   - Multi-asset correlation and cross-market signals
   - Machine learning for parameter optimization
   - Volatility scaling and dynamic risk adjustment

9. **Conclusion**
   - Overall assessment and suitability for different investor profiles
   - Final recommendation on implementation and monitoring

## FORMATTING REQUIREMENTS
- Use proper Markdown headings (# ## ### hierarchy)
- Include **bold** for key terms and **metrics**
- Use tables for data presentation
- Use `inline code` for formulas or technical details
- Use bullet points and numbered lists for clarity
- Use blockquotes > for important insights
- Use horizontal rules --- to separate major sections
- Ensure 1.5 line spacing feel (add spacing between sections)
- Professional, academic tone with clear explanations
- No markdown code fences, just clean markdown

## TONE & STYLE
- Professional and institutional (like a quantitative research report)
- Balance technical rigor with accessibility
- Use specific numbers and evidence
- Avoid casual language, use formal academic English
- Assume reader has basic financial knowledge
"""

    # =========================
    # 3. Call LLM
    # =========================
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    # =========================
    # 4. Save output
    # =========================
    with open("trade_note.md", "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"✓ Trade note generated: trade_note.md")
    return response.text

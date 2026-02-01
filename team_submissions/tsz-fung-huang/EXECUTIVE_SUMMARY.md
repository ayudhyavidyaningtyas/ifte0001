# Executive Summary: Google Stock Trading Strategy Analysis

## Project Overview
This project develops and backtests a multi-position trading strategy for Google (GOOGL) stock based on Composite RSI (CRSI) indicators and trend analysis. The strategy uses two independent positions with different risk-return profiles to optimize portfolio performance.

## Key Findings

### Strategy Performance (2015-2025)
- **Total Return**: 404.03%
- **Annualized Return**: 17.42%
- **Sharpe Ratio**: 1.055
- **Win Rate**: 71.88%
- **Maximum Drawdown**: 31.26%

### Position Breakdown

**Position 1 (Volatile Trend Strategy) - 60% Allocation**
- Signal: CRSI 40-50 → 40-50 in volatile market conditions
- Parameters: +10% take profit, -10% stop loss, 17-day max hold
- Performance: 361.79% return, 16.40% annualized, 1.036 Sharpe ratio
- Risk: 25.81% maximum drawdown

**Position 2 (Trend Transition Strategy) - 40% Allocation**
- Signal: Downtrend → Volatile trend transition
- Parameters: +80% take profit, -35% stop loss, 600-day max hold (~2.4 years)
- Performance: 467.39% return, 18.81% annualized, 0.792 Sharpe ratio
- Risk: 47.10% maximum drawdown

## Methodology

### Data Source
- **Source**: Yahoo Finance (via yfinance library)
- **Ticker**: GOOGL (Google/Alphabet Inc.)
- **Period**: October 2015 - October 2025
- **Frequency**: Daily OHLCV data

### Technical Indicators
1. **RSI (Relative Strength Index)**: 6 and 12 periods
2. **EMA (Exponential Moving Average)**: 5, 20, 30, 50, 60 periods
3. **Streak Duration**: Consecutive up/down days calculation
4. **Strike RSI**: Custom indicator based on streak duration
5. **Percentage Rank**: 20-day rolling percentile of price changes
6. **CRSI (Composite RSI)**: Weighted average of RSI, Strike RSI, and Percentage Rank
7. **Trend Indicators**: ADX, +DM, -DM, CCI, UOS, EMA crossovers

### Trend Classification
Market conditions are classified into four categories:
- **Uptrend**: Strong upward momentum (ADX > 25, +DM > -DM, EMA short > EMA long)
- **Downtrend**: Strong downward momentum (ADX > 25, -DM > +DM, EMA short < EMA long)
- **Volatile**: High volatility or conflicting signals (ADX < 20 or extreme CCI/UOS)
- **Neutral**: Weak or unclear trend signals

## Key Assumptions

1. **Data Quality**: Historical data from Yahoo Finance is accurate and complete
2. **Execution**: All trades execute at closing prices (no slippage)
3. **Transaction Costs**: Not included in backtest (would reduce returns)
4. **Market Impact**: Large orders do not affect market prices
5. **Liquidity**: Sufficient liquidity for all trade sizes
6. **Reinvestment**: All proceeds immediately reinvested
7. **No Short Selling**: Long positions only
8. **Trend Stability**: Trend classification parameters remain constant throughout the period

## Risk Considerations

1. **Maximum Drawdown**: 31.26% for combined portfolio indicates significant downside risk
2. **Position 2 Risk**: 47.10% maximum drawdown requires strong risk tolerance
3. **Market Regime Changes**: Strategy performance may vary in different market conditions
4. **Overfitting Risk**: Parameters optimized on historical data may not generalize
5. **Transaction Costs**: Real trading would incur fees reducing returns

## Conclusion

The multi-position strategy demonstrates strong risk-adjusted returns with a Sharpe ratio of 1.055 and annualized return of 17.42%. The combination of short-term volatile trend trading (Position 1) and long-term trend transition strategy (Position 2) provides diversification benefits. However, the significant maximum drawdown (31.26%) requires careful risk management and may not be suitable for all investors.


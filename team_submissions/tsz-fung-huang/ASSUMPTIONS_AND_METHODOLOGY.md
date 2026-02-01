# Assumptions and Methodology Documentation

## 1. Data Source and Content

### 1.1 Data Source
- **Provider**: Yahoo Finance
- **Access Method**: yfinance Python library (v0.2.x)
- **Ticker Symbol**: GOOGL (Alphabet Inc. Class A shares)
- **Data Type**: Historical daily OHLCV (Open, High, Low, Close, Volume) data

### 1.2 Data Period
- **Start Date**: October 1, 2015
- **End Date**: October 28, 2025
- **Total Trading Days**: ~2,533 days (approximately 10 years)
- **Frequency**: Daily (business days only, excluding weekends and holidays)

### 1.3 Data Content
Each data point contains:
- **Open**: Opening price for the trading day
- **High**: Highest price during the trading day
- **Low**: Lowest price during the trading day
- **Close**: Closing price for the trading day
- **Volume**: Number of shares traded during the day
- **Adjusted Close**: Automatically adjusted for splits and dividends (used in calculations)

### 1.4 Data Quality Assumptions
1. Yahoo Finance data is accurate and reliable
2. Missing data points are handled via forward/backward fill for prices, zero-fill for volume
3. No data manipulation or filtering beyond standard cleaning procedures
4. All prices are in USD

## 2. Technical Indicators and Calculations

### 2.1 RSI (Relative Strength Index)

**Formula**:
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss
```

**Parameters**:
- RSI6: 6-period RSI
- RSI12: 12-period RSI

**Calculation Method**: Using TA-Lib library's RSI function
```python
RSI = talib.RSI(Close, timeperiod=6)  # or 12
```

### 2.2 EMA (Exponential Moving Average)

**Formula**:
```
EMA_today = (Price_today × α) + (EMA_yesterday × (1 - α))
where α = 2 / (period + 1)
```

**Parameters Used**:
- EMA5: 5-period EMA
- EMA20: 20-period EMA
- EMA30: 30-period EMA
- EMA50: 50-period EMA
- EMA60: 60-period EMA

**Calculation Method**: Using TA-Lib library's EMA function
```python
EMA = talib.EMA(Close, timeperiod=5)  # or 20, 30, 50, 60
```

### 2.3 Streak Duration

**Definition**: Consecutive days of price movement in the same direction

**Calculation Logic**:
```
If Close[t] > Close[t-1]:
    If Streak[t-1] < 0: Streak[t] = 1  # Reversal from down to up
    Else: Streak[t] = Streak[t-1] + 1  # Continuation of up streak
Else if Close[t] < Close[t-1]:
    If Streak[t-1] > 0: Streak[t] = -1  # Reversal from up to down
    Else: Streak[t] = Streak[t-1] - 1   # Continuation of down streak
Else:
    Streak[t] = 0  # No change
```

### 2.4 Strike RSI

**Definition**: Custom indicator based on the ratio of current streak duration to total streak cycle

**Calculation**:
1. Find current streak duration and opposite streak duration
2. Calculate ratio:
   ```
   If current_duration > 0:
       duration_ratio = abs(current_duration) / (abs(opposite_duration) + abs(current_duration)) × 100
   Else if current_duration < 0:
       duration_ratio = abs(opposite_duration) / (abs(opposite_duration) + abs(current_duration)) × 100
   Else:
       duration_ratio = 50
   ```
3. Strike RSI = duration_ratio

### 2.5 Percentage Rank

**Definition**: Percentile rank of current day's price change within a rolling window

**Formula**:
```
percentage_change = abs(Close[t] - Close[t-1]) / Close[t-1]
rank = count of days in window where percentage_change < current_percentage_change
percentage_rank = (rank / window_size) × 100
```

**Parameters**:
- Window size: 20 days
- Default value: 50 (if insufficient data)

### 2.6 CRSI (Composite RSI)

**Definition**: Weighted composite indicator combining RSI, Strike RSI, and Percentage Rank

**Formula**:
```
CRSI = (3 × RSI6 + Strike_RSI + Percentage_Rank) / 5
```

**Rationale**: 
- RSI6 weighted 3x (60% weight) due to its proven effectiveness
- Strike RSI and Percentage Rank each weighted 1x (20% weight each)
- Normalized to 0-100 scale

### 2.7 Trend Classification Indicators

#### 2.7.1 ADX (Average Directional Index)
- **Period**: 21 days
- **Smoothing Window**: 5 days
- **Purpose**: Measures trend strength
- **Thresholds**:
  - Strong trend: ADX > 25
  - Weak trend: 20 < ADX ≤ 25
  - No trend: ADX ≤ 20

#### 2.7.2 +DM and -DM (Directional Movement)
- **Period**: 21 days
- **Purpose**: Identifies trend direction
- **Logic**: +DM > -DM indicates uptrend, -DM > +DM indicates downtrend

#### 2.7.3 CCI (Commodity Channel Index)
- **Period**: 20 days
- **Purpose**: Identifies overbought/oversold conditions
- **Volatile Signal**: |CCI| > 100

#### 2.7.4 UOS (Ultimate Oscillator)
- **Periods**: 14, 28, 56 days
- **Purpose**: Momentum indicator
- **Volatile Signal**: |UOS| > 50

#### 2.7.5 EMA Crossover
- **Short EMA**: 30 days
- **Long EMA**: 60 days
- **Purpose**: Confirms trend direction
- **Uptrend**: Short EMA > Long EMA
- **Downtrend**: Short EMA < Long EMA

### 2.8 Trend Classification Rules

**Uptrend**:
```
(ADX_smooth > 25 AND +DM > -DM AND EMA_short > EMA_long AND Price > EMA_short)
OR
(ADX_smooth > 20 AND ADX_smooth ≤ 25 AND +DM > -DM AND EMA_short > EMA_long)
```

**Downtrend**:
```
(ADX_smooth > 25 AND -DM > +DM AND EMA_short < EMA_long AND Price < EMA_short)
OR
(ADX_smooth > 20 AND ADX_smooth ≤ 25 AND -DM > +DM AND EMA_short < EMA_long)
```

**Volatile**:
```
ADX_smooth ≤ 20
OR
|CCI| > 100
OR
|UOS| > 50
OR
Conflicting signals (e.g., +DM > -DM but EMA_short < EMA_long)
```

**Neutral**: All other conditions

## 3. Trading Strategy Rules

### 3.1 Position 1: Volatile Trend Strategy

**Entry Signal**:
- CRSI_Prev in [40, 50) AND CRSI in [40, 50)
- Current trend = 'volatile'

**Exit Conditions**:
1. Take Profit: Price reaches +10% from entry
2. Stop Loss: Price drops to -10% from entry
3. Time-based: Hold for 17 days maximum

**Capital Allocation**: 60% of total portfolio

### 3.2 Position 2: Trend Transition Strategy

**Entry Signal**:
- Trend changes from 'downtrend' to 'volatile'

**Exit Conditions**:
1. Take Profit: Price reaches +80% from entry
2. Stop Loss: Price drops to -35% from entry
3. Time-based: Hold for 600 trading days (~2.4 years) maximum

**Capital Allocation**: 40% of total portfolio

## 4. Performance Metrics Definitions

### 4.1 Sharpe Ratio

**Formula**:
```
Sharpe Ratio = (Mean Daily Return / Std Dev of Daily Returns) × √252
```

**Interpretation**:
- > 1.0: Good risk-adjusted returns
- > 2.0: Excellent risk-adjusted returns
- < 1.0: Poor risk-adjusted returns

**Assumptions**:
- Risk-free rate = 0% (for simplicity)
- 252 trading days per year

### 4.2 Win Rate

**Formula**:
```
Win Rate = (Number of Profitable Trades / Total Number of Trades) × 100%
```

**Definition**: Percentage of trades that result in positive returns

### 4.3 Annualized Return

**Formula**:
```
Annualized Return = ((1 + Total Return)^(1/Years) - 1) × 100%
```

**Where**:
- Total Return = (Final Value - Initial Value) / Initial Value
- Years = (End Date - Start Date) / 365.25

### 4.4 Maximum Drawdown

**Formula**:
```
Maximum Drawdown = max((Peak Value - Trough Value) / Peak Value) × 100%
```

**Calculation**:
1. Calculate running maximum of portfolio value
2. Calculate drawdown at each point: (Current Value - Running Max) / Running Max
3. Maximum drawdown = minimum (most negative) drawdown value

**Interpretation**: Largest peak-to-trough decline during the period

## 5. Key Assumptions

### 5.1 Trading Assumptions
1. **Execution**: All trades execute at closing prices (no intraday trading)
2. **Slippage**: Zero slippage assumed (real trading would have execution delays)
3. **Transaction Costs**: Not included (would reduce returns by ~0.1-0.5% per trade)
4. **Market Impact**: Large orders do not affect market prices
5. **Liquidity**: Unlimited liquidity at closing prices

### 5.2 Data Assumptions
1. **Data Accuracy**: Yahoo Finance data is accurate and complete
2. **Missing Data**: Handled via forward/backward fill (prices) or zero-fill (volume)
3. **Adjustments**: Automatic adjustment for splits and dividends via yfinance
4. **Time Zone**: All data in market timezone (EST/EDT)

### 5.3 Strategy Assumptions
1. **Reinvestment**: All proceeds immediately reinvested (no cash drag)
2. **No Short Selling**: Long positions only
3. **Position Sizing**: Fixed percentage allocation (no dynamic sizing)
4. **Trend Stability**: Trend classification parameters constant throughout period
5. **Signal Timing**: Signals generated at end of trading day, executed next day

### 5.4 Market Assumptions
1. **Market Efficiency**: Semi-strong form (technical analysis can generate alpha)
2. **Regime Stability**: Historical patterns will continue (may not hold in future)
3. **No Black Swans**: Extreme market events not explicitly modeled
4. **Normal Market Conditions**: Strategy designed for normal volatility ranges

## 6. Limitations and Caveats

1. **Overfitting Risk**: Parameters optimized on historical data may not generalize
2. **Look-Ahead Bias**: No future information used, but data cleaning may introduce minor bias
3. **Survivorship Bias**: Analysis limited to GOOGL (surviving company)
4. **Transaction Costs**: Real trading costs would reduce returns
5. **Tax Implications**: Not considered (would affect after-tax returns)
6. **Market Regime Changes**: Strategy may underperform in different market conditions
7. **Sample Size**: Limited to 10 years of data (may not capture all market cycles)

## 7. Code Structure

### 7.1 Main Files
- **backtest_multi_position.py**: Main backtesting script with strategy implementation
- **statistics.py**: Statistical analysis and visualization of CRSI patterns

### 7.2 Data Files
- **googl_processed_data.parquet**: Processed data with all technical indicators
- **googl_ohlcv.parquet**: Raw OHLCV data (optional, can be regenerated)

### 7.3 Output Files
- **backtest_two_position_comparison.png**: Visualization of strategy performance
- **statistics_*.png**: Various statistical analysis charts


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from ta.momentum import RSIIndicator
from ta.trend import MACD

warnings.filterwarnings('ignore')


# =========================
# Data Ingestion & Validation
# =========================
def download_data(ticker="GOOGL", start="2015-01-01", end=None, min_data_points=250):
    """
    Download and validate stock data with enhanced error handling.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'GOOGL', 'AAPL')
    - start: Start date (YYYY-MM-DD format)
    - end: End date (None = today)
    - min_data_points: Minimum required data points for analysis
    
    Returns:
    - DataFrame with validated OHLCV data
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} from {start} to {end} ...")
    
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Validate data exists
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        if len(df) < min_data_points:
            raise ValueError(f"Insufficient data: {len(df)} points < minimum {min_data_points}")
        
        df.index = pd.to_datetime(df.index)
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        print(f"✓ Successfully downloaded {len(df)} trading days of data")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error downloading data: {str(e)}")
        raise


import pandas as pd

def preprocess_data(df, ticker="GOOGL"):
    """
    Preprocess DataFrame: handle MultiIndex columns and select ticker.
    """
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if ticker in lvl1:
            df = df.xs(ticker, axis=1, level=1)
        elif ticker in lvl0:
            df = df.xs(ticker, axis=1, level=0)
        else:
            # fallback: flatten MultiIndex
            df.columns = [f"{l0}_{l1}" for l0, l1 in df.columns]
    # Ensure columns are stripped strings
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[preprocess_data] Using columns: {df.columns.tolist()}")
    return df


def validate_data_quality(df, tolerance=0.05):
    """
    Perform comprehensive data quality validation.
    
    Parameters:
    - df: Input DataFrame
    - tolerance: Maximum allowed percentage of missing values (5% default)
    
    Returns:
    - Cleaned DataFrame with validation report
    """
    print("\n=== Data Quality Report ===")

    # --- Step 1: Missing values ---
    missing_pct = df.isnull().sum() / len(df) * 100
    if (missing_pct > tolerance).any():
        print(f"⚠ Missing values detected:\n{missing_pct[missing_pct > tolerance]}")
    else:
        print(f"✓ No significant missing values (threshold: {tolerance}%)")

    df = df.ffill().bfill()

    # --- Step 2: Invalid prices ---
    if not all(col in df.columns for col in ['Open', 'Close']):
        raise ValueError("DataFrame must have 'Open' and 'Close' columns")

    invalid_mask = df[['Open', 'Close']].le(0).any(axis=1)
    if invalid_mask.any():
        print(f"⚠ Invalid prices detected in {invalid_mask.sum()} rows")
        df = df[~invalid_mask]

    # --- Step 3: Detect outliers using IQR ---
    daily_return = df['Close'].pct_change()

    # Ensure daily_return is Series
    if isinstance(daily_return, pd.DataFrame):
        daily_return = daily_return.iloc[:, 0]

    Q1 = daily_return.quantile(0.25)
    Q3 = daily_return.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = 3 * IQR

    outliers = ((daily_return < Q1 - outlier_threshold) | 
                (daily_return > Q3 + outlier_threshold))

    # Ensure outliers is Series
    if isinstance(outliers, pd.DataFrame):
        outliers = outliers.iloc[:, 0]

    outliers = outliers.fillna(False)

    if outliers.any():
        print(f"ℹ Detected {outliers.sum()} potential outlier days (return > 3×IQR)")
        df = df.copy()
        df['Is_Outlier'] = outliers.reindex(df.index, fill_value=False)

    # --- Step 4: Data consistency ---
    def to_series(col):
        if isinstance(col, pd.DataFrame):
            return col.iloc[:, 0]
        return col

    high = to_series(df['High'])
    low = to_series(df['Low'])
    close = to_series(df['Close'])

    high_low_valid = high >= low
    close_range_valid = (close >= low) & (close <= high)

    # Debug print
    print("\n--- Debug: Data Consistency Checks ---")
    print(f"type(high_low_valid): {type(high_low_valid)}")
    print(f"type(close_range_valid): {type(close_range_valid)}")
    print(f"high_low_valid.all(): {high_low_valid.all()}")
    print(f"close_range_valid.all(): {close_range_valid.all()}")
    print("--- End Debug ---\n")

    # Keep only valid rows
    valid_mask = high_low_valid & close_range_valid
    df = df[valid_mask]

    # Clean temporary columns
    df = df.drop(columns=['Is_Outlier'], errors='ignore')

    print(f"✓ Data validation complete. Final records: {len(df)}")
    return df



# =========================
# Indicator Computation
# =========================
def compute_indicators(df, ma_short=10, ma_long=30, rsi_window=14, 
                       macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Compute technical indicators with adaptive parameters and missing value handling.
    
    Parameters:
    - df: OHLCV DataFrame
    - ma_short, ma_long: Moving average windows
    - rsi_window, macd_fast, macd_slow, macd_signal: Indicator parameters
    
    Returns:
    - DataFrame with computed indicators
    """
    df = df.copy()
    
    # === Moving Averages ===
    try:
        df['MA10'] = df['Close'].rolling(window=ma_short, min_periods=1).mean()
        df['MA30'] = df['Close'].rolling(window=ma_long, min_periods=1).mean()
        print(f"✓ Moving averages computed: MA{ma_short}, MA{ma_long}")
    except Exception as e:
        print(f"✗ Error computing MAs: {e}")
        df['MA10'] = np.nan
        df['MA30'] = np.nan
    
    # === RSI ===
    try:
        if len(df) >= rsi_window:
            rsi_indicator = RSIIndicator(df['Close'].squeeze(), window=rsi_window)
            df['RSI'] = rsi_indicator.rsi()
            df['RSI'] = df['RSI'].bfill()
            print(f"✓ RSI computed: window={rsi_window}")
        else:
            print(f"⚠ Insufficient data for RSI")
            df['RSI'] = np.nan
    except Exception as e:
        print(f"✗ Error computing RSI: {e}")
        df['RSI'] = np.nan
    
    # === MACD ===
    try:
        if len(df) >= macd_slow:
            macd_obj = MACD(df['Close'].squeeze(), 
                           window_slow=macd_slow, 
                           window_fast=macd_fast, 
                           window_sign=macd_signal)
            df["MACD"] = macd_obj.macd()
            df["MACD_Signal"] = macd_obj.macd_signal()
            df["MACD_Hist"] = macd_obj.macd_diff()
            
            df[["MACD", "MACD_Signal", "MACD_Hist"]] = \
                df[["MACD", "MACD_Signal", "MACD_Hist"]].bfill()
            
            print(f"✓ MACD computed: fast={macd_fast}, slow={macd_slow}, signal={macd_signal}")
        else:
            print(f"⚠ Insufficient data for MACD")
            df["MACD"] = np.nan
            df["MACD_Signal"] = np.nan
            df["MACD_Hist"] = np.nan
    except Exception as e:
        print(f"✗ Error computing MACD: {e}")
        df["MACD"] = np.nan
        df["MACD_Signal"] = np.nan
        df["MACD_Hist"] = np.nan
    
    # === Data integrity check ===
    null_counts = df[['MA10', 'MA30', 'RSI', 'MACD']].isnull().sum()
    if (null_counts > len(df) * 0.3).any():
        print("⚠ Warning: Significant NaN values in indicators")
    
    print(f"✓ Indicator computation complete")
    return df


# =========================
# Position Sizing
# =========================
def sigmoid_weight(rsi, mid=75, steepness=0.04):
    """
    Maps RSI values to position weights using a sigmoid function.
    RSI ↑ → weight ↓
    """
    if pd.isna(rsi):
        return 0.5
    try:
        return 1 / (1 + np.exp(steepness * (rsi - mid)))
    except (OverflowError, RuntimeWarning):
        return 0.5


def generate_dynamic_signals(df, position_cap=0.7, ma10='MA10', ma30='MA30',
                             rsi_mid=75, sigmoid_steepness=0.04, smoothing_span=2,
                             warmup_periods=30, sell_reduction=0.1):
    """
    Trend-following strategy with TRUE RSI-sigmoid-based position sizing.
    
    Parameters:
    - sell_reduction: When sell signal triggered, position reduced to 10% before gradual exit
    """  
    df = df.copy()

    # === Validation ===
    required_cols = [ma10, ma30, 'RSI', 'MACD', 'MACD_Signal']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # === Fill missing values ===
    df[ma10] = df[ma10].ffill()
    df[ma30] = df[ma30].ffill()
    df['RSI'] = df['RSI'].ffill()
    df['MACD'] = df['MACD'].ffill()
    df['MACD_Signal'] = df['MACD_Signal'].ffill()

    # === Signal conditions ===
    buy_condition = (df[ma10] > df[ma30]) & (df['MACD'] > df['MACD_Signal'])
    rsi_trend_cross = (df["RSI"] > 55) & (df["RSI"].shift(1) <= 55)
    sell_condition = (df[ma10] < df[ma30]) & rsi_trend_cross
    
    buy_signal = buy_condition.shift(1).fillna(False)
    sell_signal = sell_condition.shift(1).fillna(False)

    # === Position state (trend direction only) ===
    position_state = pd.Series(np.nan, index=df.index)
    position_state[buy_signal] = 1.0
    position_state[sell_signal] = sell_reduction
    position_state = position_state.ffill().fillna(0.0)

    # === TRUE RSI sigmoid position sizing ===
    rsi_weights = df['RSI'].apply(
        lambda x: sigmoid_weight(x, mid=rsi_mid, steepness=sigmoid_steepness)
    )

    df['Position'] = position_state * rsi_weights

    # === EMA smoothing ===
    df['Position'] = df['Position'].ewm(span=smoothing_span, adjust=False).mean()

    # === Only cap upper bound ===
    df['Position'] = df['Position'].clip(upper=position_cap)

    # === Warmup period ===
    df.loc[:df.index[warmup_periods], 'Position'] = 0.0

    # === Reporting ===
    valid_positions = df[df.index >= df.index[warmup_periods]]['Position']
    print("✓ Signal generation complete:")
    print(f"  Buy signals : {buy_signal.sum()}")
    print(f"  Sell signals: {sell_signal.sum()}")
    print(f"  Avg position: {valid_positions.mean():.2%}")
    print(f"  Max position: {valid_positions.max():.2%}")
    print(f"  Min position: {valid_positions.min():.2%}")

    return df


# =========================
# Backtest Engine
# =========================
def backtest(df, cost=0.001, initial_capital=1_000_000, trade_threshold=0.1):
    """
    Run backtest with transaction costs and performance metrics.
    """
    df = df.copy()
    
    # === Validate required columns ===
    required_cols = ['Close', 'Position']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for backtest: {missing_cols}")

    # === Daily returns ===
    df["Return"] = df["Close"].pct_change().fillna(0)

    # === Position changes & transaction costs ===
    df["Trade_Change"] = df["Position"].diff().abs().fillna(0)

    df["Effective_Trade"] = np.where(df["Trade_Change"] > trade_threshold,
                                     df["Trade_Change"], 0)
    df["Cost"] = df["Effective_Trade"] * cost

    # === Strategy returns ===
    df["Strategy_Return"] = (df["Position"].shift(1) * df["Return"] - df["Cost"]).fillna(0)

    # === Equity curve ===
    df["Equity"] = (1 + df["Strategy_Return"]).cumprod() * initial_capital

    # === Performance metrics ===
    days = max((df.index[-1] - df.index[0]).days, 1)
    years = days / 365
    start_value = df["Equity"].iloc[0]
    end_value = df["Equity"].iloc[-1]

    CAGR = (end_value / start_value) ** (1 / years) - 1

    daily_mean = df["Strategy_Return"].mean()
    daily_std = df["Strategy_Return"].std()
    Sharpe = np.sqrt(252) * (daily_mean / daily_std) if daily_std != 0 else 0
   
    df["Drawdown"] = df["Equity"] / df["Equity"].cummax() - 1
    Max_Drawdown = df["Drawdown"].min()

    WinRate_Daily = (df["Strategy_Return"] > 0).sum() / \
                    (df["Strategy_Return"] != 0).sum() if (df["Strategy_Return"] != 0).sum() > 0 else np.nan

    metrics = {
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "Max_Drawdown": Max_Drawdown,
        "WinRate_Daily": WinRate_Daily
    }

    print(f"CAGR: {CAGR:.2%}, Sharpe: {Sharpe:.2f}, "
          f"Max DD: {Max_Drawdown:.2%}, WinRate(D): {WinRate_Daily:.2%}")
    
    # Keep Effective_Trade and Drawdown for visualization
    temp_cols = ['Trade_Change', 'Cost', 'Return']
    df_clean = df.drop(columns=temp_cols, errors='ignore')

    return metrics, df_clean


# =========================
# Trade Recommendation
# =========================
def generate_trade_recommendation(df):
    """
    Generate current market recommendation based on latest data.
    Ensures all outputs are Python scalars to avoid Series ambiguity errors.
    """
    latest = df.dropna().iloc[-1]

    # === Trend ===
    ma10 = latest["MA10"].item() if hasattr(latest["MA10"], "item") else float(latest["MA10"])
    ma30 = latest["MA30"].item() if hasattr(latest["MA30"], "item") else float(latest["MA30"])
    trend = "Bullish" if ma10 > ma30 else "Bearish"

    # === Momentum (MACD) ===
    macd = latest["MACD"].item() if hasattr(latest["MACD"], "item") else float(latest["MACD"])
    macd_signal = latest["MACD_Signal"].item() if hasattr(latest["MACD_Signal"], "item") else float(latest["MACD_Signal"])
    momentum = "Positive" if macd > macd_signal else "Negative"

    # === Position-based action ===
    pos = latest["Position"].item() if hasattr(latest["Position"], "item") else float(latest["Position"])
    if pos >= 0.5:
        action = "Maintain long exposure"
    elif pos > 0.1:
        action = "Hold a reduced long position"
    else:
        action = "Stay out of the market"

    # === RSI ===
    rsi = latest["RSI"].item() if hasattr(latest["RSI"], "item") else float(latest["RSI"])

    # === Build recommendation dict with pure Python scalars ===
    recommendation = {
        "Market Regime": trend,
        "Momentum": momentum,
        "RSI": round(rsi, 2),
        "Recommended Action": action,
        "Target Position Weight": round(pos, 2)
    }

    return recommendation



# =========================
# Visualization
# =========================
def plot_equity_and_position(df):
    """
    Plot equity curve and dynamic position over time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(df["Equity"], label="Equity Curve", color="blue", linewidth=2)
    ax1.set_ylabel("Portfolio Value", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(df["Position"], label="Position", color="orange", alpha=0.6, linewidth=1.8)
    ax2.set_ylabel("Position Weight", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Equity Curve and Dynamic Position Sizing")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
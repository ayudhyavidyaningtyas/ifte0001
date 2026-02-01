"""
run_demo.py

End-to-end demo script for the AI trading agent.
This script:
1. Downloads market data
2. Validates data quality
3. Computes indicators and trading signals
4. Runs backtest and reports performance metrics
5. Generates current market recommendation
6. Generates a comprehensive LLM-based trade report
"""

import json
from agent_strategy import (
    download_data,
    validate_data_quality,
    compute_indicators,
    generate_dynamic_signals,
    backtest,
    generate_trade_recommendation,
)
from llm_report import generate_trade_note


def main():
    print("===================================")
    print("Running AI Trading Agent Demo")
    print("===================================")

    # =========================
    # 1. Data Ingestion
    # =========================
    print("Step 1: Downloading market data...")
    df = download_data(
        ticker="GOOGL",
        start="2015-01-01",
        min_data_points=250
    )

    # =========================
    # 2. Data Validation
    # =========================
    print("\nStep 2: Validating data quality...")
    df = validate_data_quality(df, tolerance=0.05)

    # =========================
    # 3. Indicator Computation
    # =========================
    print("\nStep 3: Computing technical indicators...")
    df = compute_indicators(
        df,
        ma_short=10,
        ma_long=30,
        rsi_window=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )

    # =========================
    # 4. Signal Generation & Position Sizing
    # =========================
    print("\nStep 4: Generating trading signals with dynamic position sizing...")
    df = generate_dynamic_signals(
        df,
        position_cap=0.7,
        ma10='MA10',
        ma30='MA30',
        rsi_mid=75,
        sigmoid_steepness=0.04,
        smoothing_span=2,
        warmup_periods=30,
        sell_reduction=0.1
    )

    # =========================
    # 5. Backtest Execution
    # =========================
    print("\nStep 5: Running backtest analysis...")
    metrics, bt_df = backtest(
        df,
        cost=0.001,
        initial_capital=1_000_000,
        trade_threshold=0.1
    )

    print("\n" + "="*40)
    print("BACKTEST PERFORMANCE METRICS")
    print("="*40)
    for k, v in metrics.items():
        if k == "CAGR" or k == "Max_Drawdown" or k == "WinRate_Daily":
            print(f"{k:20}: {v:.2%}")
        else:
            print(f"{k:20}: {v:.4f}")

    # Save metrics for reproducibility
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\n✓ Metrics saved to metrics.json")

    # =========================
    # 6. Trade Recommendation
    # =========================
    print("\nStep 6: Generating market recommendation...")
    recommendation = generate_trade_recommendation(bt_df)

    print("\n" + "="*40)
    print("CURRENT MARKET RECOMMENDATION")
    print("="*40)
    for k, v in recommendation.items():
        print(f"{k:25}: {v}")

    # =========================
    # 7. Strategy Configuration (for LLM)
    # =========================
    strategy_config = {
        'ticker': 'GOOGL',
        'ma_short': 10,
        'ma_long': 30,
        'rsi_window': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_mid': 75,
        'sigmoid_steepness': 0.04,
        'position_cap': 0.7,
        'smoothing_span': 2,
        'sell_reduction': 0.1,
        'transaction_cost': 0.001
    }

    # =========================
    # 8. LLM-Generated Trade Report
    # =========================
    print("\nStep 7: Generating comprehensive trade report...")
    
    trade_report = generate_trade_note(
        backtest_results=metrics,
        strategy_config=strategy_config,
        recommendation=recommendation
    )
    
    # Save the report
    with open("trade_note.md", "w", encoding="utf-8") as f:
        f.write(trade_report)
    
    print("✓ Trade report generated and saved to trade_note.md")

    print("\n" + "="*40)
    print("✓ Demo completed successfully")
    print("="*40)


if __name__ == "__main__":
    main()


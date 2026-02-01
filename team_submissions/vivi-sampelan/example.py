"""
Example usage of Vivi's Equity Research Agent.
Run this script to see a quick demo.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: Please set OPENAI_API_KEY in .env file")
    exit(1)

from src import (
    fetch_financial_data,
    calculate_dcf,
    calculate_dcf_sensitivity,
    run_agent,
    generate_equity_report_pdf
)

def demo_financial_data():
    """Demo: Fetch financial data."""
    print("=" * 60)
    print("DEMO 1: Fetching Financial Data")
    print("=" * 60)
    
    ticker = "GOOGL"
    data = fetch_financial_data(ticker)
    
    print(f"\n📊 {data['company_name']} ({data['ticker']})")
    print(f"Sector: {data['sector']}")
    print(f"Industry: {data['industry']}")
    print(f"\nCurrent Price: ${data['current_price']}")
    print(f"Market Cap: ${data['market_cap_billions']}B")
    print(f"P/E Ratio: {data['pe_ratio']}x")
    print(f"\nLatest Revenue: ${data['latest_revenue']}B")
    print(f"Latest FCF: ${data['latest_fcf']}B")
    print(f"ROE: {data['latest_roe']}%")


def demo_dcf_valuation():
    """Demo: DCF valuation."""
    print("\n" + "=" * 60)
    print("DEMO 2: DCF Valuation")
    print("=" * 60)
    
    ticker = "GOOGL"
    dcf = calculate_dcf(ticker)
    
    print(f"\n💰 DCF Analysis for {dcf['ticker']}")
    print(f"Current Price: ${dcf['current_price']}")
    print(f"Fair Value: ${dcf['fair_value_per_share']}")
    print(f"Upside: {dcf['upside_pct']:+.1f}%")
    print(f"Rating: {dcf['rating']}")
    print(f"\nDiscount Rate (WACC): {dcf['discount_rate']*100:.2f}%")
    print(f"Terminal Growth: {dcf['terminal_growth']*100:.2f}%")


def demo_sensitivity():
    """Demo: Sensitivity analysis."""
    print("\n" + "=" * 60)
    print("DEMO 3: Sensitivity Analysis")
    print("=" * 60)
    
    ticker = "GOOGL"
    sensitivity = calculate_dcf_sensitivity(ticker)
    
    print(f"\n📊 DCF Sensitivity Matrix for {ticker}")
    print("\nTerminal Growth \\ WACC")
    
    # Print header
    wacc_values = list(next(iter(sensitivity.values())).keys())
    print(f"{'':>8}", end="")
    for w in wacc_values:
        print(f"{w:>10.1f}%", end="")
    print()
    
    # Print rows
    for g, row in sensitivity.items():
        print(f"{g:>7.1f}%", end="")
        for w in wacc_values:
            fv = row[w]
            print(f"${fv:>9.0f}" if fv else f"{'N/A':>10}", end="")
        print()


def demo_full_analysis():
    """Demo: Full agent analysis."""
    print("\n" + "=" * 60)
    print("DEMO 4: Full AI Agent Analysis")
    print("=" * 60)
    
    ticker = "GOOGL"
    query = f"Analyze {ticker} and produce a comprehensive equity research report."
    
    print(f"\n🤖 Running AI agent for {ticker}...\n")
    report_text = run_agent(query, verbose=True)
    
    print("\n" + "=" * 60)
    print("GENERATED REPORT (First 500 chars)")
    print("=" * 60)
    print(report_text[:500] + "...\n")
    
    # Generate PDF
    print("📄 Generating PDF report...")
    pdf_path = generate_equity_report_pdf(
        ticker,
        report_text=report_text,
        output_path=f"outputs/{ticker}_demo_report.pdf"
    )
    
    print(f"✅ Report saved to: {pdf_path}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🚀 VIVI'S EQUITY RESEARCH AGENT - DEMO")
    print("=" * 60)
    
    try:
        # Run demos
        demo_financial_data()
        demo_dcf_valuation()
        demo_sensitivity()
        
        # Ask user if they want full analysis
        print("\n" + "=" * 60)
        response = input("\nRun full AI analysis with PDF generation? (y/n): ")
        
        if response.lower() == 'y':
            demo_full_analysis()
        else:
            print("\n✅ Demo complete! Run main.py for full analysis.")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

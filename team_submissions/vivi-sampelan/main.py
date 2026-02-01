"""
Main entry point for Vivi's Equity Research Agent.
"""

import os
from dotenv import load_dotenv
from src.agent import run_agent
from src.pdf_generator import generate_equity_report_pdf


def main():
    """Run equity research analysis."""
    # Load environment variables
    load_dotenv()
    
    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return
    
    # Get ticker from user
    ticker = input("Enter stock ticker (e.g., GOOGL): ").strip().upper()
    
    if not ticker:
        print("❌ Error: No ticker provided")
        return
    
    print(f"\n🔍 Analyzing {ticker}...\n")
    
    # Run agent analysis
    query = f"Analyze {ticker} and produce a comprehensive equity research report."
    report_text = run_agent(query, verbose=True)
    
    # Generate PDF
    print(f"\n📄 Generating PDF report...\n")
    pdf_path = generate_equity_report_pdf(
        ticker,
        report_text=report_text,
        output_path=f"outputs/{ticker}_equity_report.pdf"
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Report saved to: {pdf_path}")


if __name__ == "__main__":
    main()

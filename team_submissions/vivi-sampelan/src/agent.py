"""
LangChain agent for equity research.
"""

from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .data_fetcher import fetch_financial_data_cached, get_price_history
from .valuation import calculate_dcf
from .pdf_generator import generate_equity_report_pdf
from .utils import safe_json_dumps, format_percent


# Define LangChain tools
@tool
def get_financial_data(ticker: str) -> str:
    """Fetch financial data for a stock."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        result = fetch_financial_data_cached(ticker.upper(), today)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        summary = f"""
📊 {result['company_name']} ({result['ticker']})
Price: ${result['current_price']} | Market Cap: ${result['market_cap_billions']}B
P/E: {result['pe_ratio']}x | EV/EBIT: {result['ev_ebit']}x
Revenue: ${result['latest_revenue']}B | FCF: ${result['latest_fcf']}B
ROE: {format_percent(result['latest_roe'])}
"""
        return summary
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool  
def get_stock_price_history(ticker: str, period: str = "1y") -> str:
    """Get historical price data and technical indicators."""
    result = get_price_history(ticker, period)
    return safe_json_dumps(result, indent=2)


@tool
def calculate_valuation(ticker: str) -> str:
    """Calculate DCF valuation."""
    try:
        result = calculate_dcf(ticker.upper())
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        summary = f"""
💰 DCF VALUATION: {result['ticker']}
Fair Value: ${result['fair_value_per_share']}
Current Price: ${result['current_price']}
Upside: {result['upside_pct']:+.1f}%
RATING: {result['rating']}
"""
        return summary
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def generate_report(ticker: str) -> str:
    """Generate PDF report."""
    try:
        output_path = f"{ticker.upper()}_equity_report.pdf"
        generate_equity_report_pdf(ticker.upper(), output_path=output_path)
        return f"✅ Report generated: {output_path}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Tool registry
TOOLS = {
    "get_financial_data": get_financial_data,
    "calculate_valuation": calculate_valuation,
    "generate_report": generate_report,
    "get_stock_price_history": get_stock_price_history
}

tools_list = list(TOOLS.values())


# System prompt for the agent
SYSTEM_PROMPT = f"""You are an expert equity research analyst.
Date: {datetime.now().strftime("%d %b %Y")}

You are an expert equity research analyst at a global investment bank.

Your task is to produce a professional, institutional-quality equity research report suitable for PDF output.

You must follow this workflow strictly:
1. Retrieve and validate financial data using the provided tools.
2. Perform valuation analysis using my calculations and tool outputs only.
3. Do NOT introduce new assumptions, forecasts, or external data.
4. Synthesize analysis into a coherent equity research report.
5. Generate a structured PDF-ready report.

VALUATION FRAMEWORK:
- Primary valuation: Discounted Cash Flow (DCF)
- Secondary valuation: Dividend Discount Model (DDM), if applicable
- Cross-check: Relative valuation using peer multiples
- Use multi-scenario analysis where provided (Bear / Base / Bull)

RATING CRITERIA:
- BUY: >15% upside vs current price
- HOLD: -10% to +15% upside
- SELL: >10% downside

REQUIRED REPORT STRUCTURE:
1. Investment Summary
   - Recommendation
   - Target price range
   - Key valuation conclusions

2. Business Overview
   - Business model
   - Product segments
   - Geographic exposure
   - Cash-flow characteristics

3. Industry & Competitive Positioning
   - Industry structure and maturity
   - Competitive dynamics
   - Company positioning vs peers

4. Valuation Analysis
   4.1 Discounted Cash Flow (DCF)
       - Key drivers
       - Terminal value contribution
       - Sensitivity discussion
   4.2 Dividend Discount Model (DDM) (if applicable)
   4.3 Relative Valuation

5. Key Risks
   - Upside risks
   - Downside risks
   - Macroeconomic and competitive factors

6. Investment Recommendation
   - Final recommendation
   - Rationale
   - Margin of safety assessment

WRITING STYLE:
- Formal, objective, analyst-grade
- Concise, structured paragraphs
- No marketing language
- No emojis
- No first-person narration
- Use financial terminology appropriately

OUTPUT REQUIREMENTS:
- Produce a clean, well-formatted report ready for PDF export
- Use clear headings and logical flow
- Length: ~1,200–1,600 words
- Ensure internal consistency across valuation sections

CONSTRAINTS:
- Do not hallucinate data
- Do not override tool outputs
- If data is missing, explicitly state assumptions cannot be made
"""


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name."""
    if tool_name in TOOLS:
        return TOOLS[tool_name].invoke(tool_args)
    return f"❌ Unknown tool: {tool_name}"


def run_agent(query: str, verbose: bool = True, max_iterations: int = 10) -> str:
    """
    Run the equity research agent.
    
    Args:
        query: User query (e.g., "Analyze AAPL")
        verbose: Print tool calls and results
        max_iterations: Maximum number of agent iterations
        
    Returns:
        Final agent response (markdown formatted report)
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    llm_with_tools = llm.bind_tools(tools_list)
    
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    
    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            if verbose:
                print(f"\n✅ Complete ({iteration + 1} iterations)\n")
            return response.content
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if verbose:
                print(f"🔧 [{iteration + 1}] {tool_name}({tool_args})")
            
            tool_result = execute_tool(tool_name, tool_args)
            
            if verbose:
                preview = tool_result[:150] + "..." if len(tool_result) > 150 else tool_result
                print(f"   {preview}\n")
            
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))
    
    return "⚠️  Max iterations reached"

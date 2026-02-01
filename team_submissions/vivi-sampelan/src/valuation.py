"""
Valuation models including DCF and WACC calculations.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import yfinance as yf

from .utils import validate_ticker, safe_float, pick_series


def get_risk_free_rate() -> float:
    """Get current risk-free rate from 10-year Treasury."""
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        return tnx / 100
    except:
        return 0.04


def compute_wacc(
    ticker: str,
    erp: float = 0.04,
    rf_override: float = None
) -> Dict[str, Any]:
    """
    Compute WACC for equity-heavy tech companies.
    
    Args:
        ticker: Stock ticker symbol
        erp: Equity risk premium (default 0.04)
        rf_override: Optional override for risk-free rate
        
    Returns:
        Dictionary containing WACC and components
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        rf = rf_override if rf_override else get_risk_free_rate()
        beta = info.get("beta", 1.0)
        cost_of_equity = rf + beta * erp
        
        return {
            "wacc": round(cost_of_equity, 4),
            "beta": beta,
            "risk_free_rate": rf
        }
    except Exception as e:
        print(f"⚠️ WACC calculation failed: {e}")
        return {"wacc": 0.10}


def calculate_dcf(
    ticker: str,
    projection_years: int = 7,
    terminal_growth: float = 0.035,
    discount_rate: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate DCF valuation.
    
    Args:
        ticker: Stock ticker symbol
        projection_years: Number of years to project (default 7)
        terminal_growth: Terminal growth rate (default 0.035)
        discount_rate: Optional override for discount rate
        
    Returns:
        Dictionary containing DCF results and fair value
    """
    is_valid, validated_ticker = validate_ticker(ticker)
    if not is_valid:
        return {"error": validated_ticker, "ticker": ticker}
    
    ticker = validated_ticker
    
    try:
        if discount_rate is None:
            wacc_data = compute_wacc(ticker, erp=0.05)
            discount_rate = wacc_data.get("wacc", 0.10)
        
        stock = yf.Ticker(ticker)
        cf = stock.cashflow
        info = stock.info or {}
        
        ocf = pick_series(cf, ["Operating Cash Flow"])
        capex = pick_series(cf, ["Capital Expenditure"])
        
        if ocf is None or capex is None:
            return {"error": "Missing cash flow data", "ticker": ticker}
        
        latest_fcf = float(ocf.iloc[0] + capex.iloc[0])
        
        inc = stock.income_stmt
        revenue = pick_series(inc, ["Total Revenue", "Revenue"])
        
        if revenue is not None and len(revenue) >= 3:
            rev_sorted = revenue.sort_index()
            start_rev = float(rev_sorted.iloc[-3])
            end_rev = float(rev_sorted.iloc[-1])
            if start_rev > 0:
                growth_rate = (end_rev / start_rev) ** (1/2) - 1
                growth_rate = min(max(growth_rate, 0.08), 0.16)
            else:
                growth_rate = 0.08
        else:
            growth_rate = 0.08
        
        mature_growth = 0.08
        rates = [growth_rate + (mature_growth - growth_rate) * (i / (projection_years - 1)) 
                for i in range(projection_years)]
        
        current_year = datetime.now().year
        fcf_value = latest_fcf
        projected = []
        
        for i, g in enumerate(rates, start=1):
            fcf_value *= (1 + g)
            projected.append({
                "year": current_year + i,
                "growth_rate": round(g, 4),
                "fcf": fcf_value,
                "fcf_billions": round(fcf_value / 1e9, 2)
            })
        
        if discount_rate <= terminal_growth:
            return {"error": "Discount rate must be > terminal growth", "ticker": ticker}
        
        terminal_fcf = projected[-1]["fcf"]
        terminal_value = (terminal_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        pv_fcf = sum(p["fcf"] / ((1 + discount_rate) ** i) for i, p in enumerate(projected, start=1))
        pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
        
        enterprise_value = pv_fcf + pv_terminal
        
        bs = stock.balance_sheet
        cash = pick_series(bs, ["Cash And Cash Equivalents", "Cash"])
        debt = pick_series(bs, ["Total Debt", "Long Term Debt"])
        
        net_cash = 0.0
        if cash is not None:
            net_cash += float(cash.iloc[0])
        if debt is not None:
            net_cash -= float(debt.iloc[0])
        
        equity_value = enterprise_value + net_cash
        
        shares = info.get("sharesOutstanding")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        
        fair_value_per_share = (equity_value / shares) if shares else None
        upside_pct = ((fair_value_per_share / current_price - 1) * 100) if (fair_value_per_share and current_price) else None
        
        if upside_pct is None:
            rating = "HOLD"
        elif upside_pct > 15:
            rating = "BUY"
        elif upside_pct < -10:
            rating = "SELL"
        else:
            rating = "HOLD"
        
        return {
            "ticker": ticker,
            "latest_fcf": latest_fcf,
            "latest_fcf_billions": round(latest_fcf / 1e9, 2),
            "projected_fcf": projected,
            "enterprise_value": enterprise_value,
            "enterprise_value_billions": round(enterprise_value / 1e9, 2),
            "net_cash": net_cash,
            "net_cash_billions": round(net_cash / 1e9, 2),
            "equity_value": equity_value,
            "equity_value_billions": round(equity_value / 1e9, 2),
            "shares_outstanding": shares,
            "current_price": current_price,
            "fair_value_per_share": round(fair_value_per_share, 2) if fair_value_per_share else None,
            "upside_pct": round(upside_pct, 1) if upside_pct else None,
            "rating": rating,
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "base_growth_rate": round(growth_rate, 4)
        }
    
    except Exception as e:
        return {"error": f"DCF failed: {str(e)}", "ticker": ticker}


def calculate_dcf_sensitivity(
    ticker: str,
    wacc_values: Tuple[float, ...] = (0.08, 0.09, 0.10),
    growth_values: Tuple[float, ...] = (0.025, 0.03, 0.035)
) -> Dict[str, Dict[float, Optional[float]]]:
    """
    Calculate DCF sensitivity analysis.
    
    Args:
        ticker: Stock ticker symbol
        wacc_values: Tuple of WACC values to test
        growth_values: Tuple of terminal growth values to test
        
    Returns:
        Nested dictionary: {terminal_growth: {wacc: fair_value_per_share}}
    """
    results = {}

    for g in growth_values:
        row = {}
        for w in wacc_values:
            dcf = calculate_dcf(
                ticker,
                terminal_growth=g,
                discount_rate=w
            )
            fv = dcf.get("fair_value_per_share")
            row[round(w * 100, 1)] = fv
        results[round(g * 100, 1)] = row

    return results

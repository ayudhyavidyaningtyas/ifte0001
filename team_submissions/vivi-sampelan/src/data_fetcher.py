"""
Financial data fetching module using yfinance.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
import yfinance as yf
import pandas as pd

from .utils import (
    validate_ticker, safe_float, safe_round, pick_series
)


@lru_cache(maxsize=50)
def fetch_financial_data_cached(ticker: str, date_key: str) -> Dict[str, Any]:
    """Cached version of financial data fetch."""
    return fetch_financial_data(ticker)


def series_to_dict_billions(s: Optional[pd.Series], divisor=1e9) -> Dict[str, Optional[float]]:
    """Convert pandas series to dict with values in billions."""
    if s is None:
        return {}
    out = {}
    s_sorted = s.sort_index(ascending=False)
    for k, v in s_sorted.items():
        key = str(k.date()) if hasattr(k, "date") else str(k)
        val = safe_float(v)
        out[key] = safe_round(val / divisor, 2) if val is not None else None
    return out


def series_to_dict_pct(s: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """Convert pandas series to dict with percentage values."""
    if s is None:
        return {}
    out = {}
    s_sorted = s.sort_index(ascending=False)
    for k, v in s_sorted.items():
        key = str(k.date()) if hasattr(k, "date") else str(k)
        out[key] = safe_round(v, 2) if v is not None else None
    return out


def _latest_from_dict(d: Dict[str, Any]):
    """Get latest value from dict."""
    return next(iter(d.values())) if d else None


def fetch_financial_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive financial data for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing financial metrics and statements
    """
    is_valid, validated_ticker = validate_ticker(ticker)
    if not is_valid:
        return {"error": validated_ticker, "ticker": ticker}
    
    ticker = validated_ticker
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
            return {"error": f"No data available for {ticker}", "ticker": ticker}
        
        inc = stock.income_stmt
        bal = stock.balance_sheet
        cf = stock.cashflow
        
        revenue = pick_series(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
        ebit = pick_series(inc, ["Operating Income", "EBIT"])
        net_income = pick_series(inc, ["Net Income", "NetIncome"])
        
        equity = pick_series(bal, ["Total Stockholder Equity", "Stockholders Equity"])
        cash = pick_series(bal, ["Cash And Cash Equivalents", "Cash"])
        debt = pick_series(bal, ["Total Debt", "Long Term Debt"])
        
        ocf = pick_series(cf, ["Operating Cash Flow"])
        capex = pick_series(cf, ["Capital Expenditure"])
        
        def calc_margin(num, denom):
            if num is not None and denom is not None:
                return (num / denom * 100)
            return None
        
        ebit_margin = calc_margin(ebit, revenue)
        net_margin = calc_margin(net_income, revenue)
        roe = calc_margin(net_income, equity) if net_income is not None and equity is not None else None
        
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf + capex
        
        revenue_dict = series_to_dict_billions(revenue)
        ebit_dict = series_to_dict_billions(ebit)
        net_income_dict = series_to_dict_billions(net_income)
        fcf_dict = series_to_dict_billions(fcf)
        ebit_margin_dict = series_to_dict_pct(ebit_margin)
        net_margin_dict = series_to_dict_pct(net_margin)
        roe_dict = series_to_dict_pct(roe)
        
        market_cap = safe_float(info.get("marketCap"), 0)
        market_cap_billions = safe_round(market_cap / 1e9, 2)
        
        latest_ebit = _latest_from_dict(ebit_dict)
        latest_fcf = _latest_from_dict(fcf_dict)
        
        ev_ebit = safe_round(market_cap_billions / latest_ebit) if latest_ebit and latest_ebit != 0 else None
        p_fcf = safe_round(market_cap_billions / latest_fcf) if latest_fcf and latest_fcf != 0 else None
        
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": safe_round(info.get("currentPrice") or info.get("regularMarketPrice")),
            "market_cap_billions": market_cap_billions,
            "pe_ratio": safe_round(info.get("trailingPE")),
            "forward_pe": safe_round(info.get("forwardPE")),
            "price_to_book": safe_round(info.get("priceToBook")),
            "dividend_yield_pct": safe_round(safe_float(info.get("dividendYield"), 0) * 100, 2),
            "beta": safe_round(info.get("beta")),
            "52_week_high": safe_round(info.get("fiftyTwoWeekHigh")),
            "52_week_low": safe_round(info.get("fiftyTwoWeekLow")),
            "revenue_billions": revenue_dict,
            "ebit_billions": ebit_dict,
            "net_income_billions": net_income_dict,
            "free_cash_flow_billions": fcf_dict,
            "ebit_margin_pct": ebit_margin_dict,
            "net_margin_pct": net_margin_dict,
            "roe_pct": roe_dict,
            "latest_revenue": _latest_from_dict(revenue_dict),
            "latest_ebit": latest_ebit,
            "latest_net_income": _latest_from_dict(net_income_dict),
            "latest_fcf": latest_fcf,
            "latest_ebit_margin": _latest_from_dict(ebit_margin_dict),
            "latest_net_margin": _latest_from_dict(net_margin_dict),
            "latest_roe": _latest_from_dict(roe_dict),
            "ev_ebit": ev_ebit,
            "p_fcf": p_fcf,
            "cash_billions": safe_round(safe_float(cash.iloc[0] if cash is not None else 0) / 1e9, 2),
            "debt_billions": safe_round(safe_float(debt.iloc[0] if debt is not None else 0) / 1e9, 2),
            "data_date": datetime.now().strftime("%d %b %Y"),
            "shares_outstanding": info.get("sharesOutstanding")
        }
        
    except Exception as e:
        return {"error": f"Error: {str(e)}", "ticker": ticker}


def get_price_history(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Get historical price data and calculate key statistics.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (default "1y")
        
    Returns:
        Dictionary containing price history and technical indicators
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": "No historical data available", "ticker": str(ticker)}
        
        hist['Daily_Return'] = hist['Close'].pct_change()
        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
        hist['MA_200'] = hist['Close'].rolling(window=200).mean()
        
        current_price = float(hist['Close'].iloc[-1])
        start_price = float(hist['Close'].iloc[0])
        
        ma_50_val = hist['MA_50'].iloc[-1]
        ma_200_val = hist['MA_200'].iloc[-1]
        
        ma_50 = round(float(ma_50_val), 2) if pd.notna(ma_50_val) else None
        ma_200 = round(float(ma_200_val), 2) if pd.notna(ma_200_val) else None
        
        above_ma_50 = bool(current_price > ma_50) if ma_50 is not None else None
        above_ma_200 = bool(current_price > ma_200) if ma_200 is not None else None
        
        volatility = float(hist['Daily_Return'].std()) * (252 ** 0.5) * 100
        period_return = ((current_price / start_price) - 1) * 100
        
        return {
            "ticker": str(ticker),
            "period": str(period),
            "current_price": round(current_price, 2),
            "period_high": round(float(hist['High'].max()), 2),
            "period_low": round(float(hist['Low'].min()), 2),
            "period_return_pct": round(period_return, 2),
            "avg_daily_volume": int(hist['Volume'].mean()),
            "volatility_annualized_pct": round(volatility, 2),
            "ma_50": ma_50,
            "ma_200": ma_200,
            "above_ma_50": above_ma_50,
            "above_ma_200": above_ma_200,
            "trading_days": int(len(hist))
        }
    except Exception as e:
        return {"error": str(e), "ticker": str(ticker)}

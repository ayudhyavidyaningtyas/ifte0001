# TARGET PRICE ANALYSIS
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

if "downloader" not in globals():
    class MockDownloader:
        ticker = "GOOGL"
    downloader = MockDownloader()

ticker = str(downloader.ticker).upper().strip()

def get_automated_eps_estimates(ticker_symbol, manual_estimates=None):
    """
    Automatically fetch forward EPS estimates from Yahoo Finance.
    Falls back to manual estimates if API data unavailable.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Get current fiscal year
        current_year = date.today().year
        
        # Yahoo Finance provides these EPS fields:
        # - 'forwardEps' (next 12 months consensus)
        # - From earnings_dates: historical and forward estimates
        
        estimates = {}
        
        # Try Method 1: Get earnings calendar data
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                # earnings_dates has 'EPS Estimate' column
                eps_data = earnings_dates['EPS Estimate'].dropna()
                
                # Map dates to fiscal years
                for idx, eps_val in eps_data.items():
                    if pd.notna(eps_val) and isinstance(idx, pd.Timestamp):
                        year = idx.year
                        if year not in estimates:
                            estimates[year] = []
                        estimates[year].append(float(eps_val))
                
                # Average if multiple quarters per year
                estimates = {yr: np.mean(vals) * 4 for yr, vals in estimates.items() if len(vals) > 0}
        except Exception as e:
            print(f"  Note: earnings_dates not available ({e})")
        
        # Try Method 2: Use analyst targets and growth rates
        forward_eps = info.get('forwardEps')  # Next 12M consensus
        earnings_growth = info.get('earningsQuarterlyGrowth')  # Quarterly growth rate
        
        if forward_eps and np.isfinite(forward_eps) and forward_eps > 0:
            # Use forward EPS as anchor for FY+0
            estimates[current_year] = float(forward_eps)
            
            # Project future years using growth rate
            if earnings_growth and np.isfinite(earnings_growth):
                annual_growth = min((1 + earnings_growth) ** 4 - 1, 0.30)  # Cap at 30% annual
            else:
                annual_growth = 0.10  # Default 10% growth
            
            for i in range(1, 4):
                estimates[current_year + i] = estimates[current_year] * ((1 + annual_growth) ** i)
        
        # Try Method 3: Get from analysis page (trailingEps + growth)
        if not estimates and info.get('trailingEps'):
            trailing_eps = info.get('trailingEps')
            earnings_growth_rate = info.get('earningsGrowth', 0.10)  # Default 10%
            
            if np.isfinite(trailing_eps) and trailing_eps > 0:
                for i in range(4):
                    estimates[current_year + i] = trailing_eps * ((1 + earnings_growth_rate) ** (i + 1))
        
        # Validate we have minimum required years
        required_years = [current_year, current_year + 1, current_year + 2, current_year + 3]
        if all(year in estimates for year in required_years):
            print(f"\n✓ Automated EPS estimates downloaded from Yahoo Finance:")
            for year in sorted(estimates.keys()):
                print(f"  FY{year}: ${estimates[year]:.2f}")
            return estimates
        else:
            raise ValueError("Insufficient forward EPS data")
    
    except Exception as e:
        print(f"\n⚠️  Automated EPS download failed: {e}")
        if manual_estimates:
            print("  → Using manual estimates as fallback")
            return manual_estimates
        else:
            raise ValueError("No EPS estimates available (automated failed, no manual fallback)")

# MANUAL ESTIMATES (Fallback only)
MANUAL_EARNINGS_ESTIMATES = {
    2025: 10.57,  # FY 2025 consensus EPS
    2026: 11.24,  # FY 2026 consensus EPS
    2027: 11.95,  # FY 2027 consensus EPS
    2028: 12.70,  # FY 2028 consensus EPS
}

# Try automated download first, fall back to manual
try:
    EARNINGS_ESTIMATES = get_automated_eps_estimates(ticker, MANUAL_EARNINGS_ESTIMATES)
except Exception as e:
    print(f"\n⚠️  Using manual EPS estimates")
    EARNINGS_ESTIMATES = MANUAL_EARNINGS_ESTIMATES

# Company-specific info
FISCAL_YEAR_END_MONTH = 12  # Alphabet FY = calendar year (ends December)

# PEER SET DEFINITION
PEERS = ["MSFT", "META", "AMZN", "AAPL", "NVDA"] 

# SCENARIO DEFINITIONS - 5 CASES
EPS_SCENARIOS = {
    "Very Bear": {
        "description": "Severe recession, AI monetization fails, Search disruption",
        "fy_outer_adjustment": -0.08,
    },
    "Bear": {
        "description": "No growth scenario - macro headwinds, flat earnings",
        "fy_outer_adjustment": -0.02,
    },
    "Base": {
        "description": "In line with consensus, steady AI product traction",
        "fy_outer_adjustment": 0.00,
    },
    "Bull": {
        "description": "AI products drive upside to Cloud/Search, margin expansion",
        "fy_outer_adjustment": +0.08,
    },
    "Very Bull": {
        "description": "AI breakthrough, dominant monetization, market share gains",
        "fy_outer_adjustment": +0.12,
    },
}

# [Rest of the file remains unchanged - keep all existing functions]

def get_pe_scenarios(current_premium_to_peers):
    """Generate P/E scenarios anchored to ticker's current premium."""
    if not np.isfinite(current_premium_to_peers) or current_premium_to_peers <= 0:
        current_premium_to_peers = 1.20  # Default fallback
    
    bear_premium = current_premium_to_peers
    
    return {
        "Very Bear": bear_premium * 0.92,  # 8% de-rating
        "Bear": bear_premium * 1.00,       # Maintain premium
        "Base": bear_premium * 1.08,       # 8% re-rating
        "Bull": bear_premium * 1.17,       # 17% re-rating
        "Very Bull": bear_premium * 1.25,  # 25% re-rating
    }

def _sf(x):
    """Safe float conversion"""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except:
        return np.nan

def _get_info(ticker_str):
    """Get ticker info with error handling"""
    try:
        return yf.Ticker(ticker_str).info or {}
    except:
        return {}

def compute_eps_blend_from_date(estimates_dict, start_date, fiscal_year_end_month):
    """
    Compute time-weighted blend of current and next FY EPS starting from any date.
    
    This function accepts an arbitrary start_date and computes which fiscal years
    overlap with the 12-month window starting from that date.
    
    Args:
        estimates_dict: Dict mapping fiscal year (int) → EPS estimate (float)
        start_date: datetime.date object (the beginning of the 12-month window)
        fiscal_year_end_month: Integer 1-12 representing FY end month
    
    Returns:
        tuple: (blended_eps, weights_dict, description_string)
    """
    target_end_date = start_date + timedelta(days=365)
    
    # Determine which fiscal year "start_date" falls into
    if start_date.month <= fiscal_year_end_month:
        current_fy = start_date.year
    else:
        current_fy = start_date.year + 1
    
    # Build FY end dates
    fy_end_current = date(current_fy, fiscal_year_end_month, 28)
    if fy_end_current.month == 2:
        try:
            fy_end_current = date(current_fy, fiscal_year_end_month, 29)
        except ValueError:
            pass
    
    fy_end_next = date(current_fy + 1, fiscal_year_end_month, 28)
    if fy_end_next.month == 2:
        try:
            fy_end_next = date(current_fy + 1, fiscal_year_end_month, 29)
        except ValueError:
            pass
    
    # Calculate overlaps
    if target_end_date <= fy_end_current:
        weight_current_fy = 1.0
        weight_next_fy = 0.0
    elif start_date >= fy_end_current:
        weight_current_fy = 0.0
        weight_next_fy = 1.0
    else:
        months_in_current_fy = (fy_end_current - start_date).days / 365.0 * 12
        months_in_current_fy = max(0, min(months_in_current_fy, 12))
        
        weight_current_fy = months_in_current_fy / 12.0
        weight_next_fy = 1.0 - weight_current_fy
    
    # Get EPS estimates
    current_fy_eps = estimates_dict.get(current_fy)
    next_fy_eps = estimates_dict.get(current_fy + 1)
    
    if current_fy_eps is None or not np.isfinite(current_fy_eps):
        raise ValueError(f"Missing EPS estimate for FY {current_fy}")
    if next_fy_eps is None or not np.isfinite(next_fy_eps):
        raise ValueError(f"Missing EPS estimate for FY {current_fy + 1}")
    
    # Compute blend
    blended_eps = current_fy_eps * weight_current_fy + next_fy_eps * weight_next_fy
    
    weights = {
        current_fy: weight_current_fy,
        current_fy + 1: weight_next_fy
    }
    
    description = f"{weight_current_fy:.0%} FY{current_fy} + {weight_next_fy:.0%} FY{current_fy+1}"
    
    return blended_eps, weights, description

def get_peer_forward_pe_stats(peers):
    """Calculate peer forward P/E statistics."""
    rows = []
    for p in peers:
        info = _get_info(p)
        fwd_pe = _sf(info.get("forwardPE"))
        mcap = _sf(info.get("marketCap"))
        name = info.get("shortName", p)
        
        rows.append({
            "Ticker": p,
            "Name": name,
            "Forward_PE": fwd_pe,
            "Market_Cap": mcap
        })
    
    df = pd.DataFrame(rows)
    valid_df = df[np.isfinite(df["Forward_PE"]) & (df["Forward_PE"] > 0)].copy()
    
    if valid_df.empty:
        return df, {
            "count": 0,
            "median": np.nan,
            "mean": np.nan,
            "cap_weighted_mean": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan
        }
    
    # Calculate statistics
    count = len(valid_df)
    median = float(valid_df["Forward_PE"].median())
    mean = float(valid_df["Forward_PE"].mean())
    q1 = float(valid_df["Forward_PE"].quantile(0.25))
    q3 = float(valid_df["Forward_PE"].quantile(0.75))
    iqr = q3 - q1
    
    # Market-cap weighted average
    valid_mcap_df = valid_df[np.isfinite(valid_df["Market_Cap"]) & (valid_df["Market_Cap"] > 0)]
    
    if len(valid_mcap_df) >= 2:
        weights = valid_mcap_df["Market_Cap"].values
        values = valid_mcap_df["Forward_PE"].values
        cap_weighted_mean = float(np.average(values, weights=weights))
    else:
        cap_weighted_mean = np.nan
    
    stats = {
        "count": count,
        "median": median,
        "mean": mean,
        "cap_weighted_mean": cap_weighted_mean,
        "q1": q1,
        "q3": q3,
        "iqr": iqr
    }
    
    return valid_df, stats

# Formatting helpers
def _fmt_pe(x):
    return f"{x:.1f}x" if np.isfinite(x) else "NA"

def _fmt_pct(x):
    return f"{x:+.1%}" if np.isfinite(x) else "NA"

def _fmt_usd(x):
    return f"${x:,.0f}B" if np.isfinite(x) else "NA"

def run_targets_for_horizon(horizon_months, current_price, info, peer_stats, PE_SCENARIOS):
    """
    Run target price analysis for a specific time horizon.
    
    Args:
        horizon_months: Integer (12 or 18) representing the time horizon
        current_price: Current stock price
        info: Ticker info dict
        peer_stats: Peer group statistics
        PE_SCENARIOS: P/E multiple scenarios
    
    Returns:
        tuple: (results_df, sensitivity_df, eps_value, weight_desc)
    """
    today = date.today()
    
    # Calculate the start date for the EPS window
    if horizon_months == 12:
        # 12-month: Use today as start (next twelve months)
        eps_window_start = today
        horizon_label = "12M"
        target_date = today + timedelta(days=365)
    elif horizon_months == 18:
        # 18-month: Use today+6 months as start (months 7-18 from today)
        eps_window_start = today + relativedelta(months=6)
        horizon_label = "18M"
        target_date = today + timedelta(days=int(365 * 1.5))
    else:
        raise ValueError(f"Unsupported horizon: {horizon_months} months")
    
    # Calculate EPS blend for this horizon
    eps_blend, weights, weight_desc = compute_eps_blend_from_date(
        EARNINGS_ESTIMATES, 
        eps_window_start, 
        FISCAL_YEAR_END_MONTH
    )
    
    # Get peer median P/E for calculations
    peer_median_pe = peer_stats["median"]
    
    # Build results for each scenario
    rows = []
    for scenario_name in ["Very Bear", "Bear", "Base", "Bull", "Very Bull"]:
        premium_ratio = PE_SCENARIOS[scenario_name]  # Premium to peers (e.g., 1.20)
        fy_adjustment = EPS_SCENARIOS[scenario_name]["fy_outer_adjustment"]
        
        # Adjust EPS for scenario
        adjusted_eps = eps_blend * (1 + fy_adjustment)
        
        # Calculate target P/E: Peer Median × Premium
        target_pe = peer_median_pe * premium_ratio
        
        # Calculate target price
        target_price = adjusted_eps * target_pe
        
        # Calculate upside
        upside = (target_price / current_price) - 1 if current_price > 0 else np.nan
        
        rows.append({
            "Scenario": scenario_name,
            "Description": EPS_SCENARIOS[scenario_name]["description"],
            "P/E_Multiple": target_pe,  # Now showing actual P/E (e.g., 30x)
            "Premium_to_Peers": premium_ratio,  # Premium ratio (e.g., 1.20)
            "EPS_Adjustment": fy_adjustment,
            "Adj_EPS": adjusted_eps,
            "Target_Price": target_price,
            "Upside": upside
        })
    
    results_df = pd.DataFrame(rows)
    
    # Sensitivity analysis - create P/E range around scenarios
    # Convert premium ratios to actual P/Es
    pe_min = peer_median_pe * PE_SCENARIOS["Very Bear"] * 0.8
    pe_max = peer_median_pe * PE_SCENARIOS["Very Bull"] * 1.2
    pe_range = np.linspace(pe_min, pe_max, 15)
    
    eps_adjustments = {
        "Bear (-8%)": -0.08,
        "Base (0%)": 0.00,
        "Bull (+8%)": +0.08
    }
    
    sens_rows = []
    for eps_label, eps_adj in eps_adjustments.items():
        adj_eps = eps_blend * (1 + eps_adj)
        for pe in pe_range:
            target = adj_eps * pe
            upside = (target / current_price) - 1 if current_price > 0 else np.nan
            sens_rows.append({
                "EPS_Scenario": eps_label,
                "P/E_Multiple": pe,
                "Target_Price": target,
                "Upside": upside
            })
    
    sensitivity_df = pd.DataFrame(sens_rows)
    
    return results_df, sensitivity_df, eps_blend, weight_desc

def main():
    """Main execution function for multi-horizon target price analysis."""
    
    # Get current market data
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    
    current_price = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = _sf(info.get("marketCap"))
    company_name = info.get("longName", ticker)
    
    if not np.isfinite(current_price) or current_price <= 0:
        print(f"Error: Could not fetch current price for {ticker}")
        return None
    
    print(f"\n{'='*92}")
    print(f"MULTI-HORIZON TARGET PRICE ANALYSIS | {ticker}")
    print(f"{'='*92}")
    print(f"Company: {company_name}")
    print(f"Current Price: ${current_price:,.2f}")
    if np.isfinite(market_cap):
        print(f"Market Cap: ${market_cap/1e9:,.1f}B")
    print(f"Analysis Date: {date.today()}")
    
    # Get peer valuations
    print(f"\n{'─'*92}")
    print(f"PEER VALUATION ANALYSIS")
    print(f"{'─'*92}")
    
    peer_df, peer_stats = get_peer_forward_pe_stats(PEERS)
    
    if not peer_df.empty:
        print(peer_df.to_string(index=False))
        print(f"\nPeer Statistics:")
        print(f"  Count: {peer_stats['count']}")
        print(f"  Median Forward P/E: {_fmt_pe(peer_stats['median'])}")
        print(f"  Mean Forward P/E: {_fmt_pe(peer_stats['mean'])}")
        if np.isfinite(peer_stats['cap_weighted_mean']):
            print(f"  Cap-Weighted Mean: {_fmt_pe(peer_stats['cap_weighted_mean'])}")
    
    # Calculate ticker's forward P/E and premium
    ticker_fwd_pe = _sf(info.get("forwardPE"))
    
    if np.isfinite(ticker_fwd_pe) and np.isfinite(peer_stats["median"]):
        current_premium = ticker_fwd_pe / peer_stats["median"]
        print(f"\n{ticker} Forward P/E: {_fmt_pe(ticker_fwd_pe)}")
        print(f"Premium to Peer Median: {_fmt_pct(current_premium - 1)}")
    else:
        current_premium = 1.20  # Default
        print(f"\n⚠️ Using default premium: 1.20x")
    
    # Generate P/E scenarios
    PE_SCENARIOS = get_pe_scenarios(current_premium)
    
    # Run analysis for both horizons
    results_12m, sens_12m, eps_12m, weight_desc_12m = run_targets_for_horizon(
        12, current_price, info, peer_stats, PE_SCENARIOS
    )
    
    results_18m, sens_18m, eps_18m, weight_desc_18m = run_targets_for_horizon(
        18, current_price, info, peer_stats, PE_SCENARIOS
    )
    
    # Display results
    print(f"\n{'='*92}")
    print(f"TARGET PRICE ANALYSIS | {ticker} | 12M HORIZON")
    print(f"{'='*92}")
    print(f"EPS Blend ({weight_desc_12m}): ${eps_12m:.2f}")
    print(f"\nScenario Analysis:")
    print(results_12m.to_string(index=False))
    
    base_12m = results_12m.loc[results_12m["Scenario"] == "Base", "Target_Price"].values[0]
    base_upside_12m = results_12m.loc[results_12m["Scenario"] == "Base", "Upside"].values[0]
    print(f"\n✓ 12M Base Case Target: ${base_12m:.2f} ({_fmt_pct(base_upside_12m)} upside)")
    
    print(f"\n{'='*92}")
    print(f"TARGET PRICE ANALYSIS | {ticker} | 18M HORIZON")
    print(f"{'='*92}")
    print(f"EPS Blend ({weight_desc_18m}): ${eps_18m:.2f}")
    print(f"\nScenario Analysis:")
    print(results_18m.to_string(index=False))
    
    base_18m = results_18m.loc[results_18m["Scenario"] == "Base", "Target_Price"].values[0]
    base_upside_18m = results_18m.loc[results_18m["Scenario"] == "Base", "Upside"].values[0]
    print(f"\n✓ 18M Base Case Target: ${base_18m:.2f} ({_fmt_pct(base_upside_18m)} upside)")
    
    # Comparison summary
    print(f"\n{'='*92}")
    print(f"HORIZON COMPARISON SUMMARY")
    print(f"{'='*92}")
    
    comparison = pd.DataFrame({
        "Scenario": results_12m["Scenario"],
        "12M_Target": results_12m["Target_Price"],
        "12M_Upside": results_12m["Upside"],
        "18M_Target": results_18m["Target_Price"],
        "18M_Upside": results_18m["Upside"]
    })
    
    print(comparison.to_string(index=False))
    
    print(f"\n{'='*92}\n")
    
    # Return results
    return {
        'results_12m': results_12m,
        'sensitivity_12m': sens_12m,
        'results_18m': results_18m,
        'sensitivity_18m': sens_18m,
        'eps_12m': eps_12m,
        'eps_18m': eps_18m
    }

# Execute analysis
if __name__ == "__main__":
    main()

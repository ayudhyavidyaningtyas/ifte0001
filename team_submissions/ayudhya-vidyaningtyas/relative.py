# Relative Valuation and Dividend Discount Model
import numpy as np
import pandas as pd
import yfinance as yf

# Check if downloader exists by trying to access it
try:
    ticker = str(downloader.ticker).upper().strip()
except NameError:
    raise RuntimeError("Missing 'downloader'. Run your Data Downloader step first.")
stock = yf.Ticker(ticker)
info = stock.info or {}

print(f"\nMULTIPLES & DDM VALUATION - {ticker}")
print("=" * 80)

# Helper Functions
def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _av_get(df, period, key, default=np.nan):
    if df is None or df.empty or period not in df.index or key not in df.columns:
        return default
    return _safe_float(df.loc[period, key], default=default)

def get_latest_income_period(df):
    if df is None or df.empty:
        return None, "No Data"
    for ttm in ["2025 (TTM)", "TTM", "Trailing Twelve Months"]:
        if ttm in df.index:
            return ttm, "TTM"
    return df.index[0], str(df.index[0])

def get_latest_balance_period(df):
    if df is None or df.empty:
        return None, "No Data"
    for ttm in ["2025 (TTM)", "TTM", "Trailing Twelve Months"]:
        if ttm in df.index:
            return ttm, "Latest Quarter"
    return df.index[0], str(df.index[0])

def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s[(s > lo) & (s < hi)]

def _median_or_nan(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.median()) if len(s) else np.nan

def _pct(x):
    return f"{x:.1%}" if np.isfinite(x) else "n/a"

# Extract Company Fundamentals
latest_period, period_label = get_latest_income_period(downloader.income_statement)
balance_period, balance_label = get_latest_balance_period(downloader.balance_sheet)

# Get Shares Information
shares = _safe_float(info.get("impliedSharesOutstanding") or info.get("sharesOutstanding"), np.nan)
if not (np.isfinite(shares) and shares > 0) and latest_period in downloader.income_statement.index:
    row = downloader.income_statement.loc[latest_period]
    for k in ["weightedAverageDilutedShares", "weightedAverageSharesDiluted", "dilutedAverageShares"]:
        if k in row.index:
            s = _safe_float(row.get(k))
            if np.isfinite(s) and s > 0:
                shares = s
                break

price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"), np.nan)
if not (np.isfinite(price) and price > 0) and hasattr(downloader, "market_data"):
    price = _safe_float(downloader.market_data.get("current_price"), np.nan)

market_cap = _safe_float(info.get("marketCap"), np.nan)
if not (np.isfinite(market_cap) and market_cap > 0) and np.isfinite(price) and np.isfinite(shares):
    market_cap = float(price * shares)

revenue = _av_get(downloader.income_statement, latest_period, "totalRevenue", default=np.nan)
net_income = _av_get(downloader.income_statement, latest_period, "netIncome", default=np.nan)
ebitda = _av_get(downloader.income_statement, latest_period, "ebitda", default=np.nan)

if not np.isfinite(ebitda):
    ebit = _av_get(downloader.income_statement, latest_period, "ebit", default=np.nan)
    da = _av_get(downloader.income_statement, latest_period, "depreciationAndAmortization", default=np.nan)
    if np.isfinite(ebit) and np.isfinite(da):
        ebitda = float(ebit + da)

cash = _av_get(downloader.balance_sheet, balance_period, "cashAndCashEquivalentsAtCarryingValue", default=np.nan)
if not np.isfinite(cash):
    cash = _av_get(downloader.balance_sheet, balance_period, "cashAndCashEquivalents", default=np.nan)
sti = _av_get(downloader.balance_sheet, balance_period, "shortTermInvestments", default=0.0)
total_cash = (0.0 if not np.isfinite(cash) else cash) + (0.0 if not np.isfinite(sti) else sti)

total_debt = _av_get(downloader.balance_sheet, balance_period, "shortLongTermDebtTotal", default=np.nan)
if not np.isfinite(total_debt):
    total_debt = _av_get(downloader.balance_sheet, balance_period, "totalDebt", default=0.0)
total_debt = 0.0 if not np.isfinite(total_debt) else total_debt

net_debt = float(total_debt - total_cash)  # negative => net cash
ev = float(market_cap + net_debt) if np.isfinite(market_cap) else np.nan

print(f"Metrics Snapshot ({period_label})")
print(f"Price: ${price:,.2f} | Shares: {shares/1e9:,.2f}B | Market Cap: ${market_cap/1e9:,.1f}B")
print(f"Net Debt: ${net_debt/1e9:,.1f}B (Cash+STI: ${total_cash/1e9:,.1f}B, Debt: ${total_debt/1e9:,.1f}B)")
print("-" * 80)

# Peers
core_peers = ["MSFT", "AAPL", "META"]  
extended_add = []

core_peers = [p for p in core_peers if p.upper() != ticker.upper()]
extended_peers = [p for p in (core_peers + extended_add) if p.upper() != ticker.upper()]

def fetch_peer_multiples(peers):
    rows = []
    for p in peers:
        try:
            p_info = yf.Ticker(p).info or {}
            rows.append({
                "Ticker": p,
                "Sector": p_info.get("sector"),
                "Industry": p_info.get("industry"),
                "MktCap($T)": (_safe_float(p_info.get("marketCap")) / 1e12) if np.isfinite(_safe_float(p_info.get("marketCap"))) else np.nan,
                # Trailing multiples (explicitly trailing / TTM)
                "P/E (Trailing)": _safe_float(p_info.get("trailingPE")),
                "EV/EBITDA": _safe_float(p_info.get("enterpriseToEbitda")),
                "P/S (TTM)": _safe_float(p_info.get("priceToSalesTrailing12Months")),
            })
        except Exception:
            pass
    df = pd.DataFrame(rows)
    return df

df_core = fetch_peer_multiples(core_peers)
df_ext = fetch_peer_multiples(extended_peers)

# Outlier filters
PE_LO, PE_HI = 5, 120
EV_LO, EV_HI = 2, 60
PS_LO, PS_HI = 0.5, 30

def compute_medians(df):
    med = {}
    pe = _clip_series(df.get("P/E (Trailing)", pd.Series(dtype=float)), PE_LO, PE_HI)
    evm = _clip_series(df.get("EV/EBITDA", pd.Series(dtype=float)), EV_LO, EV_HI)
    ps = _clip_series(df.get("P/S (TTM)", pd.Series(dtype=float)), PS_LO, PS_HI)

    med["P/E (Trailing)"] = float(pe.median()) if len(pe) else np.nan
    med["EV/EBITDA"] = float(evm.median()) if len(evm) else np.nan
    med["P/S (TTM)"] = float(ps.median()) if len(ps) else np.nan

    med["n_pe"] = int(len(pe))
    med["n_ev"] = int(len(evm))
    med["n_ps"] = int(len(ps))
    return med

med_core = compute_medians(df_core)
med_ext = compute_medians(df_ext)

print("Peer Comparability Snapshot (Core)")
if df_core.empty:
    print("No peer data returned.")
else:
    print(df_core[["Ticker", "Sector", "Industry", "MktCap($T)", "P/E (Trailing)", "EV/EBITDA", "P/S (TTM)"]].to_string(index=False))

print("-" * 80)
print(f"CORE Peer Medians (after filters) -> "
      f"P/E: {med_core['P/E (Trailing)']:.1f}x (n={med_core['n_pe']}) | "
      f"EV/EBITDA: {med_core['EV/EBITDA']:.1f}x (n={med_core['n_ev']}) | "
      f"P/S: {med_core['P/S (TTM)']:.1f}x (n={med_core['n_ps']})")

print("\nPeer Comparability Snapshot (Extended; sensitivity)")
if df_ext.empty:
    print("No extended peer data returned.")
else:
    print(df_ext[["Ticker", "Sector", "Industry", "MktCap($T)", "P/E (Trailing)", "EV/EBITDA", "P/S (TTM)"]].to_string(index=False))

print("-" * 80)
print(f"EXT Peer Medians (after filters)  -> "
      f"P/E: {med_ext['P/E (Trailing)']:.1f}x (n={med_ext['n_pe']}) | "
      f"EV/EBITDA: {med_ext['EV/EBITDA']:.1f}x (n={med_ext['n_ev']}) | "
      f"P/S: {med_ext['P/S (TTM)']:.1f}x (n={med_ext['n_ps']})")

# Implied Prices
def implied_prices(med, label):
    print(f"\nIMPLIED VALUATION (Based on {label} Peer Medians)")
    print("=" * 80)

    implied_vals = []

    # P/E: (Net income * P/E) / shares
    if np.isfinite(net_income) and net_income > 0 and np.isfinite(med["P/E (Trailing)"]) and med["P/E (Trailing)"] > 0 and np.isfinite(shares) and shares > 0:
        val_pe = (net_income * med["P/E (Trailing)"]) / shares
        implied_vals.append(val_pe)
        print(f"Implied by P/E ({med['P/E (Trailing)']:.1f}x):       ${val_pe:,.2f}")
    else:
        print("Implied by P/E: N/A (missing inputs)")

    # EV/EBITDA: ((EBITDA * EV/EBITDA) - NetDebt) / shares
    if np.isfinite(ebitda) and ebitda > 0 and np.isfinite(med["EV/EBITDA"]) and med["EV/EBITDA"] > 0 and np.isfinite(shares) and shares > 0:
        implied_ev = ebitda * med["EV/EBITDA"]
        equity_val = implied_ev - net_debt
        val_ev = equity_val / shares
        implied_vals.append(val_ev)
        print(f"Implied by EV/EBITDA ({med['EV/EBITDA']:.1f}x): ${val_ev:,.2f}")
    else:
        print("Implied by EV/EBITDA: N/A (missing inputs)")

    # P/S: (Revenue * P/S) / shares
    if np.isfinite(revenue) and revenue > 0 and np.isfinite(med["P/S (TTM)"]) and med["P/S (TTM)"] > 0 and np.isfinite(shares) and shares > 0:
        val_ps = (revenue * med["P/S (TTM)"]) / shares
        implied_vals.append(val_ps)
        print(f"Implied by P/S ({med['P/S (TTM)']:.1f}x):       ${val_ps:,.2f}")
    else:
        print("Implied by P/S: N/A (missing inputs)")

    if implied_vals:
        avg_relative = float(np.mean(implied_vals))
        print("-" * 80)
        print(f"AVERAGE RELATIVE VALUE:           ${avg_relative:,.2f}")
        print(f"Current Price:                    ${price:,.2f}")
        print(f"Upside/(Downside):                {_pct(avg_relative/price - 1)}")
        return avg_relative
    else:
        print("-" * 80)
        print("No implied values computed.")
        return np.nan

avg_core = implied_prices(med_core, "CORE")
avg_ext = implied_prices(med_ext, "EXTENDED (Sensitivity)")

# DDM
print("\nDIVIDEND DISCOUNT MODEL (DDM)")
print("=" * 80)

div_rate = _safe_float(info.get("dividendRate"), 0.0)
if np.isfinite(div_rate) and div_rate > 0:
    beta = _safe_float(info.get("beta"), 1.0)
    rf = 0.0428
    erp = 0.0447
    ke = rf + beta * erp

    # For Alphabet specifically, DDM is usually not the right model because dividends are small vs buybacks.
    g = 0.04

    print(f"Annual Dividend: ${div_rate:,.2f}")
    print(f"Cost of Equity (Ke): {ke:.1%} (rf={rf:.2%}, beta={beta:.2f}, ERP={erp:.2%})")
    print(f"Assumed Dividend Growth (g): {g:.1%}")

    if ke > g:
        val_ddm = (div_rate * (1 + g)) / (ke - g)
        print(f"DDM Fair Value: ${val_ddm:,.2f}")
        print("Note: Treat DDM as non-primary for Alphabet due to low payout; buybacks dominate shareholder returns.")
    else:
        print("DDM N/A: Growth rate >= Cost of Equity (formula breaks).")
else:
    print("Dividend not available or negligible. DDM not meaningful for Alphabet; use DCF and multiples instead.")

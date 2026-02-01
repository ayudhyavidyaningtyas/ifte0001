#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import math

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Optional import for LLM generation
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# Utility helpers

from pathlib import Path
import os

def ensure_dir(path):
    # when it is not in venv
    base = Path(__file__).resolve().parent
    out_dir = base / path
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"️ Failed to create the directory in {out_dir}. Using a temporary directory instead: {e}")
        fallback = Path(os.getcwd()) / path
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return out_dir


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_env_openai_key():
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_OPENAI")


# Data ingestion
def fetch_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV (Open, High, Low, Close, Volume) using yfinance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    #Data integrity check
    if df.empty:
        raise RuntimeError(f"No market data returned for {ticker} between {start} and {end}")
    #Date index standardization: Convert "Date" to the "datetime64" type in pandas
    # to facilitate subsequent processing.
    df.index = pd.to_datetime(df.index)
    return df

def clean_financial_data(fin_data):
    """
    Convert all financial data to be uniformly expressed in millions of US dollars (USD million)
    Modify the internal structure of the original dict without changing the key names.
    """
    def scale_value(x):
        try:
            val = float(x)
            return val / 1_000_000
        except Exception:
            return x

    # Examine three reports to see if they exist；
    # Traverse through each year to ensure consistency in the data structure;
    # Use dictionary-style expressions to handle each item
    for section in ["income_annual", "bs_annual", "cf_annual"]:
        if section in fin_data:
            for year, items in fin_data[section].items():
                if isinstance(items, dict):
                    fin_data[section][year] = {
                        k: scale_value(v) for k, v in items.items()
                    }
    return fin_data

def fetch_financials_yfinance(ticker: str):
    """
    Fetch financial statements via yfinance as fallback.
    yfinance exposes `.financials` (income), `.balance_sheet`, `.cashflow` (annual)
    Returns annualized dicts (in MILLION USD).
    """
    # Retrieve the original financial statements
    tk = yf.Ticker(ticker)
    income = tk.income_stmt
    bs = tk.balance_sheet
    cf = tk.cashflow

    # Data integrity check
    if income is None or income.empty:
        raise RuntimeError("yfinance returned no income statement - consider using SEC parser")

    # Helper: convert DataFrame → dict(year → line item)
    def df_to_annual_dict(df: pd.DataFrame):
        out = {}
        for col in df.columns:
            try:
                yr = pd.to_datetime(col).year
            except Exception:
                yr = str(col)
            vals = df[col].to_dict()
            # Convert all the numerical values to millions of dollars.
            vals_m = {}
            for k, v in vals.items():
                try:
                    if pd.notna(v):
                        vals_m[k] = float(v) / 1e6
                    else:
                        vals_m[k] = None
                except Exception:
                    vals_m[k] = None
            out[yr] = vals_m
        return out

    fin_data = {
        "income_annual": df_to_annual_dict(income),
        "bs_annual": df_to_annual_dict(bs),
        "cf_annual": df_to_annual_dict(cf),
        "shares": tk.info.get("sharesOutstanding", None),
        "market_cap": tk.info.get("marketCap", None),
        "longName": tk.info.get("longName", ticker)
    }

    # fallback: compute Free Cash Flow manually (also in millions)
    if "cf_annual" in fin_data and isinstance(fin_data["cf_annual"], dict):
        fcf_calc = {}
        for year, vals in fin_data["cf_annual"].items():
            op_cf = vals.get("TotalCashFromOperatingActivities") or vals.get("OperatingCashFlow")
            capex = vals.get("CapitalExpenditures")
            if op_cf is not None and capex is not None:
                fcf_calc[int(year)] = op_cf + capex  # both already in million, capex values are negative
        fin_data["fcf_manual"] = fcf_calc
    else:
        fin_data["fcf_manual"] = {}

    print("All the financial data have been converted into the unit of Million USD.") # Confirm successful execution
    return fin_data




# Metrics computation
def compute_time_series_metrics(fin_data: dict):
    """
    Organizing the scattered financial report data into a time series DataFrame,
    and automatically calculating a complete set of key financial indicators (profitability + growth + debt repayment ability) ,
    and serves as the core bridge connecting the "data scraping module" and the "valuation model".
    """
    years = sorted(fin_data["income_annual"].keys(), reverse=False)
    rows = []

    # Extract the core data for each year in sequence
    for yr in years:
        inc = fin_data["income_annual"].get(yr, {})
        cf = fin_data["cf_annual"].get(yr, {})
        bs = fin_data["bs_annual"].get(yr, {})

        # Try to suit all possible names to avoid missing context
        revenue = _pick_first_present(inc, [
            "Total Revenue", "TotalRevenue", "Revenue", "Revenues", "totalRevenue",
            "Total revenues", "NetRevenue", "OperatingRevenue", "Operating Revenue", "SalesRevenueNet"
        ])
        net_income = _pick_first_present(inc, ["NetIncome", "Net Income", "Net income", "NetIncomeLoss"])
        op_cf = _pick_first_present(cf, ["TotalCashFromOperatingActivities", "OperatingCashFlow", "Total cash from operating activities"])
        capex = _pick_first_present(cf, ["CapitalExpenditures", "Capital Expenditures", "Payments for capital expenditures"])
        total_assets = _pick_first_present(bs, ["TotalAssets", "Assets"])
        total_liab = _pick_first_present(bs, ["TotalLiab", "Total liabilities", "Liabilities"])
        equity = _pick_first_present(bs, ["TotalStockholderEquity", "Total stockholders' equity", "Stockholders' equity", "TotalEquity"])
        current_assets = _pick_first_present(bs, ["TotalCurrentAssets", "CurrentAssets"])
        current_liab = _pick_first_present(bs, ["TotalCurrentLiabilities", "CurrentLiabilities"])
        operating_income = _pick_first_present(inc, ["OperatingIncome", "Operating Income", "OperatingProfit"])

        # Convert all to float
        revenue = _to_float_or_nan(revenue)
        net_income = _to_float_or_nan(net_income)
        total_assets = _to_float_or_nan(total_assets)
        equity = _to_float_or_nan(equity)
        total_liab = _to_float_or_nan(total_liab)
        current_assets = _to_float_or_nan(current_assets)
        current_liab = _to_float_or_nan(current_liab)
        operating_income = _to_float_or_nan(operating_income)
        op_cf = _to_float_or_nan(op_cf)
        capex = _to_float_or_nan(capex)

        # Ratio sanity checks
        # Debt-to-Equity
        if equity and not math.isnan(equity) and equity != 0:
            debt_to_equity = total_liab / equity
            if debt_to_equity > 10:
                print(f"Abnormal Debt-to-Equity Ratio ({yr}): {debt_to_equity:.2f}, possibly unit mismatch. Reset to NaN")
                debt_to_equity = float("nan")
        else:
            debt_to_equity = float("nan")

        # Current Ratio
        if current_liab and not math.isnan(current_liab) and current_liab != 0:
            current_ratio = current_assets / current_liab
            if current_ratio > 10:
                print(f"Avnormal Current Ratio ({yr}): {current_ratio:.2f}，possibly unit mismatch. Reset to NaN")
                current_ratio = float("nan")
        else:
            current_ratio = float("nan")

        # Assemble row
        row = {
            "year": int(yr) if str(yr).isdigit() else str(yr),
            "revenue": revenue,
            "net_income": net_income,
            "op_cf": op_cf,
            "capex": capex,
            "net_margin": net_income / revenue if revenue else float("nan"),
            "operating_margin": operating_income / revenue if (revenue and not math.isnan(operating_income)) else float("nan"),
            "roe": net_income / equity if equity else float("nan"),
            "roa": net_income / total_assets if total_assets else float("nan"),
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "asset_turnover": revenue / total_assets if total_assets else float("nan"),
        }
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows).set_index("year").sort_index()

    # Compute growth and FCF
    df["revenue_yoy"] = df["revenue"].pct_change() # YoY Growth = (R(i)-R(i-1))/R(i-1)

    fcf_list = []
    # If CapEx is already a negative value (a cash outflow), then simply add it directly;
    # Otherwise, calculate the difference (to ensure logical consistency).
    for i, row in df.iterrows():
        op_cf = row["op_cf"]
        capex = row["capex"]
        try:
            if pd.notna(op_cf) and pd.notna(capex):
                if float(capex) < 0:
                    fcf = float(op_cf) + float(capex)
                else:
                    fcf = float(op_cf) - float(capex)
            else:
                fcf = np.nan
        except Exception:
            fcf = np.nan
        fcf_list.append(fcf)
    df["fcf"] = fcf_list

    # Facilitates the direct invocation of the valuation model
    try:
        latest_rev = float(df["revenue"].dropna().iloc[-1])
        df.attrs["revenue_latest"] = latest_rev
    except Exception:
        pass

    df["net_income_margin"] = df["net_income"] / df["revenue"]
    return df


    #画图比例
def plot_financial_ratios(metrics_df: pd.DataFrame, out_dir: Path):
    """
    Plot key financial ratios for visualization in the report.
    Generates PNG charts for profitability, leverage, liquidity, and efficiency.
    """
    ensure_dir(out_dir)

    ratio_groups = {
        "profitability": ["net_margin", "operating_margin", "roe", "roa"],
        "leverage_liquidity": ["debt_to_equity", "current_ratio"],
        "efficiency": ["asset_turnover"],
        "growth": ["revenue_yoy"]
    }

    for group, cols in ratio_groups.items():
        available_cols = [c for c in cols if c in metrics_df.columns and metrics_df[c].notna().any()]
        if not available_cols:
            continue

        plt.figure(figsize=(6, 3))
        for c in available_cols:
            plt.plot(metrics_df.index, metrics_df[c], marker="o", label=c)
        plt.title(f"{group.replace('_', ' ').title()} Ratios")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        p = out_dir / f"{group}_ratios.png"
        plt.savefig(p)
        plt.close()

    print("Financial ratio plots saved to:", out_dir)


#"Field fault-tolerance tool function"
# To search for multiple candidate keys in a dictionary in sequence
# and return the first one that exists and has a valid value.
def _pick_first_present(d: dict, keys):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return None

# To avoid failures due to the format of data
def _to_float_or_nan(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# DCF implementation: Run a robust 5-year DCF valuation model

def run_dcf(fcf_hist: pd.Series,
            shares_outstanding: float,
            short_term_growths=None,
            wacc: float = 0.08,
            terminal_g: float = 0.025,
            terminal_method: str = "gordon",
            net_cash: float = None):

    #Robust DCF:
    #- fcf_hist: pandas Series (index=year asc or desc, values numeric)
    #- shares_outstanding: float
    #- short_term_growths: list length 5 or None (defaults provided)
    #- net_cash: optional (cash - debt) to add to equity value

    if short_term_growths is None:
        # The assumed declining growth path of FCF over the next five years in this model is based on the historical CF growth trends of large technology companies in the industry:
        # the mature growth period gradually transitions from a high growth rate to a rate close to the overall growth rate of the economy.
        # Taking Alphabet Inc. (Google) as an example, its annual FCF growth rate in recent years has reached as high as double digits (≈15%);
        # while Microsoft Corporation's average annual FCF growth rate over the past five years was approximately 10%.
        short_term_growths = [0.10, 0.08, 0.06, 0.05, 0.04]
    if len(short_term_growths) != 5:
        raise ValueError("short_term_growths must be length 5")

    # ensure fcf_hist is a Series
    if not isinstance(fcf_hist, pd.Series):
        try:
            fcf_hist = pd.Series(fcf_hist)
        except Exception:
            fcf_hist = pd.Series(dtype=float)

    last_valid = fcf_hist.dropna().sort_index()
    last_fcf = None

    if not last_valid.empty:
        # prefer most recent non-nan
        try:
            last_fcf = float(last_valid.iloc[-1])
        except Exception:
            last_fcf = None

    if last_fcf is None:
        # Try to use the average value for the last three years, if it exists.
        try:
            last_3 = last_valid.dropna().iloc[-3:]
            if not last_3.empty:
                last_fcf = float(last_3.mean())
        except Exception:
            last_fcf = None

    if last_fcf is None:
        # Try to retrieve the "revenue" from "series attrs"
        # and multiply it by the assumed margin.
        rev = None
        try:
            rev = fcf_hist.attrs.get("revenue_latest", None)
        except Exception:
            rev = None
        if rev:
            last_fcf = float(rev) * 0.25 # The average FCF Margin of Google over the past 5 years has been approximately 25%
                                         # (26% in 2023, 23% in 2022, and 28% in 2021).
        else:
            # A small value is set as a backup to avoid None.
            last_fcf = 1e9
        print("run_dcf: used fallback last_fcf =", last_fcf)

    # Generate the forcast values of FCF in 5 years
    fcfs = []
    f = last_fcf
    for g in short_term_growths:
        f = f * (1 + g)
        fcfs.append(f)

    # terminal ( Gordon Growth Method)
    if terminal_method == "gordon":
        if wacc <= terminal_g:
            # Protection, avoidance of division by zero or negative values
            terminal_value = fcfs[-1] * 10.0
        else:
            terminal_value = fcfs[-1] * (1 + terminal_g) / (wacc - terminal_g)
    else:
        terminal_value = fcfs[-1] * 10.0

    # discount
    pv = 0.0
    for i, c in enumerate(fcfs, start=1):
        pv += c / ((1 + wacc) ** i)
    pv += terminal_value / ((1 + wacc) ** len(fcfs))

    equity_value = pv
    if net_cash is not None:
        equity_value += float(net_cash)

    implied_price = None
    if shares_outstanding and shares_outstanding > 0:
        implied_price = equity_value / shares_outstanding

    result = {
        "last_fcf": float(last_fcf),
        "projected_fcfs": [float(x) for x in fcfs],
        "terminal_value": float(terminal_value),
        "pv_total": float(pv),
        "equity_value": float(equity_value),
        "implied_price": float(implied_price) if implied_price is not None else None,
        "wacc_used": wacc,
        "terminal_g": terminal_g,
        "assumptions": {"short_term_growths": short_term_growths}
    }

    # sensitivity grid for WACC and Gordon
    # These two parameters have the greatest impact on the valuation.
    # Therefore, we need to examine how the intrinsic value would change
    # if we slightly increase or decrease these parameters.
    wacc_grid = np.round(np.linspace(max(0.03, wacc - 0.02), wacc + 0.04, 5), 4)
    g_grid = np.round(np.linspace(max(0.0, terminal_g - 0.01), terminal_g + 0.02, 5), 4)
    sens = {}
    for wa in wacc_grid:
        row = {}
        for g in g_grid:
            if wa <= g + 1e-6:
                row[str(g)] = None
            else:
                tv = fcfs[-1] * (1 + g) / (wa - g)
                pv_local = 0.0
                for i, c in enumerate(fcfs, start=1):
                    pv_local += c / ((1 + wa) ** i)
                pv_local += tv / ((1 + wa) ** len(fcfs))
                if shares_outstanding and shares_outstanding > 0:
                    row[str(g)] = pv_local / shares_outstanding
                else:
                    row[str(g)] = None
        sens[str(wa)] = row
    result["sensitivity"] = sens
    return result



# Simple multiples (peer fallback)

def run_relative_valuation(revenue_ttm, peer_ev_revenue=5.0, shares_outstanding=None):

    # Used as a market-based cross-check to complement the DCF model.
    # A minimal EV/Revenue multiple valuation.
    # - peer_ev_revenue: peer median EV/Revenue
    # - revenue_ttm: latest trailing twelve month revenue (in same units)

    ev = revenue_ttm * peer_ev_revenue
    implied_price = None
    if shares_outstanding and shares_outstanding > 0:
        implied_price = ev / shares_outstanding
    return {"peer_ev_revenue": peer_ev_revenue, "ev": float(ev), "implied_price": float(implied_price) if implied_price is not None else None}


# Memo generation (LLM or template)

from openai import OpenAI
import json

def generate_memo_facts(ticker, metrics_df: pd.DataFrame, dcf_res: dict, rel_res: dict, meta: dict):
    #Assemble facts dict used to feed LLM or template
    latest_year = metrics_df.index.max()

    def safe_float(val):
        try:
            return float(val) if pd.notna(val) and val is not None else None
        except Exception:
            return None

    fcf_val = None
    if latest_year in metrics_df.index and "fcf" in metrics_df.columns:
        fcf_val = safe_float(metrics_df.loc[latest_year, "fcf"])
        # If it is empty, then attempt to retrieve it from the fcf_manual field of the meta.
        if fcf_val is None and "fcf_manual" in meta:
            manual_fcf = meta["fcf_manual"].get(int(latest_year))
            fcf_val = safe_float(manual_fcf)

    facts = {
        "ticker": ticker,
        "company_name": meta.get("longName") or ticker,
        "latest_year": str(latest_year),
        "revenue_latest": safe_float(metrics_df.loc[latest_year, "revenue"]) if latest_year in metrics_df.index else None,
        "net_income_latest": safe_float(metrics_df.loc[latest_year, "net_income"]) if latest_year in metrics_df.index else None,
        "fcf_latest": fcf_val,
        "implied_price_dcf": dcf_res.get("implied_price"),
        "implied_price_relative": rel_res.get("implied_price"),
        "assumptions": dcf_res.get("assumptions"),
    }

    return facts


def template_generate_memo(facts: dict, metrics_df: pd.DataFrame = None):
    # Enhanced investment memo generator with ratio commentary. No external LLM required.

    lines = []
    lines.append(f"{facts.get('company_name')} ({facts.get('ticker')}) — Investment Memo\n")

    # 1. Investment Thesis
    lines.append("Investment Thesis:")
    r = facts.get("revenue_latest")
    fcf = facts.get("fcf_latest")
    dcf_p = facts.get("implied_price_dcf")
    rel_p = facts.get("implied_price_relative")

    lines.append(f"- Large-scale leader in digital advertising and cloud computing; LTM revenue ≈ {fmt_n(r)} (source: Facts).")
    if fcf:
        lines.append(f"- Strong free cash flow generation: ≈ {fmt_n(fcf)} last fiscal year.")
    else:
        lines.append("- Positive operating cash flow trend supports reinvestment capacity.")
    lines.append(f"- Valuation: DCF implied price = {fmt_n(dcf_p)}; Relative (EV/Rev) implied = {fmt_n(rel_p)}. (methods: DCF & EV/Revenue).")

    # 2. Financial Performance Analysis
    if metrics_df is not None:
        latest_year = metrics_df.index.max()
        ratios = {}
        for key in ["net_margin", "roe", "roa", "debt_to_equity", "current_ratio"]:
            if key in metrics_df.columns:
                val = metrics_df.loc[latest_year, key]
                ratios[key] = round(val * 100, 2) if not pd.isna(val) else None

        lines.append("\nFinancial Performance Overview:")
        nm = ratios.get("net_margin")
        roe = ratios.get("roe")
        roa = ratios.get("roa")
        debt_eq = ratios.get("debt_to_equity")
        curr = ratios.get("current_ratio")

        lines.append(f"- Profitability: Net margin of {nm}% and ROE of {roe}%, reflecting solid earnings quality." if nm and roe else "- Profitability stable over recent years.")
        if roa:
            lines.append(f"- Efficiency: Return on Assets (ROA) around {roa}%, suggesting effective asset utilization.")
        if debt_eq:
            lines.append(f"- Leverage: Debt-to-Equity ratio at {round(debt_eq,2)}x, indicating balanced capital structure.")
        if curr:
            lines.append(f"- Liquidity: Current ratio of {round(curr,2)}x, implying adequate short-term solvency.")

    # 3. Growth Drivers
    lines.append("\nKey Growth Drivers:")
    lines.append("- Expansion of Google Cloud and enterprise AI services.")
    lines.append("- Continued dominance in search and advertising.")
    lines.append("- Operational leverage and ongoing share repurchases.")

    # 4. Top Risks
    lines.append("\nTop Risks:")
    lines.append("- Regulatory and antitrust scrutiny in the US and EU.")
    lines.append("- Rising competition in AI and cloud infrastructure.")
    lines.append("- Slower ad market recovery or shifts in digital spending.")

    # 5. Catalysts
    lines.append("\nCatalysts:")
    lines.append("- Accelerating profitability in Google Cloud and AI monetization.")
    lines.append("- Regulatory clarity or favorable capital return policies.")

    return "\n".join(lines)

def llm_generate_memo_openai(facts: dict, model="gpt-4o-mini"):

    # Use OpenAI official SDK (v1.x) to generate a memo.
    # Compatible with `from openai import OpenAI`.

    api_key = read_env_openai_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    # Initialize the new SDK client
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a buy-side equity analyst. Write a concise professional investment memo "
        "using only the provided facts (financials, valuation). Use structured, formal tone."
    )

    facts_json = json.dumps(facts, indent=2)
    user_prompt = f"""
    Produce a concise one-page (~300-400 words) investment memo for {facts.get('company_name')} ({facts.get('ticker')}).

    Structure:
    1) Investment thesis (3 bullets)
    2) Valuation summary (include DCF and relative)
    3) Key growth drivers (3 bullets)
    4) Top 3 risks
    5) Catalysts

    Facts:
    {facts_json}
    """

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("！！！OpenAI call failed inside llm_generate_memo_openai:", str(e))
        return None

def fmt_n(x):
    try:
        if x is None:
            return "N/A"
        # if it's a small number but possibly large currency, format in billions if >1e9
        x = float(x)
        if math.isnan(x):
            return "N/A"
        if abs(x) >= 1e9:
            return f"${x/1e9:,.2f}B"
        if abs(x) >= 1e6:
            return f"${x/1e6:,.2f}M"
        return f"${x:,.2f}"

    except Exception:
        return str(x)


# Plot helpers

def plot_timeseries_metrics(metrics_df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    # Revenue
    try:
        revenue = metrics_df["revenue"].dropna()
        if not revenue.empty:
            plt.figure(figsize=(6,3))
            revenue.plot(marker='o')
            plt.title("Revenue (annual)")
            plt.ylabel("USD")
            plt.tight_layout()
            p = out_dir / "revenue.png"
            plt.savefig(p)
            plt.close()
    except Exception:
        pass
    # FCF
    try:
        fcf = metrics_df["fcf"].dropna()
        if not fcf.empty:
            plt.figure(figsize=(6,3))
            fcf.plot(marker='o')
            plt.title("Free Cash Flow (annual)")
            plt.tight_layout()
            p = out_dir / "fcf.png"
            plt.savefig(p)
            plt.close()
    except Exception:
        pass


# Main pipeline
# Function: Automatically execute the complete process from data collection
# → financial analysis → valuation modeling → chart visualization → report generation

def pipeline(ticker: str, start: str, end: str, out: Path, model_choice: str = "local"):
    out = Path(out) / ticker
    ensure_dir(out)
    ts = datetime.utcnow().isoformat()
    meta = {"run_timestamp": ts, "ticker": ticker}

    print(f"[1/6] Fetching market data for {ticker}...")
    market = fetch_market_data(ticker, start, end)
    market.to_csv(out / "market_data.csv")

    print(f"[2/6] Fetching financials (yfinance fallback)...")
    fin = fetch_financials_yfinance(ticker)
    save_json(fin, out / "raw_fin_yfinance.json")

    print(f"[3/6] Computing metrics...")
    metrics_df = compute_time_series_metrics(fin)
    metrics_df.to_csv(out / "metrics_timeseries.csv")
    metrics_summary = metrics_df.tail(5).to_dict(orient="index")
    save_json(metrics_summary, out / "metrics_summary.json")

    print(f"[4/6] Running DCF and relative valuation...")
    shares = fin.get("shares")
    if shares is None or shares == 0:
        last_close = market["Close"].iloc[-1]
        market_cap = fin.get("market_cap")
        if market_cap and last_close:
            try:
                shares = market_cap / last_close
            except Exception:
                shares = None

    # Obtain the FCF sequence
    fcf_series = metrics_df["fcf"].dropna().astype(float)
    try:
        fcf_series.attrs["revenue_latest"] = metrics_df["revenue"].dropna().iloc[-1]
    except Exception:
        pass

    # If the FCF data is empty, try to calculate it from the cash flow statement.

    if fcf_series.empty:
        print("No FCF was found. Attempting to calculate manually...")
        try:
            cashflow = fin.get("cf_annual", {})
            fcf_data = {}
            for year, vals in cashflow.items():
                op_cf = (vals.get("Operating Cash Flow")
                         or vals.get("TotalCashFromOperatingActivities")
                         or vals.get("Cash Flow From Continuing Operating Activities")
                         or vals.get("OperatingCashFlow")
                         or vals.get("Free Cash Flow")

                )
                capex = (
                        vals.get("Capital Expenditure")
                        or vals.get("CapitalExpenditures")
                        or vals.get("Net PPE Purchase And Sale")
                        or vals.get("Purchase Of PPE")
                )
                if op_cf is not None and capex is not None:
                    fcf_data[int(year)] = op_cf + capex  # CapEx 通常是负值
            fcf_series = pd.Series(fcf_data).sort_index()
            print("The free cash flow has been successfully calculated from the cash flow statement.")
            print("shares_outstanding:", shares)
            print("FCF series is as follow：")
            print(fcf_series)
        except Exception as e:
            print("Can not obtain FCF:", e)
            fcf_series = pd.Series(dtype=float)


    # If it remains empty, the process will not terminate
    # but instead continue to generate the report (with DCF set to NaN)
    if fcf_series.empty:
        print("Warning: Unable to obtain FCF data. DCF valuation will be skipped and only relative valuation will be output.")


        dcf_res = {"implied_price": None, "note": "No FCF data available"}
    else:
        dcf_res = {"implied_price": None}


    # Obtain net cash (cash and short-term investments - total debt)
    try:
        bs = fin.get("bs_annual", {})
        latest_year = max(bs.keys()) if isinstance(bs, dict) and len(bs) > 0 else None
        if latest_year and isinstance(bs[latest_year], dict):
            cash = (
                bs[latest_year].get("Cash Cash Equivalents And Short Term Investments")
                or bs[latest_year].get("CashAndCashEquivalents")
                or bs[latest_year].get("End Cash Position")
                or 0
            )
            debt = (
                bs[latest_year].get("Total Debt")
                or bs[latest_year].get("Long Term Debt")
                or 0
            )
            net_cash = cash - debt
            print(f"Net Cash: {net_cash:,.0f}")
        else:
            net_cash = 0
    except Exception as e:
        print("Unable to extract net cash information:", e)
        net_cash = 0

    # When invoking DCF, pass the net cash to adjust the valuation.
    # Invoke DCF and relative valuation (both restored to US dollar units)
    if not fcf_series.empty:
        # Convert millions of units back to US dollars units
        fcf_usd = fcf_series * 1e6
        dcf_res = run_dcf(fcf_usd, shares_outstanding=shares, wacc=0.08)
        if dcf_res.get("implied_price"):
            dcf_res["implied_price"] += net_cash / shares  # Increase in net cash value per share
            print(f"The per-share DCF valuation has been adjusted to take into account the net cash contribution.：+{net_cash / shares:.2f}")
    else:
        dcf_res = {"implied_price": None, "note": "No FCF data available"}

    # Similarly: Relative valuation has also returned to the dollar unit
    latest_revenue_million = metrics_df["revenue"].iloc[-1] if "revenue" in metrics_df.columns else None
    revenue_usd = latest_revenue_million * 1e6 if latest_revenue_million is not None else None

    rel_res = run_relative_valuation(
        revenue_ttm=revenue_usd,
        peer_ev_revenue=5.0,
        shares_outstanding=shares
    )

    save_json(dcf_res, out / "dcf_results.json")
    save_json(rel_res, out / "relative_results.json")

    print(f"[5/6] Creating plots...")
    plot_timeseries_metrics(metrics_df, out)

    print(f"[6/6] Generating memo...")
    facts = generate_memo_facts(ticker, metrics_df, dcf_res, rel_res, fin)
    save_json(facts, out / "memo_facts.json")

    memo_text = None
    # Check the model selection and API key
    if model_choice.lower().startswith("openai") or read_env_openai_key():
        try:
            print("Calling OpenAI for memo generation (ensure OPENAI_API_KEY is set)...")
            memo_text = llm_generate_memo_openai(facts, model="gpt-4o-mini")
            if memo_text:
                print("✅ LLM memo generated successfully!")
            else:
                print("⚠️ LLM returned empty text, falling back to template.")
                memo_text = template_generate_memo(facts, metrics_df)

        except Exception as e:
            print("⚠️ OpenAI call failed:", str(e))
            print("Falling back to template memo.")
            memo_text = template_generate_memo(facts, metrics_df)
    else:
        print("Using template memo (no OpenAI model selected).")
        memo_text = template_generate_memo(facts, metrics_df)

    # write in file
    with open(out / "investment_memo.txt", "w", encoding="utf-8") as f:
        f.write(memo_text)

    print("✅ Pipeline complete. Outputs saved to:", out)
    summary = {
        "ticker": ticker,
        "run_timestamp": ts,
        "latest_close": float(market["Close"].iloc[-1]),
        "implied_price_dcf": dcf_res.get("implied_price"),
        "implied_price_relative": rel_res.get("implied_price")
    }
    save_json(summary, out / "run_summary.json")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Run demo Fundamental Agent (minimal).")
    p.add_argument("--ticker", type=str, default="GOOGL", help="Ticker，e.g.,GOOGL")
    p.add_argument("--start", type=str, default="2020-01-01", help="Fixed start date")
    p.add_argument("--end", type=str, default="2025-01-01", help="Fixed end date")
    p.add_argument("--out", type=str, default="outputs", help="Output folder root")
    p.add_argument("--model", type=str, default="local", help="Model: 'local' or 'openai'")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline(args.ticker, args.start, args.end, args.out, model_choice=args.model)

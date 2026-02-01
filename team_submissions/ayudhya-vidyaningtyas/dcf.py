# DCF Valuation
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "rating_cap": "AA+",
    "base_erp": 0.0447,
    "effective_tax": 0.162,      # For NOPAT
    "marginal_tax": 0.21,        # For WACC shield
}

def _num(x, default=np.nan):
    if x is None: return default
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
            if x.endswith("%"): return float(x[:-1]) / 100.0
        if pd.isna(x): return default
        return float(x)
    except: return default

def _safe_float(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else float(default)
    except: return float(default)

def _first_existing(row: pd.Series, keys, default=np.nan):
    for k in keys:
        if k in row.index:
            v = _num(row.get(k), default=default)
            if not pd.isna(v): return v
    return default

def _annual_years(df: pd.DataFrame):
    return sorted([y for y in df.index if isinstance(y, (int, np.integer))])

def _latest_annual_year_in(df: pd.DataFrame):
    ys = _annual_years(df)
    return ys[-1] if ys else df.index[0]

def _cash_and_sti(row: pd.Series) -> tuple:
    cash = _first_existing(row, ["cashAndCashEquivalentsAtCarryingValue", "cashAndCashEquivalents"], default=np.nan)
    sti = _first_existing(row, ["shortTermInvestments"], default=np.nan)
    if np.isnan(cash) and np.isnan(sti):
        csti = _first_existing(row, ["cashAndShortTermInvestments"], default=0.0)
        return float(csti), 0.0
    return float(0.0 if np.isnan(cash) else cash), float(0.0 if np.isnan(sti) else sti)

class RiskFreeRateFetcher:
    @staticmethod
    def _tnx_to_decimal(x: float) -> float:
        x = float(x)
        if x < 1: return x
        if x < 20: return x / 100.0
        return x / 1000.0

    @staticmethod
    def get_10y_treasury(months=12, use_average=True) -> dict:
        try:
            tnx = yf.Ticker("^TNX")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(months * 30))
            hist = tnx.history(start=start_date, end=end_date)
            if hist.empty: return {"chosen": 0.0430}
            current = RiskFreeRateFetcher._tnx_to_decimal(hist["Close"].iloc[-1])
            avg = RiskFreeRateFetcher._tnx_to_decimal(hist["Close"].mean())
            return {"chosen": avg if use_average else current}
        except: return {"chosen": 0.0430}

class MarketDataFetcher:
    @staticmethod
    def fetch_beta(ticker: str) -> tuple:
        try:
            stock = yf.Ticker(ticker)
            raw_beta = stock.info.get("beta")
            raw_beta = 1.0 if raw_beta is None else float(raw_beta)
        except: raw_beta = 1.0
        adjusted_beta = (2.0/3.0) * raw_beta + (1.0/3.0) * 1.0
        return raw_beta, adjusted_beta

class InterestExpenseFetcher:
    @staticmethod
    def _pick_row_by_keywords(df, keywords):
        if df is None or df.empty: return None
        idx = [str(i).strip() for i in df.index]
        for kw in keywords:
            for i, name in enumerate(idx):
                if kw in name.lower(): return df.iloc[i]
        return None

    @staticmethod
    def get_ttm_interest_and_ebit_yf(ticker: str) -> dict:
        try:
            tk = yf.Ticker(ticker)
            q = tk.quarterly_financials
            if q is None or q.empty or q.shape[1] < 4: return {"ok": False}
            q = q.copy()
            q.columns = [pd.to_datetime(c) for c in q.columns]
            q = q.sort_index(axis=1)
            last4 = q.iloc[:, -4:]
            
            interest_row = InterestExpenseFetcher._pick_row_by_keywords(last4, ["interest expense", "interestexpense"])
            ebit_row = InterestExpenseFetcher._pick_row_by_keywords(last4, ["operating income", "ebit"])
            
            interest = np.nan
            if interest_row is not None:
                interest = abs(float(np.nansum(pd.to_numeric(interest_row, errors="coerce").values)))
            
            ebit = np.nan
            if ebit_row is not None:
                ebit = float(np.nansum(pd.to_numeric(ebit_row, errors="coerce").values))
            
            ok = (np.isfinite(interest) and interest > 0 and np.isfinite(ebit))
            return {"interest": interest, "ebit": ebit, "ok": ok}
        except: return {"ok": False}

def _rating_table():
    return [(8.5, "AAA", 0.0060), (6.5, "AA", 0.0075), (5.5, "A+", 0.0090),
            (4.25, "A", 0.0110), (3.0, "A-", 0.0130), (2.5, "BBB+", 0.0160),
            (2.0, "BBB", 0.0190), (1.75, "BBB-", 0.0230), (1.5, "BB+", 0.0280),
            (1.25, "BB", 0.0330), (1.0, "BB-", 0.0410), (0.8, "B+", 0.0520),
            (0.65, "B", 0.0620), (0.5, "B-", 0.0750), (0.2, "CCC", 0.0950)]

def estimate_rd_synthetic_icr(ticker, income, balance, rf, rating_cap):
    yf_ttm = InterestExpenseFetcher.get_ttm_interest_and_ebit_yf(ticker)
    if yf_ttm["ok"]:
        ebit, interest = yf_ttm["ebit"], yf_ttm["interest"]
    else:
        year = _latest_annual_year_in(income)
        row = income.loc[year]
        ebit = _first_existing(row, ["ebit", "operatingIncome"], default=0.0)
        ie = _first_existing(row, ["interestExpense", "interestAndDebtExpense"], default=np.nan)
        interest = abs(_num(ie, default=np.nan))
    
    icr = (ebit / interest) if (np.isfinite(interest) and interest > 0) else 100.0
    rating, spread = "BBB", 0.0190
    for cutoff, r, s in _rating_table():
        if icr >= cutoff: rating, spread = r, s; break
    
    cap_spread = 0.0
    for _, r, s in _rating_table():
        if r == rating_cap: cap_spread = s
    if spread < cap_spread: rating, spread = rating_cap, cap_spread
    
    return {"rd": max(rf + spread, 0.0), "rating": rating, "icr": icr, "spread": spread}

class AssumptionEngine:
    def __init__(self, downloader):
        self.downloader = downloader
        self.income = downloader.income_statement
        self.balance = downloader.balance_sheet
        self.cashflow = downloader.cash_flow
        self.hist_ebitda_margin = 0.0
        self.hist_tax_rate = 0.162
        self.hist_capex_ratio = 0.12
        self.hist_da_ratio = 0.05

    def operating_nwc_dollars(self, period_label) -> float:
        row = self.balance.loc[period_label]
        tca = _first_existing(row, ["totalCurrentAssets"], default=0)
        cash, sti = _cash_and_sti(row)
        op_ca = tca - cash - sti
        tcl = _first_existing(row, ["totalCurrentLiabilities"], default=0)
        std = _first_existing(row, ["shortTermDebt"], default=0)
        cpltd = _first_existing(row, ["currentPortionOfLongTermDebt"], default=0)
        op_cl = tcl - std - cpltd
        return float(op_ca - op_cl)

    def calculate_historicals(self):
        common_periods = [p for p in self.income.index if p in self.balance.index]
        recent = sorted(common_periods, key=lambda x: str(x), reverse=True)[:3]
        
        nwc_ratios, margin_ratios, da_ratios, tax_ratios, capex_ratios = [], [], [], [], []
        
        print("\n" + "="*80)
        print("HISTORICAL ANALYSIS (3-Year Averages)")
        print("="*80)
        print(f"{'Period':<12} {'Revenue':>10} {'NWC %':>8} {'EBIT %':>8} {'D&A %':>8} {'Tax %':>8} {'Capex %':>8}")
        print("-" * 80)
        
        for p in recent:
            row_income = self.income.loc[p]
            rev = _num(row_income.get("totalRevenue"), default=0)
            if rev == 0: continue
            
            ebit = _num(row_income.get("ebit"), default=0)
            if ebit == 0: ebit = _num(row_income.get("operatingIncome"), default=0)
            
            da = _num(row_income.get("depreciationAndAmortization"), default=0)
            if da == 0 and p in self.cashflow.index:
                cf_row = self.cashflow.loc[p]
                da = _num(cf_row.get("depreciationAndAmortization"), default=0)
                if da == 0: da = _num(cf_row.get("depreciation"), default=0)
            
            tax_exp = _num(row_income.get("incomeTaxExpense"), default=0)
            pretax = _num(row_income.get("incomeBeforeTax"), default=0)
            if pretax == 0: pretax = ebit
            tax_rate = tax_exp / pretax if pretax > 0 else 0.21
            if tax_rate < 0.05 or tax_rate > 0.50: tax_rate = 0.21
            
            capex = 0
            if p in self.cashflow.index:
                cf_row = self.cashflow.loc[p]
                capex = abs(_num(cf_row.get("capitalExpenditures"), default=0))
            
            nwc = self.operating_nwc_dollars(p)
            
            nwc_ratios.append(nwc / rev)
            margin_ratios.append(ebit / rev)
            da_ratios.append(da / rev)
            tax_ratios.append(tax_rate)
            capex_ratios.append(capex / rev)
            
            print(f"{str(p):<12} ${rev/1e9:>9.1f}B {nwc/rev:>8.1%} {ebit/rev:>8.1%} {da/rev:>8.1%} {tax_rate:>8.1%} {capex/rev:>8.1%}")
        
        avg_nwc = float(np.mean(nwc_ratios)) if nwc_ratios else 0.0
        avg_margin = float(np.mean(margin_ratios)) if margin_ratios else 0.30
        avg_da = float(np.mean(da_ratios)) if da_ratios else 0.05
        avg_tax = float(np.mean(tax_ratios)) if tax_ratios else 0.162
        avg_capex = float(np.mean(capex_ratios)) if capex_ratios else 0.12
        avg_tax = max(avg_tax, 0.15)
        
        print("-" * 80)
        print(f"Avg → NWC: {avg_nwc:.1%} | EBIT: {avg_margin:.1%} | D&A: {avg_da:.1%} | Tax: {avg_tax:.1%} | Capex: {avg_capex:.1%}")
        print("="*80)
        
        return avg_nwc, avg_margin, avg_da, avg_tax, avg_capex

    def create_scenarios(self) -> dict:
        hist_nwc, hist_margin, hist_da, hist_tax, hist_capex = self.calculate_historicals()
        self.hist_ebitda_margin = hist_margin + hist_da
        self.hist_tax_rate = hist_tax
        self.hist_capex_ratio = hist_capex
        self.hist_da_ratio = hist_da
        
        # Separate AI build capex from steady-state
        # Steady-state capex = D&A + incremental growth capex
        # For tech: typically D&A + 2-3% of revenue
        steady_state_capex = hist_da + 0.025  # D&A + 2.5% growth capex
        
        print(f"\nCAPEX FRAMEWORK:")
        print(f"  Historical Capex:     {hist_capex:.1%}")
        print(f"  Historical D&A:       {hist_da:.1%}")
        print(f"  Steady-State Target:  {steady_state_capex:.1%} (D&A + 2.5%)")
        print(f"  → AI Build Premium:   {hist_capex - steady_state_capex:.1%}\n")
        
        return {
            "Bear": {
                "name": "Bear",
                "erp_shift": +0.005,
                
                # Growth
                "growth_y1": 0.09, "growth_y5": 0.07, "growth_y10": 0.03, "terminal_g": 0.03,
                
                # Profitability
                "ebitda_margin": max(self.hist_ebitda_margin - 0.02, 0.20),
                "ebitda_margin_improve": 0.0,
                
                # NWC
                "nwc_pct": hist_nwc + 0.01,
                
                # D&A
                "da_pct": hist_da,
                
                # CAPEX: AI build cycle (Y1-5) → Steady-state (Y6-10) → Mature terminal
                "capex_ai_build": hist_capex + 0.02,           # 14% (elevated AI spend)
                "capex_transition": steady_state_capex + 0.02, # 9.5% (normalizing)
                "capex_terminal": steady_state_capex + 0.01,   # 8.5% (mature)
                
                # Tax
                "tax_forecast": hist_tax + 0.02, 
            },
            
            "Base": {
                "name": "Base",
                "erp_shift": 0.00,
                
                # Growth
                "growth_y1": 0.13, "growth_y5": 0.11, "growth_y10": 0.035, "terminal_g": 0.035,
                
                # Profitability
                "ebitda_margin": self.hist_ebitda_margin,
                "ebitda_margin_improve": 0.001,
                
                # NWC
                "nwc_pct": hist_nwc,
                "nwc_terminal": hist_nwc * 0.75,  # More efficient in maturity
                
                # D&A
                "da_pct": hist_da,
                
                # CAPEX: AI build (Y1-5) → Transition (Y6-9) → Mature (Y10 & terminal)
                "capex_ai_build": hist_capex,                  # 12% (current reality)
                "capex_transition": steady_state_capex + 0.01, # 8.5% (normalizing)
                "capex_terminal": steady_state_capex,          # 7.5% (D&A + growth capex)
                
                # Tax
                "tax_forecast": hist_tax,  # NOPAT only
            },
            
            "Bullish": {
                "name": "Bull",
                "erp_shift": -0.005,
                
                # Growth
                "growth_y1": 0.15, "growth_y5": 0.13, "growth_y10": 0.0375, "terminal_g": 0.0375,
                
                # Profitability
                "ebitda_margin": self.hist_ebitda_margin + 0.01,
                "ebitda_margin_improve": 0.0015,
                
                # NWC
                "nwc_pct": hist_nwc - 0.01,
                "nwc_terminal": hist_nwc * 0.7,  # Very efficient
                
                # D&A
                "da_pct": hist_da,
                
                # CAPEX: Efficient AI build → Fast normalization
                "capex_ai_build": hist_capex,                  # 12% (current reality)
                "capex_transition": steady_state_capex,        # 7.5% (quick normalize)
                "capex_terminal": steady_state_capex - 0.01,   # 6.5% (very efficient)
                
                # Tax
                "tax_forecast": hist_tax,
            },
        }

class DCFModel:
    def __init__(self, downloader, use_avg_treasury=True):
        self.downloader = downloader
        self.ticker = str(downloader.ticker).upper().strip()
        self.income = downloader.income_statement
        self.balance = downloader.balance_sheet
        self.use_avg_treasury = use_avg_treasury
        md = getattr(downloader, "market_data", {})
        self.market_cap = _safe_float(md.get("market_cap", 0))
        self.current_price = _safe_float(md.get("current_price", 0))
        self.shares_outstanding = _safe_float(md.get("shares_outstanding", 0))

    def _latest_period_label(self, df):
        for lab in ["2025 (TTM)", "TTM", "Trailing Twelve Months"]:
            if lab in df.index: return lab
        yrs = [y for y in df.index if isinstance(y, (int, np.integer))]
        return max(yrs) if yrs else df.index[0]

    def _net_debt(self):
        lab = self._latest_period_label(self.balance)
        row = self.balance.loc[lab]
        debt = _first_existing(row, ["shortLongTermDebtTotal", "totalDebt"], default=0)
        cash, sti = _cash_and_sti(row)
        return float(debt - (cash + sti))
    
    def _gross_debt(self):
        lab = self._latest_period_label(self.balance)
        row = self.balance.loc[lab]
        debt = _first_existing(row, ["shortLongTermDebtTotal", "totalDebt"], default=0)
        return float(debt)

    def calculate_wacc(self, erp, marginal_tax=None):
        # IMPROVEMENT #3: Clear separation of tax rates
        beta_raw, beta_adj = MarketDataFetcher.fetch_beta(self.ticker)
        rf = float(RiskFreeRateFetcher.get_10y_treasury(use_average=self.use_avg_treasury)["chosen"])
        re = rf + beta_adj * erp
        credit = estimate_rd_synthetic_icr(self.ticker, self.income, self.balance, rf, CONFIG["rating_cap"])
        rd = credit["rd"]
        
        # Use gross debt for WACC weights
        gross_debt = self._gross_debt()
        E = self.market_cap
        V = E + gross_debt
        
        # Marginal tax for shield (statutory 21%)
        tax_shield = marginal_tax if marginal_tax is not None else CONFIG["marginal_tax"]
        
        wacc = (E/V) * re + (gross_debt/V) * rd * (1 - tax_shield) if V > 0 else re
        
        self.wacc_components = {
            "rf": rf, "beta_raw": beta_raw, "beta_adj": beta_adj, "re": re, "rd": rd,
            "rating": credit["rating"], "icr": credit["icr"],
            "E": E, "D": gross_debt, "V": V,
            "tax_shield": tax_shield,  # For WACC
            "wacc": wacc
        }
        return wacc

    def _project_10y(self, s, base_rev, base_op_nwc):
        # IMPROVEMENT #1 & #2: Phased capex with mature terminal economics
        rows = []
        for t in range(1, 11):
            # Revenue growth (interpolated)
            w = (t-1) / 4.0 if t <= 5 else (t-5) / 5.0
            g = (1-w) * s["growth_y1"] + w * s["growth_y5"] if t <= 5 else (1-w) * s["growth_y5"] + w * s["growth_y10"]
            rev = base_rev * (1 + g) if t == 1 else rows[-1]["Revenue"] * (1 + g)
            
            # Profitability
            margin = np.clip(s["ebitda_margin"] + s["ebitda_margin_improve"] * t, 0, 0.60)
            ebitda = rev * margin
            da = rev * s["da_pct"]
            ebit = ebitda - da
            
            # NOPAT (uses effective tax)
            nopat = ebit * (1 - s["tax_forecast"])
            
            # CAPEX PHASING: AI build (Y1-5) → Transition (Y6-9) → Terminal (Y10)
            if t <= 5:
                capex_pct = s["capex_ai_build"]
            elif t < 10:
                # Linear transition from Y6 to Y9
                fade_pct = (t - 5) / 4.0
                capex_pct = (1 - fade_pct) * s["capex_ai_build"] + fade_pct * s["capex_transition"]
            else:
                # Y10 uses terminal capex (matches Gordon growth assumption)
                capex_pct = s["capex_terminal"]
            
            capex = max(rev * capex_pct, da)
            
            # NWC: Transition to mature level in terminal year
            if "nwc_terminal" in s and t == 10:
                nwc_pct = s["nwc_terminal"]  # Mature NWC
            else:
                nwc_pct = s["nwc_pct"]
            
            nwc_this = rev * nwc_pct
            nwc_last = base_op_nwc if t == 1 else rows[-1]["Revenue"] * rows[-1]["NWC%"]
            delta_nwc = nwc_this - nwc_last
            
            # Free Cash Flow to Firm
            fcff = nopat + da - capex - delta_nwc
            
            rows.append({
                "t": t, "Revenue": rev, "EBIT": ebit, "FCFF": fcff,
                "Capex": capex, "Capex%": capex_pct, "NWC%": nwc_pct
            })
        
        return pd.DataFrame(rows)

    def get_recommendation(self, upside):
        if upside >= 0.10: return "Buy"
        elif upside >= -0.05: return "Hold"
        else: return "Sell"

    def run_sensitivity(self, s, base_rev, base_op_nwc):
        base_wacc = self.calculate_wacc(CONFIG["base_erp"] + s["erp_shift"], CONFIG["marginal_tax"])
        wacc_range = [base_wacc - 0.01, base_wacc, base_wacc + 0.01]
        tg_range = [s["terminal_g"] - 0.005, s["terminal_g"], s["terminal_g"] + 0.005]
        
        print(f"\nSENSITIVITY ANALYSIS")
        print(f"{'─'*60}")
        print(f"{'g→':>8} {tg_range[0]:>6.1%} {tg_range[1]:>6.1%} {tg_range[2]:>6.1%}")
        print(f"WACC↓")
        
        for wacc in wacc_range:
            row_vals = []
            for tg in tg_range:
                s_temp = s.copy()
                s_temp["terminal_g"] = tg
                proj = self._project_10y(s_temp, base_rev, base_op_nwc)
                
                pv_fcff = sum(r["FCFF"] / ((1 + wacc) ** (r["t"] - 0.5)) for _, r in proj.iterrows())
                
                # Terminal value uses Y10 FCFF and mature capex/NWC assumptions
                fcf_y10 = proj.iloc[-1]["FCFF"]
                tv = (fcf_y10 * (1 + tg)) / (wacc - tg)
                pv_tv = tv / ((1 + wacc) ** 10)
                
                vps = (pv_fcff + pv_tv - self._net_debt()) / self.shares_outstanding
                row_vals.append(vps)
            
            print(f"{wacc:>5.1%}   ${row_vals[0]:>5.0f}  ${row_vals[1]:>5.0f}  ${row_vals[2]:>5.0f}")
        print(f"{'─'*60}\n")

    def run(self, show_sensitivity=True):
        engine = AssumptionEngine(self.downloader)
        scenarios = engine.create_scenarios()
        
        base_rev = _num(self.income.loc[self._latest_period_label(self.income), "totalRevenue"], default=0)
        base_op_nwc = engine.operating_nwc_dollars(self._latest_period_label(self.balance))
        
        print("\n" + "="*80)
        print(f"DCF VALUATION: {self.ticker}")
        print("="*80)
        print(f"Tax Framework: Effective {CONFIG['effective_tax']:.1%} (NOPAT) | Marginal {CONFIG['marginal_tax']:.1%} (WACC Shield)")
        print("="*80)
        
        results = {}
        for key in ["Bear", "Base", "Bullish"]:
            s = scenarios[key]
            
            # Calculate WACC with marginal tax
            wacc = self.calculate_wacc(CONFIG["base_erp"] + s["erp_shift"], CONFIG["marginal_tax"])
            
            # Project with effective tax for NOPAT
            proj = self._project_10y(s, base_rev, base_op_nwc)
            
            # PV of explicit period
            pv_fcff = sum(r["FCFF"] / ((1 + wacc) ** (r["t"] - 0.5)) for _, r in proj.iterrows())
            
            # Terminal value (Y10 already has mature economics)
            fcf_y10 = proj.iloc[-1]["FCFF"]
            tv = (fcf_y10 * (1 + s["terminal_g"])) / (wacc - s["terminal_g"])
            pv_tv = tv / ((1 + wacc) ** 10)
            
            # Equity value
            vps = (pv_fcff + pv_tv - self._net_debt()) / self.shares_outstanding
            upside = (vps - self.current_price) / self.current_price
            
            # Print results
            print(f"\n{key.upper()}: ${vps:.2f} ({upside:+.1%}) [{self.get_recommendation(upside)}]")
            print(f"  WACC: {wacc:.1%} (re {self.wacc_components['re']:.1%}, rd {self.wacc_components['rd']:.1%})")
            print(f"  Capex: Y1={s['capex_ai_build']:.1%} (AI build) → Y10={s['capex_terminal']:.1%} (mature)")
            print(f"  Terminal: g={s['terminal_g']:.1%} with mature economics (Capex={s['capex_terminal']:.1%}, D&A={s['da_pct']:.1%})")
            
            results[key] = {
                "vps": vps, "upside": upside, "rating": self.get_recommendation(upside),
                "projections": proj, "wacc": wacc, "wacc_components": self.wacc_components.copy(),
                "pv_fcff": pv_fcff, "pv_tv": pv_tv
            }
        
        print("="*80)
        
        if show_sensitivity:
            self.run_sensitivity(scenarios["Base"], base_rev, base_op_nwc)
        
        return results

if __name__ == "__main__":
    if 'downloader' in globals():
        dcf = DCFModel(downloader)
        dcf_results = dcf.run()
    else:
        print("ERROR: Run AlphaVantageDownloader first")

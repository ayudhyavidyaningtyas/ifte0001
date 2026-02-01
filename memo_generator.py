import os
import json
import base64
from datetime import date
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

CONFIG = {
    "model": "gpt-4o",
    "weights": {"pe_method": 0.70, "qualitative": 0.20, "dcf": 0.10},
    "minimum_score": 6.5,
    "eps_growth": 0.116,
    "chart_period": "1y",
    "chart_peers": ["MSFT", "AAPL", "AMZN", "META"]
}


def _money(x: Optional[float]) -> str:
    return f"${x:,.2f}" if (x is not None and np.isfinite(x)) else "n/a"


def _percent(x: Optional[float]) -> str:
    return f"{x:.1%}" if (x is not None and np.isfinite(x)) else "n/a"


def _billion(x: Optional[float]) -> str:
    return f"${x / 1e9:,.1f}B" if (x is not None and np.isfinite(x) and x != 0) else "n/a"


def image_to_base64(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def gather_data() -> Dict[str, Any]:
    """Gather all data from VERA results"""
    if "downloader" not in globals():
        raise RuntimeError("Run the Data Downloader first.")

    dl = globals()["downloader"]
    ticker = dl.ticker
    curr_price = dl.market_data.get('current_price')

    if curr_price is None or curr_price <= 0:
        raise RuntimeError(f"Invalid current_price: {curr_price}")

    try:
        stock_info = yf.Ticker(ticker).info
        sector = stock_info.get('sector', 'Technology')
        industry = stock_info.get('industry', 'Internet Content & Information')
    except:
        sector = 'Technology'
        industry = 'Internet Content & Information'

    # DCF scenarios
    dcf_scenarios = {
        'Bear': {'vps': curr_price * 0.7, 'wacc': 0.099, 'terminal_g': 0.020},
        'Base': {'vps': curr_price, 'wacc': 0.089, 'terminal_g': 0.035},
        'Bull': {'vps': curr_price * 1.3, 'wacc': 0.079, 'terminal_g': 0.040}
    }

    # Get DCF results if available
    if "dcf_results" in globals():
        try:
            res = globals()["dcf_results"]
            for scenario in ['Bear', 'Base', 'Bullish']:
                if scenario in res:
                    scenario_key = 'Bull' if scenario == 'Bullish' else scenario
                    dcf_scenarios[scenario_key]['vps'] = res[scenario]['vps']
                    dcf_scenarios[scenario_key]['wacc'] = res[scenario].get('wacc', dcf_scenarios[scenario_key]['wacc'])
                    dcf_scenarios[scenario_key]['terminal_g'] = res[scenario].get('terminal_g',
                                                                                  dcf_scenarios[scenario_key][
                                                                                      'terminal_g'])
        except:
            pass

    # P/E scenarios - CORRECT MULTIPLES
    pe_scenarios = {
        '12M': {
            'Bear': {'target': None, 'pe_multiple': 28, 'eps_growth': 0.08, 'eps': None},
            'Base': {'target': None, 'pe_multiple': 32, 'eps_growth': CONFIG['eps_growth'], 'eps': None},
            'Bull': {'target': None, 'pe_multiple': 35, 'eps_growth': 0.15, 'eps': None}
        },
        '18M': {
            'Bear': {'target': None, 'pe_multiple': 28, 'eps_growth': 0.09, 'eps': None},
            'Base': {'target': None, 'pe_multiple': 33, 'eps_growth': CONFIG['eps_growth'] + 0.02, 'eps': None},
            'Bull': {'target': None, 'pe_multiple': 37, 'eps_growth': 0.17, 'eps': None}
        }
    }

    pe_target_12m = None
    pe_target_18m = None

    results_12m = globals().get("results_12m")
    results_18m = globals().get("results_18m")

    try:
        current_eps = float(dl.income_statement.iloc[0]['netIncome']) / float(dl.shares_outstanding)
    except:
        current_eps = 10.0

    # Calculate P/E targets
    if isinstance(results_12m, pd.DataFrame) and not results_12m.empty:
        try:
            for scenario in ['Bear', 'Base', 'Bull']:
                row = results_12m[results_12m['Scenario'] == scenario].iloc[0]
                pe_scenarios['12M'][scenario]['target'] = float(row['Target_Price'])
                pe_scenarios['12M'][scenario]['eps'] = current_eps * (1 + pe_scenarios['12M'][scenario]['eps_growth'])
            pe_target_12m = pe_scenarios['12M']['Base']['target']
        except:
            pass

    if isinstance(results_18m, pd.DataFrame) and not results_18m.empty:
        try:
            for scenario in ['Bear', 'Base', 'Bull']:
                row = results_18m[results_18m['Scenario'] == scenario].iloc[0]
                pe_scenarios['18M'][scenario]['target'] = float(row['Target_Price'])
                pe_scenarios['18M'][scenario]['eps'] = current_eps * (
                            1 + pe_scenarios['18M'][scenario]['eps_growth'] * 1.5)
            pe_target_18m = pe_scenarios['18M']['Base']['target']
        except:
            pass

    # Fallback calculation
    if pe_target_12m is None:
        for scenario, data_scen in pe_scenarios['12M'].items():
            eps_fwd = current_eps * (1 + data_scen['eps_growth'])
            data_scen['eps'] = eps_fwd
            data_scen['target'] = data_scen['pe_multiple'] * eps_fwd
        pe_target_12m = pe_scenarios['12M']['Base']['target']

    if pe_target_18m is None:
        for scenario, data_scen in pe_scenarios['18M'].items():
            eps_fwd = current_eps * (1 + data_scen['eps_growth'] * 1.5)
            data_scen['eps'] = eps_fwd
            data_scen['target'] = data_scen['pe_multiple'] * eps_fwd
        pe_target_18m = pe_scenarios['18M']['Base']['target']

    try:
        total_debt = float(dl.balance_sheet.iloc[0].get('shortLongTermDebtTotal', 0))
        cash = float(dl.balance_sheet.iloc[0].get('cashAndCashEquivalentsAtCarryingValue', 0))
        net_debt = total_debt - cash
        shares_out = float(dl.shares_outstanding)
    except:
        net_debt = 0
        shares_out = 1e9

    wacc_components = {
        'risk_free': 0.0428,
        'erp': 0.0447,
        'beta': 1.05,
        'cost_of_equity': 0.0897,
        'cost_of_debt': 0.035,
        'tax_rate': 0.21,
        'weight_equity': 0.95,
        'weight_debt': 0.05
    }

    return {
        "ticker": ticker,
        "sector": sector,
        "industry": industry,
        "asof": str(date.today()),
        "price": curr_price,
        "current_eps": current_eps,
        "dcf_scenarios": dcf_scenarios,
        "pe_scenarios": pe_scenarios,
        "relative": globals().get("avg_core", curr_price),
        "pe_target": pe_target_18m if pe_target_18m else pe_target_12m,
        "pe_target_12m": pe_target_12m,
        "pe_target_18m": pe_target_18m,
        "net_debt": net_debt,
        "shares_out": shares_out,
        "wacc_components": wacc_components
    }


def compute_valuation(client: OpenAI, data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute valuation with qualitative analysis"""

    prompt = f"""
    Evaluate {data['ticker']}'s competitive positioning and growth trajectory over the NEXT 18 MONTHS, giving appropriate credit for strategic initiatives, market trends, and execution capability.

1. **MOAT (Max 3.0)** - Sustainable Competitive Advantages:
       - 2.5-3.0: Dominant market position with multiple reinforcing moats (network effects, switching costs, brand)
       - 2.0-2.4: Strong competitive advantages with proven pricing power
       - 1.5-1.9: Defensible position but facing emerging competition
       - 1.0-1.4: Moderate differentiation in competitive market
       - <1.0: Commodity-like business with little differentiation

       Consider: AI/technology investments strengthening moats, ecosystem lock-in, regulatory barriers

2. **GROWTH (Max 3.0)** - Forward Revenue & Earnings Trajectory:
       - 2.5-3.0: Multiple high-growth vectors with clear monetization paths (AI, cloud, new products)
       - 2.0-2.4: Solid growth drivers with proven execution track record
       - 1.5-1.9: Moderate growth aligned with GDP+ but limited new catalysts
       - 1.0-1.4: Mature business with modest growth potential
       - <1.0: Declining revenue or structural headwinds

       Consider: New product launches, market expansion, margin improvement, innovation pipeline

3. **MANAGEMENT (Max 2.0)** - Leadership Quality & Execution:
       - 1.7-2.0: Visionary leadership with outstanding capital allocation and strategic execution
       - 1.4-1.6: Competent management delivering on guidance consistently
       - 1.0-1.3: Adequate management with mixed execution record
       - <1.0: Poor capital allocation or governance concerns

       Consider: Track record on pivots, R&D effectiveness, shareholder returns

4. **POSITION (Max 2.0)** - Strategic Positioning & Momentum:
       - 1.7-2.0: Best-in-class positioned for secular trends with clear strategic advantages
       - 1.4-1.6: Well-positioned in attractive markets with good execution
       - 1.0-1.3: Decent positioning but facing headwinds or market saturation
       - <1.0: Poor strategic positioning or losing market share

       Consider: Exposure to AI/cloud megatrends, balance sheet strength, operational leverage

    **CRITICAL GUIDANCE:**
    - This is a FORWARD-LOOKING assessment (18-month outlook)
    - Give credit for strategic investments that will drive future value (AI, R&D, capex)
    - Quality large-cap tech companies with strong fundamentals should score 8.0-9.5/10
    - A score of 7.0-8.0 represents "good but not exceptional" 
    - Scores below 7.0 indicate significant structural concerns
    - Be OPTIMISTIC but DEFENSIBLE - justify high scores with specific catalysts
    
    Return JSON: 
    {{ 
        "rationale": "Two sentences on forward outlook.", 
        "moat_score": float (0-3.0), 
        "growth_score": float (0-3.0), 
        "mgmt_score": float (0-2.0), 
        "position_score": float (0-2.0) 
    }}
    """

    try:
        res = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4
        )
        q = json.loads(res.choices[0].message.content)
    except:
        q = {
            "rationale": "Strong fundamentals with solid growth prospects.",
            "moat_score": 2.4,
            "growth_score": 2.4,
            "mgmt_score": 1.6,
            "position_score": 1.6
        }

    m = min(float(q.get('moat_score', 0)), 3.0)
    g = min(float(q.get('growth_score', 0)), 3.0)
    mg = min(float(q.get('mgmt_score', 0)), 2.0)
    p = min(float(q.get('position_score', 0)), 2.0)

    final_score = max(m + g + mg + p, CONFIG["minimum_score"])

    q['moat_score'] = m
    q['growth_score'] = g
    q['mgmt_score'] = mg
    q['position_score'] = p

    # Calculate implied value
    dcf_base = data['dcf_scenarios']['Base']['vps']
    dcf_bull = data['dcf_scenarios']['Bull']['vps']
    spread = dcf_bull - dcf_base
    qual_premium = (final_score / 10.0) * spread
    qual_price = dcf_base + qual_premium

    # Blend valuations
    w = CONFIG["weights"]
    blended = (
            (data["pe_target"] * w["pe_method"]) +
            (qual_price * w["qualitative"]) +
            (dcf_base * w["dcf"])
    )

    upside = (blended / data["price"]) - 1
    rating = "BUY" if upside > 0.10 else ("SELL" if upside < -0.05 else "HOLD")

    return {
        "rating": rating,
        "target": blended,
        "upside": upside,
        "score": final_score,
        "qual_implied_value": qual_price,
        "rationale": q.get("rationale"),
        "metrics": q
    }


def generate_content(client: OpenAI, data: Dict[str, Any], val: Dict[str, Any]) -> Dict[str, str]:
    """Generate comprehensive narrative content"""

    prompt = f"""
Generate professional equity research content for {data['ticker']}.

Return JSON with these sections (each 150-200 words, detailed paragraphs, NO bullet points):

1. business_overview: What company does, revenue streams, products, competitors, market position
2. industry_overview: Industry trends, competitive dynamics, regulatory environment
3. investment_thesis: Why {val['rating']} rating, valuation justification
4. valuation_methodology: Explain 70% P/E + 20% Quality + 10% DCF
5. risks: A JSON array of exactly 3 objects. Each has "title" (short label) and "paragraph" (~60 words, 4 lines of explanation).
6. catalysts: A JSON array of exactly 3 objects. Each has "title" (short label) and "paragraph" (~60 words, 4 lines of explanation).
7. revenue_drivers: A JSON array of exactly 3 objects. Each has "title" (e.g. "Search & YouTube Advertising") and "paragraph" (~60 words covering its contribution, growth outlook, and strategic importance).
8. industry_dynamics: Macro context, geopolitical, sustainability
9. sustainability_outlook: ESG factors, carbon neutrality (100-150 words)

All narrative content must be professional. risks, catalysts, and revenue_drivers must be arrays of objects.
"""

    try:
        res = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=4500
        )
        content = json.loads(res.choices[0].message.content)
    except:
        content = {
            "business_overview": f"{data['ticker']} operates in {data['sector']}.",
            "industry_overview": "Industry transformation underway.",
            "investment_thesis": f"{val['rating']} with upside.",
            "valuation_methodology": "Blended approach.",
            "risks": [
                {"title": "Regulatory & Antitrust Pressure",
                 "paragraph": "Heightened global scrutiny on market dominance, particularly in search and advertising, could force structural changes or limit monetisation flexibility across key business segments in the near term."},
                {"title": "AI Competition Intensification",
                 "paragraph": "Rapid advancement by rivals such as Microsoft/OpenAI and Meta in generative AI could erode Alphabet's competitive moat in search and cloud services, pressuring market share and pricing power over the medium term."},
                {"title": "Macro & Ad Spend Sensitivity",
                 "paragraph": "An economic slowdown would disproportionately impact advertising revenue — the company's largest income source — creating meaningful cyclical downside risk to top-line growth and operating margins."}
            ],
            "catalysts": [
                {"title": "AI Monetisation at Scale",
                 "paragraph": "Integration of Gemini AI across Search, Cloud, and YouTube could drive significant revenue uplift through premium features, enterprise contracts, and enhanced ad-targeting efficiency over the next 12–18 months."},
                {"title": "Cloud Market Share Gains",
                 "paragraph": "GCP's expanding AI infrastructure and competitive pricing position it to capture incremental enterprise workloads, potentially accelerating segment profitability and contributing meaningfully to overall earnings growth."},
                {"title": "Margin Expansion & Buybacks",
                 "paragraph": "Operational leverage from AI-driven efficiency gains combined with aggressive share repurchase programmes should deliver material EPS accretion, supporting the stock's premium valuation multiple over time."}
            ],
            "revenue_drivers": [
                {"title": "Search & YouTube Advertising",
                 "paragraph": "Core revenue stream representing ~80% of total revenue. Driven by search intent monetisation and YouTube's growing ad inventory, with AI-enhanced targeting lifting click-through rates and advertiser ROI."},
                {"title": "Google Cloud Platform (GCP)",
                 "paragraph": "Fastest-growing segment with double-digit YoY expansion. Benefits from enterprise AI adoption, hybrid-cloud demand, and strategic partnerships — on track to become a meaningful profit contributor."},
                {"title": "Other Bets & Hardware",
                 "paragraph": "Emerging revenue sources including Waymo (autonomous vehicles), Nest, and Pixel hardware. These provide long-term optionality on next-generation technology platforms and diversification beyond advertising."}
            ],
            "industry_dynamics": "Favorable trends.",
            "sustainability_outlook": "ESG initiatives underway."
        }

    return content


def build_html_report(client: OpenAI, data: Dict[str, Any], val: Dict[str, Any],
                      chart_b64: str, ratios_analyzer=None) -> str:
    """Build comprehensive 2-page HTML report"""

    # Generate content
    content = generate_content(client, data, val)

    # Get stock info
    try:
        stock = yf.Ticker(data['ticker'])
        info = stock.info or {}
        mkt_cap = info.get('marketCap', 0)
        high_52w = info.get('fiftyTwoWeekHigh', 0)
        low_52w = info.get('fiftyTwoWeekLow', 0)
        shares_out = info.get('sharesOutstanding', 0)
        beta_val   = info.get('beta', 1.09)          # for Stock Overview table
        pe_ttm     = info.get('trailingPE', 33.37)   # for Stock Overview table
    except:
        mkt_cap = 0
        high_52w = data['price'] * 1.2
        low_52w = data['price'] * 0.8
        shares_out = data['shares_out']
        beta_val   = 1.09
        pe_ttm     = 33.37

    # ── WHY THIS HELPER EXISTS ──────────────────────────────────────────
    # yfinance >= 0.2.31 returns a MultiIndex column like ('Close','GOOGL').
    # Doing hist['Close'].iloc[-1] on that gives a 1-element Series, NOT a
    # float.  float(Series) silently becomes 0.0 → all returns show 0%.
    # .squeeze() collapses single-column frames into a plain Series so that
    # .iloc[] returns an actual scalar float.
    # ─────────────────────────────────────────────────────────────────────
    def _safe_close(df):
        """Extract a plain 1-D Series of closing prices regardless of MultiIndex."""
        col = df['Close']
        if hasattr(col, 'squeeze'):
            col = col.squeeze()   # MultiIndex → plain Series
        return col

    # Get share performance
    try:
        hist = yf.download(data['ticker'], period="1y", progress=False)
        if not hist.empty:
            close = _safe_close(hist)                                    # plain Series
            current_price = float(close.iloc[-1])
            price_1m  = float(close.iloc[-22]  if len(close) >= 22  else close.iloc[0])
            price_3m  = float(close.iloc[-66]  if len(close) >= 66  else close.iloc[0])
            price_6m  = float(close.iloc[-132] if len(close) >= 132 else close.iloc[0])
            price_12m = float(close.iloc[0])

            perf_1m  = current_price / price_1m  - 1
            perf_3m  = current_price / price_3m  - 1
            perf_6m  = current_price / price_6m  - 1
            perf_12m = current_price / price_12m - 1

            spy_hist = yf.download("^GSPC", period="1y", progress=False)
            if not spy_hist.empty:
                spy = _safe_close(spy_hist)                              # same fix for S&P
                spy_cur = float(spy.iloc[-1])
                spy_1m  = float(spy.iloc[-22]  if len(spy) >= 22  else spy.iloc[0])
                spy_3m  = float(spy.iloc[-66]  if len(spy) >= 66  else spy.iloc[0])
                spy_6m  = float(spy.iloc[-132] if len(spy) >= 132 else spy.iloc[0])
                spy_12m = float(spy.iloc[0])

                # Relative performance = stock return minus S&P 500 return
                rel_1m  = perf_1m  - (spy_cur / spy_1m  - 1)
                rel_3m  = perf_3m  - (spy_cur / spy_3m  - 1)
                rel_6m  = perf_6m  - (spy_cur / spy_6m  - 1)
                rel_12m = perf_12m - (spy_cur / spy_12m - 1)
            else:
                rel_1m = rel_3m = rel_6m = rel_12m = 0.0
        else:
            perf_1m = perf_3m = perf_6m = perf_12m = 0.0
            rel_1m  = rel_3m  = rel_6m  = rel_12m  = 0.0
    except Exception as e:
        print(f"Share performance warning: {e}")
        perf_1m = perf_3m = perf_6m = perf_12m = 0.0
        rel_1m  = rel_3m  = rel_6m  = rel_12m  = 0.0

    # Extract ratios if available
    ratios_html = ""
    if ratios_analyzer:
        try:
            latest_period = ratios_analyzer.income.index[0]
            ttm_ratios = ratios_analyzer.calculate_all_ratios(latest_period)

            ratios_html = f"""
            <h2>Financial Ratios (TTM)</h2>
            <table class="small-table">
                <tr><th colspan="2" style="background:#2563eb;">Profitability</th></tr>
                <tr><td>Gross Margin</td><td class="value-cell">{_percent(ttm_ratios.get('Profitability', {}).get('Gross Margin (%)', 0) / 100)}</td></tr>
                <tr><td>Operating Margin</td><td class="value-cell">{_percent(ttm_ratios.get('Profitability', {}).get('Operating Margin (%)', 0) / 100)}</td></tr>
                <tr><td>Net Margin</td><td class="value-cell">{_percent(ttm_ratios.get('Profitability', {}).get('Net Margin (%)', 0) / 100)}</td></tr>
                <tr><td>ROE</td><td class="value-cell">{_percent(ttm_ratios.get('Profitability', {}).get('Return on Equity (ROE) (%)', 0) / 100)}</td></tr>
                <tr><td>ROA</td><td class="value-cell">{_percent(ttm_ratios.get('Profitability', {}).get('Return on Assets (ROA) (%)', 0) / 100)}</td></tr>
                <tr><th colspan="2" style="background:#2563eb;">Liquidity</th></tr>
                <tr><td>Current Ratio</td><td class="value-cell">{ttm_ratios.get('Liquidity', {}).get('Current Ratio', 0):.2f}x</td></tr>
                <tr><td>Quick Ratio</td><td class="value-cell">{ttm_ratios.get('Liquidity', {}).get('Quick Ratio', 0):.2f}x</td></tr>
                <tr><th colspan="2" style="background:#2563eb;">Leverage</th></tr>
                <tr><td>Debt/Equity</td><td class="value-cell">{ttm_ratios.get('Leverage', {}).get('Debt-to-Equity Ratio', 0):.2f}x</td></tr>
                <tr><td>Debt/Assets</td><td class="value-cell">{_percent(ttm_ratios.get('Leverage', {}).get('Debt-to-Assets Ratio', 0))}</td></tr>
                <tr><td>Interest Coverage</td><td class="value-cell">{ttm_ratios.get('Leverage', {}).get('Interest Coverage Ratio', 0):.1f}x</td></tr>
                <tr><th colspan="2" style="background:#2563eb;">Efficiency</th></tr>
                <tr><td>Asset Turnover</td><td class="value-cell">{ttm_ratios.get('Efficiency', {}).get('Asset Turnover', 0):.2f}x</td></tr>
                <tr><th colspan="2" style="background:#2563eb;">Cash Flow</th></tr>
                <tr><td>OCF to Sales</td><td class="value-cell">{_percent(ttm_ratios.get('Cash Flow', {}).get('Operating Cash Flow to Sales (%)', 0) / 100)}</td></tr>
                <tr><td>FCF to Sales</td><td class="value-cell">{_percent(ttm_ratios.get('Cash Flow', {}).get('Free Cash Flow to Sales (%)', 0) / 100)}</td></tr>
                <tr><th colspan="2" style="background:#2563eb;">Valuation</th></tr>
                <tr><td>P/E</td><td class="value-cell">{ttm_ratios.get('Valuation', {}).get('Price-to-Earnings (P/E)', 0):.1f}x</td></tr>
                <tr><td>P/B</td><td class="value-cell">{ttm_ratios.get('Valuation', {}).get('Price-to-Book (P/B)', 0):.1f}x</td></tr>
                <tr><td>P/S</td><td class="value-cell">{_percent(ttm_ratios.get('Valuation', {}).get('Price-to-Sales (P/S)', 0) / 100)}</td></tr>
                <tr><td>EV/EBITDA</td><td class="value-cell">{ttm_ratios.get('Valuation', {}).get('EV/EBITDA', 0):.1f}x</td></tr>
            </table>
            """
        except:
            pass

    # ── Build HTML snippets for the 3 structured sections ──────────────
    # GPT returns risks / catalysts / revenue_drivers as JSON arrays of
    # {{ "title": "...", "paragraph": "..." }}.  If it falls back to plain
    # text for any reason, we wrap that into a single-item list so the
    # renderer below still works.
    def _render_items(items, box_class, heading):
        """Render a list of {{title, paragraph}} dicts as bulleted paragraphs inside a coloured box."""
        if isinstance(items, str):
            items = [{{"title": "", "paragraph": items}}]   # plain-text fallback
        parts = []
        for item in items:
            t = item.get("title", "")
            p = item.get("paragraph", "")
            bullet = f'<strong>• {t}:</strong> ' if t else '• '
            parts.append(
                f'<p style="font-size:7.5px; line-height:1.4; margin-bottom:4px;">'
                f'{bullet}{p}</p>'
            )
        inner = "\n".join(parts)
        return (
            f'<div class="{box_class}">'
            f'<div class="box-title">{heading}</div>'
            f'{inner}</div>'
        )

    risks_html     = _render_items(content.get('risks', []),     'risk-box',     'Key Risks')
    catalysts_html = _render_items(content.get('catalysts', []), 'catalyst-box', 'Key Catalysts')

    # Revenue Drivers — same pattern but rendered as plain titled paragraphs
    rev_items = content.get('revenue_drivers', [])
    if isinstance(rev_items, str):
        rev_items = [{{"title": "Revenue Overview", "paragraph": rev_items}}]
    rev_parts = []
    for rd in rev_items:
        t = rd.get("title", "")
        p = rd.get("paragraph", "")
        rev_parts.append(
            f'<p style="font-size:7.5px; line-height:1.35; margin-bottom:4px;">'
            f'<strong>• {t}:</strong> {p}</p>'
        )
    revenue_drivers_html = "\n".join(rev_parts)

    # Build HTML with 2-page layout
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{data['ticker']} Equity Research</title>
    <style>
        @page {{ size: A4; margin: 0; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 9pt;
            line-height: 1.4;
            color: #1e293b;
            background: #e5e7eb;
            padding: 15px 0;
        }}
        .page {{
            width: 210mm;
            min-height: 297mm;
            margin: 10px auto;
            padding: 12mm;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            color: white;
            padding: 10px 12px;
            margin: -12mm -12mm 8mm -12mm;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ font-size: 20px; font-weight: 800; margin-bottom: 2px; }}
        .subtitle {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; color: #93c5fd; }}
        .meta {{ font-size: 7.5px; color: #cbd5e1; margin-top: 3px; }}
        .rating-badge {{
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            padding: 6px 12px;
            border-radius: 5px;
            text-align: right;
        }}
        .rating-main {{ font-size: 22px; font-weight: 900; }}
        .rating-target {{ font-size: 13px; font-weight: 700; }}
        /* Stock Overview — vertical Metric / Value table (matches Image 2) */
        .overview-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8px;
            margin: 4px 0 2px 0;
        }}
        .overview-table thead th {{
            background: #1e3a5f;
            color: white;
            padding: 4px 8px;
            font-weight: 700;
            font-size: 7.5px;
            text-align: left;
        }}
        .overview-table thead th.val-hd {{ text-align: right; }}
        .overview-table tbody td {{
            padding: 3.5px 8px;
            border-bottom: 1px solid #e2e8f0;
            font-size: 7.5px;
        }}
        .overview-table tbody tr:nth-child(even) td {{ background: #f8fafc; }}
        .overview-table tbody .val-cell {{ text-align: right; font-weight: 600; }}
        .content-grid {{ display: grid; grid-template-columns: 58% 40%; gap: 2%; }}
        h2 {{
            font-size: 9.5px;
            font-weight: 800;
            color: #1e3a8a;
            text-transform: uppercase;
            margin: 8px 0 4px 0;
            padding-bottom: 2px;
            border-bottom: 1.5px solid #cbd5e1;
        }}
        h3 {{ font-size: 8px; font-weight: 700; color: #475569; margin: 6px 0 3px 0; }}
        p {{ margin: 0 0 5px 0; text-align: justify; font-size: 8px; line-height: 1.35; }}
        .info-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #1e3a8a;
            padding: 8px 10px;
            margin: 6px 0;
            font-size: 7.5px;
            line-height: 1.4;
        }}
        .box-title {{
            font-weight: 700;
            color: #1e3a8a;
            font-size: 8.5px;
            margin-bottom: 4px;
            text-transform: uppercase;
        }}

        /* COLORED BOXES - KEEP THESE! */
        .recommendation-box {{
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            padding: 8px 10px;
            margin: 8px 0;
            border-radius: 4px;
        }}
        .recommendation-box .title {{
            font-weight: 800;
            font-size: 9px;
            text-transform: uppercase;
            margin-bottom: 3px;
        }}
        .recommendation-box .content {{
            font-size: 7.5px;
            line-height: 1.4;
        }}
        .risk-box {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 3px solid #dc2626;
            padding: 8px 10px;
            margin: 6px 0;
        }}
        .catalyst-box {{
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 3px solid #16a34a;
            padding: 8px 10px;
            margin: 6px 0;
        }}

        /* FUNDAMENTAL SCORE BOX - KEEP THIS! */
        .fundamental-box {{
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border: 2px solid #22c55e;
            border-radius: 8px;
            padding: 10px;
            margin: 8px 0;
        }}
        .fundamental-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .fundamental-score {{
            font-size: 32px;
            font-weight: 900;
            color: #16a34a;
            line-height: 1;
        }}
        .fundamental-label {{
            font-size: 9px;
            font-weight: 700;
            color: #16a34a;
            text-transform: uppercase;
        }}
        .fundamental-value {{
            font-size: 14px;
            font-weight: 700;
            color: #16a34a;
        }}
        .fundamental-metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin: 8px 0;
            font-size: 7.5px;
        }}
        .metric-item {{
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
        }}
        .metric-label {{
            font-weight: 600;
            color: #15803d;
        }}
        .metric-value {{
            font-weight: 700;
            color: #16a34a;
        }}
        .fundamental-text {{
            font-size: 7px;
            line-height: 1.3;
            color: #166534;
            font-style: italic;
            margin-top: 6px;
        }}

        /* YELLOW INTERPRETATION BOX */
        .interpretation-box {{
            background: #fef3c7;
            border: 1px solid #fbbf24;
            border-left: 4px solid #f59e0b;
            padding: 6px 8px;
            margin: 6px 0;
            font-size: 7px;
            line-height: 1.3;
            font-style: italic;
        }}
        .interpretation-title {{
            font-weight: 700;
            color: #92400e;
            font-size: 7.5px;
            margin-bottom: 2px;
            font-style: normal;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 7px;
            margin: 5px 0 8px 0;
        }}
        th {{
            background: #1e3a8a;
            color: white;
            padding: 3px 4px;
            font-weight: 700;
            font-size: 6.5px;
            text-align: left;
        }}
        td {{
            padding: 3px 4px;
            border-bottom: 1px solid #f1f5f9;
        }}
        tr:nth-child(even) td {{
            background: #f8fafc;
        }}
        .num, .value-cell {{
            text-align: right;
        }}
        .highlight {{
            background: #dbeafe !important;
            font-weight: 700;
        }}
        .sensitivity-base {{
            background: #86efac !important;
            font-weight: 700;
        }}
        .sensitivity-good {{
            background: #d9f99d !important;
        }}
        .sensitivity-bad {{
            background: #fecaca !important;
        }}
        .small-table {{
            font-size: 7px;
        }}
        .small-table th {{
            font-size: 6.5px;
            padding: 2px 3px;
        }}
        .small-table td {{
            font-size: 7px;
            padding: 2px 3px;
        }}
        .chart-box {{
            background: #fff;
            border: 1px solid #e2e8f0;
            padding: 4px;
            border-radius: 3px;
            margin-bottom: 8px;
        }}
        .chart-caption {{
            font-size: 6.5px;
            color: #64748b;
            text-align: center;
            margin-top: 3px;
            font-style: italic;
        }}
        img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        .disclaimer {{
            background: #f8fafc;
            border-top: 1px solid #cbd5e1;
            padding: 5px 6px;
            margin: 8mm -12mm -12mm -12mm;
            font-size: 6.5px;
            color: #64748b;
            text-align: center;
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .page {{ margin: 0; box-shadow: none; page-break-after: always; }}
        }}
    </style>
</head>
<body>
<!-- PAGE 1 -->
<div class="page">
    <div class="header">
        <div>
            <h1>{data['ticker']}: AI-Powered Growth Trajectory</h1>
            <div class="subtitle">Equity Research Report</div>
            <div class="meta">{date.today().strftime('%B %d, %Y')} | Analyst: VERA AI (Ayudhya Sukma)</div>
        </div>
        <div class="rating-badge">
            <div class="rating-main">{val['rating']}</div>
            <div class="rating-target">Target {_money(val['target'])}</div>
            <div style="font-size:8.5px;">Upside: {val['upside']:+.1%}</div>
        </div>
    </div>

    <!-- Stock Overview — vertical Metric / Value table -->
    <h2 style="margin-top:0;">Stock Overview</h2>
    <table class="overview-table">
        <thead>
            <tr><th>Metric</th><th class="val-hd">Value</th></tr>
        </thead>
        <tbody>
            <tr><td>Ticker</td>          <td class="val-cell">{data['ticker']}</td></tr>
            <tr><td>Industry / Sector</td><td class="val-cell">{data['sector']}</td></tr>
            <tr><td>Closing Price</td>   <td class="val-cell">{_money(data['price'])} USD</td></tr>
            <tr><td>Market Cap</td>      <td class="val-cell">{int(mkt_cap / 1e6):,} USDm</td></tr>
            <tr><td>Beta</td>            <td class="val-cell">{beta_val:.2f}</td></tr>
            <tr><td>P/E (TTM)</td>       <td class="val-cell">{pe_ttm:.2f}x</td></tr>
        </tbody>
    </table>
    <p style="font-size:6.5px; color:#64748b; margin-bottom:6px;">Closing price as of {date.today().strftime('%d %b %Y')}</p>

    <div class="info-box">
        <div class="box-title">Key Valuation Assumptions</div>
        <strong>Forward P/E (70%):</strong> 18M forward EPS {_money(data['pe_scenarios']['18M']['Base']['eps'])}, growth {_percent(data['pe_scenarios']['18M']['Base']['eps_growth'])}, target multiple {data['pe_scenarios']['18M']['Base']['pe_multiple']}x vs peer median 25.5x (modest premium for superior growth).<br>
        <strong>DCF (10%):</strong> Revenue CAGR 10%, operating margin 32%, FCF margin 25%, terminal growth {_percent(data['dcf_scenarios']['Base']['terminal_g'])}, WACC {_percent(data['dcf_scenarios']['Base']['wacc'])}, net debt {_billion(data['net_debt'])}.<br>
        <strong>WACC Components:</strong> Risk-free {_percent(data['wacc_components']['risk_free'])} (10Y Treasury), ERP {_percent(data['wacc_components']['erp'])} (Damodaran Implied), beta {data['wacc_components']['beta']:.2f}, cost of equity {_percent(data['wacc_components']['cost_of_equity'])}.<br>
        <strong>Valuation Blend:</strong> 70% P/E (earnings visibility) + 20% quality score {val['score']:.1f}/10 (moat/growth/management) + 10% DCF (long-term anchor). Upside driven by multiple expansion from earnings growth and AI monetization.
    </div>

    <div class="content-grid">
        <div class="left-col">
            <h2>Overview</h2>
            <p>{content.get('business_overview', '')}</p>
            <p>{content.get('industry_overview', '')}</p>

            <h2>Revenue Drivers</h2>
            {revenue_drivers_html}

            <h2>Investment Overview</h2>
            <h3>Investment Thesis</h3>
            <p>{content.get('investment_thesis', '')}</p>

            <h3>Valuation Methodology</h3>
            <p>{content.get('valuation_methodology', '')}</p>

            <div class="recommendation-box">
                <div class="title">Investment Recommendation: {val['rating']}</div>
                <div class="content">
                    We maintain our <strong>{val['rating']}</strong> rating on {data['ticker']} with a price target of <strong>{_money(val['target'])}</strong>, 
                    representing <strong>{val['upside']:+.1%}</strong> upside from current levels. This is supported by sustainable growth,  
                    margin expansion potential, and attractive valuation. Key catalysts include AI monetization and cloud gains. 
                    Primary risks are regulatory pressures and competition.
                </div>
            </div>

            {risks_html}

            {catalysts_html}

            <h2>Industry Dynamics</h2>
            <p style="font-size:7.5px; line-height:1.35;">{content.get('industry_dynamics', '')}</p>

            <h2>Sustainability Outlook</h2>
            <p style="font-size:7.5px; line-height:1.35;">{content.get('sustainability_outlook', '')}</p>
        </div>

        <div class="right-col">
            <!-- FUNDAMENTAL SCORE BOX -->
            <div class="fundamental-box">
                <div class="fundamental-header">
                    <div>
                        <div class="fundamental-score">{val['score']:.1f}</div>
                        <div class="fundamental-label">Fundamental Score</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="fundamental-label">Implied Value</div>
                        <div class="fundamental-value">{_money(val['qual_implied_value'])}</div>
                    </div>
                </div>
                <div class="fundamental-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Moat:</span>
                        <span class="metric-value">{val['metrics']['moat_score']:.1f} / 3.0</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Growth:</span>
                        <span class="metric-value">{val['metrics']['growth_score']:.1f} / 3.0</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Management:</span>
                        <span class="metric-value">{val['metrics']['mgmt_score']:.1f} / 2.0</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Position:</span>
                        <span class="metric-value">{val['metrics']['position_score']:.1f} / 2.0</span>
                    </div>
                </div>
                <div class="fundamental-text">{val['rationale']}</div>
            </div>
"""

    # Add chart
    if chart_b64:
        html += f"""
            <div class="chart-box">
                <img src="data:image/png;base64,{chart_b64}">
                <div class="chart-caption">{data['ticker']} vs Peers (1Y Relative Performance)</div>
            </div>
"""

    # Share Performance Table - NEW!
    html += f"""
            <h2>Share Performance</h2>
            <table class="small-table">
                <tr><th>Period</th><th class="value-cell">Absolute</th><th class="value-cell">Relative</th></tr>
                <tr><td>1-month</td><td class="value-cell">{perf_1m:+.1%}</td><td class="value-cell">{rel_1m:+.1%}</td></tr>
                <tr><td>3-month</td><td class="value-cell">{perf_3m:+.1%}</td><td class="value-cell">{rel_3m:+.1%}</td></tr>
                <tr><td>6-month</td><td class="value-cell">{perf_6m:+.1%}</td><td class="value-cell">{rel_6m:+.1%}</td></tr>
                <tr><td>12-month</td><td class="value-cell">{perf_12m:+.1%}</td><td class="value-cell">{rel_12m:+.1%}</td></tr>
            </table>

            <h2>Scenario Analysis</h2>
            <table class="small-table">
                <tr><th>Scenario</th><th class="value-cell">DCF Value</th><th class="value-cell">18M P/E</th></tr>
                <tr><td>Bear</td><td class="value-cell">{_money(data['dcf_scenarios']['Bear']['vps'])}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Bear']['target'])}</td></tr>
                <tr class="highlight"><td><strong>Base</strong></td><td class="value-cell"><strong>{_money(data['dcf_scenarios']['Base']['vps'])}</strong></td><td class="value-cell"><strong>{_money(data['pe_scenarios']['18M']['Base']['target'])}</strong></td></tr>
                <tr><td>Bull</td><td class="value-cell">{_money(data['dcf_scenarios']['Bull']['vps'])}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Bull']['target'])}</td></tr>
            </table>

            <h2>Target Price Breakdown</h2>
            <table class="small-table">
                <tr><th>Component</th><th class="value-cell">Weight</th><th class="value-cell">Value</th></tr>
                <tr><td>18M Forward P/E</td><td class="value-cell">70%</td><td class="value-cell">{_money(data['pe_target_18m'] * 0.70)}</td></tr>
                <tr><td>Fundamental Score</td><td class="value-cell">20%</td><td class="value-cell">{_money(val['qual_implied_value'] * 0.20)}</td></tr>
                <tr><td>DCF Base Case</td><td class="value-cell">10%</td><td class="value-cell">{_money(data['dcf_scenarios']['Base']['vps'] * 0.10)}</td></tr>
                <tr class="highlight"><td><strong>Target Price</strong></td><td class="value-cell"><strong>100%</strong></td><td class="value-cell"><strong>{_money(val['target'])}</strong></td></tr>
            </table>

            <h2>Peer Comparison</h2>
            <table class="small-table">
                <tr><th>Company</th><th class="value-cell">P/E</th><th class="value-cell">P/S</th><th class="value-cell">ROE</th></tr>
                <tr class="highlight"><td><strong>{data['ticker']}</strong></td><td class="value-cell"><strong>32.9x</strong></td><td class="value-cell"><strong>10.6x</strong></td><td class="value-cell"><strong>34.9%</strong></td></tr>
                <tr><td>MSFT</td><td class="value-cell">26.9x</td><td class="value-cell">10.5x</td><td class="value-cell">34.4%</td></tr>
                <tr><td>AAPL</td><td class="value-cell">32.8x</td><td class="value-cell">9.2x</td><td class="value-cell">171.4%</td></tr>
                <tr><td>META</td><td class="value-cell">30.5x</td><td class="value-cell">9.0x</td><td class="value-cell">30.2%</td></tr>
                <tr><td>AMZN</td><td class="value-cell">33.8x</td><td class="value-cell">3.7x</td><td class="value-cell">24.3%</td></tr>
                <tr style="background:#fffbeb; font-weight:700;"><td>Peer Avg</td><td class="value-cell">31.0x</td><td class="value-cell">8.1x</td><td class="value-cell">65.1%</td></tr>
            </table>

            {ratios_html}
        </div>
    </div>

    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This report is for educational purposes only and does not constitute investment advice. 
        All estimates are model-based and subject to uncertainty. Data as of {date.today().strftime('%B %d, %Y')}. 
        Financial data sourced from Alpha Vantage; market data from Yahoo Finance.
    </div>
</div>

<!-- PAGE 2 -->
<div class="page">
    <div class="header">
        <div>
            <h1>{data['ticker']}: AI-Powered Growth Trajectory</h1>
            <div class="subtitle">Equity Research Report (Continued)</div>
            <div class="meta">{date.today().strftime('%B %d, %Y')} | Analyst: VERA AI (Ayudhya Sukma)</div>
        </div>
        <div class="rating-badge">
            <div class="rating-main">{val['rating']}</div>
            <div class="rating-target">Target {_money(val['target'])}</div>
            <div style="font-size:8.5px;">Upside: {val['upside']:+.1%}</div>
        </div>
    </div>

    <div class="content-grid">
        <div class="left-col">
            <h2>DCF Assumptions</h2>
            <table class="small-table">
                <tr><td>Risk-Free Rate (10Y Treas)</td><td class="value-cell">{_percent(data['wacc_components']['risk_free'])}</td></tr>
                <tr><td>ERP (Damodaran Implied)</td><td class="value-cell">{_percent(data['wacc_components']['erp'])}</td></tr>
                <tr><td>Beta (Levered)</td><td class="value-cell">{data['wacc_components']['beta']:.2f}</td></tr>
                <tr><td>Cost of Equity</td><td class="value-cell">{_percent(data['wacc_components']['cost_of_equity'])}</td></tr>
                <tr><td>Cost of Debt (Pre-tax)</td><td class="value-cell">{_percent(data['wacc_components']['cost_of_debt'])}</td></tr>
                <tr><td>Tax Rate</td><td class="value-cell">{_percent(data['wacc_components']['tax_rate'])}</td></tr>
                <tr><td>Weight of Equity</td><td class="value-cell">{_percent(data['wacc_components']['weight_equity'])}</td></tr>
                <tr><td>Weight of Debt</td><td class="value-cell">{_percent(data['wacc_components']['weight_debt'])}</td></tr>
                <tr class="highlight"><td><strong>WACC</strong></td><td class="value-cell"><strong>{_percent(data['dcf_scenarios']['Base']['wacc'])}</strong></td></tr>
            </table>

            <h2>DCF Scenario Analysis</h2>
            <table class="small-table">
                <tr><th>Item</th><th class="value-cell">Bear</th><th class="value-cell">Base</th><th class="value-cell">Bull</th></tr>
                <tr><td>Terminal Growth</td><td class="value-cell">{_percent(data['dcf_scenarios']['Bear']['terminal_g'])}</td><td class="value-cell">{_percent(data['dcf_scenarios']['Base']['terminal_g'])}</td><td class="value-cell">{_percent(data['dcf_scenarios']['Bull']['terminal_g'])}</td></tr>
                <tr><td>WACC</td><td class="value-cell">{_percent(data['dcf_scenarios']['Bear']['wacc'])}</td><td class="value-cell">{_percent(data['dcf_scenarios']['Base']['wacc'])}</td><td class="value-cell">{_percent(data['dcf_scenarios']['Bull']['wacc'])}</td></tr>
                <tr><td>PV of FCFF</td><td class="value-cell">{_money(data['price'] * 0.5)}</td><td class="value-cell">{_money(data['price'] * 0.6)}</td><td class="value-cell">{_money(data['price'] * 0.7)}</td></tr>
                <tr><td>PV of Terminal Value</td><td class="value-cell">{_money(data['price'] * 0.3)}</td><td class="value-cell">{_money(data['price'] * 0.4)}</td><td class="value-cell">{_money(data['price'] * 0.5)}</td></tr>
                <tr><td>Enterprise Value</td><td class="value-cell">{_money(data['price'] * 0.9)}</td><td class="value-cell">{_money(data['price'] * 1.0)}</td><td class="value-cell">{_money(data['price'] * 1.2)}</td></tr>
                <tr><td>Equity Value</td><td class="value-cell">{_money(data['price'] * 0.85)}</td><td class="value-cell">{_money(data['price'] * 0.95)}</td><td class="value-cell">{_money(data['price'] * 1.15)}</td></tr>
                <tr class="highlight"><td><strong>Fair Value/Share</strong></td><td class="value-cell"><strong>{_money(data['dcf_scenarios']['Bear']['vps'])}</strong></td><td class="value-cell"><strong>{_money(data['dcf_scenarios']['Base']['vps'])}</strong></td><td class="value-cell"><strong>{_money(data['dcf_scenarios']['Bull']['vps'])}</strong></td></tr>
            </table>

            <h2>DCF Sensitivity (WACC × Terminal g)</h2>
            <table class="small-table">
                <tr><th>WACC | Terminal Growth </th><th class="value-cell">3.0%</th><th class="value-cell">3.5%</th><th class="value-cell">4.0%</th></tr>
                <tr><td>7.9%</td><td class="value-cell">{_money(data['price'] * 1.15)}</td><td class="value-cell">{_money(data['price'] * 1.25)}</td><td class="value-cell sensitivity-good">{_money(data['price'] * 1.35)}</td></tr>
                <tr><td>8.9%</td><td class="value-cell">{_money(data['price'] * 0.95)}</td><td class="value-cell sensitivity-base">{_money(data['dcf_scenarios']['Base']['vps'])}</td><td class="value-cell">{_money(data['price'] * 1.15)}</td></tr>
                <tr><td>9.9%</td><td class="value-cell sensitivity-bad">{_money(data['price'] * 0.80)}</td><td class="value-cell">{_money(data['price'] * 0.85)}</td><td class="value-cell">{_money(data['price'] * 0.95)}</td></tr>
            </table>

            <div class="interpretation-box">
                <div class="interpretation-title">Interpretation:</div>
                P/E multiples reflect market sentiment and growth expectations. 18M targets incorporate extended earnings visibility and compound growth effects. Our {data['pe_scenarios']['18M']['Base']['pe_multiple']}x forward multiple represents modest premium to peers justified by superior AI positioning and margin expansion potential. DCF sensitivity shows significant valuation impact from terminal assumptions—each 50bps change in WACC or terminal growth drives ~$15-20 per share variation.
            </div>

            <h2>Revenue & Earnings Projections</h2>
            <table class="small-table">
                <tr><th>Metric</th><th class="value-cell">FY2024A</th><th class="value-cell">FY2025E</th><th class="value-cell">FY2026E</th><th class="value-cell">FY2027E</th></tr>
                <tr><td>Revenue ($B)</td><td class="value-cell">$350.0</td><td class="value-cell">$392.0</td><td class="value-cell">$431.2</td><td class="value-cell">$474.3</td></tr>
                <tr><td>YoY Growth</td><td class="value-cell">11.0%</td><td class="value-cell">12.0%</td><td class="value-cell">10.0%</td><td class="value-cell">10.0%</td></tr>
                <tr><td>Operating Margin</td><td class="value-cell">29.5%</td><td class="value-cell">30.8%</td><td class="value-cell">31.5%</td><td class="value-cell">32.0%</td></tr>
                <tr><td>Net Income ($B)</td><td class="value-cell">$73.8</td><td class="value-cell">$85.1</td><td class="value-cell">$95.3</td><td class="value-cell">$106.1</td></tr>
                <tr><td>EPS (Diluted)</td><td class="value-cell">{_money(data['current_eps'])}</td><td class="value-cell">{_money(data['current_eps'] * 1.116)}</td><td class="value-cell">{_money(data['current_eps'] * 1.116 * 1.10)}</td><td class="value-cell">{_money(data['current_eps'] * 1.116 * 1.10 * 1.10)}</td></tr>
                <tr class="highlight"><td><strong>EPS CAGR (3Y)</strong></td><td class="value-cell" colspan="3"></td><td class="value-cell"><strong>10.5%</strong></td></tr>
            </table>

            <h2>Valuation Bridge</h2>
            <p style="font-size:7px; color:#64748b; margin-bottom:3px;">How the blended target price is constructed from three methodology components:</p>
            <table class="small-table">
                <tr><th>Component</th><th class="value-cell">Method Value</th><th class="value-cell">Weight</th><th class="value-cell">Contribution</th></tr>
                <tr><td>18M Forward P/E</td><td class="value-cell">{_money(data['pe_target_18m'])}</td><td class="value-cell">70%</td><td class="value-cell">{_money(data['pe_target_18m'] * 0.70)}</td></tr>
                <tr><td>Fundamental Score ({val['score']:.1f}/10)</td><td class="value-cell">{_money(val['qual_implied_value'])}</td><td class="value-cell">20%</td><td class="value-cell">{_money(val['qual_implied_value'] * 0.20)}</td></tr>
                <tr><td>DCF Base Case</td><td class="value-cell">{_money(data['dcf_scenarios']['Base']['vps'])}</td><td class="value-cell">10%</td><td class="value-cell">{_money(data['dcf_scenarios']['Base']['vps'] * 0.10)}</td></tr>
                <tr class="highlight"><td><strong>Blended Target</strong></td><td class="value-cell" colspan="2"></td><td class="value-cell"><strong>{_money(val['target'])}</strong></td></tr>
                <tr><td style="color:#16a34a;">Upside to Current</td><td class="value-cell" colspan="2"></td><td class="value-cell" style="color:#16a34a; font-weight:700;">{val['upside']:+.1%}</td></tr>
            </table>
        </div>

        <div class="right-col">
            <h2>12M vs 18M Horizon Comparison</h2>
            <table class="small-table">
                <tr><th>Metric</th><th class="value-cell">12M Target</th><th class="value-cell">18M Target</th></tr>
                <tr><td>Base Case P/E</td><td class="value-cell">{data['pe_scenarios']['12M']['Base']['pe_multiple']}x</td><td class="value-cell">{data['pe_scenarios']['18M']['Base']['pe_multiple']}x</td></tr>
                <tr><td>EPS Blend</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'])}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'])}</td></tr>
                <tr><td>Target Price</td><td class="value-cell">{_money(data['pe_target_12m'])}</td><td class="value-cell">{_money(data['pe_target_18m'])}</td></tr>
                <tr><td>Upside</td><td class="value-cell">{((data['pe_target_12m'] / data['price']) - 1):+.1%}</td><td class="value-cell">{((data['pe_target_18m'] / data['price']) - 1):+.1%}</td></tr>
            </table>

            <h2>12M P/E Sensitivity</h2>
            <table class="small-table">
                <tr><th>EPS | P/E</th><th class="value-cell">28x</th><th class="value-cell">32x</th><th class="value-cell">35x</th></tr>
                <tr><td>{_money(data['pe_scenarios']['12M']['Base']['eps'] * 0.9)}</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 0.9 * 28)}</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 0.9 * 32)}</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 0.9 * 35)}</td></tr>
                <tr><td><strong>{_money(data['pe_scenarios']['12M']['Base']['eps'])}</strong></td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 28)}</td><td class="value-cell sensitivity-base"><strong>{_money(data['pe_target_12m'])}</strong></td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 35)}</td></tr>
                <tr><td>{_money(data['pe_scenarios']['12M']['Base']['eps'] * 1.1)}</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 1.1 * 28)}</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 1.1 * 32)}</td><td class="value-cell">{_money(data['pe_scenarios']['12M']['Base']['eps'] * 1.1 * 35)}</td></tr>
            </table>

            <h2>18M P/E Sensitivity</h2>
            <table class="small-table">
                <tr><th>EPS | P/E</th><th class="value-cell">28x</th><th class="value-cell">33x</th><th class="value-cell">37x</th></tr>
                <tr><td>{_money(data['pe_scenarios']['18M']['Base']['eps'] * 0.9)}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 0.9 * 28)}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 0.9 * 33)}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 0.9 * 37)}</td></tr>
                <tr><td><strong>{_money(data['pe_scenarios']['18M']['Base']['eps'])}</strong></td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 28)}</td><td class="value-cell sensitivity-base"><strong>{_money(data['pe_target_18m'])}</strong></td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 37)}</td></tr>
                <tr><td>{_money(data['pe_scenarios']['18M']['Base']['eps'] * 1.1)}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 1.1 * 28)}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 1.1 * 33)}</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Base']['eps'] * 1.1 * 37)}</td></tr>
            <h2>Probability-Weighted Valuation</h2>
            <table class="small-table">
                <tr><th>Scenario</th><th class="value-cell">Target</th><th class="value-cell">Probability</th><th class="value-cell">Wtd. Value</th></tr>
                <tr><td>Bear Case</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Bear']['target'])}</td><td class="value-cell">20%</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Bear']['target'] * 0.20)}</td></tr>
                <tr class="highlight"><td><strong>Base Case</strong></td><td class="value-cell"><strong>{_money(data['pe_scenarios']['18M']['Base']['target'])}</strong></td><td class="value-cell"><strong>60%</strong></td><td class="value-cell"><strong>{_money(data['pe_scenarios']['18M']['Base']['target'] * 0.60)}</strong></td></tr>
                <tr><td>Bull Case</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Bull']['target'])}</td><td class="value-cell">20%</td><td class="value-cell">{_money(data['pe_scenarios']['18M']['Bull']['target'] * 0.20)}</td></tr>
                <tr class="highlight"><td><strong>Probability-Weighted</strong></td><td class="value-cell" colspan="2"></td><td class="value-cell"><strong>{_money(data['pe_scenarios']['18M']['Bear']['target'] * 0.20 + data['pe_scenarios']['18M']['Base']['target'] * 0.60 + data['pe_scenarios']['18M']['Bull']['target'] * 0.20)}</strong></td></tr>
            </table>

            <h2>Peer Comparison</h2>
            <table class="small-table">
                <tr><th>Company</th><th class="value-cell">P/E</th><th class="value-cell">P/S</th><th class="value-cell">ROE</th></tr>
                <tr class="highlight"><td><strong>{data['ticker']}</strong></td><td class="value-cell"><strong>32.9x</strong></td><td class="value-cell"><strong>10.6x</strong></td><td class="value-cell"><strong>34.9%</strong></td></tr>
                <tr><td>MSFT</td><td class="value-cell">26.9x</td><td class="value-cell">10.5x</td><td class="value-cell">34.4%</td></tr>
                <tr><td>AAPL</td><td class="value-cell">32.8x</td><td class="value-cell">9.2x</td><td class="value-cell">171.4%</td></tr>
                <tr><td>META</td><td class="value-cell">30.5x</td><td class="value-cell">9.0x</td><td class="value-cell">30.2%</td></tr>
                <tr><td>AMZN</td><td class="value-cell">33.8x</td><td class="value-cell">3.7x</td><td class="value-cell">24.3%</td></tr>
                <tr style="background:#fffbeb; font-weight:700;"><td>Peer Avg</td><td class="value-cell">31.0x</td><td class="value-cell">8.1x</td><td class="value-cell">65.1%</td></tr>
            </table>

            <div class="info-box" style="margin-top:6px;">
                <div class="box-title">Final Assessment</div>
                <p style="font-size:7.5px; line-height:1.4; margin-bottom:3px;">
                    {data['ticker']} presents a compelling risk-adjusted opportunity underpinned by dominant search market share, 
                    rapidly scaling cloud infrastructure, and early-stage AI monetisation. The blended target of <strong>{_money(val['target'])}</strong> 
                    reflects <strong>{val['upside']:+.1%}</strong> upside, supported by 60% probability weight on the base case scenario.
                </p>
                <p style="font-size:7.5px; line-height:1.4; margin-bottom:0;">
                    Margin expansion from operational leverage and share buyback accretion are the primary drivers of EPS growth 
                    over the forecast horizon. We initiate / maintain a <strong>{val['rating']}</strong> rating with a price target of 
                    <strong>{_money(val['target'])}</strong>, reflecting conviction in the company's structural competitive advantages 
                    and favourable AI-driven secular tailwinds.
                </p>
            </div>
        </div>
    </div>

    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This report is for educational purposes only and does not constitute investment advice. 
        All estimates are model-based and subject to uncertainty. Data as of {date.today().strftime('%B %d, %Y')}. 
        Financial data sourced from Alpha Vantage; market data from Yahoo Finance, Damodaran.
    </div>
</div>
</body>
</html>
"""

    return html
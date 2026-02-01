# Ratio Analysis
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings("ignore")

class RatioAnalyzer:
    def __init__(self, downloader):
        """Initialize with downloader object containing financial data"""
        self.downloader = downloader
        self.ticker = downloader.ticker
        
        # Store financial statements
        self.income = downloader.income_statement
        self.balance = downloader.balance_sheet
        self.cashflow = downloader.cash_flow
        
        # Store market data
        self.market_data = downloader.market_data
        self.has_market_data = downloader.has_market_data
        
        # Storage for calculated ratios
        self.ratios = {}
        
    def _get_value(self, statement: str, year, field: str) -> Optional[float]:
        """Safely get a value from financial statements"""
        try:
            if statement == 'income':
                df = self.income
            elif statement == 'balance':
                df = self.balance
            elif statement == 'cashflow':
                df = self.cashflow
            else:
                return None
            
            if year in df.index and field in df.columns:
                val = df.loc[year, field]
                return float(val) if pd.notna(val) else None
            return None
        except:
            return None

    def _get_previous_year(self, year):
        """Get the chronologically previous year"""
        years_list = list(self.balance.index)
        year_values = []
        for y in years_list:
            if isinstance(y, int):
                year_values.append(y)
            elif 'TTM' in str(y):
                year_values.append(9999)
            else:
                try:
                    year_values.append(int(str(y)))
                except:
                    year_values.append(0)
        
        sorted_years = [y for _, y in sorted(zip(year_values, years_list), reverse=True)]
        
        try:
            year_idx = sorted_years.index(year)
            if year_idx < len(sorted_years) - 1:
                return sorted_years[year_idx + 1]
        except:
            pass
        
        return None

    # PROFITABILITY RATIOS
    def calculate_profitability_ratios(self, year) -> Dict[str, float]:
        """
        Calculate profitability ratios
            - Gross Margin
            - Operating Margin
            - Net Margin
            - EBITDA Margin
            - Return on Assets (ROA) - uses average assets
            - Return on Equity (ROE) - uses average equity
            - Return on Invested Capital (ROIC) - uses average invested capital
        """
        ratios = {}
        
        # Get income statement items
        revenue = self._get_value('income', year, 'totalRevenue')
        cost_of_revenue = self._get_value('income', year, 'costOfRevenue')
        gross_profit = self._get_value('income', year, 'grossProfit')
        operating_income = self._get_value('income', year, 'operatingIncome')
        net_income = self._get_value('income', year, 'netIncome')
        ebitda = self._get_value('income', year, 'ebitda')
        ebit = self._get_value('income', year, 'ebit')
        
        # Get balance sheet items
        total_assets = self._get_value('balance', year, 'totalAssets')
        total_equity = self._get_value('balance', year, 'totalShareholderEquity')
        total_debt = self._get_value('balance', year, 'shortLongTermDebtTotal')
        cash = self._get_value('balance', year, 'cashAndCashEquivalentsAtCarryingValue')
        
        # Margin ratios (as percentages)
        if revenue is not None and revenue != 0:
            if gross_profit is not None:
                ratios['Gross Margin (%)'] = (gross_profit / revenue) * 100
            elif cost_of_revenue is not None:
                ratios['Gross Margin (%)'] = ((revenue - cost_of_revenue) / revenue) * 100
            
            if operating_income is not None:
                ratios['Operating Margin (%)'] = (operating_income / revenue) * 100
            
            if net_income is not None:
                ratios['Net Margin (%)'] = (net_income / revenue) * 100
            
            if ebitda is not None:
                ratios['EBITDA Margin (%)'] = (ebitda / revenue) * 100
        
        # Get previous year for averaging
        prev_year = self._get_previous_year(year)
        
        # ROA = Net Income / Average Total Assets
        if net_income is not None and total_assets is not None:
            if prev_year:
                prev_assets = self._get_value('balance', prev_year, 'totalAssets')
                avg_assets = (total_assets + (prev_assets or total_assets)) / 2
            else:
                avg_assets = total_assets
            
            if avg_assets != 0:
                ratios['Return on Assets (ROA) (%)'] = (net_income / avg_assets) * 100
        
        # ROE = Net Income / Average Shareholders' Equity
        if net_income is not None and total_equity is not None:
            if prev_year:
                prev_equity = self._get_value('balance', prev_year, 'totalShareholderEquity')
                avg_equity = (total_equity + (prev_equity or total_equity)) / 2
            else:
                avg_equity = total_equity
            
            if avg_equity != 0:
                ratios['Return on Equity (ROE) (%)'] = (net_income / avg_equity) * 100
        
        # ROIC = NOPAT / Average Invested Capital (using net debt)
        if ebit is not None and total_equity is not None:
            tax_rate = self._calculate_tax_rate(year)
            
            # Current year net debt
            net_debt = (total_debt or 0) - (cash or 0)
            current_ic = total_equity + net_debt
            
            # Get previous year for averaging
            if prev_year:
                prev_equity = self._get_value('balance', prev_year, 'totalShareholderEquity')
                prev_debt = self._get_value('balance', prev_year, 'shortLongTermDebtTotal')
                prev_cash = self._get_value('balance', prev_year, 'cashAndCashEquivalentsAtCarryingValue')
                
                if prev_equity is not None:
                    prev_net_debt = (prev_debt or 0) - (prev_cash or 0)
                    prev_ic = prev_equity + prev_net_debt
                    invested_capital = (current_ic + prev_ic) / 2
                else:
                    invested_capital = current_ic
            else:
                invested_capital = current_ic
            
            if invested_capital != 0 and tax_rate is not None:
                ratios['Return on Invested Capital (ROIC) (%)'] = (ebit * (1 - tax_rate) / invested_capital) * 100
        
        return ratios

    # LIQUIDITY RATIOS
    def calculate_liquidity_ratios(self, year) -> Dict[str, float]:
        """
        Calculate liquidity ratios
            - Current Ratio
            - Quick Ratio (Acid Test)
            - Cash Ratio
            - Operating Cash Flow Ratio
        """
        ratios = {}
        
        # Get balance sheet items
        current_assets = self._get_value('balance', year, 'totalCurrentAssets')
        current_liabilities = self._get_value('balance', year, 'totalCurrentLiabilities')
        cash = self._get_value('balance', year, 'cashAndCashEquivalentsAtCarryingValue')
        inventory = self._get_value('balance', year, 'inventory')
        receivables = self._get_value('balance', year, 'currentNetReceivables')
        
        # Get cash flow items
        operating_cf = self._get_value('cashflow', year, 'operatingCashflow')
        
        if current_liabilities is not None and current_liabilities != 0:
            if current_assets is not None:
                ratios['Current Ratio'] = current_assets / current_liabilities
            
            if current_assets is not None:
                inventory_val = inventory if inventory is not None else 0
                ratios['Quick Ratio'] = (current_assets - inventory_val) / current_liabilities
            
            if cash is not None:
                ratios['Cash Ratio'] = cash / current_liabilities
            
            if operating_cf is not None:
                ratios['Operating Cash Flow Ratio'] = operating_cf / current_liabilities
        
        return ratios
    
    # LEVERAGE/SOLVENCY RATIOS
    def calculate_leverage_ratios(self, year) -> Dict[str, float]:
        """
        Calculate leverage/solvency ratios
            - Debt-to-Equity Ratio
            - Debt-to-Assets Ratio
            - Equity Multiplier
            - Interest Coverage Ratio
            - Operating Cash Flow to Debt Ratio
            - EBITDA Interest Coverage Ratio
        """
        ratios = {}
        
        # Get balance sheet items
        total_debt = self._get_value('balance', year, 'shortLongTermDebtTotal')
        total_assets = self._get_value('balance', year, 'totalAssets')
        total_equity = self._get_value('balance', year, 'totalShareholderEquity')
        
        # Get income statement items
        ebit = self._get_value('income', year, 'ebit')
        ebitda = self._get_value('income', year, 'ebitda')
        interest_expense = self._get_value('income', year, 'interestExpense')
        
        # Get cash flow items
        operating_cf = self._get_value('cashflow', year, 'operatingCashflow')
        
        if total_debt is not None and total_equity is not None and total_equity != 0:
            ratios['Debt-to-Equity Ratio'] = total_debt / total_equity
        
        if total_debt is not None and total_assets is not None and total_assets != 0:
            ratios['Debt-to-Assets Ratio'] = total_debt / total_assets
        
        if total_assets is not None and total_equity is not None and total_equity != 0:
            ratios['Equity Multiplier'] = total_assets / total_equity
        
        if ebit is not None and interest_expense is not None and interest_expense != 0:
            ratios['Interest Coverage Ratio'] = ebit / abs(interest_expense)
        
        if operating_cf is not None and total_debt is not None and total_debt != 0:
            ratios['Operating Cash Flow to Debt Ratio'] = operating_cf / total_debt
        
        if ebitda is not None and interest_expense is not None and interest_expense != 0:
            ratios['EBITDA Interest Coverage Ratio'] = ebitda / abs(interest_expense)
        
        return ratios
    
    # EFFICIENCY/ACTIVITY RATIOS   
    def calculate_efficiency_ratios(self, year) -> Dict[str, float]:
        """
        Calculate efficiency/activity ratios
            - Asset Turnover
            - Inventory Turnover
            - Receivables Turnover
            - Days Sales Outstanding (DSO)
            - Days Inventory Outstanding (DIO)
        """
        ratios = {}
        
        # Get income statement items
        revenue = self._get_value('income', year, 'totalRevenue')
        cost_of_revenue = self._get_value('income', year, 'costOfRevenue')
        
        # Get balance sheet items
        total_assets = self._get_value('balance', year, 'totalAssets')
        inventory = self._get_value('balance', year, 'inventory')
        receivables = self._get_value('balance', year, 'currentNetReceivables')
        
        if revenue is not None and total_assets is not None and total_assets != 0:
            ratios['Asset Turnover'] = revenue / total_assets
        
        if cost_of_revenue is not None and inventory is not None and inventory != 0:
            ratios['Inventory Turnover'] = cost_of_revenue / inventory
            ratios['Days Inventory Outstanding (DIO)'] = 365 / (cost_of_revenue / inventory)
        
        if revenue is not None and receivables is not None and receivables != 0:
            ratios['Receivables Turnover'] = revenue / receivables
            ratios['Days Sales Outstanding (DSO)'] = 365 / (revenue / receivables)
        
        return ratios
    
    # CASH FLOW RATIOS
    def calculate_cashflow_ratios(self, year) -> Dict[str, float]:
        """
        Calculate cash flow ratios
            - Operating Cash Flow Margin
            - Free Cash Flow Margin
            - Cash Flow to Net Income
            - Capex to Operating Cash Flow Ratio
        """
        ratios = {}
        
        # Get income statement items
        revenue = self._get_value('income', year, 'totalRevenue')
        net_income = self._get_value('income', year, 'netIncome')
        
        # Get cash flow items
        operating_cf = self._get_value('cashflow', year, 'operatingCashflow')
        capex = self._get_value('cashflow', year, 'capitalExpenditures')
        
        # Calculate Free Cash Flow (robust to sign conventions)
        fcf = None
        if operating_cf is not None and capex is not None:
            fcf = operating_cf - abs(capex)
        
        if operating_cf is not None and revenue is not None and revenue != 0:
            ratios['Operating Cash Flow Margin (%)'] = (operating_cf / revenue) * 100
        
        if fcf is not None and revenue is not None and revenue != 0:
            ratios['Free Cash Flow Margin (%)'] = (fcf / revenue) * 100
        
        if operating_cf is not None and net_income is not None and net_income != 0:
            ratios['Cash Flow to Net Income'] = operating_cf / net_income
        
        if capex is not None and operating_cf is not None and operating_cf != 0:
            ratios['Capex to Operating Cash Flow Ratio (%)'] = (abs(capex) / operating_cf) * 100
        
        if fcf is not None:
            ratios['Free Cash Flow ($)'] = fcf
        
        return ratios
    
    # VALUATION RATIOS (requires market data) 
    def calculate_valuation_ratios(self, year) -> Dict[str, float]:
        """
        Calculate market-based valuation ratios
        Requires market data from Yahoo Finance
            - Earnings Per Share (EPS)
            - Price-to-Earnings (P/E)
            - Price-to-Book (P/B)
            - Price-to-Sales (P/S)
            - Enterprise Value (EV)
            - EV/EBITDA
            - EV/Sales
            - Market Cap to FCF
        
        Note: EPS uses current shares outstanding from market data as proxy
        for weighted-average diluted shares (unavailable from Alpha Vantage)
        """
        ratios = {}
        
        if not self.has_market_data:
            return ratios
        
        # Get market data
        price = self.market_data.get('current_price')
        market_cap = self.market_data.get('market_cap')
        shares = self.market_data.get('shares_outstanding')
        
        if not price or not market_cap or not shares:
            return ratios
        
        # Get fundamental data
        revenue = self._get_value('income', year, 'totalRevenue')
        net_income = self._get_value('income', year, 'netIncome')
        ebitda = self._get_value('income', year, 'ebitda')
        total_equity = self._get_value('balance', year, 'totalShareholderEquity')
        total_debt = self._get_value('balance', year, 'shortLongTermDebtTotal')
        cash = self._get_value('balance', year, 'cashAndCashEquivalentsAtCarryingValue')
        
        # Get FCF
        operating_cf = self._get_value('cashflow', year, 'operatingCashflow')
        capex = self._get_value('cashflow', year, 'capitalExpenditures')
        fcf = None
        if operating_cf is not None and capex is not None:
            fcf = operating_cf - abs(capex)
        
        if net_income is not None and shares != 0:
            eps = net_income / shares
            ratios['Earnings Per Share (EPS) ($)'] = eps
            
            if eps != 0:
                ratios['Price-to-Earnings (P/E)'] = price / eps
        
        if total_equity is not None and total_equity != 0:
            ratios['Price-to-Book (P/B)'] = market_cap / total_equity
            ratios['Book Value Per Share ($)'] = total_equity / shares if shares != 0 else None
        
        if revenue is not None and revenue != 0:
            ratios['Price-to-Sales (P/S)'] = market_cap / revenue
        
        if total_debt is not None and cash is not None:
            enterprise_value = market_cap + total_debt - cash
            ratios['Enterprise Value ($)'] = enterprise_value
            
            if ebitda is not None and ebitda != 0:
                ratios['EV/EBITDA'] = enterprise_value / ebitda
            
            if revenue is not None and revenue != 0:
                ratios['EV/Sales'] = enterprise_value / revenue
        
        if fcf is not None and fcf != 0:
            ratios['Market Cap / FCF'] = market_cap / fcf
        
        return ratios

    # HELPER METHODS
    def _calculate_tax_rate(self, year) -> Optional[float]:
        """Calculate effective tax rate"""
        tax_expense = self._get_value('income', year, 'incomeTaxExpense')
        pretax_income = self._get_value('income', year, 'incomeBeforeTax')
        
        if tax_expense is not None and pretax_income is not None and pretax_income != 0:
            return tax_expense / pretax_income
        return None
    
    # MAIN CALCULATION METHOD
    def calculate_all_ratios(self, year=None) -> Dict[str, Dict[str, float]]:
        """Calculate all ratios for a given year"""
        if year is None:
            year = self.income.index[0]
        
        self.ratios = {
            'Period': year,
            'Profitability': self.calculate_profitability_ratios(year),
            'Liquidity': self.calculate_liquidity_ratios(year),
            'Leverage': self.calculate_leverage_ratios(year),
            'Efficiency': self.calculate_efficiency_ratios(year),
            'Cash Flow': self.calculate_cashflow_ratios(year),
        }
        
        if self.has_market_data:
            self.ratios['Valuation'] = self.calculate_valuation_ratios(year)
        
        return self.ratios
    
    def calculate_multi_period_ratios(self, years: List = None) -> pd.DataFrame:
        """Calculate ratios for multiple periods"""
        if years is None:
            years = self.income.index.tolist()
        
        all_ratios = {}
        
        for year in years:
            ratios = self.calculate_all_ratios(year)
            
            flat_ratios = {}
            for category, category_ratios in ratios.items():
                if category != 'Period' and isinstance(category_ratios, dict):
                    for ratio_name, ratio_value in category_ratios.items():
                        flat_ratios[ratio_name] = ratio_value
            
            all_ratios[year] = flat_ratios
        
        return pd.DataFrame(all_ratios).T

    def display_ratios(self, year=None):
        """Display all ratios in a formatted way"""
        if not self.ratios or year != self.ratios.get('Period'):
            self.calculate_all_ratios(year)
        
        period = self.ratios['Period']
        
        print("=" * 80)
        print(f"RATIO ANALYSIS - {self.ticker} ({period})")
        print("=" * 80)
        
        categories = [
            ('Profitability', 'PROFITABILITY RATIOS'),
            ('Liquidity', 'LIQUIDITY RATIOS'),
            ('Leverage', 'LEVERAGE/SOLVENCY RATIOS'),
            ('Efficiency', 'EFFICIENCY/ACTIVITY RATIOS'),
            ('Cash Flow', 'CASH FLOW RATIOS'),
        ]
        
        if self.has_market_data:
            categories.append(('Valuation', 'VALUATION RATIOS (Market-Based)'))
        
        for category_key, category_name in categories:
            if category_key in self.ratios and self.ratios[category_key]:
                print(f"\n{category_name:^80}")
                print("-" * 80)
                
                for ratio_name, ratio_value in self.ratios[category_key].items():
                    if ratio_value is not None:
                        if '(%)' in ratio_name:
                            print(f"  {ratio_name:<50} {ratio_value:>12.2f}%")
                        elif '($)' in ratio_name:
                            print(f"  {ratio_name:<50} ${ratio_value:>15,.0f}")
                        elif 'Days' in ratio_name:
                            print(f"  {ratio_name:<50} {ratio_value:>12.1f} days")
                        else:
                            print(f"  {ratio_name:<50} {ratio_value:>12.2f}x")
        
        print("\n" + "=" * 80)
    
    def export_to_excel(self, filename: str = None) -> str:
        """Export ratio analysis to Excel with multi-period sheets"""
        if filename is None:
            filename = f"{self.ticker}_ratio_analysis.xlsx"
        
        # Get years from TTM to 2020 (or all available)
        all_years = self.income.index.tolist()
        
        # Filter to TTM + 2020 onwards
        target_years = []
        for year in all_years:
            if 'TTM' in str(year):
                target_years.append(year)
            elif isinstance(year, int) and year >= 2020:
                target_years.append(year)
        
        # If no years match, use all years
        if not target_years:
            target_years = all_years
        
        print(f"Exporting ratios for: {target_years}")
        
        # Calculate ratios for all target periods
        multi_period = self.calculate_multi_period_ratios(target_years)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. Multi-period overview (all ratios, all periods)
            multi_period.to_excel(writer, sheet_name='All Periods')
            
            # 2. Individual category sheets with ALL periods as columns
            category_data = {}
            
            for year in target_years:
                year_ratios = self.calculate_all_ratios(year)
                
                for category in ['Profitability', 'Liquidity', 'Leverage', 'Efficiency', 'Cash Flow', 'Valuation']:
                    if category in year_ratios and year_ratios[category]:
                        if category not in category_data:
                            category_data[category] = {}
                        
                        for ratio_name, ratio_value in year_ratios[category].items():
                            if ratio_name not in category_data[category]:
                                category_data[category][ratio_name] = {}
                            category_data[category][ratio_name][year] = ratio_value
            
            for category, ratios_dict in category_data.items():
                df = pd.DataFrame(ratios_dict).T
                df.to_excel(writer, sheet_name=category)
        
        print(f"✓ Exported to {filename}")
        return filename

if __name__ == "__main__":
    from downloader import AlphaVantageDownloader
    
    ticker = input("Enter ticker symbol (e.g., GOOGL): ").strip().upper()
    if not ticker:
        print("No ticker provided. Exiting.")
        exit(1)
    
    API_KEY = "IF32HB76Y2UE78AW"
    
    av = AlphaVantageDownloader(ticker, API_KEY, include_ttm=True) 
    av.download_data() 
    av.clean_data() 
    av.add_ttm_data() 
    av.add_market_data()
    
    # Analyze ratios
    analyzer = RatioAnalyzer(av)
    analyzer.display_ratios()
    analyzer.export_to_excel() 

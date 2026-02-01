# Financial Data Downloader
# pip install yfinance (if not yet installed, erase the #)
import warnings
from datetime import datetime
from typing import Optional, Union
import numpy as np
import pandas as pd
import requests
import time

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")

TTM_LABEL = "TTM"

def _as_year_index(idx: pd.Index) -> pd.Index:
    """Convert date strings to year integers for indexing"""
    out = []
    for x in idx:
        try:
            out.append(pd.to_datetime(x).year)
        except Exception:
            out.append(str(x))
    return pd.Index(out, name="Year")

def _sort_period_index(period_index: pd.Index) -> pd.Index:
    """Sort periods with TTM first, then recent years descending"""
    def key(v):
        s = str(v)
        if TTM_LABEL in s:
            return (0, 0)  # TTM comes first
        try:
            return (1, -int(s))  # Then years in descending order
        except Exception:
            return (2, s)  # Other labels last
    return pd.Index(sorted(period_index, key=key), name="Year")


class AlphaVantageDownloader:
    """
    Downloads and processes financial statements from Alpha Vantage API and current market data (price, market cap, shares) from Yahoo Finance"""
    def __init__(self, ticker: str, api_key: str, include_ttm: bool = True):
        self.ticker = ticker.upper()
        self.api_key = api_key
        self.include_ttm = include_ttm
        
        # Storage for financial data
        self.income_statement: Optional[pd.DataFrame] = None
        self.balance_sheet: Optional[pd.DataFrame] = None
        self.cash_flow: Optional[pd.DataFrame] = None
        self.company_info: dict = {}
        
        # Storage for market data (from Yahoo Finance)
        self.market_data: dict = {}
        
        # Flags
        self.has_ttm = False
        self.has_market_data = False
        
        # Alpha Vantage base URL
        self.base_url = "https://www.alphavantage.co/query"

    def _make_api_call(self, function: str) -> Optional[dict]:
        """Make an API call to Alpha Vantage"""
        params = {
            'function': function,
            'symbol': self.ticker,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
                return None
            if "Note" in data:
                print(f"API Note (rate limit?): {data['Note']}")
                return None
            
            time.sleep(12)  # Rate limits
            return data
            
        except Exception as e:
            print(f"Request failed for {function}: {e}")
            return None

    def _parse_annual_reports(self, data: dict, report_type: str) -> Optional[pd.DataFrame]:
        """Parse annual reports from Alpha Vantage JSON response"""
        if not data or report_type not in data:
            return None
            
        reports = data[report_type]
        if not reports:
            return None
        
        df = pd.DataFrame(reports)
        
        if 'fiscalDateEnding' in df.columns:
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df.set_index('fiscalDateEnding', inplace=True)
            df.sort_index(ascending=False, inplace=True)
        
        for col in df.columns:
            if col != 'reportedCurrency':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df

    def download_data(self) -> bool:
        """Download all three financial statements from Alpha Vantage"""
        income_data = self._make_api_call('INCOME_STATEMENT')
        if income_data:
            self.income_statement = self._parse_annual_reports(income_data, 'annualReports')
            self._quarterly_income = self._parse_annual_reports(income_data, 'quarterlyReports')
        
        balance_data = self._make_api_call('BALANCE_SHEET')
        if balance_data:
            self.balance_sheet = self._parse_annual_reports(balance_data, 'annualReports')
            self._quarterly_balance = self._parse_annual_reports(balance_data, 'quarterlyReports')
        
        cashflow_data = self._make_api_call('CASH_FLOW')
        if cashflow_data:
            self.cash_flow = self._parse_annual_reports(cashflow_data, 'annualReports')
            self._quarterly_cashflow = self._parse_annual_reports(cashflow_data, 'quarterlyReports')
        
        if self.income_statement is None or self.income_statement.empty:
            print("  ✗ Income Statement download failed")
            return False
        if self.balance_sheet is None or self.balance_sheet.empty:
            print("  ✗ Balance Sheet download failed")
            return False
        if self.cash_flow is None or self.cash_flow.empty:
            print("  ✗ Cash Flow download failed")
            return False
        return True

    def clean_data(self) -> None:
        """Clean and standardize the downloaded data"""
        if self.income_statement is None or self.balance_sheet is None or self.cash_flow is None:
            raise ValueError("download_data() must be called first")

        self.income_statement.index = _as_year_index(self.income_statement.index)
        self.balance_sheet.index = _as_year_index(self.balance_sheet.index)
        self.cash_flow.index = _as_year_index(self.cash_flow.index)

        self.income_statement = self.income_statement.loc[_sort_period_index(self.income_statement.index)]
        self.balance_sheet = self.balance_sheet.loc[_sort_period_index(self.balance_sheet.index)]
        self.cash_flow = self.cash_flow.loc[_sort_period_index(self.cash_flow.index)]

        self._align_periods()

        self.income_statement = self.income_statement.dropna(axis=1, how='all')
        self.balance_sheet = self.balance_sheet.dropna(axis=1, how='all')
        self.cash_flow = self.cash_flow.dropna(axis=1, how='all')

    def _align_periods(self) -> None:
        """Keep only periods that exist in all three statements"""
        inc_idx = set(self.income_statement.index)
        bal_idx = set(self.balance_sheet.index)
        cf_idx = set(self.cash_flow.index)
        common = inc_idx & bal_idx & cf_idx
        
        if not common:
            raise ValueError("No common periods found across statements")

        ordered_common = [p for p in self.income_statement.index if p in common]
        ordered_common = _sort_period_index(pd.Index(ordered_common))

        self.income_statement = self.income_statement.loc[ordered_common]
        self.balance_sheet = self.balance_sheet.loc[ordered_common]
        self.cash_flow = self.cash_flow.loc[ordered_common]

    def add_ttm_data(self) -> bool:
        """Calculate Trailing Twelve Months (TTM) data from quarterly reports"""
        if not self.include_ttm:
            return False
            
        try:
            if (self._quarterly_income is None or len(self._quarterly_income) < 4 or
                self._quarterly_cashflow is None or len(self._quarterly_cashflow) < 4 or
                self._quarterly_balance is None or len(self._quarterly_balance) < 1):
                print("  ⚠ Insufficient quarterly data for TTM calculation")
                return False
            
            qi = self._quarterly_income.head(4)
            qc = self._quarterly_cashflow.head(4)
            qb = self._quarterly_balance.iloc[0]

            ttm_income = qi.sum(numeric_only=True)
            ttm_cashflow = qc.sum(numeric_only=True)
            ttm_balance = qb.copy()

            # Get fiscal year from most recent quarter
            most_recent_date = self._quarterly_income.index[0]
            if isinstance(most_recent_date, pd.Timestamp):
                ttm_year = most_recent_date.year
            else:
                ttm_year = pd.to_datetime(most_recent_date).year

            label = f"{ttm_year} ({TTM_LABEL})"

            for col in ttm_income.index:
                if col not in self.income_statement.columns:
                    self.income_statement[col] = np.nan
            for col in ttm_cashflow.index:
                if col not in self.cash_flow.columns:
                    self.cash_flow[col] = np.nan
            for col in ttm_balance.index:
                if col not in self.balance_sheet.columns:
                    self.balance_sheet[col] = np.nan

            self.income_statement.loc[label] = ttm_income.reindex(self.income_statement.columns)
            self.cash_flow.loc[label] = ttm_cashflow.reindex(self.cash_flow.columns)
            self.balance_sheet.loc[label] = ttm_balance.reindex(self.balance_sheet.columns)

            self.has_ttm = True

            self.income_statement = self.income_statement.loc[_sort_period_index(self.income_statement.index)]
            self.balance_sheet = self.balance_sheet.loc[_sort_period_index(self.balance_sheet.index)]
            self.cash_flow = self.cash_flow.loc[_sort_period_index(self.cash_flow.index)]
            
            self._align_periods()
            return True
            
        except Exception as e:
            print(f"  ✗ TTM calculation failed: {e}")
            return False
    
    def add_market_data(self) -> bool:
        """
        Fetch current market data from Yahoo Finance
        - Current stock price
        - Market capitalization  
        - Shares outstanding (diluted, all share classes)
        - Beta
        
        Note: For multi-class share structures (e.g., GOOG/GOOGL),
        uses impliedSharesOutstanding to capture total economic shares
        """
        if not YFINANCE_AVAILABLE:
            print("  yfinance not installed, run 'pip install yfinance'")
            return False
        
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            # Get market cap and price first
            market_cap = info.get('marketCap')
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            # Get diluted shares outstanding (handles multi-class structures)
            shares = info.get('impliedSharesOutstanding')
            
            # Fallback: Calculate from market cap / price
            if shares is None or shares == 0:
                if market_cap and price and price != 0:
                    shares = market_cap / price
            
            # Last resort: Use basic sharesOutstanding
            if shares is None or shares == 0:
                shares = info.get('sharesOutstanding')
            
            self.market_data = {
                'current_price': price,
                'market_cap': market_cap,
                'shares_outstanding': shares,
                'beta': info.get('beta'),
                'pe_ratio': info.get('trailingPE'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'company_name': info.get('longName'),
            }
            
            self.has_market_data = True
            return True
            
        except Exception as e:
            print(f"   Market data fetch failed: {e}")
            return False
    
    @property
    def current_price(self) -> Optional[float]:
        """Current stock price from Yahoo Finance"""
        return self.market_data.get('current_price')
    
    @property
    def market_cap(self) -> Optional[float]:
        """Market capitalization from Yahoo Finance"""
        return self.market_data.get('market_cap')
    
    @property
    def shares_outstanding(self) -> Optional[float]:
        """Shares outstanding from Yahoo Finance"""
        return self.market_data.get('shares_outstanding')
    
    @property
    def beta(self) -> Optional[float]:
        """Beta from Yahoo Finance"""
        return self.market_data.get('beta')

    # Add these getter methods to AlphaVantageDownloader class
    def get_income_statement(self) -> pd.DataFrame:
        """Return income statement data"""
        if self.income_statement is None:
            raise ValueError("No income statement data. Run download_data() first.")
        return self.income_statement.copy()

    def get_balance_sheet(self) -> pd.DataFrame:
        """Return balance sheet data"""
        if self.balance_sheet is None:
            raise ValueError("No balance sheet data. Run download_data() first.")
        return self.balance_sheet.copy()

    def get_cash_flow(self) -> pd.DataFrame:
        """Return cash flow data"""
        if self.cash_flow is None:
            raise ValueError("No cash flow data. Run download_data() first.")
        return self.cash_flow.copy()

    
    # HELPER FUNCTIONS
    def is_ttm_period(self, period: Union[int, str]) -> bool:
        """Check if a period is TTM"""
        return TTM_LABEL in str(period)

    def get_annual_periods(self) -> pd.Index:
        """Get only annual periods (excluding TTM)"""
        return pd.Index([p for p in self.income_statement.index if not self.is_ttm_period(p)])

    def create_summary_sheet(self) -> pd.DataFrame:
        """Create a summary of downloaded data"""
        years = [y for y in self.income_statement.index if isinstance(y, (int, np.integer))]
        years = sorted(years) if years else []

        if years:
            year_range = f"{min(years)}-{max(years)}"
            if self.has_ttm:
                year_range += "+TTM"
        else:
            year_range = "n/a"

        summary = {
            "Ticker": [self.ticker],
            "Periods": [year_range],
            "Source": ["Alpha Vantage"],
        }
        
        # Add market data
        if self.has_market_data:
            summary["Current Price"] = [f"${self.current_price:.2f}" if self.current_price else "N/A"]
            summary["Market Cap"] = [f"${self.market_cap:,.0f}" if self.market_cap else "N/A"]
            summary["Shares Out"] = [f"{self.shares_outstanding:,.0f}" if self.shares_outstanding else "N/A"]

        return pd.DataFrame(summary)

    def export_to_excel(self, filename: Optional[str] = None) -> Optional[str]:
        """Export all statements to Excel file"""
        if filename is None:
            filename = f"{self.ticker}_financialdata.xlsx"

        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Summary sheet
                self.create_summary_sheet().to_excel(writer, sheet_name="Summary", index=False)
                
                # Financial statements
                self.income_statement.to_excel(writer, sheet_name="Income")
                self.balance_sheet.to_excel(writer, sheet_name="Balance")
                self.cash_flow.to_excel(writer, sheet_name="CashFlow")
                
                # Market data sheet (if available)
                if self.has_market_data:
                    market_df = pd.DataFrame([self.market_data]).T
                    market_df.columns = ['Value']
                    market_df.to_excel(writer, sheet_name="MarketData")
            
            return filename
        except Exception as e:
            print(f"Export failed: {e}")
            return None

if __name__ == "__main__":
    API_KEY = "IF32HB76Y2UE78AW" #INSERT API KEY HERE
    
    if not ticker:
        print("No ticker provided. Exiting.")
        exit()
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠ WARNING: Please set your Alpha Vantage API key first!")
        exit()
    
    # Create downloader
    downloader = AlphaVantageDownloader(
        ticker=ticker,
        api_key=API_KEY,
        include_ttm=True
    )
    
    # Download financial statements 
    if not downloader.download_data():
        print("\n✗ Download failed.")
        exit()
    
    downloader.clean_data()
    
    if downloader.include_ttm:
        downloader.add_ttm_data()
    
    # Download market data
    downloader.add_market_data()
    
    # Export everything
    filename = downloader.export_to_excel()
    
    # Display summary
    print("\nCOMPLETE SUMMARY")
    print("="*70)
    print(downloader.create_summary_sheet().to_string(index=False))
    
    # Show financial metrics
    latest = downloader.income_statement.index[0]
    print(f"\nFINANCIAL DATA ({latest})")
    print("="*70)
    
    revenue = downloader.income_statement.loc[latest, 'totalRevenue']
    net_income = downloader.income_statement.loc[latest, 'netIncome']
    ebitda = downloader.income_statement.loc[latest, 'ebitda']
    
    print(f"Revenue:           ${revenue:>15,.0f}")
    print(f"Net Income:        ${net_income:>15,.0f}")
    print(f"EBITDA:            ${ebitda:>15,.0f}")
    
    # Show market data
    if downloader.has_market_data:
        print("\nMARKET DATA (Yahoo Finance)")
        print("="*70)
        print(f"Current Price:     ${downloader.current_price:>15.2f}")
        print(f"Market Cap:        ${downloader.market_cap:>15,.0f}")
        print(f"Shares Out:        {downloader.shares_outstanding:>16,.0f}")
        print(f"Beta:              {downloader.beta:>16.2f}")
        
        # Calculate valuation ratios
        print("\nVALUATION RATIOS (Combined)")
        print("="*70)
        
        # P/E Ratio
        eps = net_income / downloader.shares_outstanding
        pe = downloader.current_price / eps
        print(f"EPS:               ${eps:>15.2f}")
        print(f"P/E Ratio:         {pe:>16.2f}x")
        
        # EV/EBITDA
        total_debt = downloader.balance_sheet.loc[latest, 'shortLongTermDebtTotal']
        cash = downloader.balance_sheet.loc[latest, 'cashAndCashEquivalentsAtCarryingValue']
        enterprise_value = downloader.market_cap + total_debt - cash
        ev_ebitda = enterprise_value / ebitda
        print(f"Enterprise Value:  ${enterprise_value:>15,.0f}")
        print(f"EV/EBITDA:         {ev_ebitda:>16.2f}x")
    
    if filename:
        print("\n" + "="*70)
        print(f"✓ Data saved to: {filename}")
        print("="*70)
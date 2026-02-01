# check_yfinance_fields.py
import yfinance as yf

# 1️⃣ 选择你要分析的公司，比如 Google
ticker = "GOOGL"

# 2️⃣ 下载该公司的财报对象
tk = yf.Ticker(ticker)

# 3️⃣ 输出各类财报的字段名
print("=== INCOME STATEMENT ===")
print(list(tk.financials.index))

print("\n=== CASHFLOW STATEMENT ===")
print(list(tk.cashflow.index))

print("\n=== BALANCE SHEET ===")
print(list(tk.balance_sheet.index))

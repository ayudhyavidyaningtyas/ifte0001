import sys
import os

print("=" * 80)
sys.stdout.flush()
print("VERA - Virtual Equity Research Assistant")
sys.stdout.flush()
print("=" * 80)
sys.stdout.flush()

from vera_main import main as run_analysis

try:
    results = run_analysis()
except Exception as e:
    print(f"\nAnalysis failed: {e}")
    sys.stdout.flush()
    sys.exit(1)

# Extract results
downloader = results['downloader']
ratios = results.get('ratios')
dcf_results = results.get('dcf')
relative_results = results.get('relative_valuation')
target_prices = results.get('target_prices')

# Generate Memo
print("\nGenerate memo? (Y/N): ", end='', flush=True)
generate = input().strip().lower()

if generate != 'y':
    print("\nAnalysis complete.")
    sys.stdout.flush()
    sys.exit(0)

# Get OpenAI API key
print("\nEnter your OpenAI API key:", flush=True)
print("API Key: ", end='', flush=True)
api_key = input().strip()

if not api_key:
    print("\nNo API key provided. Exiting.", flush=True)
    sys.exit(0)

# Generate memo
try:
    import types
    import pandas as pd

    with open('memo_generator.py', 'r') as f:
        memo_code = f.read()

    # Create namespace and inject results
    memo_ns = types.ModuleType('memo_module')
    memo_ns.downloader = downloader

    if ratios:
        memo_ns.ratios = ratios
        memo_ns.ratios_analyzer = ratios

    if dcf_results:
        memo_ns.dcf_results = dcf_results

    if target_prices is not None:
        if isinstance(target_prices, dict):
            memo_ns.results_12m = target_prices.get('12m')
            memo_ns.results_18m = target_prices.get('18m')
            if target_prices.get('12m') is not None:
                memo_ns.results_df = target_prices.get('12m')
        else:
            memo_ns.results_df = target_prices

    if relative_results:
        avg_core, avg_ext = relative_results
        if avg_core:
            memo_ns.avg_core = avg_core

    os.environ['OPENAI_API_KEY'] = api_key

    # Execute memo generator
    exec(memo_code, memo_ns.__dict__)

    # Generate
    print("\nGenerating memo...", flush=True)
    data = memo_ns.gather_data()

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    val = memo_ns.compute_valuation(client, data)

    # Chart
    print("Generating chart...", flush=True)
    import yfinance as yf
    import matplotlib.pyplot as plt

    try:
        target_ticker = data['ticker']
        peer_tickers = memo_ns.CONFIG.get("chart_peers", ["MSFT", "AAPL", "AMZN", "META"])
        all_tickers = [target_ticker] + ["^GSPC", "^IXIC"] + peer_tickers

        hist = yf.download(" ".join(all_tickers),
                           period=memo_ns.CONFIG.get("chart_period", "1y"),
                           progress=False, ignore_tz=True)

        if isinstance(hist, pd.DataFrame):
            if 'Close' in hist.columns:
                price_data = hist['Close']
            elif isinstance(hist.columns, pd.MultiIndex):
                price_data = hist.xs('Close', axis=1, level=0)
            else:
                price_data = hist

        norm = (price_data / price_data.iloc[0] - 1) * 100

        fig, ax = plt.subplots(figsize=(4.5, 2.5))

        COLORS = {
            'TARGET': '#1e3a8a', 'SP500': '#B22222', 'NASDAQ': '#FF8C00',
            'MSFT': '#00A4EF', 'AAPL': '#A3AAAE', 'AMZN': '#FF9900',
            'META': '#0866FF', 'NVDA': '#76B900', 'TSLA': '#E82127',
            'GOOG': '#4285F4', 'NFLX': '#E50914'
        }

        for ticker in peer_tickers:
            if ticker in norm.columns:
                ax.plot(norm.index, norm[ticker],
                        color=COLORS.get(ticker, '#94a3b8'),
                        linewidth=1.2, alpha=0.7, label=ticker)

        if "^GSPC" in norm.columns:
            ax.plot(norm.index, norm["^GSPC"], label="S&P 500",
                    color=COLORS['SP500'], linewidth=1.5,
                    linestyle='--', alpha=0.9)

        if "^IXIC" in norm.columns:
            ax.plot(norm.index, norm["^IXIC"], label="NASDAQ",
                    color=COLORS['NASDAQ'], linewidth=1.5,
                    linestyle='-.', alpha=0.9)

        if target_ticker in norm.columns:
            ax.plot(norm.index, norm[target_ticker], label=target_ticker,
                    color=COLORS['TARGET'], linewidth=2.5,
                    alpha=1.0, zorder=10)

        ax.set_facecolor('#fafafa')
        ax.grid(True, alpha=0.15, color='#cbd5e1')
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
        ax.set_ylabel('Return (%)', fontsize=8, color='#475569')
        ax.set_xlabel('Date', fontsize=8, color='#475569')
        ax.legend(fontsize='x-small', frameon=True, facecolor='white',
                  edgecolor='#e2e8f0', loc='upper left', ncol=2)
        ax.tick_params(labelsize=7)

        for spine in ax.spines.values():
            spine.set_edgecolor('#cbd5e1')
            spine.set_linewidth(0.6)

        plt.tight_layout()
        plt.savefig("chart_temp.png", dpi=120, bbox_inches='tight')
        b64_chart = memo_ns.image_to_base64("chart_temp.png")

        if os.path.exists("chart_temp.png"):
            os.remove("chart_temp.png")

    except Exception as e:
        print(f"Chart warning: {e}", flush=True)
        b64_chart = ""

    # Build HTML
    print("Building HTML report...", flush=True)
    html_content = memo_ns.build_html_report(client, data, val, b64_chart, ratios)

    if html_content:
        output_filename = f"{downloader.ticker}_Research_Note.html"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n✓ Memo saved: {output_filename}", flush=True)
        print(f"  Rating: {val['rating']} | Target: ${val['target']:.2f} | Upside: {val['upside']:+.1%}", flush=True)
    else:
        print("\nError: HTML generation failed", flush=True)
        sys.exit(1)

except Exception as e:
    print(f"\nError: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)

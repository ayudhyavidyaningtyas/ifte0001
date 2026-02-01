import sys
import os

API_KEY = "IF32HB76Y2UE78AW"  # If error, generate new Alpha Vantage API Key and change it


def main():
    """Run complete analysis with single API call."""

    # STEP 1: Get Ticker Input  
    ticker = input("\nEnter ticker symbol (e.g., GOOGL): ").strip().upper()

    if not ticker:
        print("No ticker provided. Exiting.")
        sys.exit(1)

    if API_KEY in ["YOUR_API_KEY_HERE"]:
        print("\n️  WARNING: Please update API_KEY in this file")
        print("   Line 24: API_KEY = 'YOUR_KEY_HERE'")
        print("   Get free key: https://www.alphavantage.co/support/#api-key")
        sys.exit(1)

    # STEP 2: Download Financial Data
    try:
        # Import downloader from extracted Cell 1
        from downloader import AlphaVantageDownloader
    except ImportError:
        print("Error, downloader.py not found.")
        print("Make sure it's in the same directory as this script.")
        sys.exit(1)

    # Create downloader
    try:
        downloader = AlphaVantageDownloader(
            ticker=ticker,
            api_key=API_KEY,
            include_ttm=True
        )

        # Download data
        if not downloader.download_data():
            print("\nData download failed!")
            print("\nCommon causes:")
            print("  • API rate limit (5/min, 25/day for free tier)")
            print("  • Invalid ticker symbol")
            print("  • Network issue")
            print("\nTry:")
            print("  • Wait 1 minute and retry")
            print("  • Verify ticker is correct")
            print("  • Check API key")
            sys.exit(1)

        # Process data
        downloader.clean_data()

        if downloader.include_ttm:
            downloader.add_ttm_data()

        downloader.add_market_data()

        print("Data downloaded successfully!")
        print(f"   • Income Statement: {len(downloader.income_statement)} periods")
        print(f"   • Balance Sheet: {len(downloader.balance_sheet)} periods")
        print(f"   • Cash Flow: {len(downloader.cash_flow)} periods")
        print(f"   • Market Data: {'Available' if downloader.has_market_data else 'N/A'}")

        # Export financial data to Excel
        print("\n   > Exporting financial data to Excel...")
        financial_data_file = downloader.export_to_excel()
        if financial_data_file:
            print(f"   ✓ Financial data exported to: {financial_data_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Store results
    results = {'downloader': downloader}

    # STEP 3: Run Analysis 1 - Ratio Analysis
    print(f"\n{'=' * 80}")
    print("STEP 2: RATIO ANALYSIS")
    print(f"{'=' * 80}")

    try:
        from ratios import RatioAnalyzer

        print(f"   > Initializing RatioAnalyzer for {ticker}...")
        analyzer = RatioAnalyzer(downloader)

        print(f"   > Calculating ratios...")
        analyzer.display_ratios()

        print(f"   > Exporting to Excel...")
        excel_file = analyzer.export_to_excel()

        results['ratios'] = analyzer
        print(f"\n✓ Ratio analysis complete, saved to {excel_file}.")

    except ImportError as e:
        print(f"\nFailed to import RatioAnalyzer: {e}")
        print("   Make sure ratios.py is in the same directory.")
        import traceback
        traceback.print_exc()
        results['ratios'] = None
    except Exception as e:
        print(f"\nRatio analysis failed: {e}")
        print("   Error details:")
        import traceback
        traceback.print_exc()
        results['ratios'] = None

    # DEBUG: Check if ratios was successfully stored
    if results.get('ratios') is not None:
        print(f"   ✓ DEBUG: Ratios analyzer successfully stored in results")
        print(f"   ✓ DEBUG: Analyzer type: {type(results['ratios'])}")
    else:
        print(f"DEBUG: Ratios is None - will not be available for memo generation")

    # STEP 4: Run Analysis 2 - DCF Valuation
    print(f"\n{'=' * 80}")
    print("STEP 3: DCF VALUATION")
    print(f"{'=' * 80}")

    try:
        from dcf import DCFModel

        dcf = DCFModel(downloader)
        dcf_results = dcf.run(show_sensitivity=True)

        results['dcf'] = dcf_results

    except Exception as e:
        print(f"\nDCF valuation failed: {e}")
        results['dcf'] = None

    # STEP 5: Run Analysis 3 - Relative Valuation

    print(f"\n{'=' * 80}")
    print("STEP 4: RELATIVE VALUATION")
    print(f"{'=' * 80}")

    try:
        import types

        with open('relative.py', 'r') as f:
            cell4_code = f.read()

        # Create module and inject downloader
        cell4_ns = types.ModuleType('cell4_module')
        cell4_ns.downloader = downloader

        # Execute cell code
        exec(cell4_code, cell4_ns.__dict__)

        # Extract results
        avg_core = getattr(cell4_ns, 'avg_core', None)
        avg_ext = getattr(cell4_ns, 'avg_ext', None)

        results['relative_valuation'] = (avg_core, avg_ext)

    except FileNotFoundError:
        print("\nrelative.py not found, relative analysis and DDM skipped.")
        results['relative_valuation'] = None
    except Exception as e:
        print(f"\nRelative valuation failed: {e}")
        results['relative_valuation'] = None

    # STEP 6: Run Analysis 4 - Target Price
    print(f"\n{'=' * 80}")
    print("STEP 5: TARGET PRICE ANALYSIS (12M & 18M)")
    print(f"{'=' * 80}")

    try:
        import types

        # Try new multi-horizon script first, fall back to old target.py
        try:
            with open('target.py', 'r') as f:
                target_code = f.read()
            using_multi_horizon = True
        except FileNotFoundError:
            with open('target.py', 'r') as f:
                target_code = f.read()
            using_multi_horizon = False

        # Create module and inject downloader
        target_ns = types.ModuleType('target_module')
        target_ns.downloader = downloader

        # Execute target code
        exec(target_code, target_ns.__dict__)

        if using_multi_horizon:
            # Extract multi-horizon results (returned as dict from main())
            if hasattr(target_ns, 'main'):
                target_results = target_ns.main()
                if target_results:
                    results['target_prices'] = {
                        '12m': target_results.get('results_12m'),
                        '18m': target_results.get('results_18m'),
                        'sensitivity_12m': target_results.get('sensitivity_12m'),
                        'sensitivity_18m': target_results.get('sensitivity_18m'),
                        'eps_12m': target_results.get('eps_12m'),
                        'eps_18m': target_results.get('eps_18m')
                    }
                else:
                    results['target_prices'] = None
            else:
                results['target_prices'] = None
        else:
            # Fall back to old single-horizon extraction
            results_df = getattr(target_ns, 'results_df', None)
            targets_df = getattr(target_ns, 'targets_df', None)
            results['target_prices'] = results_df if results_df is not None else targets_df

    except FileNotFoundError:
        print("\n⚠️  target script not found, target price calculation skipped.")
        results['target_prices'] = None
    except Exception as e:
        print(f"\n⚠️  Target price analysis failed: {e}")
        import traceback
        traceback.print_exc()
        results['target_prices'] = None

    return results


# RUN

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

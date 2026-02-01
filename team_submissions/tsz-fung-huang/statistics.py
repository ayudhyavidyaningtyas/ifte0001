import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load processed data (run backtest.py first to generate the data file)
# Alternatively, you can import googl_data directly from backtest.py if running in the same session
try:
    googl_data = pd.read_parquet('googl_processed_data.parquet')
    print("Loaded processed data from googl_processed_data.parquet")
except FileNotFoundError:
    print("Error: googl_processed_data.parquet not found. Please run backtest.py first to generate the data.")
    raise

# Statistical Analysis Code
# Directional and Crossover CRSI Analysis: 2D Bucket Statistics (with baseline comparison, significance tests, trading rules)

print("\n" + "=" * 60)
print("CRSI 2D Bucket Statistics: Directional and Crossover Analysis")
print("=" * 60)

# ========== 1. Define "Hit" Definition (aligned with backtest parameters) ==========
# Backtest parameters:
TAKE_PROFIT_PCT = 10.0  # Take profit at +10%
MAX_HOLD_DAYS = 20      # Maximum holding period: 20 days
HIT_DEFINITION = "max_price"  # Options: "max_price" or "close_price"

# Calculate maximum and minimum prices within next MAX_HOLD_DAYS (relative to current day's close)
googl_data['Future_20d_High'] = googl_data['High'].rolling(window=MAX_HOLD_DAYS, min_periods=1).max().shift(-(MAX_HOLD_DAYS-1))
googl_data['Future_20d_Low'] = googl_data['Low'].rolling(window=MAX_HOLD_DAYS, min_periods=1).min().shift(-(MAX_HOLD_DAYS-1))

# Calculate maximum return within MAX_HOLD_DAYS relative to current day's close
googl_data['Future_20d_Max_Return'] = (googl_data['Future_20d_High'] / googl_data['Close'] - 1) * 100

# Calculate MAX_HOLD_DAYS close return relative to current day's close
googl_data['Future_20d_Close_Return'] = (googl_data['Close'].shift(-MAX_HOLD_DAYS) / googl_data['Close'] - 1) * 100

# Mark whether take profit threshold is reached based on definition
if HIT_DEFINITION == "max_price":
    googl_data['Hit_TakeProfit'] = googl_data['Future_20d_Max_Return'] >= TAKE_PROFIT_PCT
    hit_description = f"Maximum price reached +{TAKE_PROFIT_PCT}% within {MAX_HOLD_DAYS} days"
else:
    googl_data['Hit_TakeProfit'] = googl_data['Future_20d_Close_Return'] >= TAKE_PROFIT_PCT
    hit_description = f"Close price >= +{TAKE_PROFIT_PCT}% after {MAX_HOLD_DAYS} days"

print(f"\nHit Definition (aligned with backtest): {hit_description}")
print(f"Take Profit: +{TAKE_PROFIT_PCT}%")
print(f"Max Hold Days: {MAX_HOLD_DAYS}")

# Previous day CRSI
googl_data['CRSI_Prev'] = googl_data['CRSI'].shift(1)

# Trend Detection: Add trend column to dataframe (same improved logic as backtest)
# Improved trend detection with longer periods and multiple confirmations
# Uses ADX, DM, EMA, and price action for more accurate trend identification

# Check if EMA columns exist, if not calculate them
if 'EMA_Short' not in googl_data.columns:
    import talib
    EMA_SHORT = 30  # Increased from 20 for more stability
    EMA_LONG = 60   # Increased from 50 for more stability
    googl_data['EMA_Short'] = talib.EMA(googl_data['Close'], timeperiod=EMA_SHORT)
    googl_data['EMA_Long'] = talib.EMA(googl_data['Close'], timeperiod=EMA_LONG)

# Trend detection parameters (same as backtest)
ADX_TREND_THRESHOLD = 25  # Standard threshold for strong trend
ADX_WEAK_THRESHOLD = 20    # Lower threshold for weak trend
CCI_VOLATILE_THRESHOLD = 100
UOS_VOLATILE_THRESHOLD = 50

# Smooth ADX using moving average to reduce noise (increased window for more stability)
if 'ADX_Smooth' not in googl_data.columns:
    googl_data['ADX_Smooth'] = googl_data['ADX'].rolling(window=5, min_periods=1).mean()

googl_data['Trend'] = 'neutral'

# Fill NaN values with 0 for comparison
adx_smooth = googl_data['ADX_Smooth'].fillna(0)
plus_dm_filled = googl_data['+DM'].fillna(0)
minus_dm_filled = googl_data['-DM'].fillna(0)
cci_filled = googl_data['CCI'].fillna(0)
uos_filled = googl_data['UOS'].fillna(0)
ema_short = googl_data['EMA_Short'].fillna(googl_data['Close'])
ema_long = googl_data['EMA_Long'].fillna(googl_data['Close'])
price = googl_data['Close']

# Improved trend detection with multiple confirmations (same as backtest)
strong_uptrend = (adx_smooth > ADX_TREND_THRESHOLD) & \
                 (plus_dm_filled > minus_dm_filled) & \
                 (ema_short > ema_long) & \
                 (price > ema_short)

strong_downtrend = (adx_smooth > ADX_TREND_THRESHOLD) & \
                   (minus_dm_filled > plus_dm_filled) & \
                   (ema_short < ema_long) & \
                   (price < ema_short)

weak_uptrend = (adx_smooth > ADX_WEAK_THRESHOLD) & \
               (adx_smooth <= ADX_TREND_THRESHOLD) & \
               (plus_dm_filled > minus_dm_filled) & \
               (ema_short > ema_long)

weak_downtrend = (adx_smooth > ADX_WEAK_THRESHOLD) & \
                 (adx_smooth <= ADX_TREND_THRESHOLD) & \
                 (minus_dm_filled > plus_dm_filled) & \
                 (ema_short < ema_long)

volatile_condition = (adx_smooth <= ADX_WEAK_THRESHOLD) | \
                     (abs(cci_filled) > CCI_VOLATILE_THRESHOLD) | \
                     (abs(uos_filled) > UOS_VOLATILE_THRESHOLD) | \
                     ((plus_dm_filled > minus_dm_filled) & (ema_short < ema_long)) | \
                     ((minus_dm_filled > plus_dm_filled) & (ema_short > ema_long))

# Assign trends (priority: strong uptrend > strong downtrend > weak uptrend > weak downtrend > volatile > neutral)
googl_data.loc[strong_uptrend, 'Trend'] = 'uptrend'
googl_data.loc[strong_downtrend & ~strong_uptrend, 'Trend'] = 'downtrend'
googl_data.loc[weak_uptrend & ~strong_uptrend & ~strong_downtrend, 'Trend'] = 'uptrend'
googl_data.loc[weak_downtrend & ~strong_uptrend & ~strong_downtrend & ~weak_uptrend, 'Trend'] = 'downtrend'
googl_data.loc[volatile_condition & ~strong_uptrend & ~strong_downtrend & ~weak_uptrend & ~weak_downtrend, 'Trend'] = 'volatile'

# Print trend distribution
print("\n" + "=" * 60)
print("Trend Distribution")
print("=" * 60)
trend_counts = googl_data['Trend'].value_counts()
for trend, count in trend_counts.items():
    pct = (count / len(googl_data)) * 100
    print(f"{trend}: {count} ({pct:.2f}%)")

# Remove data without previous day CRSI
if HIT_DEFINITION == "max_price":
    valid_data = googl_data[googl_data['CRSI_Prev'].notna() & googl_data['Future_20d_Max_Return'].notna()].copy()
else:
    valid_data = googl_data[googl_data['CRSI_Prev'].notna() & googl_data['Future_20d_Close_Return'].notna()].copy()

print(f"\nTotal Sample Size: {len(googl_data)}")
print(f"Valid Statistical Sample Size (with previous day CRSI and {MAX_HOLD_DAYS}-day future data): {len(valid_data)}")

# ========== 2. Set Baseline ==========
baseline_hit_count = valid_data['Hit_TakeProfit'].sum()
baseline_hit_rate = (baseline_hit_count / len(valid_data)) * 100
baseline_avg_return = valid_data['Future_20d_Close_Return'].mean()

print(f"\n{'='*60}")
print("[Baseline Statistics] Overall Sample Performance")
print(f"{'='*60}")
print(f"Total Sample Size: {len(valid_data)}")
print(f"Hit Count: {baseline_hit_count}")
print(f"Baseline Hit Rate: {baseline_hit_rate:.2f}%")
print(f"Baseline Average Return: {baseline_avg_return:.2f}%")

# ========== Significance Test Functions ==========
def binomial_test(hit_count, sample_count, baseline_rate):
    """Binomial test: compare group hit rate vs baseline hit rate"""
    p_value = stats.binom_test(hit_count, sample_count, baseline_rate / 100, alternative='two-sided')
    return p_value

def t_test_against_baseline(subset_returns, baseline_mean):
    """t-test: compare group average return vs baseline average return"""
    if len(subset_returns) < 2:
        return None, None
    t_stat, p_value = stats.ttest_1samp(subset_returns, baseline_mean)
    return t_stat, p_value

# Define CRSI bucket intervals (0-10, 10-20, ..., 90-100)
bins = list(range(0, 101, 10))
bin_labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]

# Assign CRSI_t-1 and CRSI_t to buckets
valid_data['CRSI_Prev_Bin'] = pd.cut(valid_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
valid_data['CRSI_Current_Bin'] = pd.cut(valid_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)

# Create 2D bucket statistics table
print("\n" + "=" * 60)
print("2D Bucket Statistics Table (CRSI_t-1 × CRSI_t)")
print("=" * 60)

# Create statistics table
stats_table = []
for prev_bin in bin_labels:
    for curr_bin in bin_labels:
        mask = (valid_data['CRSI_Prev_Bin'] == prev_bin) & (valid_data['CRSI_Current_Bin'] == curr_bin)
        subset = valid_data[mask]
        
        if len(subset) > 0:
            sample_count = len(subset)
            hit_count = subset['Hit_TakeProfit'].sum()
            hit_rate = (hit_count / sample_count) * 100 if sample_count > 0 else 0
            avg_return = subset['Future_20d_Close_Return'].mean()
            
            # Significance tests
            binom_p = binomial_test(hit_count, sample_count, baseline_hit_rate) if sample_count > 0 else None
            t_stat, t_p = t_test_against_baseline(subset['Future_20d_Close_Return'].values, baseline_avg_return)
            
            # Compare with baseline
            hit_rate_diff = hit_rate - baseline_hit_rate
            return_diff = avg_return - baseline_avg_return
            
            stats_table.append({
                'CRSI_t-1': prev_bin,
                'CRSI_t': curr_bin,
                'Sample_Count': sample_count,
                'Hit_Count': hit_count,
                'Hit_Rate(%)': round(hit_rate, 2),
                'Hit_Rate_Diff(%)': round(hit_rate_diff, 2),
                'Hit_Rate_p_value': round(binom_p, 4) if binom_p is not None else None,
                'Avg_Return(%)': round(avg_return, 2),
                'Return_Diff(%)': round(return_diff, 2),
                'Return_p_value': round(t_p, 4) if t_p is not None else None
            })

stats_df = pd.DataFrame(stats_table)

if len(stats_df) > 0:
    print("\nStatistics Table (sorted by sample count, top 20):")
    print(stats_df.sort_values('Sample_Count', ascending=False).head(20).to_string(index=False))
    
    # ========== 3. Merge Adjacent Intervals to Increase Sample Size ==========
    print("\n" + "=" * 60)
    print("Merged Interval Analysis (increase sample size to >=30)")
    print("=" * 60)
    
    # Define merge rules
    merged_groups = []
    mid_to_submid = pd.DataFrame()
    high_persist = pd.DataFrame()
    
    # Merge rule 1: Mid to Sub-mid (50-60→40-50 and 40-50→30-40)
    mid_to_submid = valid_data[
        ((valid_data['CRSI_Prev'] >= 50) & (valid_data['CRSI_Prev'] < 60) & 
         (valid_data['CRSI'] >= 40) & (valid_data['CRSI'] < 50)) |
        ((valid_data['CRSI_Prev'] >= 40) & (valid_data['CRSI_Prev'] < 50) & 
         (valid_data['CRSI'] >= 30) & (valid_data['CRSI'] < 40))
    ].copy()
    
    if len(mid_to_submid) >= 30:
        hit_count = mid_to_submid['Hit_TakeProfit'].sum()
        hit_rate = (hit_count / len(mid_to_submid)) * 100
        avg_return = mid_to_submid['Future_20d_Close_Return'].mean()
        binom_p = binomial_test(hit_count, len(mid_to_submid), baseline_hit_rate)
        t_stat, t_p = t_test_against_baseline(mid_to_submid['Future_20d_Close_Return'].values, baseline_avg_return)
        
        merged_groups.append({
            'Group': 'Mid to Sub-mid',
            'Sample_Count': len(mid_to_submid),
            'Hit_Rate(%)': round(hit_rate, 2),
            'Hit_Rate_Diff(%)': round(hit_rate - baseline_hit_rate, 2),
            'Hit_Rate_p_value': round(binom_p, 4),
            'Avg_Return(%)': round(avg_return, 2),
            'Return_Diff(%)': round(avg_return - baseline_avg_return, 2),
            'Return_p_value': round(t_p, 4) if t_p is not None else None
        })
    
    # Merge rule 2: High persistence (60-70→60-70 and 70-80→70-80)
    high_persist = valid_data[
        ((valid_data['CRSI_Prev'] >= 60) & (valid_data['CRSI_Prev'] < 70) & 
         (valid_data['CRSI'] >= 60) & (valid_data['CRSI'] < 70)) |
        ((valid_data['CRSI_Prev'] >= 70) & (valid_data['CRSI_Prev'] < 80) & 
         (valid_data['CRSI'] >= 70) & (valid_data['CRSI'] < 80))
    ].copy()
    
    if len(high_persist) >= 30:
        hit_count = high_persist['Hit_TakeProfit'].sum()
        hit_rate = (hit_count / len(high_persist)) * 100
        avg_return = high_persist['Future_20d_Close_Return'].mean()
        binom_p = binomial_test(hit_count, len(high_persist), baseline_hit_rate)
        t_stat, t_p = t_test_against_baseline(high_persist['Future_20d_Close_Return'].values, baseline_avg_return)
        
        merged_groups.append({
            'Group': 'High Persistence',
            'Sample_Count': len(high_persist),
            'Hit_Rate(%)': round(hit_rate, 2),
            'Hit_Rate_Diff(%)': round(hit_rate - baseline_hit_rate, 2),
            'Hit_Rate_p_value': round(binom_p, 4),
            'Avg_Return(%)': round(avg_return, 2),
            'Return_Diff(%)': round(avg_return - baseline_avg_return, 2),
            'Return_p_value': round(t_p, 4) if t_p is not None else None
        })
    
    if len(merged_groups) > 0:
        merged_df = pd.DataFrame(merged_groups)
        print("\nMerged Interval Statistics (compared with baseline):")
        print(merged_df.to_string(index=False))
        
        # Mark groups significantly better than baseline (p < 0.05 and higher hit rate/return)
        print("\nGroups Significantly Better than Baseline (p < 0.05):")
        significant = merged_df[
            (merged_df['Hit_Rate_p_value'] < 0.05) & (merged_df['Hit_Rate_Diff(%)'] > 0) |
            ((merged_df['Return_p_value'] < 0.05) & (merged_df['Return_Diff(%)'] > 0))
        ]
        if len(significant) > 0:
            print(significant[['Group', 'Sample_Count', 'Hit_Rate(%)', 'Hit_Rate_Diff(%)', 'Avg_Return(%)', 'Return_Diff(%)']].to_string(index=False))
        else:
            print("No groups significantly better than baseline")
    
    # Create pivot tables for visualization
    pivot_sample = stats_df.pivot(index='CRSI_t-1', columns='CRSI_t', values='Sample_Count').fillna(0)
    pivot_hit_rate = stats_df.pivot(index='CRSI_t-1', columns='CRSI_t', values='Hit_Rate(%)').fillna(0)
    pivot_avg_return = stats_df.pivot(index='CRSI_t-1', columns='CRSI_t', values='Avg_Return(%)').fillna(0)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sample count heatmap
    sns.heatmap(pivot_sample, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Sample Count'})
    axes[0].set_title('Sample Count by CRSI Bins\n(CRSI_t-1 × CRSI_t)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('CRSI_t', fontsize=10)
    axes[0].set_ylabel('CRSI_t-1', fontsize=10)
    
    # Hit rate heatmap
    sns.heatmap(pivot_hit_rate, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1], cbar_kws={'label': 'Hit Rate (%)'})
    axes[1].set_title(f'Hit Rate (%) by CRSI Bins\n(>={TAKE_PROFIT_PCT}% in {MAX_HOLD_DAYS} days)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('CRSI_t', fontsize=10)
    axes[1].set_ylabel('CRSI_t-1', fontsize=10)
    
    # Average return heatmap
    sns.heatmap(pivot_avg_return, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[2], cbar_kws={'label': 'Avg Return (%)'})
    axes[2].set_title('Average Return (%) by CRSI Bins\n(20-day close return)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('CRSI_t', fontsize=10)
    axes[2].set_ylabel('CRSI_t-1', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('crsi_2d_bucket_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n2D bucket heatmap saved to: crsi_2d_bucket_analysis.png")
    plt.close()
    
    # ========== 5. Trend Transition Analysis ==========
    print("\n" + "=" * 60)
    print("Trend Transition Analysis: Hit Rate by Trend Changes")
    print("=" * 60)
    
    # Add previous day trend
    googl_data['Trend_Prev'] = googl_data['Trend'].shift(1)
    
    # Analyze trend transitions
    # Add previous day trend to valid_data
    valid_data_with_trend = valid_data.copy()
    valid_data_with_trend['Trend_Prev'] = googl_data.loc[valid_data.index, 'Trend_Prev']
    valid_data_with_trend['Trend_Current'] = googl_data.loc[valid_data.index, 'Trend']
    
    trend_transitions = []
    trends_list = ['uptrend', 'downtrend', 'volatile', 'neutral']
    
    for prev_trend in trends_list:
        for curr_trend in trends_list:
            transition_data = valid_data_with_trend[
                (valid_data_with_trend['Trend_Prev'] == prev_trend) & 
                (valid_data_with_trend['Trend_Current'] == curr_trend)
            ]
            
            if len(transition_data) >= 10:
                hit_count = transition_data['Hit_TakeProfit'].sum()
                hit_rate = (hit_count / len(transition_data)) * 100
                avg_return = transition_data['Future_20d_Close_Return'].mean()
                
                trend_transitions.append({
                    'From_Trend': prev_trend,
                    'To_Trend': curr_trend,
                    'Sample_Count': len(transition_data),
                    'Hit_Rate(%)': round(hit_rate, 2),
                    'Avg_Return(%)': round(avg_return, 2)
                })
    
    if len(trend_transitions) > 0:
        transition_df = pd.DataFrame(trend_transitions)
        print("\nTrend Transition Statistics:")
        print(transition_df.sort_values('Hit_Rate(%)', ascending=False).to_string(index=False))
        
        # Find best transitions
        print("\nTop 10 Trend Transitions by Hit Rate:")
        print(transition_df.nlargest(10, 'Hit_Rate(%)').to_string(index=False))
        
        # Create transition heatmap
        transition_pivot = transition_df.pivot(index='From_Trend', columns='To_Trend', values='Hit_Rate(%)')
        
        fig_trans, ax_trans = plt.subplots(figsize=(10, 8))
        sns.heatmap(transition_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=baseline_hit_rate, cbar_kws={'label': 'Hit Rate (%)'},
                   ax=ax_trans, linewidths=0.5, linecolor='gray', annot_kws={'size': 10})
        ax_trans.set_title(f'Trend Transition Hit Rate Analysis\n(20 days, +{TAKE_PROFIT_PCT}%)', 
                          fontsize=13, fontweight='bold')
        ax_trans.set_xlabel('To Trend', fontsize=11)
        ax_trans.set_ylabel('From Trend', fontsize=11)
        plt.tight_layout()
        plt.savefig('statistics_trend_transition.png', dpi=300, bbox_inches='tight')
        print(f"\nTrend transition heatmap saved to: statistics_trend_transition.png")
        plt.close()
    
    # ========== 6. Trend-Based Analysis ==========
    print("\n" + "=" * 60)
    print("Trend-Based CRSI Interval Analysis")
    print("=" * 60)
    
    # Analyze by trend
    trends = ['uptrend', 'downtrend', 'volatile', 'neutral']
    trend_baseline_hit_rates = {}  # Store baseline hit rates for visualization
    
    for trend in trends:
        trend_data = valid_data[valid_data['Trend'] == trend].copy()
        
        if len(trend_data) < 10:
            print(f"\n[{trend.upper()}] Sample size too small ({len(trend_data)}), skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{trend.upper()}] Analysis")
        print(f"{'='*60}")
        
        # Baseline for this trend
        trend_baseline_hit_rate = (trend_data['Hit_TakeProfit'].sum() / len(trend_data)) * 100
        trend_baseline_avg_return = trend_data['Future_20d_Close_Return'].mean()
        trend_baseline_hit_rates[trend] = trend_baseline_hit_rate  # Store for visualization
        
        print(f"Sample Size: {len(trend_data)}")
        print(f"Baseline Hit Rate: {trend_baseline_hit_rate:.2f}%")
        print(f"Baseline Average Return: {trend_baseline_avg_return:.2f}%")
        
        # 2D bucket analysis for this trend
        trend_stats_table = []
        for prev_bin in bin_labels:
            for curr_bin in bin_labels:
                mask = (trend_data['CRSI_Prev_Bin'] == prev_bin) & (trend_data['CRSI_Current_Bin'] == curr_bin)
                subset = trend_data[mask]
                
                if len(subset) >= 10:  # Minimum sample size for trend analysis
                    sample_count = len(subset)
                    hit_count = subset['Hit_TakeProfit'].sum()
                    hit_rate = (hit_count / sample_count) * 100 if sample_count > 0 else 0
                    avg_return = subset['Future_20d_Close_Return'].mean()
                    
                    # Significance tests
                    binom_p = binomial_test(hit_count, sample_count, trend_baseline_hit_rate) if sample_count > 0 else None
                    t_stat, t_p = t_test_against_baseline(subset['Future_20d_Close_Return'].values, trend_baseline_avg_return)
                    
                    # Compare with trend baseline
                    hit_rate_diff = hit_rate - trend_baseline_hit_rate
                    return_diff = avg_return - trend_baseline_avg_return
                    
                    trend_stats_table.append({
                        'CRSI_t-1': prev_bin,
                        'CRSI_t': curr_bin,
                        'Sample_Count': sample_count,
                        'Hit_Rate(%)': round(hit_rate, 2),
                        'Hit_Rate_Diff(%)': round(hit_rate_diff, 2),
                        'Hit_Rate_p_value': round(binom_p, 4) if binom_p is not None else None,
                        'Avg_Return(%)': round(avg_return, 2),
                        'Return_Diff(%)': round(return_diff, 2),
                        'Return_p_value': round(t_p, 4) if t_p is not None else None
                    })
        
        if len(trend_stats_table) > 0:
            trend_stats_df = pd.DataFrame(trend_stats_table)
            
            # Find best intervals for this trend (hit rate > 60% or significantly better than baseline)
            best_intervals = trend_stats_df[
                (trend_stats_df['Hit_Rate(%)'] > 60) | 
                ((trend_stats_df['Hit_Rate_Diff(%)'] > 5) & (trend_stats_df['Hit_Rate_p_value'] < 0.05))
            ].sort_values('Hit_Rate(%)', ascending=False)
            
            if len(best_intervals) > 0:
                print(f"\nBest CRSI Intervals for {trend.upper()} (Hit Rate > 60% or significantly better):")
                print(best_intervals.head(10).to_string(index=False))
            else:
                print(f"\nNo intervals with hit rate > 60% found for {trend.upper()}")
                print("Top 5 intervals by hit rate:")
                print(trend_stats_df.sort_values('Hit_Rate(%)', ascending=False).head(5).to_string(index=False))
    
    # ========== 7. Specific Scenario Analysis ==========
    print("\n" + "=" * 60)
    print("Specific Scenario Analysis")
    print("=" * 60)
    
    # Scenario 1: Volatile Trend - 2.5% gain in 5 days
    print("\n" + "-" * 60)
    print("Scenario 1: Volatile Trend - 2.5% gain in 5 days")
    print("-" * 60)
    
    # Calculate 5-day max return for volatile trend
    googl_data['Future_5d_High'] = googl_data['High'].rolling(window=5, min_periods=1).max().shift(-4)
    googl_data['Future_5d_Max_Return'] = (googl_data['Future_5d_High'] / googl_data['Close'] - 1) * 100
    googl_data['Hit_2_5pct_5d'] = googl_data['Future_5d_Max_Return'] >= 2.5
    
    volatile_data = valid_data[valid_data['Trend'] == 'volatile'].copy()
    if len(volatile_data) > 0:
        # Add 5-day hit data using index alignment
        volatile_data['Hit_2_5pct_5d'] = googl_data.loc[volatile_data.index, 'Hit_2_5pct_5d']
        volatile_data['Future_5d_Max_Return'] = googl_data.loc[volatile_data.index, 'Future_5d_Max_Return']
        
        volatile_hit_count = volatile_data['Hit_2_5pct_5d'].sum()
        volatile_hit_rate = (volatile_hit_count / len(volatile_data)) * 100 if len(volatile_data) > 0 else 0
        volatile_avg_return = volatile_data['Future_5d_Max_Return'].mean()
        
        print(f"Sample Size: {len(volatile_data)}")
        print(f"Hit Count: {volatile_hit_count}")
        print(f"Hit Rate: {volatile_hit_rate:.2f}%")
        print(f"Average 5-day Max Return: {volatile_avg_return:.2f}%")
        
        # Analyze by CRSI intervals
        volatile_data['CRSI_Prev_Bin'] = pd.cut(volatile_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        volatile_data['CRSI_Current_Bin'] = pd.cut(volatile_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        scenario1_results = []
        for prev_bin in bin_labels:
            for curr_bin in bin_labels:
                mask = (volatile_data['CRSI_Prev_Bin'] == prev_bin) & (volatile_data['CRSI_Current_Bin'] == curr_bin)
                subset = volatile_data[mask]
                if len(subset) >= 5:
                    hit_count = subset['Hit_2_5pct_5d'].sum()
                    hit_rate = (hit_count / len(subset)) * 100
                    avg_return = subset['Future_5d_Max_Return'].mean()
                    scenario1_results.append({
                        'CRSI_t-1': prev_bin,
                        'CRSI_t': curr_bin,
                        'Sample_Count': len(subset),
                        'Hit_Rate(%)': round(hit_rate, 2),
                        'Avg_Return(%)': round(avg_return, 2)
                    })
        
        if len(scenario1_results) > 0:
            scenario1_df = pd.DataFrame(scenario1_results)
            print("\nTop 10 CRSI Intervals by Hit Rate:")
            print(scenario1_df.sort_values('Hit_Rate(%)', ascending=False).head(10).to_string(index=False))
    else:
        print("No volatile trend data available")
    
    # Scenario 2: Downtrend - 3% gain in 5 days
    print("\n" + "-" * 60)
    print("Scenario 2: Downtrend - 3% gain in 5 days")
    print("-" * 60)
    
    # Calculate 3% hit for downtrend (different from volatile trend's 2.5%)
    googl_data['Hit_3pct_5d_downtrend'] = googl_data['Future_5d_Max_Return'] >= 3.0
    
    downtrend_data = valid_data[valid_data['Trend'] == 'downtrend'].copy()
    if len(downtrend_data) > 0:
        # Add 5-day hit data using index alignment
        downtrend_data['Hit_3pct_5d'] = googl_data.loc[downtrend_data.index, 'Hit_3pct_5d_downtrend']
        downtrend_data['Future_5d_Max_Return'] = googl_data.loc[downtrend_data.index, 'Future_5d_Max_Return']
        
        downtrend_hit_count = downtrend_data['Hit_3pct_5d'].sum()
        downtrend_hit_rate = (downtrend_hit_count / len(downtrend_data)) * 100 if len(downtrend_data) > 0 else 0
        downtrend_avg_return = downtrend_data['Future_5d_Max_Return'].mean()
        
        print(f"Sample Size: {len(downtrend_data)}")
        print(f"Hit Count: {downtrend_hit_count}")
        print(f"Hit Rate: {downtrend_hit_rate:.2f}%")
        print(f"Average 5-day Max Return: {downtrend_avg_return:.2f}%")
        
        # Analyze by CRSI intervals
        downtrend_data['CRSI_Prev_Bin'] = pd.cut(downtrend_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        downtrend_data['CRSI_Current_Bin'] = pd.cut(downtrend_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        scenario2_results = []
        for prev_bin in bin_labels:
            for curr_bin in bin_labels:
                mask = (downtrend_data['CRSI_Prev_Bin'] == prev_bin) & (downtrend_data['CRSI_Current_Bin'] == curr_bin)
                subset = downtrend_data[mask]
                if len(subset) >= 5:
                    hit_count = subset['Hit_3pct_5d'].sum()
                    hit_rate = (hit_count / len(subset)) * 100
                    avg_return = subset['Future_5d_Max_Return'].mean()
                    scenario2_results.append({
                        'CRSI_t-1': prev_bin,
                        'CRSI_t': curr_bin,
                        'Sample_Count': len(subset),
                        'Hit_Rate(%)': round(hit_rate, 2),
                        'Avg_Return(%)': round(avg_return, 2)
                    })
        
        if len(scenario2_results) > 0:
            scenario2_df = pd.DataFrame(scenario2_results)
            print("\nTop 10 CRSI Intervals by Hit Rate:")
            print(scenario2_df.sort_values('Hit_Rate(%)', ascending=False).head(10).to_string(index=False))
    else:
        print("No downtrend data available")
    
    # Scenario 3: Uptrend - 8% gain in 15 days
    print("\n" + "-" * 60)
    print("Scenario 3: Uptrend - 8% gain in 15 days")
    print("-" * 60)
    
    # Calculate 15-day max return for uptrend
    googl_data['Future_15d_High'] = googl_data['High'].rolling(window=15, min_periods=1).max().shift(-14)
    googl_data['Future_15d_Max_Return'] = (googl_data['Future_15d_High'] / googl_data['Close'] - 1) * 100
    googl_data['Hit_8pct_15d'] = googl_data['Future_15d_Max_Return'] >= 8.0
    
    uptrend_data = valid_data[valid_data['Trend'] == 'uptrend'].copy()
    if len(uptrend_data) > 0:
        # Add 15-day hit data using index alignment
        uptrend_data['Hit_8pct_15d'] = googl_data.loc[uptrend_data.index, 'Hit_8pct_15d']
        uptrend_data['Future_15d_Max_Return'] = googl_data.loc[uptrend_data.index, 'Future_15d_Max_Return']
        
        uptrend_hit_count = uptrend_data['Hit_8pct_15d'].sum()
        uptrend_hit_rate = (uptrend_hit_count / len(uptrend_data)) * 100 if len(uptrend_data) > 0 else 0
        uptrend_avg_return = uptrend_data['Future_15d_Max_Return'].mean()
        
        print(f"Sample Size: {len(uptrend_data)}")
        print(f"Hit Count: {uptrend_hit_count}")
        print(f"Hit Rate: {uptrend_hit_rate:.2f}%")
        print(f"Average 15-day Max Return: {uptrend_avg_return:.2f}%")
        
        # Analyze by CRSI intervals
        uptrend_data['CRSI_Prev_Bin'] = pd.cut(uptrend_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        uptrend_data['CRSI_Current_Bin'] = pd.cut(uptrend_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        scenario3_results = []
        for prev_bin in bin_labels:
            for curr_bin in bin_labels:
                mask = (uptrend_data['CRSI_Prev_Bin'] == prev_bin) & (uptrend_data['CRSI_Current_Bin'] == curr_bin)
                subset = uptrend_data[mask]
                if len(subset) >= 5:
                    hit_count = subset['Hit_8pct_15d'].sum()
                    hit_rate = (hit_count / len(subset)) * 100
                    avg_return = subset['Future_15d_Max_Return'].mean()
                    scenario3_results.append({
                        'CRSI_t-1': prev_bin,
                        'CRSI_t': curr_bin,
                        'Sample_Count': len(subset),
                        'Hit_Rate(%)': round(hit_rate, 2),
                        'Avg_Return(%)': round(avg_return, 2)
                    })
        
        if len(scenario3_results) > 0:
            scenario3_df = pd.DataFrame(scenario3_results)
            print("\nTop 10 CRSI Intervals by Hit Rate:")
            print(scenario3_df.sort_values('Hit_Rate(%)', ascending=False).head(10).to_string(index=False))
    else:
        print("No uptrend data available")
    
    # ========== 8. Detailed Optimization Analysis for Uptrend and Downtrend ==========
    print("\n" + "=" * 60)
    print("Detailed Optimization Analysis for Uptrend and Downtrend")
    print("=" * 60)
    
    # 8.1 Uptrend: Multiple Target/Period Combinations
    print("\n" + "-" * 60)
    print("8.1 Uptrend: Optimal Target/Period Combinations")
    print("-" * 60)
    
    uptrend_data = valid_data[valid_data['Trend'] == 'uptrend'].copy()
    if len(uptrend_data) > 0:
        # Calculate returns for multiple periods
        for period in [3, 5, 7, 10, 12, 15, 20]:
            googl_data[f'Future_{period}d_High'] = googl_data['High'].rolling(window=period, min_periods=1).max().shift(-(period-1))
            googl_data[f'Future_{period}d_Max_Return'] = (googl_data[f'Future_{period}d_High'] / googl_data['Close'] - 1) * 100
            googl_data[f'Future_{period}d_Low'] = googl_data['Low'].rolling(window=period, min_periods=1).min().shift(-(period-1))
            googl_data[f'Future_{period}d_Min_Return'] = (googl_data[f'Future_{period}d_Low'] / googl_data['Close'] - 1) * 100
            googl_data[f'Future_{period}d_Close_Return'] = (googl_data['Close'].shift(-period) / googl_data['Close'] - 1) * 100
        
        # Add data to uptrend_data
        for period in [3, 5, 7, 10, 12, 15, 20]:
            uptrend_data[f'Future_{period}d_Max_Return'] = googl_data.loc[uptrend_data.index, f'Future_{period}d_Max_Return']
            uptrend_data[f'Future_{period}d_Min_Return'] = googl_data.loc[uptrend_data.index, f'Future_{period}d_Min_Return']
            uptrend_data[f'Future_{period}d_Close_Return'] = googl_data.loc[uptrend_data.index, f'Future_{period}d_Close_Return']
        
        # Test different target/period combinations
        uptrend_combinations = [
            (3, 5, 'Quick: 3% in 5 days'),
            (5, 7, 'Short: 5% in 7 days'),
            (8, 10, 'Medium: 8% in 10 days'),
            (8, 12, 'Medium-Long: 8% in 12 days'),
            (10, 15, 'Long: 10% in 15 days'),
            (10, 20, 'Very Long: 10% in 20 days')
        ]
        
        uptrend_opt_results = []
        for target_pct, period, label in uptrend_combinations:
            hit_mask = uptrend_data[f'Future_{period}d_Max_Return'] >= target_pct
            hit_rate = (hit_mask.sum() / len(uptrend_data)) * 100
            avg_return = uptrend_data[f'Future_{period}d_Close_Return'].mean()
            max_dd = uptrend_data[f'Future_{period}d_Min_Return'].min()
            
            # Analyze by best CRSI intervals
            uptrend_data['CRSI_Prev_Bin'] = pd.cut(uptrend_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
            uptrend_data['CRSI_Current_Bin'] = pd.cut(uptrend_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
            
            best_intervals = []
            for prev_bin in bin_labels:
                for curr_bin in bin_labels:
                    subset = uptrend_data[
                        (uptrend_data['CRSI_Prev_Bin'] == prev_bin) & 
                        (uptrend_data['CRSI_Current_Bin'] == curr_bin)
                    ]
                    if len(subset) >= 5:
                        interval_hit_rate = (subset[f'Future_{period}d_Max_Return'] >= target_pct).sum() / len(subset) * 100
                        interval_avg_return = subset[f'Future_{period}d_Close_Return'].mean()
                        best_intervals.append({
                            'Interval': f'{prev_bin}→{curr_bin}',
                            'Sample': len(subset),
                            'Hit_Rate(%)': round(interval_hit_rate, 2),
                            'Avg_Return(%)': round(interval_avg_return, 2)
                        })
            
            if best_intervals:
                best_intervals_df = pd.DataFrame(best_intervals)
                top_interval = best_intervals_df.nlargest(1, 'Hit_Rate(%)').iloc[0]
                uptrend_opt_results.append({
                    'Strategy': label,
                    'Target/Period': f'{target_pct}%/{period}d',
                    'Overall_Hit_Rate(%)': round(hit_rate, 2),
                    'Best_Interval': top_interval['Interval'],
                    'Best_Hit_Rate(%)': top_interval['Hit_Rate(%)'],
                    'Best_Sample': int(top_interval['Sample']),
                    'Best_Avg_Return(%)': top_interval['Avg_Return(%)'],
                    'Overall_Avg_Return(%)': round(avg_return, 2),
                    'Max_Drawdown(%)': round(max_dd, 2)
                })
        
        if len(uptrend_opt_results) > 0:
            uptrend_opt_df = pd.DataFrame(uptrend_opt_results)
            print("\nUptrend Strategy Comparison:")
            print(uptrend_opt_df.to_string(index=False))
            
            # Find best strategy
            best_strategy = uptrend_opt_df.loc[uptrend_opt_df['Best_Hit_Rate(%)'].idxmax()]
            print(f"\nRecommended Uptrend Strategy:")
            print(f"  Strategy: {best_strategy['Strategy']}")
            print(f"  Target/Period: {best_strategy['Target/Period']}")
            print(f"  Best Interval: {best_strategy['Best_Interval']}")
            print(f"  Hit Rate: {best_strategy['Best_Hit_Rate(%)']:.2f}%")
            print(f"  Sample Size: {best_strategy['Best_Sample']}")
            print(f"  Avg Return: {best_strategy['Best_Avg_Return(%)']:.2f}%")
            print(f"  Max Drawdown: {best_strategy['Max_Drawdown(%)']:.2f}%")
    
    # 8.2 Downtrend: Multiple Target/Period Combinations
    print("\n" + "-" * 60)
    print("8.2 Downtrend: Optimal Target/Period Combinations")
    print("-" * 60)
    
    downtrend_data = valid_data[valid_data['Trend'] == 'downtrend'].copy()
    if len(downtrend_data) > 0:
        # Add data to downtrend_data
        for period in [3, 5, 7, 10, 12, 15, 20]:
            downtrend_data[f'Future_{period}d_Max_Return'] = googl_data.loc[downtrend_data.index, f'Future_{period}d_Max_Return']
            downtrend_data[f'Future_{period}d_Min_Return'] = googl_data.loc[downtrend_data.index, f'Future_{period}d_Min_Return']
            downtrend_data[f'Future_{period}d_Close_Return'] = googl_data.loc[downtrend_data.index, f'Future_{period}d_Close_Return']
        
        # Test different target/period combinations
        downtrend_combinations = [
            (3, 5, 'Quick: 3% in 5 days'),
            (5, 7, 'Short: 5% in 7 days'),
            (8, 10, 'Medium: 8% in 10 days'),
            (8, 12, 'Medium-Long: 8% in 12 days'),
            (10, 15, 'Long: 10% in 15 days'),
            (10, 20, 'Very Long: 10% in 20 days')
        ]
        
        downtrend_opt_results = []
        for target_pct, period, label in downtrend_combinations:
            hit_mask = downtrend_data[f'Future_{period}d_Max_Return'] >= target_pct
            hit_rate = (hit_mask.sum() / len(downtrend_data)) * 100
            avg_return = downtrend_data[f'Future_{period}d_Close_Return'].mean()
            max_dd = downtrend_data[f'Future_{period}d_Min_Return'].min()
            
            # Analyze by best CRSI intervals
            downtrend_data['CRSI_Prev_Bin'] = pd.cut(downtrend_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
            downtrend_data['CRSI_Current_Bin'] = pd.cut(downtrend_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
            
            best_intervals = []
            for prev_bin in bin_labels:
                for curr_bin in bin_labels:
                    subset = downtrend_data[
                        (downtrend_data['CRSI_Prev_Bin'] == prev_bin) & 
                        (downtrend_data['CRSI_Current_Bin'] == curr_bin)
                    ]
                    if len(subset) >= 5:
                        interval_hit_rate = (subset[f'Future_{period}d_Max_Return'] >= target_pct).sum() / len(subset) * 100
                        interval_avg_return = subset[f'Future_{period}d_Close_Return'].mean()
                        best_intervals.append({
                            'Interval': f'{prev_bin}→{curr_bin}',
                            'Sample': len(subset),
                            'Hit_Rate(%)': round(interval_hit_rate, 2),
                            'Avg_Return(%)': round(interval_avg_return, 2)
                        })
            
            if best_intervals:
                best_intervals_df = pd.DataFrame(best_intervals)
                top_interval = best_intervals_df.nlargest(1, 'Hit_Rate(%)').iloc[0]
                downtrend_opt_results.append({
                    'Strategy': label,
                    'Target/Period': f'{target_pct}%/{period}d',
                    'Overall_Hit_Rate(%)': round(hit_rate, 2),
                    'Best_Interval': top_interval['Interval'],
                    'Best_Hit_Rate(%)': top_interval['Hit_Rate(%)'],
                    'Best_Sample': int(top_interval['Sample']),
                    'Best_Avg_Return(%)': top_interval['Avg_Return(%)'],
                    'Overall_Avg_Return(%)': round(avg_return, 2),
                    'Max_Drawdown(%)': round(max_dd, 2)
                })
        
        if len(downtrend_opt_results) > 0:
            downtrend_opt_df = pd.DataFrame(downtrend_opt_results)
            print("\nDowntrend Strategy Comparison:")
            print(downtrend_opt_df.to_string(index=False))
            
            # Find best strategy
            best_strategy = downtrend_opt_df.loc[downtrend_opt_df['Best_Hit_Rate(%)'].idxmax()]
            print(f"\nRecommended Downtrend Strategy:")
            print(f"  Strategy: {best_strategy['Strategy']}")
            print(f"  Target/Period: {best_strategy['Target/Period']}")
            print(f"  Best Interval: {best_strategy['Best_Interval']}")
            print(f"  Hit Rate: {best_strategy['Best_Hit_Rate(%)']:.2f}%")
            print(f"  Sample Size: {best_strategy['Best_Sample']}")
            print(f"  Avg Return: {best_strategy['Best_Avg_Return(%)']:.2f}%")
            print(f"  Max Drawdown: {best_strategy['Max_Drawdown(%)']:.2f}%")
    
    # 8.3 Stop Loss Optimization Analysis
    print("\n" + "-" * 60)
    print("8.3 Stop Loss Optimization Analysis")
    print("-" * 60)
    
    # Analyze optimal stop loss for each trend
    for trend_name, trend_data in [('Uptrend', uptrend_data), ('Downtrend', downtrend_data)]:
        if len(trend_data) > 0:
            print(f"\n{trend_name} Stop Loss Analysis:")
            
            # Test different stop loss levels
            stop_loss_levels = [-2, -3, -4, -5, -6, -8, -10]
            sl_analysis = []
            
            for period in [5, 10, 15, 20]:
                max_return_col = f'Future_{period}d_Max_Return'
                min_return_col = f'Future_{period}d_Min_Return'
                
                if max_return_col in trend_data.columns:
                    for sl in stop_loss_levels:
                        # Count how many would hit stop loss before reaching target
                        # For different targets
                        for target in [3, 5, 8, 10]:
                            if target <= abs(sl) * 2:  # Only test if target makes sense
                                hit_target = (trend_data[max_return_col] >= target).sum()
                                hit_sl = (trend_data[min_return_col] <= sl).sum()
                                hit_sl_before_target = ((trend_data[min_return_col] <= sl) & 
                                                       (trend_data[max_return_col] < target)).sum()
                                
                                if len(trend_data) > 0:
                                    sl_analysis.append({
                                        'Period': period,
                                        'Target(%)': target,
                                        'Stop_Loss(%)': sl,
                                        'Hit_Target': hit_target,
                                        'Hit_SL': hit_sl,
                                        'Hit_SL_Before_Target': hit_sl_before_target,
                                        'Effective_Hit_Rate(%)': round((hit_target - hit_sl_before_target) / len(trend_data) * 100, 2)
                                    })
            
            if len(sl_analysis) > 0:
                sl_df = pd.DataFrame(sl_analysis)
                # Find best stop loss for each target/period
                print("\nOptimal Stop Loss by Target/Period:")
                for period in [5, 10, 15, 20]:
                    for target in [3, 5, 8, 10]:
                        subset = sl_df[(sl_df['Period'] == period) & (sl_df['Target(%)'] == target)]
                        if len(subset) > 0:
                            best_sl = subset.loc[subset['Effective_Hit_Rate(%)'].idxmax()]
                            print(f"  {target}% in {period}d: Stop Loss {best_sl['Stop_Loss(%)']:.0f}% "
                                  f"(Effective Hit Rate: {best_sl['Effective_Hit_Rate(%)']:.2f}%)")
    
    # 8.4 Risk-Adjusted Return Analysis
    print("\n" + "-" * 60)
    print("8.4 Risk-Adjusted Return Analysis")
    print("-" * 60)
    
    for trend_name, trend_data in [('Uptrend', uptrend_data), ('Downtrend', downtrend_data)]:
        if len(trend_data) > 0:
            print(f"\n{trend_name} Risk-Adjusted Metrics:")
            
            risk_metrics = []
            for period in [5, 10, 15, 20]:
                if f'Future_{period}d_Close_Return' in trend_data.columns:
                    returns = trend_data[f'Future_{period}d_Close_Return'].values
                    returns = returns[~np.isnan(returns)]
                    
                    if len(returns) > 0:
                        mean_return = returns.mean()
                        std_return = returns.std()
                        sharpe = mean_return / std_return if std_return > 0 else 0
                        win_rate = (returns > 0).sum() / len(returns) * 100
                        max_dd = returns.min()
                        
                        risk_metrics.append({
                            'Period': period,
                            'Mean_Return(%)': round(mean_return, 2),
                            'Std_Return(%)': round(std_return, 2),
                            'Sharpe_Ratio': round(sharpe, 3),
                            'Win_Rate(%)': round(win_rate, 2),
                            'Max_DD(%)': round(max_dd, 2)
                        })
            
            if len(risk_metrics) > 0:
                risk_df = pd.DataFrame(risk_metrics)
                print(risk_df.to_string(index=False))
                
                # Find best period by Sharpe ratio
                best_period = risk_df.loc[risk_df['Sharpe_Ratio'].idxmax()]
                print(f"\nBest Period (by Sharpe Ratio): {int(best_period['Period'])} days")
                print(f"  Sharpe Ratio: {best_period['Sharpe_Ratio']:.3f}")
                print(f"  Mean Return: {best_period['Mean_Return(%)']:.2f}%")
                print(f"  Win Rate: {best_period['Win_Rate(%)']:.2f}%")
    
    # 8.5 Fine-Grained CRSI Interval Analysis (5-point bins)
    print("\n" + "-" * 60)
    print("8.5 Fine-Grained CRSI Interval Analysis (5-point bins)")
    print("-" * 60)
    
    # Use 5-point bins for more granular analysis
    fine_bins = list(range(0, 101, 5))
    fine_bin_labels = [f'{i}-{i+5}' for i in range(0, 100, 5)]
    
    for trend_name, trend_data in [('Uptrend', uptrend_data), ('Downtrend', downtrend_data)]:
        if len(trend_data) > 0:
            print(f"\n{trend_name} Fine-Grained Analysis:")
            
            # Add fine-grained bins
            trend_data['CRSI_Prev_Fine'] = pd.cut(trend_data['CRSI_Prev'], bins=fine_bins, labels=fine_bin_labels, include_lowest=True)
            trend_data['CRSI_Current_Fine'] = pd.cut(trend_data['CRSI'], bins=fine_bins, labels=fine_bin_labels, include_lowest=True)
            
            # Analyze for best target/period combination
            if trend_name == 'Uptrend':
                target, period = 3, 5  # Quick strategy
            else:
                target, period = 3, 5  # Quick strategy
            
            hit_col = f'Future_{period}d_Max_Return'
            return_col = f'Future_{period}d_Close_Return'
            
            fine_results = []
            for prev_bin in fine_bin_labels:
                for curr_bin in fine_bin_labels:
                    subset = trend_data[
                        (trend_data['CRSI_Prev_Fine'] == prev_bin) & 
                        (trend_data['CRSI_Current_Fine'] == curr_bin)
                    ]
                    if len(subset) >= 3:  # Lower threshold for fine-grained
                        hit_rate = (subset[hit_col] >= target).sum() / len(subset) * 100
                        avg_return = subset[return_col].mean()
                        if not np.isnan(hit_rate) and hit_rate > 0:
                            fine_results.append({
                                'Interval': f'{prev_bin}→{curr_bin}',
                                'Sample': len(subset),
                                'Hit_Rate(%)': round(hit_rate, 2),
                                'Avg_Return(%)': round(avg_return, 2)
                            })
            
            if len(fine_results) > 0:
                fine_df = pd.DataFrame(fine_results)
                top_fine = fine_df.nlargest(5, 'Hit_Rate(%)')
                print(f"\nTop 5 Intervals for {target}% in {period} days:")
                print(top_fine.to_string(index=False))
    
    # 8.6 Simulated Backtest Results for Different Parameter Combinations
    print("\n" + "-" * 60)
    print("8.6 Simulated Backtest Results for Different Parameter Combinations")
    print("-" * 60)
    
    def simulate_backtest(df, buy_signals, take_profit_pct, stop_loss_pct, max_hold_days):
        """Simulate backtest to estimate performance"""
        trades = []
        buy_indices = df[buy_signals].index
        
        for buy_idx in buy_indices:
            try:
                buy_loc = df.index.get_loc(buy_idx)
                if buy_loc >= len(df) - 1:
                    continue
                
                buy_price = df.loc[buy_idx, 'Close']
                
                # Find sell date
                sold = False
                for i in range(1, min(max_hold_days + 1, len(df) - buy_loc)):
                    future_loc = buy_loc + i
                    if future_loc >= len(df):
                        break
                    
                    future_idx = df.index[future_loc]
                    future_high = df.loc[future_idx, 'High']
                    future_low = df.loc[future_idx, 'Low']
                    future_close = df.loc[future_idx, 'Close']
                    
                    high_return = (future_high / buy_price - 1) * 100
                    low_return = (future_low / buy_price - 1) * 100
                    
                    # Check take profit
                    if high_return >= take_profit_pct:
                        return_pct = take_profit_pct
                        reason = 'Take Profit'
                        days_held = i
                        trades.append({
                            'Return(%)': return_pct,
                            'Days_Held': days_held,
                            'Reason': reason
                        })
                        sold = True
                        break
                    
                    # Check stop loss
                    if low_return <= stop_loss_pct:
                        return_pct = stop_loss_pct
                        reason = 'Stop Loss'
                        days_held = i
                        trades.append({
                            'Return(%)': return_pct,
                            'Days_Held': days_held,
                            'Reason': reason
                        })
                        sold = True
                        break
                
                # Check expiry if not sold
                if not sold and buy_loc + max_hold_days < len(df):
                    future_idx = df.index[buy_loc + max_hold_days]
                    future_close = df.loc[future_idx, 'Close']
                    return_pct = (future_close / buy_price - 1) * 100
                    reason = 'Expiry'
                    days_held = max_hold_days
                    trades.append({
                        'Return(%)': return_pct,
                        'Days_Held': days_held,
                        'Reason': reason
                    })
            except (KeyError, IndexError):
                continue
        
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        total_return = trades_df['Return(%)'].sum()
        win_rate = (trades_df['Return(%)'] > 0).sum() / len(trades_df) * 100
        avg_return = trades_df['Return(%)'].mean()
        take_profit_count = (trades_df['Reason'] == 'Take Profit').sum()
        stop_loss_count = (trades_df['Reason'] == 'Stop Loss').sum()
        expiry_count = (trades_df['Reason'] == 'Expiry').sum()
        
        return {
            'Total_Trades': len(trades_df),
            'Total_Return(%)': round(total_return, 2),
            'Win_Rate(%)': round(win_rate, 2),
            'Avg_Return(%)': round(avg_return, 2),
            'Take_Profit_Count': take_profit_count,
            'Stop_Loss_Count': stop_loss_count,
            'Expiry_Count': expiry_count
        }
    
    # Test different parameter combinations for uptrend
    if len(uptrend_data) > 0:
        print("\nUptrend Parameter Optimization:")
        # Create signal on full dataframe
        uptrend_signal = (googl_data['Trend'] == 'uptrend') & \
                        (googl_data['CRSI_Prev'] >= 50) & (googl_data['CRSI_Prev'] < 60) & \
                        (googl_data['CRSI'] >= 30) & (googl_data['CRSI'] < 40)
        
        param_combinations = [
            (3, -3, 5, 'Quick: 3%/-3%/5d'),
            (5, -4, 7, 'Short: 5%/-4%/7d'),
            (8, -5, 10, 'Medium: 8%/-5%/10d'),
            (10, -5, 15, 'Long: 10%/-5%/15d'),
            (10, -5, 20, 'Very Long: 10%/-5%/20d')
        ]
        
        uptrend_sim_results = []
        for tp, sl, hold, label in param_combinations:
            result = simulate_backtest(googl_data, uptrend_signal, tp, sl, hold)
            if result:
                result['Strategy'] = label
                result['TP/SL/Hold'] = f'{tp}%/{sl}%/{hold}d'
                uptrend_sim_results.append(result)
        
        if len(uptrend_sim_results) > 0:
            uptrend_sim_df = pd.DataFrame(uptrend_sim_results)
            print(uptrend_sim_df.to_string(index=False))
    
    # Test different parameter combinations for downtrend
    if len(downtrend_data) > 0:
        print("\nDowntrend Parameter Optimization:")
        # Create signal on full dataframe
        downtrend_signal = (googl_data['Trend'] == 'downtrend') & \
                          (googl_data['CRSI_Prev'] >= 30) & (googl_data['CRSI_Prev'] < 40) & \
                          (googl_data['CRSI'] >= 30) & (googl_data['CRSI'] < 40)
        
        param_combinations = [
            (3, -2, 5, 'Quick: 3%/-2%/5d'),
            (3, -3, 5, 'Quick: 3%/-3%/5d'),
            (5, -3, 7, 'Short: 5%/-3%/7d'),
            (5, -4, 7, 'Short: 5%/-4%/7d')
        ]
        
        downtrend_sim_results = []
        for tp, sl, hold, label in param_combinations:
            result = simulate_backtest(googl_data, downtrend_signal, tp, sl, hold)
            if result:
                result['Strategy'] = label
                result['TP/SL/Hold'] = f'{tp}%/{sl}%/{hold}d'
                downtrend_sim_results.append(result)
        
        if len(downtrend_sim_results) > 0:
            downtrend_sim_df = pd.DataFrame(downtrend_sim_results)
            print(downtrend_sim_df.to_string(index=False))
    
    # ========== 9. Advanced Strategy Analysis ==========
    print("\n" + "=" * 60)
    print("Advanced Strategy Analysis")
    print("=" * 60)
    
    # 8.1 Win Rate and Profit/Loss Ratio Analysis
    print("\n" + "-" * 60)
    print("8.1 Win Rate and Profit/Loss Ratio by Trend")
    print("-" * 60)
    
    for trend in ['uptrend', 'volatile', 'downtrend']:
        trend_data = valid_data[valid_data['Trend'] == trend].copy()
        if len(trend_data) >= 10:
            # Calculate returns
            returns = trend_data['Future_20d_Close_Return'].values
            profitable = returns > 0
            losing = returns < 0
            
            win_rate = (profitable.sum() / len(returns)) * 100 if len(returns) > 0 else 0
            avg_win = returns[profitable].mean() if profitable.sum() > 0 else 0
            avg_loss = abs(returns[losing].mean()) if losing.sum() > 0 else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            print(f"\n{trend.upper()}:")
            print(f"  Sample Size: {len(trend_data)}")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Average Win: {avg_win:.2f}%")
            print(f"  Average Loss: -{avg_loss:.2f}%")
            print(f"  Profit/Loss Ratio: {profit_loss_ratio:.2f}")
            best_return = returns.max() if len(returns) > 0 and not np.isnan(returns.max()) else 0
            worst_return = returns.min() if len(returns) > 0 and not np.isnan(returns.min()) else 0
            print(f"  Best Return: {best_return:.2f}%")
            print(f"  Worst Return: {worst_return:.2f}%")
    
    # 8.2 Optimal Holding Period Analysis
    print("\n" + "-" * 60)
    print("8.2 Optimal Holding Period Analysis")
    print("-" * 60)
    
    # Calculate returns for different holding periods
    holding_periods = [5, 10, 15, 20, 25, 30]
    holding_analysis = []
    
    for period in holding_periods:
        googl_data[f'Future_{period}d_High'] = googl_data['High'].rolling(window=period, min_periods=1).max().shift(-(period-1))
        googl_data[f'Future_{period}d_Max_Return'] = (googl_data[f'Future_{period}d_High'] / googl_data['Close'] - 1) * 100
        googl_data[f'Future_{period}d_Close_Return'] = (googl_data['Close'].shift(-period) / googl_data['Close'] - 1) * 100
        
        # Analyze by trend
        for trend in ['uptrend', 'volatile', 'downtrend']:
            trend_mask = valid_data['Trend'] == trend
            trend_data = valid_data[trend_mask].copy()
            
            if len(trend_data) >= 10:
                # Get returns for this period
                period_max_returns = googl_data.loc[trend_data.index, f'Future_{period}d_Max_Return']
                period_close_returns = googl_data.loc[trend_data.index, f'Future_{period}d_Close_Return']
                
                # Calculate hit rate for different targets
                hit_5pct = (period_max_returns >= 5.0).sum() / len(trend_data) * 100
                hit_10pct = (period_max_returns >= 10.0).sum() / len(trend_data) * 100
                avg_return = period_close_returns.mean()
                
                holding_analysis.append({
                    'Trend': trend,
                    'Holding_Days': period,
                    'Hit_Rate_5pct(%)': round(hit_5pct, 2),
                    'Hit_Rate_10pct(%)': round(hit_10pct, 2),
                    'Avg_Return(%)': round(avg_return, 2),
                    'Sample_Size': len(trend_data)
                })
    
    if len(holding_analysis) > 0:
        holding_df = pd.DataFrame(holding_analysis)
        print("\nHolding Period Analysis by Trend:")
        print(holding_df.to_string(index=False))
        
        # Find optimal holding period for each trend
        print("\nOptimal Holding Periods (by Hit Rate 10%):")
        for trend in ['uptrend', 'volatile', 'downtrend']:
            trend_holding = holding_df[holding_df['Trend'] == trend]
            if len(trend_holding) > 0:
                best = trend_holding.loc[trend_holding['Hit_Rate_10pct(%)'].idxmax()]
                print(f"  {trend.upper()}: {int(best['Holding_Days'])} days (Hit Rate: {best['Hit_Rate_10pct(%)']:.2f}%)")
    
    # 8.3 CRSI Momentum Analysis
    print("\n" + "-" * 60)
    print("8.3 CRSI Momentum Analysis")
    print("-" * 60)
    
    # Calculate CRSI change
    valid_data['CRSI_Change'] = valid_data['CRSI'] - valid_data['CRSI_Prev']
    valid_data['CRSI_Change_Abs'] = abs(valid_data['CRSI_Change'])
    
    # Analyze by CRSI change magnitude
    momentum_bins = [0, 5, 10, 15, 20, 30, 50, 100]
    momentum_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-50', '50+']
    valid_data['CRSI_Change_Bin'] = pd.cut(valid_data['CRSI_Change_Abs'], bins=momentum_bins, labels=momentum_labels, include_lowest=True)
    
    momentum_analysis = []
    for change_bin in momentum_labels:
        for trend in ['uptrend', 'volatile', 'downtrend']:
            subset = valid_data[(valid_data['CRSI_Change_Bin'] == change_bin) & (valid_data['Trend'] == trend)]
            if len(subset) >= 10:
                hit_rate = (subset['Hit_TakeProfit'].sum() / len(subset)) * 100
                avg_return = subset['Future_20d_Close_Return'].mean()
                momentum_analysis.append({
                    'Trend': trend,
                    'CRSI_Change': change_bin,
                    'Sample_Size': len(subset),
                    'Hit_Rate(%)': round(hit_rate, 2),
                    'Avg_Return(%)': round(avg_return, 2)
                })
    
    if len(momentum_analysis) > 0:
        momentum_df = pd.DataFrame(momentum_analysis)
        print("\nCRSI Momentum Analysis:")
        print(momentum_df.to_string(index=False))
    
    # 8.4 Risk Metrics Analysis
    print("\n" + "-" * 60)
    print("8.4 Risk Metrics by Trend and Best Intervals")
    print("-" * 60)
    
    # Calculate risk metrics for best intervals
    best_intervals_by_trend = {
        'volatile': {'CRSI_Prev': [40, 50], 'CRSI': [40, 50]},
        'downtrend': {'CRSI_Prev': [30, 40], 'CRSI': [30, 40]},
        'uptrend': {'CRSI_Prev': [50, 60], 'CRSI': [40, 50]}
    }
    
    risk_analysis = []
    for trend, intervals in best_intervals_by_trend.items():
        trend_data = valid_data[valid_data['Trend'] == trend].copy()
        interval_data = trend_data[
            (trend_data['CRSI_Prev'] >= intervals['CRSI_Prev'][0]) & 
            (trend_data['CRSI_Prev'] < intervals['CRSI_Prev'][1]) &
            (trend_data['CRSI'] >= intervals['CRSI'][0]) & 
            (trend_data['CRSI'] < intervals['CRSI'][1])
        ]
        
        if len(interval_data) >= 10:
            returns = interval_data['Future_20d_Close_Return'].values
            
            # Risk metrics
            max_drawdown = returns.min()
            volatility = returns.std()
            sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0
            win_rate = (returns > 0).sum() / len(returns) * 100
            avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0
            
            risk_analysis.append({
                'Trend': trend,
                'Interval': f"{intervals['CRSI_Prev'][0]}-{intervals['CRSI_Prev'][1]} → {intervals['CRSI'][0]}-{intervals['CRSI'][1]}",
                'Sample_Size': len(interval_data),
                'Win_Rate(%)': round(win_rate, 2),
                'Avg_Return(%)': round(returns.mean(), 2),
                'Max_Drawdown(%)': round(max_drawdown, 2),
                'Volatility(%)': round(volatility, 2),
                'Sharpe_Ratio': round(sharpe_ratio, 3),
                'Avg_Win(%)': round(avg_win, 2),
                'Avg_Loss(%)': round(avg_loss, 2)
            })
    
    if len(risk_analysis) > 0:
        risk_df = pd.DataFrame(risk_analysis)
        print("\nRisk Metrics for Best Intervals:")
        print(risk_df.to_string(index=False))
    
    # 8.5 Signal Strength Analysis
    print("\n" + "-" * 60)
    print("8.5 Signal Strength Analysis (Multiple Confirmations)")
    print("-" * 60)
    
    # Analyze signals with multiple confirmations
    signal_analysis = []
    
    # Strong signals: trend + best CRSI interval + momentum
    for trend in ['uptrend', 'volatile', 'downtrend']:
        trend_data = valid_data[valid_data['Trend'] == trend].copy()
        
        # Get best interval for this trend
        if trend == 'volatile':
            signal_data = trend_data[
                (trend_data['CRSI_Prev'] >= 40) & (trend_data['CRSI_Prev'] < 50) &
                (trend_data['CRSI'] >= 40) & (trend_data['CRSI'] < 50)
            ]
        elif trend == 'downtrend':
            signal_data = trend_data[
                (trend_data['CRSI_Prev'] >= 30) & (trend_data['CRSI_Prev'] < 40) &
                (trend_data['CRSI'] >= 30) & (trend_data['CRSI'] < 40)
            ]
        else:  # uptrend
            signal_data = trend_data[
                (trend_data['CRSI_Prev'] >= 50) & (trend_data['CRSI_Prev'] < 60) &
                (trend_data['CRSI'] >= 40) & (trend_data['CRSI'] < 50)
            ]
        
        if len(signal_data) >= 5:
            # Add momentum filter (CRSI change)
            strong_momentum = signal_data[signal_data['CRSI_Change_Abs'] >= 5]
            weak_momentum = signal_data[signal_data['CRSI_Change_Abs'] < 5]
            
            for momentum_type, momentum_data in [('Strong Momentum', strong_momentum), ('Weak Momentum', weak_momentum)]:
                if len(momentum_data) >= 5:
                    hit_rate = (momentum_data['Hit_TakeProfit'].sum() / len(momentum_data)) * 100
                    avg_return = momentum_data['Future_20d_Close_Return'].mean()
                    signal_analysis.append({
                        'Trend': trend,
                        'Signal_Type': momentum_type,
                        'Sample_Size': len(momentum_data),
                        'Hit_Rate(%)': round(hit_rate, 2),
                        'Avg_Return(%)': round(avg_return, 2)
                    })
    
    if len(signal_analysis) > 0:
        signal_df = pd.DataFrame(signal_analysis)
        print("\nSignal Strength Analysis:")
        print(signal_df.to_string(index=False))
    
    # ========== 9. Visualization: Trend Time Series and Hit Rate Heatmaps ==========
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    # Ensure we use the same bins and labels as in statistics
    bins = list(range(0, 101, 10))
    bin_labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Trend Time Series (Top Left)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Map trends to numeric values for plotting
    trend_map = {'uptrend': 3, 'downtrend': 1, 'volatile': 2, 'neutral': 0}
    trend_colors = {'uptrend': 'green', 'downtrend': 'red', 'volatile': 'orange', 'neutral': 'gray'}
    
    # Plot price
    ax1_twin = ax1.twinx()
    ax1_twin.plot(googl_data.index, googl_data['Close'], label='GOOGL Close Price', 
                  linewidth=1.5, alpha=0.6, color='black')
    ax1_twin.set_ylabel('Price ($)', fontsize=11, color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black')
    
    # Plot trend as colored background
    for trend, value in trend_map.items():
        trend_mask = googl_data['Trend'] == trend
        if trend_mask.any():
            ax1.fill_between(googl_data.index, value-0.4, value+0.4, 
                           where=trend_mask, alpha=0.3, color=trend_colors[trend], 
                           label=trend.capitalize())
    
    ax1.set_ylabel('Trend', fontsize=11)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_yticks(list(trend_map.values()))
    ax1.set_yticklabels(list(trend_map.keys()))
    ax1.set_title('Trend Distribution Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2-4. Hit Rate Heatmaps for Each Trend (using same data as statistics)
    trends_for_plot = ['uptrend', 'volatile', 'downtrend']
    trend_titles = ['Uptrend', 'Volatile', 'Downtrend']
    
    for idx, (trend, title) in enumerate(zip(trends_for_plot, trend_titles)):
        ax = fig.add_subplot(gs[1 + idx//2, idx % 2])
        
        # Use valid_data with same filtering as statistics
        trend_data = valid_data[valid_data['Trend'] == trend].copy()
        
        if len(trend_data) < 10:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {title}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{title} - Hit Rate Heatmap\n(20 days, +10%)', fontsize=11, fontweight='bold')
            continue
        
        # Create bins using same method as statistics
        trend_data['CRSI_Prev_Bin'] = pd.cut(trend_data['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        trend_data['CRSI_Current_Bin'] = pd.cut(trend_data['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Calculate hit rates for each bin combination (same as statistics)
        heatmap_data = []
        for prev_bin in bin_labels:
            row = []
            for curr_bin in bin_labels:
                mask = (trend_data['CRSI_Prev_Bin'] == prev_bin) & (trend_data['CRSI_Current_Bin'] == curr_bin)
                subset = trend_data[mask]
                if len(subset) >= 5:  # Minimum sample size (same as statistics)
                    hit_rate = (subset['Hit_TakeProfit'].sum() / len(subset)) * 100
                    row.append(hit_rate)
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=bin_labels, columns=bin_labels)
        
        # Plot heatmap
        if not heatmap_df.isna().all().all():
            baseline_hr = trend_baseline_hit_rates.get(trend, 0)
            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                       vmin=0, vmax=100, center=baseline_hr, cbar_kws={'label': 'Hit Rate (%)'},
                       ax=ax, linewidths=0.5, linecolor='gray', annot_kws={'size': 7})
            ax.set_title(f'{title} - Hit Rate Heatmap (20d, +10%)\n(Baseline: {baseline_hr:.1f}%)', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('CRSI_t (Current)', fontsize=10)
            ax.set_ylabel('CRSI_t-1 (Previous)', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'No sufficient data\nfor {title} heatmap', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{title} - Hit Rate Heatmap', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'CRSI Interval Analysis by Trend\n(Take Profit: +{TAKE_PROFIT_PCT}%, Max Hold: {MAX_HOLD_DAYS} days)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('statistics_trend_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: statistics_trend_analysis.png")
    plt.close()
    
    # Additional chart: Trend distribution over time (monthly aggregation)
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10))
    
    # Monthly trend distribution
    googl_data['YearMonth'] = googl_data.index.to_period('M')
    monthly_trends = googl_data.groupby('YearMonth')['Trend'].value_counts().unstack(fill_value=0)
    monthly_trends_pct = monthly_trends.div(monthly_trends.sum(axis=1), axis=0) * 100
    
    ax2_1 = axes2[0]
    monthly_trends_pct.plot(kind='area', ax=ax2_1, stacked=True, 
                           color={'uptrend': 'green', 'downtrend': 'red', 'volatile': 'orange', 'neutral': 'gray'},
                           alpha=0.6)
    ax2_1.set_title('Trend Distribution Over Time (Monthly)', fontsize=13, fontweight='bold')
    ax2_1.set_ylabel('Percentage (%)', fontsize=11)
    ax2_1.set_xlabel('Date', fontsize=11)
    ax2_1.legend(title='Trend', loc='upper left', fontsize=9)
    ax2_1.grid(True, alpha=0.3)
    ax2_1.set_ylim(0, 100)
    
    # Hit rate by trend (bar chart) - using same data as statistics
    ax2_2 = axes2[1]
    trend_hit_rates = {}
    trend_sample_sizes = {}
    for trend in ['uptrend', 'volatile', 'downtrend']:
        trend_data = valid_data[valid_data['Trend'] == trend]
        if len(trend_data) > 0:
            hit_rate = (trend_data['Hit_TakeProfit'].sum() / len(trend_data)) * 100
            trend_hit_rates[trend] = hit_rate
            trend_sample_sizes[trend] = len(trend_data)
    
    if trend_hit_rates:
        bars = ax2_2.bar(trend_hit_rates.keys(), trend_hit_rates.values(), 
                        color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2_2.axhline(y=baseline_hit_rate, color='blue', linestyle='--', linewidth=2, 
                     label=f'Overall Baseline ({baseline_hit_rate:.1f}%)')
        ax2_2.set_title(f'Hit Rate by Trend (20 days, +{TAKE_PROFIT_PCT}%)', fontsize=13, fontweight='bold')
        ax2_2.set_ylabel('Hit Rate (%)', fontsize=11)
        ax2_2.set_xlabel('Trend', fontsize=11)
        ax2_2.legend(fontsize=10)
        ax2_2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars with sample sizes
        for bar, trend in zip(bars, trend_hit_rates.keys()):
            height = bar.get_height()
            sample_size = trend_sample_sizes.get(trend, 0)
            ax2_2.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.1f}%\n(n={sample_size})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('statistics_trend_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved to: statistics_trend_distribution.png")
    plt.close()
    
    # ========== 9. Specific Scenario Visualization ==========
    print("Generating Specific Scenario Visualizations...")
    
    # Create scenario comparison chart
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scenario 1: Volatile - 2.5% in 5 days
    ax3_1 = axes3[0, 0]
    volatile_scenario = valid_data[valid_data['Trend'] == 'volatile'].copy()
    if len(volatile_scenario) > 0:
        volatile_scenario['Hit_2_5pct_5d'] = googl_data.loc[volatile_scenario.index, 'Hit_2_5pct_5d']
        volatile_scenario['CRSI_Prev_Bin'] = pd.cut(volatile_scenario['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        volatile_scenario['CRSI_Current_Bin'] = pd.cut(volatile_scenario['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        scenario1_heatmap = []
        for prev_bin in bin_labels:
            row = []
            for curr_bin in bin_labels:
                mask = (volatile_scenario['CRSI_Prev_Bin'] == prev_bin) & (volatile_scenario['CRSI_Current_Bin'] == curr_bin)
                subset = volatile_scenario[mask]
                if len(subset) >= 5:
                    hit_rate = (subset['Hit_2_5pct_5d'].sum() / len(subset)) * 100
                    row.append(hit_rate)
                else:
                    row.append(np.nan)
            scenario1_heatmap.append(row)
        
        scenario1_df = pd.DataFrame(scenario1_heatmap, index=bin_labels, columns=bin_labels)
        if not scenario1_df.isna().all().all():
            sns.heatmap(scenario1_df, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                       center=45.52, cbar_kws={'label': 'Hit Rate (%)'}, ax=ax3_1, 
                       linewidths=0.5, linecolor='gray', annot_kws={'size': 6})
            ax3_1.set_title('Volatile: 2.5% in 5 days\n(Baseline: 45.52%)', fontsize=11, fontweight='bold')
            ax3_1.set_xlabel('CRSI_t', fontsize=9)
            ax3_1.set_ylabel('CRSI_t-1', fontsize=9)
    
    # Scenario 2: Downtrend - 3% in 5 days
    ax3_2 = axes3[0, 1]
    downtrend_scenario = valid_data[valid_data['Trend'] == 'downtrend'].copy()
    if len(downtrend_scenario) > 0:
        downtrend_scenario['Hit_3pct_5d'] = googl_data.loc[downtrend_scenario.index, 'Hit_3pct_5d_downtrend']
        downtrend_scenario['CRSI_Prev_Bin'] = pd.cut(downtrend_scenario['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        downtrend_scenario['CRSI_Current_Bin'] = pd.cut(downtrend_scenario['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        scenario2_heatmap = []
        for prev_bin in bin_labels:
            row = []
            for curr_bin in bin_labels:
                mask = (downtrend_scenario['CRSI_Prev_Bin'] == prev_bin) & (downtrend_scenario['CRSI_Current_Bin'] == curr_bin)
                subset = downtrend_scenario[mask]
                if len(subset) >= 5:
                    hit_rate = (subset['Hit_3pct_5d'].sum() / len(subset)) * 100
                    row.append(hit_rate)
                else:
                    row.append(np.nan)
            scenario2_heatmap.append(row)
        
        scenario2_df = pd.DataFrame(scenario2_heatmap, index=bin_labels, columns=bin_labels)
        if not scenario2_df.isna().all().all():
            sns.heatmap(scenario2_df, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                       center=55.37, cbar_kws={'label': 'Hit Rate (%)'}, ax=ax3_2,
                       linewidths=0.5, linecolor='gray', annot_kws={'size': 6})
            ax3_2.set_title('Downtrend: 3% in 5 days\n(Baseline: 55.37%)', fontsize=11, fontweight='bold')
            ax3_2.set_xlabel('CRSI_t', fontsize=9)
            ax3_2.set_ylabel('CRSI_t-1', fontsize=9)
    
    # Scenario 3: Uptrend - 8% in 15 days
    ax3_3 = axes3[1, 0]
    uptrend_scenario = valid_data[valid_data['Trend'] == 'uptrend'].copy()
    if len(uptrend_scenario) > 0:
        uptrend_scenario['Hit_8pct_15d'] = googl_data.loc[uptrend_scenario.index, 'Hit_8pct_15d']
        uptrend_scenario['CRSI_Prev_Bin'] = pd.cut(uptrend_scenario['CRSI_Prev'], bins=bins, labels=bin_labels, include_lowest=True)
        uptrend_scenario['CRSI_Current_Bin'] = pd.cut(uptrend_scenario['CRSI'], bins=bins, labels=bin_labels, include_lowest=True)
        
        scenario3_heatmap = []
        for prev_bin in bin_labels:
            row = []
            for curr_bin in bin_labels:
                mask = (uptrend_scenario['CRSI_Prev_Bin'] == prev_bin) & (uptrend_scenario['CRSI_Current_Bin'] == curr_bin)
                subset = uptrend_scenario[mask]
                if len(subset) >= 5:
                    hit_rate = (subset['Hit_8pct_15d'].sum() / len(subset)) * 100
                    row.append(hit_rate)
                else:
                    row.append(np.nan)
            scenario3_heatmap.append(row)
        
        scenario3_df = pd.DataFrame(scenario3_heatmap, index=bin_labels, columns=bin_labels)
        if not scenario3_df.isna().all().all():
            sns.heatmap(scenario3_df, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                       center=11.19, cbar_kws={'label': 'Hit Rate (%)'}, ax=ax3_3,
                       linewidths=0.5, linecolor='gray', annot_kws={'size': 6})
            ax3_3.set_title('Uptrend: 8% in 15 days\n(Baseline: 11.19%)', fontsize=11, fontweight='bold')
            ax3_3.set_xlabel('CRSI_t', fontsize=9)
            ax3_3.set_ylabel('CRSI_t-1', fontsize=9)
    
    # Scenario comparison bar chart
    ax3_4 = axes3[1, 1]
    scenario_baselines = {
        'Volatile\n(2.5%, 5d)': 45.52,
        'Downtrend\n(3%, 5d)': 55.37,
        'Uptrend\n(8%, 15d)': 11.19
    }
    bars = ax3_4.bar(scenario_baselines.keys(), scenario_baselines.values(),
                     color=['orange', 'red', 'green'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3_4.set_title('Scenario Baseline Hit Rates', fontsize=13, fontweight='bold')
    ax3_4.set_ylabel('Hit Rate (%)', fontsize=11)
    ax3_4.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3_4.text(bar.get_x() + bar.get_width()/2., height,
                  f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Specific Scenario Analysis - Hit Rate Heatmaps', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('statistics_scenario_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved to: statistics_scenario_analysis.png")
    plt.close()
    
    # ========== 10. Advanced Analysis Visualizations ==========
    print("Generating Advanced Analysis Visualizations...")
    
    # Create comprehensive advanced analysis chart
    fig4, axes4 = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Win Rate and Profit/Loss Ratio by Trend
    ax4_1 = axes4[0, 0]
    win_rate_data = []
    pl_ratio_data = []
    trend_names = []
    
    for trend in ['uptrend', 'volatile', 'downtrend']:
        trend_data = valid_data[valid_data['Trend'] == trend]
        if len(trend_data) >= 10:
            returns = trend_data['Future_20d_Close_Return'].values
            profitable = returns > 0
            win_rate = (profitable.sum() / len(returns)) * 100 if len(returns) > 0 else 0
            avg_win = returns[profitable].mean() if profitable.sum() > 0 else 0
            avg_loss = abs(returns[~profitable].mean()) if (~profitable).sum() > 0 else 0
            pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            win_rate_data.append(win_rate)
            pl_ratio_data.append(pl_ratio)
            trend_names.append(trend.capitalize())
    
    if win_rate_data:
        x = np.arange(len(trend_names))
        width = 0.35
        bars1 = ax4_1.bar(x - width/2, win_rate_data, width, label='Win Rate (%)', color='green', alpha=0.7)
        ax4_1_twin = ax4_1.twinx()
        bars2 = ax4_1_twin.bar(x + width/2, pl_ratio_data, width, label='P/L Ratio', color='blue', alpha=0.7)
        ax4_1.set_xlabel('Trend', fontsize=11)
        ax4_1.set_ylabel('Win Rate (%)', fontsize=11, color='green')
        ax4_1_twin.set_ylabel('Profit/Loss Ratio', fontsize=11, color='blue')
        ax4_1.set_xticks(x)
        ax4_1.set_xticklabels(trend_names)
        ax4_1.set_title('Win Rate and P/L Ratio by Trend', fontsize=12, fontweight='bold')
        ax4_1.legend(loc='upper left')
        ax4_1_twin.legend(loc='upper right')
        ax4_1.grid(True, alpha=0.3, axis='y')
    
    # 2. Optimal Holding Period Analysis
    ax4_2 = axes4[0, 1]
    if len(holding_analysis) > 0:
        holding_df_plot = pd.DataFrame(holding_analysis)
        for trend in ['uptrend', 'volatile', 'downtrend']:
            trend_holding = holding_df_plot[holding_df_plot['Trend'] == trend]
            if len(trend_holding) > 0:
                ax4_2.plot(trend_holding['Holding_Days'], trend_holding['Hit_Rate_10pct(%)'], 
                          marker='o', label=trend.capitalize(), linewidth=2)
        ax4_2.set_xlabel('Holding Days', fontsize=11)
        ax4_2.set_ylabel('Hit Rate 10% (%)', fontsize=11)
        ax4_2.set_title('Optimal Holding Period Analysis', fontsize=12, fontweight='bold')
        ax4_2.legend()
        ax4_2.grid(True, alpha=0.3)
    
    # 3. Risk Metrics Comparison
    ax4_3 = axes4[0, 2]
    if len(risk_analysis) > 0:
        risk_df_plot = pd.DataFrame(risk_analysis)
        x = np.arange(len(risk_df_plot))
        width = 0.25
        ax4_3.bar(x - width, risk_df_plot['Win_Rate(%)'], width, label='Win Rate (%)', color='green', alpha=0.7)
        ax4_3.bar(x, risk_df_plot['Sharpe_Ratio'] * 10, width, label='Sharpe Ratio (×10)', color='blue', alpha=0.7)
        ax4_3.bar(x + width, risk_df_plot['Avg_Return(%)'], width, label='Avg Return (%)', color='orange', alpha=0.7)
        ax4_3.set_xlabel('Best Interval by Trend', fontsize=11)
        ax4_3.set_ylabel('Value', fontsize=11)
        ax4_3.set_title('Risk Metrics Comparison', fontsize=12, fontweight='bold')
        ax4_3.set_xticks(x)
        ax4_3.set_xticklabels(risk_df_plot['Trend'], rotation=45, ha='right')
        ax4_3.legend()
        ax4_3.grid(True, alpha=0.3, axis='y')
    
    # 4. CRSI Momentum Analysis
    ax4_4 = axes4[1, 0]
    if len(momentum_analysis) > 0:
        momentum_df_plot = pd.DataFrame(momentum_analysis)
        momentum_pivot = momentum_df_plot.pivot(index='CRSI_Change', columns='Trend', values='Hit_Rate(%)')
        sns.heatmap(momentum_pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4_4, 
                   cbar_kws={'label': 'Hit Rate (%)'}, linewidths=0.5, linecolor='gray')
        ax4_4.set_title('CRSI Momentum vs Hit Rate', fontsize=12, fontweight='bold')
        ax4_4.set_xlabel('Trend', fontsize=11)
        ax4_4.set_ylabel('CRSI Change Magnitude', fontsize=11)
    
    # 5. Signal Strength Comparison
    ax4_5 = axes4[1, 1]
    if len(signal_analysis) > 0:
        signal_df_plot = pd.DataFrame(signal_analysis)
        signal_pivot = signal_df_plot.pivot(index='Trend', columns='Signal_Type', values='Hit_Rate(%)')
        signal_pivot.plot(kind='bar', ax=ax4_5, width=0.8, alpha=0.7)
        ax4_5.set_title('Signal Strength Analysis', fontsize=12, fontweight='bold')
        ax4_5.set_xlabel('Trend', fontsize=11)
        ax4_5.set_ylabel('Hit Rate (%)', fontsize=11)
        ax4_5.legend(title='Signal Type')
        ax4_5.grid(True, alpha=0.3, axis='y')
        ax4_5.set_xticklabels(ax4_5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Holding Period vs Hit Rate (5% and 10%)
    ax4_6 = axes4[1, 2]
    if len(holding_analysis) > 0:
        holding_df_plot = pd.DataFrame(holding_analysis)
        for trend in ['uptrend', 'volatile', 'downtrend']:
            trend_holding = holding_df_plot[holding_df_plot['Trend'] == trend]
            if len(trend_holding) > 0:
                ax4_6.plot(trend_holding['Holding_Days'], trend_holding['Hit_Rate_5pct(%)'], 
                          marker='s', label=f'{trend.capitalize()} (5%)', linestyle='--', linewidth=2)
                ax4_6.plot(trend_holding['Holding_Days'], trend_holding['Hit_Rate_10pct(%)'], 
                          marker='o', label=f'{trend.capitalize()} (10%)', linewidth=2)
        ax4_6.set_xlabel('Holding Days', fontsize=11)
        ax4_6.set_ylabel('Hit Rate (%)', fontsize=11)
        ax4_6.set_title('Holding Period: 5% vs 10% Target', fontsize=12, fontweight='bold')
        ax4_6.legend()
        ax4_6.grid(True, alpha=0.3)
    
    plt.suptitle('Advanced Strategy Analysis - Comprehensive Metrics', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('statistics_advanced_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved to: statistics_advanced_analysis.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Important Notes:")
    print("=" * 60)
    print("1. The above rules are based on historical data backtesting. Actual trading should consider market environment changes")
    print("2. It is recommended to combine with other technical indicators and fundamental analysis")
    print("3. Strictly execute stop loss and take profit rules to control risk")
    print(f"4. Current hit definition: {hit_description}")
    print("5. It is recommended to verify rule effectiveness in a simulated environment before live trading")
    
else:
    print("\nNo valid data samples found")


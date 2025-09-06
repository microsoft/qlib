#!/usr/bin/env python3
"""
Enhanced Qlib Data Visualizations
=================================
Creates comprehensive visualizations for the Qlib dataset analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def initialize_qlib():
    """Initialize Qlib with error handling."""
    try:
        import qlib
        from qlib.constant import REG_CN
        mount_path = "/workspace/qlib/data/qlib_data/cn_data"
        qlib.init(provider_uri=mount_path, region=REG_CN)
        return True
    except Exception as e:
        print(f"Failed to initialize Qlib: {e}")
        return False

def get_trading_calendar():
    """Get trading calendar data."""
    try:
        from qlib.data import D
        calendar = D.calendar(start_time='2005-01-01', end_time='2025-12-31', freq='day')
        return pd.Series(calendar).to_list()
    except Exception as e:
        print(f"Failed to get trading calendar: {e}")
        return None

def create_enhanced_calendar_visualization():
    """Create enhanced trading calendar visualization."""
    try:
        calendar_dates = get_trading_calendar()
        if not calendar_dates:
            return
            
        print("📅 Creating enhanced trading calendar visualization...")
        
        # Convert to DataFrame
        df = pd.DataFrame({'date': calendar_dates})
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday  # 0=Monday, 6=Sunday
        df['quarter'] = df['date'].dt.quarter
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Chinese Stock Market Trading Calendar Analysis', fontsize=16, fontweight='bold')
        
        # 1. Trading days per year
        yearly_counts = df.groupby('year').size()
        yearly_counts = yearly_counts[yearly_counts.index >= 2010]  # Focus on recent years
        
        axes[0,0].bar(yearly_counts.index, yearly_counts.values, color='steelblue', alpha=0.8)
        axes[0,0].set_title('Trading Days Per Year (2010-2025)', fontweight='bold')
        axes[0,0].set_ylabel('Number of Trading Days')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(yearly_counts.index, yearly_counts.values, 1)
        p = np.poly1d(z)
        axes[0,0].plot(yearly_counts.index, p(yearly_counts.index), "r--", alpha=0.8, label='Trend')
        axes[0,0].legend()
        
        # 2. Trading days by month (all years combined)
        monthly_avg = df.groupby('month').size() / len(df['year'].unique())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[0,1].bar(range(1, 13), monthly_avg.values, color='orange', alpha=0.8)
        axes[0,1].set_title('Average Trading Days Per Month', fontweight='bold')
        axes[0,1].set_ylabel('Average Trading Days')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_xticks(range(1, 13))
        axes[0,1].set_xticklabels(month_names)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Trading days by weekday
        weekday_counts = df.groupby('weekday').size()
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_labels = [weekday_names[i] for i in weekday_counts.index]
        
        axes[1,0].bar(range(len(weekday_counts)), weekday_counts.values, color='green', alpha=0.8)
        axes[1,0].set_title('Trading Days by Weekday (Total)', fontweight='bold')
        axes[1,0].set_ylabel('Number of Trading Days')
        axes[1,0].set_xlabel('Weekday')
        axes[1,0].set_xticks(range(len(weekday_counts)))
        axes[1,0].set_xticklabels(weekday_labels)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Recent trading activity (last 2 years)
        recent_df = df[df['year'] >= 2023].copy()
        recent_df['month_year'] = recent_df['date'].dt.to_period('M')
        recent_monthly = recent_df.groupby('month_year').size()
        
        axes[1,1].plot(range(len(recent_monthly)), recent_monthly.values, 
                      marker='o', linewidth=2, markersize=6, color='red')
        axes[1,1].set_title('Recent Trading Activity (2023-2025)', fontweight='bold')
        axes[1,1].set_ylabel('Trading Days per Month')
        axes[1,1].set_xlabel('Month-Year')
        axes[1,1].grid(True, alpha=0.3)
        
        # Set x-tick labels for recent activity
        x_labels = [str(p) for p in recent_monthly.index[::3]]  # Every 3rd month
        axes[1,1].set_xticks(range(0, len(recent_monthly), 3))
        axes[1,1].set_xticklabels(x_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/enhanced_trading_calendar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Enhanced trading calendar visualization created")
        
    except Exception as e:
        print(f"⚠️  Failed to create calendar visualization: {e}")

def create_market_overview():
    """Create comprehensive market overview."""
    try:
        from qlib.data import D
        
        print("🏢 Creating comprehensive market overview...")
        
        # Get instrument counts for different segments
        segments = {
            'CSI 300': 'csi300',
            'CSI 500': 'csi500', 
            'CSI 800': 'csi800',
            'CSI 1000': 'csi1000'
        }
        
        segment_data = {}
        for name, code in segments.items():
            try:
                instruments = D.instruments(code)
                count = len(D.list_instruments(
                    instruments=instruments, 
                    start_time='2024-01-01', 
                    end_time='2025-09-06', 
                    as_list=True
                ))
                segment_data[name] = count
                print(f"   {name}: {count} instruments")
            except Exception as e:
                print(f"   ⚠️  Failed to get {name}: {e}")
                segment_data[name] = 0
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Bar chart of market segments
        names = list(segment_data.keys())
        counts = list(segment_data.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = axes[0].bar(names, counts, color=colors, alpha=0.8)
        axes[0].set_title('Chinese Stock Market Index Coverage', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Stocks')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Pie chart showing market composition
        # Remove segments with 0 stocks
        pie_data = {k: v for k, v in segment_data.items() if v > 0}
        
        if pie_data:
            axes[1].pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%',
                       colors=colors[:len(pie_data)], startangle=90)
            axes[1].set_title('Market Segment Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/market_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Market overview visualization created")
        return segment_data
        
    except Exception as e:
        print(f"⚠️  Failed to create market overview: {e}")
        return {}

def create_stock_analysis():
    """Create detailed stock price analysis."""
    try:
        from qlib.data import D
        
        print("📈 Creating detailed stock analysis...")
        
        # Get some popular stocks
        instruments = D.instruments('csi300')
        available_stocks = D.list_instruments(
            instruments=instruments, 
            start_time='2024-01-01', 
            end_time='2025-09-06', 
            as_list=True
        )
        
        if not available_stocks:
            print("⚠️  No stocks available for analysis")
            return
            
        # Select first 3 stocks with good data
        analysis_stocks = available_stocks[:3]
        
        fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
        
        # Create individual stock analysis
        for i, stock in enumerate(analysis_stocks):
            try:
                data = D.features(
                    instruments=[stock], 
                    fields=fields,
                    start_time='2024-06-01', 
                    end_time='2025-09-06'
                )
                
                if data.empty:
                    continue
                    
                # Reset index to work with datetime
                data_df = data.reset_index()
                
                # Create comprehensive analysis for this stock
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Comprehensive Analysis - {stock}', fontsize=16, fontweight='bold')
                
                # 1. Price chart with VWAP
                axes[0,0].plot(data_df['datetime'], data_df['$close'], label='Close Price', linewidth=2)
                axes[0,0].plot(data_df['datetime'], data_df['$vwap'], label='VWAP', linewidth=2, alpha=0.8)
                axes[0,0].fill_between(data_df['datetime'], data_df['$low'], data_df['$high'], 
                                      alpha=0.3, label='High-Low Range')
                axes[0,0].set_title('Price Movement with VWAP')
                axes[0,0].set_ylabel('Price')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].tick_params(axis='x', rotation=45)
                
                # 2. Volume analysis
                axes[0,1].bar(data_df['datetime'], data_df['$volume'], alpha=0.6, color='orange')
                # Add volume moving average
                data_df['volume_ma'] = data_df['$volume'].rolling(window=10).mean()
                axes[0,1].plot(data_df['datetime'], data_df['volume_ma'], color='red', linewidth=2, label='10-day MA')
                axes[0,1].set_title('Volume Analysis')
                axes[0,1].set_ylabel('Volume')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].tick_params(axis='x', rotation=45)
                
                # 3. Daily returns
                data_df['returns'] = data_df['$close'].pct_change() * 100
                axes[1,0].plot(data_df['datetime'][1:], data_df['returns'][1:], alpha=0.8)
                axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[1,0].set_title('Daily Returns (%)')
                axes[1,0].set_ylabel('Return (%)')
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].tick_params(axis='x', rotation=45)
                
                # 4. Price volatility (20-day rolling std)
                data_df['volatility'] = data_df['returns'].rolling(window=20).std()
                axes[1,1].plot(data_df['datetime'], data_df['volatility'], color='purple', linewidth=2)
                axes[1,1].set_title('20-Day Rolling Volatility')
                axes[1,1].set_ylabel('Volatility (%)')
                axes[1,1].grid(True, alpha=0.3)
                axes[1,1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'qlib_analysis_output/stock_analysis_{stock}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Analysis created for {stock}")
                
                # Also create an interactive candlestick chart
                create_interactive_candlestick(stock, data_df)
                
            except Exception as e:
                print(f"⚠️  Failed to analyze {stock}: {e}")
                continue
                
    except Exception as e:
        print(f"⚠️  Failed to create stock analysis: {e}")

def create_interactive_candlestick(stock_symbol, data_df):
    """Create interactive candlestick chart."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=[f'{stock_symbol} - Candlestick Chart', 'Volume'],
            vertical_spacing=0.1
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data_df['datetime'],
                open=data_df['$open'],
                high=data_df['$high'],
                low=data_df['$low'],
                close=data_df['$close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # VWAP line
        fig.add_trace(
            go.Scatter(
                x=data_df['datetime'],
                y=data_df['$vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Volume
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(data_df['$close'], data_df['$open'])]
        
        fig.add_trace(
            go.Bar(
                x=data_df['datetime'],
                y=data_df['$volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Interactive Analysis - {stock_symbol}',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        fig.write_html(f'qlib_analysis_output/interactive_{stock_symbol}.html')
        
        print(f"✅ Interactive chart created for {stock_symbol}")
        
    except Exception as e:
        print(f"⚠️  Failed to create interactive chart for {stock_symbol}: {e}")

def create_summary_statistics():
    """Create summary statistics visualization."""
    try:
        from qlib.data import D
        
        print("📊 Creating summary statistics...")
        
        # Get sample of stocks for statistics
        instruments = D.instruments('csi300')
        available_stocks = D.list_instruments(
            instruments=instruments, 
            start_time='2024-01-01', 
            end_time='2025-09-06', 
            as_list=True
        )
        
        if not available_stocks:
            print("⚠️  No stocks available for statistics")
            return
            
        # Sample 10 stocks for analysis
        sample_stocks = available_stocks[:10]
        
        # Collect statistics
        stats_data = []
        
        for stock in sample_stocks:
            try:
                data = D.features(
                    instruments=[stock], 
                    fields=['$close', '$volume'],
                    start_time='2024-01-01', 
                    end_time='2025-09-06'
                )
                
                if not data.empty:
                    data_df = data.reset_index()
                    returns = data_df['$close'].pct_change().dropna() * 100
                    
                    stats_data.append({
                        'Stock': stock,
                        'Avg_Price': data_df['$close'].mean(),
                        'Price_Std': data_df['$close'].std(),
                        'Avg_Return': returns.mean(),
                        'Return_Std': returns.std(),
                        'Avg_Volume': data_df['$volume'].mean(),
                        'Max_Price': data_df['$close'].max(),
                        'Min_Price': data_df['$close'].min()
                    })
                    
            except Exception as e:
                print(f"⚠️  Failed to get stats for {stock}: {e}")
                continue
        
        if not stats_data:
            print("⚠️  No statistics data collected")
            return
            
        stats_df = pd.DataFrame(stats_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Market Statistics Summary (Sample Stocks)', fontsize=16, fontweight='bold')
        
        # 1. Average returns
        axes[0,0].bar(range(len(stats_df)), stats_df['Avg_Return'], alpha=0.7, color='blue')
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0,0].set_title('Average Daily Returns (%)')
        axes[0,0].set_ylabel('Return (%)')
        axes[0,0].set_xticks(range(len(stats_df)))
        axes[0,0].set_xticklabels(stats_df['Stock'], rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Volatility (return std)
        axes[0,1].bar(range(len(stats_df)), stats_df['Return_Std'], alpha=0.7, color='orange')
        axes[0,1].set_title('Return Volatility (Std Dev)')
        axes[0,1].set_ylabel('Volatility (%)')
        axes[0,1].set_xticks(range(len(stats_df)))
        axes[0,1].set_xticklabels(stats_df['Stock'], rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Average volume
        axes[1,0].bar(range(len(stats_df)), stats_df['Avg_Volume'], alpha=0.7, color='green')
        axes[1,0].set_title('Average Daily Volume')
        axes[1,0].set_ylabel('Volume')
        axes[1,0].set_xticks(range(len(stats_df)))
        axes[1,0].set_xticklabels(stats_df['Stock'], rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Price ranges
        price_ranges = stats_df['Max_Price'] - stats_df['Min_Price']
        axes[1,1].bar(range(len(stats_df)), price_ranges, alpha=0.7, color='red')
        axes[1,1].set_title('Price Ranges (Max - Min)')
        axes[1,1].set_ylabel('Price Range')
        axes[1,1].set_xticks(range(len(stats_df)))
        axes[1,1].set_xticklabels(stats_df['Stock'], rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save statistics table
        stats_df.round(4).to_csv('qlib_analysis_output/summary_statistics.csv', index=False)
        
        print("✅ Summary statistics visualization created")
        print("✅ Statistics table saved as CSV")
        
    except Exception as e:
        print(f"⚠️  Failed to create summary statistics: {e}")

def main():
    """Main execution function."""
    print("🚀 Creating Enhanced Qlib Data Visualizations")
    print("="*60)
    
    if not initialize_qlib():
        print("❌ Cannot proceed without Qlib initialization")
        return
    
    # Create output directory
    os.makedirs('qlib_analysis_output', exist_ok=True)
    
    # Create all visualizations
    create_enhanced_calendar_visualization()
    market_data = create_market_overview()
    create_stock_analysis()
    create_summary_statistics()
    
    print("\n" + "="*60)
    print("🎉 ENHANCED VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("📁 All files saved to 'qlib_analysis_output' directory:")
    print("   📊 enhanced_trading_calendar.png")
    print("   🏢 market_overview.png")  
    print("   📈 stock_analysis_[SYMBOL].png (individual stocks)")
    print("   🌐 interactive_[SYMBOL].html (interactive charts)")
    print("   📋 summary_statistics.png")
    print("   📄 summary_statistics.csv")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Workspace Qlib Data Analysis Script
===================================

This script runs comprehensive Qlib data analysis using the workspace data location.
It imports configuration from qlib_config.py for easy path management.

Author: AI Assistant  
Date: 2025-09-06
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our configuration module
from qlib_config import initialize_qlib, ensure_output_dir, get_sample_stocks, print_config

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_trading_calendar():
    """Analyze and display trading calendar information."""
    try:
        from qlib.data import D
        
        print("\n" + "="*60)
        print("📅 TRADING CALENDAR ANALYSIS")
        print("="*60)
        
        # Get trading calendar for a wide date range
        calendar = D.calendar(start_time='2005-01-01', end_time='2025-12-31', freq='day')
        calendar_list = pd.Series(calendar).to_list()
        
        print(f"📊 Total trading days available: {len(calendar_list)}")
        print(f"📅 Date range: {min(calendar_list)} to {max(calendar_list)}")
        print(f"🎯 Latest 10 trading days:")
        
        latest_days = calendar_list[-10:]
        for day in latest_days:
            print(f"   {day.strftime('%Y-%m-%d (%A)')}")
            
        # Calculate some statistics
        start_year = min(calendar_list).year
        end_year = max(calendar_list).year
        years_span = end_year - start_year + 1
        avg_days_per_year = len(calendar_list) / years_span
        
        print(f"\n📈 Statistics:")
        print(f"   Years covered: {years_span} years ({start_year}-{end_year})")
        print(f"   Average trading days per year: {avg_days_per_year:.1f}")
        
        return calendar_list
        
    except Exception as e:
        print(f"❌ Failed to analyze trading calendar: {e}")
        return None

def analyze_instruments():
    """Analyze available instruments (stocks) in the dataset."""
    try:
        from qlib.data import D
        
        print("\n" + "="*60)
        print("🏢 INSTRUMENTS (STOCKS) ANALYSIS")
        print("="*60)
        
        # Analyze different market segments
        market_segments = {
            'csi300': 'CSI 300 (Large Cap)',
            'csi500': 'CSI 500 (Mid Cap)', 
            'csi800': 'CSI 800 (Large + Mid Cap)',
            'csi1000': 'CSI 1000 (Small Cap)',
            'all': 'All Available Stocks'
        }
        
        instrument_stats = {}
        
        for segment, description in market_segments.items():
            try:
                instruments = D.instruments(segment)
                instrument_list = D.list_instruments(
                    instruments=instruments, 
                    start_time='2024-01-01', 
                    end_time='2025-09-06', 
                    as_list=True
                )
                
                instrument_stats[segment] = {
                    'count': len(instrument_list),
                    'description': description,
                    'sample': instrument_list[:5] if instrument_list else []
                }
                
                print(f"\n📊 {description}:")
                print(f"   Total instruments: {len(instrument_list)}")
                print(f"   Sample symbols: {', '.join(instrument_list[:5])}")
                
            except Exception as e:
                print(f"   ⚠️  Could not analyze {segment}: {e}")
                continue
        
        return instrument_stats
        
    except Exception as e:
        print(f"❌ Failed to analyze instruments: {e}")
        return None

def analyze_sample_stocks():
    """Analyze sample stocks with detailed data."""
    try:
        from qlib.data import D
        
        print("\n" + "="*60)
        print("📊 SAMPLE STOCK ANALYSIS")
        print("="*60)
        
        # Get sample stocks
        sample_stocks = get_sample_stocks('csi300', 3)
        
        if not sample_stocks:
            print("⚠️  No sample stocks available")
            return None
            
        print(f"📈 Analyzing {len(sample_stocks)} sample stocks:")
        for stock in sample_stocks:
            print(f"   {stock}")
        
        # Basic OHLCV fields
        fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
        
        stock_data = {}
        
        for stock in sample_stocks:
            try:
                data = D.features(
                    instruments=[stock], 
                    fields=fields,
                    start_time='2024-06-01', 
                    end_time='2025-09-06'
                )
                
                if not data.empty:
                    stock_data[stock] = data
                    data_df = data.reset_index()
                    
                    # Calculate basic statistics
                    avg_price = data_df['$close'].mean()
                    price_std = data_df['$close'].std()
                    returns = data_df['$close'].pct_change().dropna() * 100
                    avg_return = returns.mean()
                    return_std = returns.std()
                    avg_volume = data_df['$volume'].mean()
                    
                    print(f"\n   📊 {stock} Statistics:")
                    print(f"      Average Price: {avg_price:.2f}")
                    print(f"      Price Std Dev: {price_std:.2f}")
                    print(f"      Avg Daily Return: {avg_return:.2f}%")
                    print(f"      Return Volatility: {return_std:.2f}%")
                    print(f"      Average Volume: {avg_volume:.0f}")
                    print(f"      Data Points: {len(data_df)} days")
                    
            except Exception as e:
                print(f"   ⚠️  Failed to analyze {stock}: {e}")
                continue
        
        return stock_data
        
    except Exception as e:
        print(f"❌ Failed to analyze sample stocks: {e}")
        return None

def create_summary_visualization(calendar, instruments, stock_data):
    """Create a summary visualization combining all analyses."""
    try:
        output_dir = ensure_output_dir()
        
        print(f"\n📊 Creating summary visualization...")
        print(f"📁 Output directory: {output_dir}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Qlib Workspace Data Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Trading calendar (yearly data)
        if calendar:
            calendar_df = pd.DataFrame({'date': calendar})
            calendar_df['year'] = calendar_df['date'].dt.year
            yearly_counts = calendar_df.groupby('year').size()
            recent_years = yearly_counts[yearly_counts.index >= 2020]
            
            axes[0,0].bar(recent_years.index, recent_years.values, alpha=0.8, color='steelblue')
            axes[0,0].set_title('Trading Days Per Year (Recent)', fontweight='bold')
            axes[0,0].set_ylabel('Trading Days')
            axes[0,0].grid(True, alpha=0.3)
        else:
            axes[0,0].text(0.5, 0.5, 'Calendar data not available', ha='center', va='center')
            axes[0,0].set_title('Trading Calendar')
        
        # 2. Market segments
        if instruments:
            segments = []
            counts = []
            for segment, data in instruments.items():
                if segment != 'all':
                    segments.append(segment.upper())
                    counts.append(data['count'])
            
            if segments and counts:
                bars = axes[0,1].bar(segments, counts, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                axes[0,1].set_title('Market Segments Coverage', fontweight='bold')
                axes[0,1].set_ylabel('Number of Stocks')
                axes[0,1].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                                  f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Sample stock prices (if available)
        if stock_data:
            for i, (stock, data) in enumerate(stock_data.items()):
                if i >= 3:  # Limit to 3 stocks
                    break
                data_df = data.reset_index()
                axes[1,0].plot(data_df['datetime'], data_df['$close'], 
                              label=stock, linewidth=2, alpha=0.8)
            
            axes[1,0].set_title('Sample Stock Prices', fontweight='bold')
            axes[1,0].set_ylabel('Price')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Volume comparison (if available)
        if stock_data:
            stock_names = []
            avg_volumes = []
            
            for stock, data in stock_data.items():
                data_df = data.reset_index()
                stock_names.append(stock)
                avg_volumes.append(data_df['$volume'].mean())
            
            if stock_names and avg_volumes:
                axes[1,1].bar(range(len(stock_names)), avg_volumes, alpha=0.8, color='orange')
                axes[1,1].set_title('Average Daily Volume', fontweight='bold')
                axes[1,1].set_ylabel('Volume')
                axes[1,1].set_xticks(range(len(stock_names)))
                axes[1,1].set_xticklabels(stock_names, rotation=45)
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/workspace_data_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Summary visualization saved")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to create summary visualization: {e}")
        return False

def run_comprehensive_analysis():
    """Run the complete analysis workflow."""
    print("🚀 Starting Workspace Qlib Data Analysis")
    print("="*60)
    
    # Print current configuration
    print_config()
    
    # Initialize Qlib
    if not initialize_qlib():
        print("❌ Cannot proceed without Qlib initialization")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = ensure_output_dir()
    print(f"📁 Output directory: {output_dir}")
    
    # Run analyses
    print("\n🔍 Running comprehensive data analysis...")
    
    calendar = analyze_trading_calendar()
    instruments = analyze_instruments()
    stock_data = analyze_sample_stocks()
    
    # Create summary visualization
    create_summary_visualization(calendar, instruments, stock_data)
    
    print("\n" + "="*60)
    print("🎉 WORKSPACE DATA ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📁 Results saved to: {output_dir}")
    print("📊 Summary visualization: workspace_data_summary.png")
    print("✅ Qlib is ready for quantitative analysis!")

if __name__ == "__main__":
    run_comprehensive_analysis()
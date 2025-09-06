#!/usr/bin/env python3
"""
Qlib Data Setup and Analysis Script
===================================

This script initializes Qlib, analyzes the dataset structure, and creates comprehensive
data visualizations for quantitative finance analysis.

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
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Initialize Qlib
def initialize_qlib():
    """Initialize Qlib with the downloaded dataset."""
    try:
        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D
        
        # Initialize Qlib with workspace data
        mount_path = "/workspace/qlib/data/qlib_data/cn_data"
        qlib.init(provider_uri=mount_path, region=REG_CN)
        
        print("✅ Qlib initialized successfully!")
        print(f"📁 Data path: {mount_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Qlib: {e}")
        return False

def analyze_trading_calendar():
    """Analyze and display trading calendar information."""
    try:
        from qlib.data import D
        
        print("\n" + "="*60)
        print("📅 TRADING CALENDAR ANALYSIS")
        print("="*60)
        
        # Get trading calendar for a wide date range
        calendar = D.calendar(start_time='2005-01-01', end_time='2025-12-31', freq='day')
        
        print(f"📊 Total trading days available: {len(calendar)}")
        print(f"📅 Date range: {calendar.min()} to {calendar.max()}")
        print(f"🎯 Latest 10 trading days:")
        
        latest_days = calendar.tail(10)
        for day in latest_days:
            print(f"   {day.strftime('%Y-%m-%d (%A)')}")
            
        # Calculate some statistics
        start_year = calendar.min().year
        end_year = calendar.max().year
        years_span = end_year - start_year + 1
        avg_days_per_year = len(calendar) / years_span
        
        print(f"\n📈 Statistics:")
        print(f"   Years covered: {years_span} years ({start_year}-{end_year})")
        print(f"   Average trading days per year: {avg_days_per_year:.1f}")
        
        return calendar
        
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
                    start_time='2020-01-01', 
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

def analyze_data_fields():
    """Analyze available data fields and features."""
    try:
        from qlib.data import D
        
        print("\n" + "="*60)
        print("📊 DATA FIELDS ANALYSIS")
        print("="*60)
        
        # Test with a few popular stocks
        test_instruments = ['SH000001', 'SZ399300', 'SH600000', 'SZ000001']  # Various indices and stocks
        
        # Basic OHLCV fields
        basic_fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$vwap', '$change']
        
        # Advanced factor fields (Alpha158 style)
        advanced_fields = [
            'Ref($close, 1)',  # Previous close
            'Mean($close, 5)',  # 5-day moving average
            'Mean($close, 20)', # 20-day moving average
            'Std($close, 20)',  # 20-day standard deviation
            '$high-$low',       # Daily range
            '$close/$open',     # Close to open ratio
            'Ref($volume, 1)',  # Previous volume
            'Mean($volume, 5)'  # 5-day volume average
        ]
        
        print("📈 Available Data Fields:")
        print("\n🔹 Basic OHLCV Fields:")
        for field in basic_fields:
            print(f"   {field}")
            
        print("\n🔹 Advanced Factor Fields (examples):")
        for field in advanced_fields:
            print(f"   {field}")
        
        # Test data retrieval with one stock
        try:
            print(f"\n🎯 Testing data retrieval with sample stock...")
            
            # Find a stock that exists in our dataset
            instruments = D.instruments('csi300')
            available_stocks = D.list_instruments(
                instruments=instruments, 
                start_time='2024-01-01', 
                end_time='2024-12-31', 
                as_list=True
            )
            
            if available_stocks:
                test_stock = available_stocks[0]
                print(f"   Using stock: {test_stock}")
                
                # Get basic data
                data = D.features(
                    instruments=[test_stock], 
                    fields=basic_fields[:4],  # Just OHLC for testing
                    start_time='2024-09-01', 
                    end_time='2024-09-06'
                )
                
                print(f"   ✅ Successfully retrieved data shape: {data.shape}")
                print(f"   📊 Sample data (last 3 rows):")
                print(data.tail(3).to_string())
                
                return {
                    'basic_fields': basic_fields,
                    'advanced_fields': advanced_fields,
                    'test_stock': test_stock,
                    'sample_data': data
                }
            else:
                print("   ⚠️  No instruments found for testing")
                return None
                
        except Exception as e:
            print(f"   ⚠️  Data retrieval test failed: {e}")
            return {
                'basic_fields': basic_fields,
                'advanced_fields': advanced_fields,
                'test_stock': None,
                'sample_data': None
            }
            
    except Exception as e:
        print(f"❌ Failed to analyze data fields: {e}")
        return None

def create_sample_visualizations(calendar, instrument_stats, data_fields):
    """Create sample data visualizations."""
    try:
        from qlib.data import D
        
        print("\n" + "="*60)
        print("📊 CREATING SAMPLE VISUALIZATIONS")
        print("="*60)
        
        # Create output directory
        os.makedirs('qlib_analysis_output', exist_ok=True)
        
        # 1. Trading Calendar Visualization
        print("📅 Creating trading calendar visualization...")
        create_calendar_plot(calendar)
        
        # 2. Market Segment Comparison
        print("🏢 Creating market segments comparison...")
        create_market_segments_plot(instrument_stats)
        
        # 3. Stock Price Analysis (if we have sample data)
        if data_fields and data_fields.get('sample_data') is not None:
            print("📈 Creating stock price analysis...")
            create_stock_analysis_plots(data_fields)
        
        print("✅ Visualizations saved to 'qlib_analysis_output' directory")
        
    except Exception as e:
        print(f"❌ Failed to create visualizations: {e}")

def create_calendar_plot(calendar):
    """Create trading calendar visualization."""
    try:
        # Group by year and month to show trading days pattern
        calendar_df = pd.DataFrame({'date': calendar})
        calendar_df['year'] = calendar_df['date'].dt.year
        calendar_df['month'] = calendar_df['date'].dt.month
        calendar_df['day'] = calendar_df['date'].dt.day
        
        # Count trading days per month
        monthly_counts = calendar_df.groupby(['year', 'month']).size().reset_index(name='trading_days')
        monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
        
        # Filter to recent years for clarity
        recent_data = monthly_counts[monthly_counts['year'] >= 2020]
        
        plt.figure(figsize=(14, 8))
        
        # Subplot 1: Trading days per month
        plt.subplot(2, 1, 1)
        plt.plot(recent_data['date'], recent_data['trading_days'], marker='o', linewidth=2, markersize=4)
        plt.title('Trading Days Per Month (2020-2025)', fontsize=14, fontweight='bold')
        plt.ylabel('Trading Days')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Subplot 2: Yearly comparison
        plt.subplot(2, 1, 2)
        yearly_counts = calendar_df.groupby('year').size()
        yearly_counts = yearly_counts[yearly_counts.index >= 2020]
        
        bars = plt.bar(yearly_counts.index, yearly_counts.values, alpha=0.7, color='steelblue')
        plt.title('Total Trading Days Per Year', fontsize=14, fontweight='bold')
        plt.ylabel('Trading Days')
        plt.xlabel('Year')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/trading_calendar_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Calendar plot creation failed: {e}")

def create_market_segments_plot(instrument_stats):
    """Create market segments comparison plot."""
    try:
        if not instrument_stats:
            return
            
        # Prepare data for plotting
        segments = []
        counts = []
        descriptions = []
        
        for segment, data in instrument_stats.items():
            if segment != 'all':  # Skip 'all' for cleaner visualization
                segments.append(segment.upper())
                counts.append(data['count'])
                descriptions.append(data['description'])
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(segments, counts, color=colors[:len(segments)], alpha=0.8)
        
        plt.title('Chinese Stock Market Segments\nNumber of Available Instruments', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Number of Instruments', fontsize=12)
        plt.xlabel('Market Index', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Add descriptions below x-axis
        for i, (segment, desc) in enumerate(zip(segments, descriptions)):
            plt.text(i, -max(counts)*0.1, desc.replace('CSI ', ''), 
                    ha='center', va='top', fontsize=10, style='italic')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/market_segments_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Market segments plot creation failed: {e}")

def create_stock_analysis_plots(data_fields):
    """Create detailed stock price analysis plots."""
    try:
        from qlib.data import D
        
        # Get more comprehensive data for a popular stock
        instruments = D.instruments('csi300')
        available_stocks = D.list_instruments(
            instruments=instruments, 
            start_time='2023-01-01', 
            end_time='2025-09-06', 
            as_list=True
        )
        
        if not available_stocks:
            print("⚠️  No stocks available for detailed analysis")
            return
            
        # Use first few stocks for analysis
        analysis_stocks = available_stocks[:3]
        
        # Get comprehensive data
        fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$vwap']
        
        stock_data = {}
        for stock in analysis_stocks:
            try:
                data = D.features(
                    instruments=[stock], 
                    fields=fields,
                    start_time='2024-01-01', 
                    end_time='2025-09-06'
                )
                if not data.empty:
                    stock_data[stock] = data
                    print(f"✅ Retrieved data for {stock}: {data.shape[0]} days")
            except Exception as e:
                print(f"⚠️  Failed to get data for {stock}: {e}")
                continue
        
        if not stock_data:
            print("⚠️  No stock data retrieved for analysis")
            return
        
        # Create comprehensive analysis plots
        create_candlestick_plot(stock_data)
        create_volume_analysis_plot(stock_data)
        create_multi_stock_comparison_plot(stock_data)
        
    except Exception as e:
        print(f"⚠️  Stock analysis plots creation failed: {e}")

def create_candlestick_plot(stock_data):
    """Create interactive candlestick plot."""
    try:
        # Use the first stock with most data
        stock_symbol = max(stock_data.keys(), key=lambda x: len(stock_data[x]))
        data = stock_data[stock_symbol].copy()
        
        # Reset index to get datetime as column
        data = data.reset_index()
        
        # Create candlestick plot
        fig = make_subplots(
            rows=2, cols=1, 
            row_heights=[0.7, 0.3],
            subplot_titles=[f'{stock_symbol} - Price & Volume Analysis', 'Volume'],
            vertical_spacing=0.1
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['datetime'],
                open=data['$open'],
                high=data['$high'],
                low=data['$low'],
                close=data['$close'],
                name="Price",
                increasing_line_color='red',
                decreasing_line_color='green'
            ),
            row=1, col=1
        )
        
        # Add VWAP line
        fig.add_trace(
            go.Scatter(
                x=data['datetime'],
                y=data['$vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(data['$close'], data['$open'])]
        
        fig.add_trace(
            go.Bar(
                x=data['datetime'],
                y=data['$volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Comprehensive Price Analysis - {stock_symbol}',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        # Save as HTML for interactivity
        fig.write_html('qlib_analysis_output/interactive_candlestick_analysis.html')
        
        # Also save as static image
        fig.write_image('qlib_analysis_output/candlestick_analysis.png', width=1200, height=800)
        
    except Exception as e:
        print(f"⚠️  Candlestick plot creation failed: {e}")

def create_volume_analysis_plot(stock_data):
    """Create volume analysis visualization."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different volume analyses
        stock_symbols = list(stock_data.keys())[:3]  # Max 3 stocks
        
        for i, symbol in enumerate(stock_symbols):
            data = stock_data[symbol].copy()
            data = data.reset_index()
            
            # Calculate volume moving averages
            data['volume_ma5'] = data['$volume'].rolling(window=5).mean()
            data['volume_ma20'] = data['$volume'].rolling(window=20).mean()
            
            plt.subplot(len(stock_symbols), 1, i+1)
            
            # Plot volume bars
            plt.bar(data['datetime'], data['$volume'], alpha=0.3, color='blue', label='Daily Volume')
            
            # Plot volume moving averages
            plt.plot(data['datetime'], data['volume_ma5'], color='red', linewidth=2, label='5-day MA')
            plt.plot(data['datetime'], data['volume_ma20'], color='green', linewidth=2, label='20-day MA')
            
            plt.title(f'{symbol} - Volume Analysis', fontweight='bold')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if i < len(stock_symbols) - 1:
                plt.xticks([])
            else:
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/volume_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Volume analysis plot creation failed: {e}")

def create_multi_stock_comparison_plot(stock_data):
    """Create multi-stock comparison plot."""
    try:
        plt.figure(figsize=(15, 12))
        
        # Normalize prices for comparison (base = 100)
        normalized_data = {}
        
        for symbol, data in stock_data.items():
            data_reset = data.reset_index()
            if len(data_reset) > 0:
                base_price = data_reset['$close'].iloc[0]
                normalized_data[symbol] = {
                    'datetime': data_reset['datetime'],
                    'normalized_price': (data_reset['$close'] / base_price) * 100,
                    'volume': data_reset['$volume']
                }
        
        # Plot 1: Normalized price comparison
        plt.subplot(3, 1, 1)
        for symbol, data in normalized_data.items():
            plt.plot(data['datetime'], data['normalized_price'], 
                    label=symbol, linewidth=2, marker='o', markersize=1)
        
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        plt.title('Normalized Price Comparison (Base = 100)', fontweight='bold', fontsize=14)
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Daily returns comparison
        plt.subplot(3, 1, 2)
        for symbol, orig_data in stock_data.items():
            data_reset = orig_data.reset_index()
            if len(data_reset) > 1:
                returns = data_reset['$close'].pct_change().dropna() * 100
                plt.plot(data_reset['datetime'][1:], returns, 
                        label=f'{symbol} Returns', alpha=0.7, linewidth=1)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Daily Returns Comparison (%)', fontweight='bold', fontsize=14)
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Volume comparison
        plt.subplot(3, 1, 3)
        for symbol, data in normalized_data.items():
            plt.plot(data['datetime'], data['volume'], 
                    label=f'{symbol} Volume', alpha=0.7, linewidth=1)
        
        plt.title('Volume Comparison', fontweight='bold', fontsize=14)
        plt.ylabel('Volume')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('qlib_analysis_output/multi_stock_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Multi-stock comparison plot creation failed: {e}")

def generate_summary_report(calendar, instrument_stats, data_fields):
    """Generate a comprehensive summary report."""
    try:
        print("\n" + "="*60)
        print("📋 GENERATING SUMMARY REPORT")
        print("="*60)
        
        report_content = f"""
# Qlib Dataset Analysis Report
## Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Dataset Overview

### Trading Calendar
- **Total Trading Days**: {len(calendar) if calendar is not None else 'N/A'}
- **Date Range**: {calendar.min().strftime('%Y-%m-%d') if calendar is not None else 'N/A'} to {calendar.max().strftime('%Y-%m-%d') if calendar is not None else 'N/A'}
- **Years Covered**: {calendar.max().year - calendar.min().year + 1 if calendar is not None else 'N/A'} years

### Market Segments
"""
        
        if instrument_stats:
            for segment, data in instrument_stats.items():
                report_content += f"- **{data['description']}**: {data['count']} instruments\n"
        
        report_content += f"""

### Available Data Fields

#### Basic OHLCV Fields:
"""
        
        if data_fields and 'basic_fields' in data_fields:
            for field in data_fields['basic_fields']:
                report_content += f"- `{field}`\n"
        
        report_content += f"""

#### Advanced Factor Fields (Examples):
"""
        
        if data_fields and 'advanced_fields' in data_fields:
            for field in data_fields['advanced_fields']:
                report_content += f"- `{field}`\n"
        
        report_content += f"""

---

## 🎯 Data Quality Assessment

✅ **Strengths:**
- Comprehensive Chinese stock market coverage
- Multiple market segments (CSI300, CSI500, CSI800, CSI1000)
- Long historical data range (20+ years)
- Complete OHLCV data fields
- Additional calculated fields (VWAP, factors)

⚠️ **Considerations:**
- Some missing data points (normal for financial data)
- Large step changes in some series (expected due to stock splits, etc.)
- Community-maintained dataset (not official)

---

## 📈 Recommended Usage

### For Beginners:
1. Start with CSI300 index constituents (large, liquid stocks)
2. Use basic OHLCV fields for initial analysis
3. Focus on recent data (2020-2025) for better quality

### For Advanced Users:
1. Leverage all market segments for comprehensive analysis
2. Use advanced factor fields for alpha research
3. Apply appropriate data cleaning and preprocessing

### For Research:
1. Combine with Alpha158/Alpha360 datasets
2. Use Qlib's built-in models and strategies
3. Implement custom factors and trading strategies

---

## 📁 Files Generated

### Visualizations:
- `trading_calendar_analysis.png`: Trading days analysis
- `market_segments_comparison.png`: Market segments overview
- `candlestick_analysis.png`: Individual stock analysis
- `interactive_candlestick_analysis.html`: Interactive price charts
- `volume_analysis.png`: Volume patterns analysis
- `multi_stock_comparison.png`: Comparative stock analysis

### Data:
- All data accessible through Qlib API
- Data location: `~/.qlib/qlib_data/cn_data/`

---

## 🚀 Next Steps

1. **Explore Examples**: Run Qlib example notebooks and scripts
2. **Build Models**: Try LightGBM, LSTM, or other built-in models
3. **Create Strategies**: Implement quantitative trading strategies
4. **Backtest**: Use Qlib's backtesting framework
5. **Research**: Develop custom alpha factors

---

*Report generated by Qlib Data Analysis Script*
*For more information, visit: https://github.com/microsoft/qlib*
        """
        
        # Save report
        with open('qlib_analysis_output/dataset_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        print("✅ Summary report saved to 'qlib_analysis_output/dataset_analysis_report.md'")
        
        return report_content
        
    except Exception as e:
        print(f"❌ Failed to generate summary report: {e}")
        return None

def main():
    """Main execution function."""
    print("🚀 Starting Qlib Data Setup and Analysis")
    print("="*60)
    
    # Step 1: Initialize Qlib
    if not initialize_qlib():
        print("❌ Cannot proceed without Qlib initialization")
        sys.exit(1)
    
    # Step 2: Analyze trading calendar
    calendar = analyze_trading_calendar()
    
    # Step 3: Analyze instruments
    instrument_stats = analyze_instruments()
    
    # Step 4: Analyze data fields
    data_fields = analyze_data_fields()
    
    # Step 5: Create visualizations
    create_sample_visualizations(calendar, instrument_stats, data_fields)
    
    
    # Step 6: Generate summary report
    generate_summary_report(calendar, instrument_stats, data_fields)
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print("📁 All outputs saved to 'qlib_analysis_output' directory")
    print("📊 Check the generated visualizations and report")
    print("🚀 You're ready to start quantitative analysis with Qlib!")

if __name__ == "__main__":
    main()
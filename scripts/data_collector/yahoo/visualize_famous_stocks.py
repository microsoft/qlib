#!/usr/bin/env python3
"""
Famous Stocks Visualization Script
Creates K-line (candlestick) charts and other visualizations for major stocks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import seaborn as sns
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')

class StockVisualizer:
    """Professional stock data visualizer with K-line charts"""
    
    def __init__(self, data_dir: str = '/workspace/qlib/data/comprehensive_yahoo_data',
                 output_dir: str = '/workspace/qlib/visualization_output'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Famous stocks to analyze
        self.famous_stocks = {
            'NVDA': 'NVIDIA Corporation',
            'TSLA': 'Tesla Inc',
            'AAPL': 'Apple Inc',
            'GOOGL': 'Alphabet Inc',
            'META': 'Meta Platforms',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc',
            'NFLX': 'Netflix Inc',
            'AMD': 'Advanced Micro Devices',
            'CRM': 'Salesforce Inc'
        }
        
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
    def load_stock_data(self, symbol: str, days_back: int = 252) -> Optional[pd.DataFrame]:
        """Load stock data for given symbol"""
        try:
            file_path = self.data_dir / f"{symbol}.csv"
            if not file_path.exists():
                print(f"❌ Data not found for {symbol}")
                return None
            
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Get recent data
            if days_back:
                df = df.tail(days_back)
            
            # Ensure we have required columns
            required_cols = ['$open', '$high', '$low', '$close', '$volume']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ Missing required columns for {symbol}")
                return None
            
            print(f"✅ Loaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")
            return None
    
    def create_candlestick_chart(self, symbol: str, df: pd.DataFrame, 
                               interactive: bool = True) -> None:
        """Create candlestick (K-line) chart"""
        
        if interactive:
            # Interactive Plotly chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{self.famous_stocks.get(symbol, symbol)} - Stock Price', 'Volume'),
                row_width=[0.2, 0.7]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['date'],
                    open=df['$open'],
                    high=df['$high'],
                    low=df['$low'],
                    close=df['$close'],
                    name=symbol,
                    increasing_line_color='#00FF88',
                    decreasing_line_color='#FF3366'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            df['ma_20'] = df['$close'].rolling(window=20).mean()
            df['ma_50'] = df['$close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['ma_20'], 
                          line=dict(color='orange', width=1),
                          name='MA20'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['ma_50'], 
                          line=dict(color='cyan', width=1),
                          name='MA50'),
                row=1, col=1
            )
            
            # Volume chart
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(df['$close'], df['$open'])]
            
            fig.add_trace(
                go.Bar(x=df['date'], y=df['$volume'], 
                      marker_color=colors, name='Volume', opacity=0.7),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - {self.famous_stocks.get(symbol, symbol)} Candlestick Chart',
                template='plotly_dark',
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            # Save interactive chart
            output_file = self.output_dir / f'{symbol}_interactive_candlestick.html'
            fig.write_html(output_file)
            print(f"✅ Saved interactive chart: {output_file}")
        
        else:
            # Static matplotlib chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Candlestick chart
            for idx, row in df.iterrows():
                color = '#00FF88' if row['$close'] >= row['$open'] else '#FF3366'
                
                # Candlestick body
                body_height = abs(row['$close'] - row['$open'])
                body_bottom = min(row['$open'], row['$close'])
                
                ax1.add_patch(Rectangle((idx-0.3, body_bottom), 0.6, body_height, 
                                      facecolor=color, edgecolor=color))
                
                # Wicks
                ax1.plot([idx, idx], [row['$low'], row['$high']], 
                        color=color, linewidth=1)
            
            # Moving averages
            df['ma_20'] = df['$close'].rolling(window=20).mean()
            df['ma_50'] = df['$close'].rolling(window=50).mean()
            
            ax1.plot(df.index, df['ma_20'], color='orange', linewidth=1, label='MA20')
            ax1.plot(df.index, df['ma_50'], color='cyan', linewidth=1, label='MA50')
            
            ax1.set_title(f'{symbol} - {self.famous_stocks.get(symbol, symbol)} Candlestick Chart', 
                         fontsize=16, color='white')
            ax1.set_ylabel('Price ($)', color='white')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            colors = ['#00FF88' if close >= open else '#FF3366' 
                     for close, open in zip(df['$close'], df['$open'])]
            ax2.bar(df.index, df['$volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('Volume', color='white')
            ax2.set_xlabel('Days', color='white')
            
            plt.tight_layout()
            
            # Save static chart
            output_file = self.output_dir / f'{symbol}_candlestick.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.close()
            print(f"✅ Saved static chart: {output_file}")
    
    def create_comparison_chart(self, symbols: List[str], days_back: int = 252) -> None:
        """Create comparison chart for multiple stocks"""
        
        stock_data = {}
        for symbol in symbols:
            df = self.load_stock_data(symbol, days_back)
            if df is not None:
                # Normalize to percentage change from start
                start_price = df['$close'].iloc[0]
                df['normalized'] = (df['$close'] / start_price - 1) * 100
                stock_data[symbol] = df
        
        if not stock_data:
            print("❌ No data available for comparison")
            return
        
        # Interactive comparison
        fig = go.Figure()
        
        for symbol, df in stock_data.items():
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['normalized'],
                    mode='lines',
                    name=f'{symbol} - {self.famous_stocks.get(symbol, symbol)}',
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title='Famous Stocks Performance Comparison (Normalized)',
            template='plotly_dark',
            height=600,
            xaxis_title='Date',
            yaxis_title='Performance (%)',
            showlegend=True
        )
        
        # Save comparison chart
        output_file = self.output_dir / 'famous_stocks_comparison.html'
        fig.write_html(output_file)
        print(f"✅ Saved comparison chart: {output_file}")
        
        # Static version
        plt.figure(figsize=(15, 8))
        for symbol, df in stock_data.items():
            plt.plot(df.index, df['normalized'], linewidth=2, 
                    label=f'{symbol} - {self.famous_stocks.get(symbol, symbol)}')
        
        plt.title('Famous Stocks Performance Comparison (Normalized)', 
                 fontsize=16, color='white')
        plt.xlabel('Days', color='white')
        plt.ylabel('Performance (%)', color='white')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.output_dir / 'famous_stocks_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        print(f"✅ Saved static comparison: {output_file}")
    
    def create_volatility_analysis(self, symbols: List[str], days_back: int = 252) -> None:
        """Create volatility analysis for stocks"""
        
        volatility_data = []
        
        for symbol in symbols:
            df = self.load_stock_data(symbol, days_back)
            if df is not None:
                # Calculate returns and volatility
                df['returns'] = df['$close'].pct_change()
                volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized
                avg_return = df['returns'].mean() * 252 * 100  # Annualized
                
                volatility_data.append({
                    'Symbol': symbol,
                    'Name': self.famous_stocks.get(symbol, symbol),
                    'Volatility (%)': volatility,
                    'Average Return (%)': avg_return
                })
        
        if not volatility_data:
            print("❌ No data for volatility analysis")
            return
        
        vol_df = pd.DataFrame(volatility_data)
        
        # Risk-Return scatter plot
        fig = px.scatter(
            vol_df, 
            x='Volatility (%)', 
            y='Average Return (%)',
            text='Symbol',
            title='Risk-Return Analysis (Annualized)',
            template='plotly_dark',
            height=600
        )
        
        fig.update_traces(textposition="top center", marker_size=10)
        fig.update_layout(showlegend=False)
        
        # Save risk-return chart
        output_file = self.output_dir / 'risk_return_analysis.html'
        fig.write_html(output_file)
        print(f"✅ Saved risk-return analysis: {output_file}")
        
        # Print results
        print(f"\n📊 Risk-Return Analysis:")
        print(vol_df.round(2).to_string(index=False))
    
    def create_technical_analysis(self, symbol: str, df: pd.DataFrame) -> None:
        """Create technical analysis with indicators"""
        
        # Calculate technical indicators
        df = df.copy()
        
        # Moving averages
        df['ma_20'] = df['$close'].rolling(window=20).mean()
        df['ma_50'] = df['$close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['$close'].rolling(window=20).mean()
        bb_std = df['$close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # RSI
        delta = df['$close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['$close'].ewm(span=12).mean()
        exp2 = df['$close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                f'{symbol} - Price & Bollinger Bands',
                'Volume',
                'RSI',
                'MACD'
            ],
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price chart with Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['$open'],
                high=df['$high'], 
                low=df['$low'],
                close=df['$close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_upper'], 
                      line=dict(color='gray', width=1), name='BB Upper'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_lower'], 
                      line=dict(color='gray', width=1), name='BB Lower', 
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['ma_20'], 
                      line=dict(color='orange', width=1), name='MA20'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['ma_50'], 
                      line=dict(color='cyan', width=1), name='MA50'),
            row=1, col=1
        )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['$close'], df['$open'])]
        fig.add_trace(
            go.Bar(x=df['date'], y=df['$volume'], 
                  marker_color=colors, name='Volume', opacity=0.7),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['rsi'], 
                      line=dict(color='purple', width=2), name='RSI'),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['macd'], 
                      line=dict(color='blue', width=2), name='MACD'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['macd_signal'], 
                      line=dict(color='red', width=2), name='Signal'),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=df['date'], y=df['macd_histogram'], 
                  marker_color='gray', name='Histogram', opacity=0.6),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {self.famous_stocks.get(symbol, symbol)} Technical Analysis',
            template='plotly_dark',
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Save technical analysis
        output_file = self.output_dir / f'{symbol}_technical_analysis.html'
        fig.write_html(output_file)
        print(f"✅ Saved technical analysis: {output_file}")
    
    def generate_summary_report(self, symbols: List[str]) -> None:
        """Generate comprehensive summary report"""
        
        report_data = []
        
        for symbol in symbols:
            df = self.load_stock_data(symbol, 252)  # 1 year of data
            if df is not None:
                # Calculate key metrics
                current_price = df['$close'].iloc[-1]
                price_52w_high = df['$high'].max()
                price_52w_low = df['$low'].min()
                
                returns = df['$close'].pct_change()
                ytd_return = (df['$close'].iloc[-1] / df['$close'].iloc[0] - 1) * 100
                volatility = returns.std() * np.sqrt(252) * 100
                avg_volume = df['$volume'].mean()
                
                # Recent performance
                recent_30d = df.tail(30)
                monthly_return = (recent_30d['$close'].iloc[-1] / recent_30d['$close'].iloc[0] - 1) * 100
                
                report_data.append({
                    'Symbol': symbol,
                    'Name': self.famous_stocks.get(symbol, symbol)[:20],
                    'Current Price': f"${current_price:.2f}",
                    '52W High': f"${price_52w_high:.2f}",
                    '52W Low': f"${price_52w_low:.2f}",
                    'YTD Return (%)': f"{ytd_return:.1f}",
                    '30D Return (%)': f"{monthly_return:.1f}",
                    'Volatility (%)': f"{volatility:.1f}",
                    'Avg Volume': f"{avg_volume/1000000:.1f}M"
                })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            
            # Save to CSV
            csv_file = self.output_dir / 'famous_stocks_summary.csv'
            report_df.to_csv(csv_file, index=False)
            
            print(f"\n📊 FAMOUS STOCKS SUMMARY REPORT")
            print("=" * 80)
            print(report_df.to_string(index=False))
            print(f"\n✅ Summary saved to: {csv_file}")
    
    def create_all_visualizations(self, days_back: int = 252) -> None:
        """Create all visualizations for famous stocks"""
        
        print("🚀 CREATING COMPREHENSIVE STOCK VISUALIZATIONS")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📅 Using last {days_back} days of data")
        print()
        
        available_stocks = []
        
        # Create individual charts for each stock
        for symbol in self.famous_stocks.keys():
            print(f"📈 Processing {symbol} - {self.famous_stocks[symbol]}...")
            
            df = self.load_stock_data(symbol, days_back)
            if df is not None:
                available_stocks.append(symbol)
                
                # Create candlestick charts
                self.create_candlestick_chart(symbol, df, interactive=True)
                self.create_candlestick_chart(symbol, df, interactive=False)
                
                # Create technical analysis
                self.create_technical_analysis(symbol, df)
                
                print(f"   ✅ Completed visualizations for {symbol}")
            else:
                print(f"   ❌ Skipped {symbol} (no data)")
            
            print()
        
        # Create comparison charts
        if len(available_stocks) >= 2:
            print("📊 Creating comparison and analysis charts...")
            self.create_comparison_chart(available_stocks, days_back)
            self.create_volatility_analysis(available_stocks, days_back)
            self.generate_summary_report(available_stocks)
        
        print(f"\n🎉 VISUALIZATION COMPLETE!")
        print(f"✅ Processed {len(available_stocks)} stocks")
        print(f"📁 All charts saved to: {self.output_dir}")
        print(f"🌐 Open the .html files in your browser for interactive charts")


def main():
    """Main execution function"""
    
    # Create visualizer
    visualizer = StockVisualizer()
    
    # Generate all visualizations
    visualizer.create_all_visualizations(days_back=252)  # 1 year of data
    
    print(f"\n📝 Generated files:")
    output_files = list(visualizer.output_dir.glob('*'))
    for file in sorted(output_files):
        print(f"   📄 {file.name}")


if __name__ == "__main__":
    main()
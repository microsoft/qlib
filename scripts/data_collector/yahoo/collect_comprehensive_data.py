#!/usr/bin/env python3
"""
Comprehensive Yahoo Finance Data Collector for Qlib
Collects data for 8k+ stocks from comprehensive universe
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from tqdm import tqdm
import time
import warnings
from datetime import datetime, timedelta
import argparse
import json

# Suppress warnings
warnings.filterwarnings('ignore')

class ComprehensiveYahooCollector:
    """Comprehensive data collector for Yahoo Finance with 8k+ symbols"""
    
    def __init__(self, universe_file, output_dir, start_date='1990-01-01', end_date='2025-01-01'):
        self.universe_file = Path(universe_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date
        self.delay = 0.05  # Reduced delay for faster collection
        
        # Load symbols
        self.symbols = self.load_symbols()
        
        # Statistics
        self.stats = {
            'total_symbols': len(self.symbols),
            'successful': 0,
            'failed': 0,
            'empty_data': 0,
            'errors': [],
            'start_time': datetime.now()
        }
        
    def load_symbols(self):
        """Load symbols from universe file"""
        if not self.universe_file.exists():
            raise FileNotFoundError(f"Universe file not found: {self.universe_file}")
            
        with open(self.universe_file, 'r') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
            
        print(f"📈 Loaded {len(symbols)} symbols from {self.universe_file.name}")
        return symbols
    
    def get_stock_data(self, symbol, retries=3):
        """Download data for a single symbol with retries"""
        for attempt in range(retries):
            try:
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,  # Use adjusted prices
                    back_adjust=True   # Back adjust splits/dividends
                )
                
                if data.empty:
                    return None, "No data available"
                
                # Rename columns to Qlib format
                data = data.rename(columns={
                    'Open': '$open',
                    'High': '$high', 
                    'Low': '$low',
                    'Close': '$close',  # This is already adjusted
                    'Volume': '$volume'
                })
                
                # Calculate additional fields
                data['$factor'] = 1.0  # Factor is 1 since we're using adjusted prices
                data['$vwap'] = (data['$high'] + data['$low'] + data['$close']) / 3
                
                # Reset index to make date a column
                data = data.reset_index()
                data['date'] = data['Date']
                
                # Select required columns
                columns_to_keep = ['date', '$open', '$high', '$low', '$close', '$volume', '$vwap', '$factor']
                data = data[[col for col in columns_to_keep if col in data.columns]]
                
                # Sort by date
                data = data.sort_values('date')
                
                return data, None
                
            except Exception as e:
                error_msg = str(e)
                if attempt < retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return None, error_msg
        
        return None, "Max retries exceeded"
    
    def collect_batch(self, symbol_batch, batch_num, total_batches):
        """Collect data for a batch of symbols"""
        batch_results = {
            'successful': 0,
            'failed': 0,
            'empty_data': 0,
            'errors': []
        }
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(symbol_batch)} symbols)")
        
        for symbol in tqdm(symbol_batch, desc=f"Batch {batch_num}"):
            try:
                # Clean symbol for filename
                clean_symbol = symbol.replace('^', '_').replace('/', '_').replace('.', '_')
                output_file = self.output_dir / f"{clean_symbol}.csv"
                
                # Skip if file already exists
                if output_file.exists():
                    batch_results['successful'] += 1
                    continue
                
                # Get data
                data, error = self.get_stock_data(symbol)
                
                if error:
                    batch_results['failed'] += 1
                    batch_results['errors'].append(f"{symbol}: {error}")
                    continue
                
                if data is None or len(data) == 0:
                    batch_results['empty_data'] += 1
                    continue
                
                # Save to CSV
                data.to_csv(output_file, index=False)
                batch_results['successful'] += 1
                
                # Add delay to avoid rate limiting
                time.sleep(self.delay)
                
            except Exception as e:
                batch_results['failed'] += 1
                batch_results['errors'].append(f"{symbol}: {str(e)}")
                continue
        
        return batch_results
    
    def collect_all_data(self, batch_size=100, max_symbols=None):
        """Collect data for all symbols in batches"""
        print("🚀 COMPREHENSIVE YAHOO FINANCE DATA COLLECTION")
        print("=" * 80)
        print(f"📈 Total symbols: {len(self.symbols)}")
        print(f"📅 Date range: {self.start_date} to {self.end_date}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"📦 Batch size: {batch_size}")
        
        # Limit symbols for testing if specified
        symbols_to_process = self.symbols
        if max_symbols:
            symbols_to_process = self.symbols[:max_symbols]
            print(f"🧪 Testing mode: limiting to first {max_symbols} symbols")
        
        # Split into batches
        batches = [symbols_to_process[i:i+batch_size] for i in range(0, len(symbols_to_process), batch_size)]
        total_batches = len(batches)
        
        print(f"📦 Processing {total_batches} batches...")
        
        # Process each batch
        for i, batch in enumerate(batches, 1):
            batch_results = self.collect_batch(batch, i, total_batches)
            
            # Update global stats
            self.stats['successful'] += batch_results['successful']
            self.stats['failed'] += batch_results['failed'] 
            self.stats['empty_data'] += batch_results['empty_data']
            self.stats['errors'].extend(batch_results['errors'])
            
            # Print batch summary
            print(f"   ✅ Successful: {batch_results['successful']}")
            print(f"   ❌ Failed: {batch_results['failed']}")
            print(f"   📭 Empty: {batch_results['empty_data']}")
            
            # Save progress periodically
            if i % 10 == 0 or i == total_batches:
                self.save_progress_report()
        
        # Final summary
        self.print_final_summary()
        self.save_final_report()
        
        return self.stats
    
    def save_progress_report(self):
        """Save progress report"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'processed': self.stats['successful'] + self.stats['failed'] + self.stats['empty_data'],
            'total': self.stats['total_symbols'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'empty_data': self.stats['empty_data'],
            'success_rate': self.stats['successful'] / max(1, self.stats['successful'] + self.stats['failed']) * 100,
            'recent_errors': self.stats['errors'][-20:] if self.stats['errors'] else []
        }
        
        progress_file = self.output_dir / 'collection_progress.json'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def print_final_summary(self):
        """Print final collection summary"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        total_processed = self.stats['successful'] + self.stats['failed'] + self.stats['empty_data']
        success_rate = self.stats['successful'] / max(1, total_processed) * 100
        
        print(f"\n🎉 COLLECTION COMPLETE!")
        print(f"=" * 60)
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Total processed: {total_processed}/{self.stats['total_symbols']}")
        print(f"✅ Successful: {self.stats['successful']} ({success_rate:.1f}%)")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"📭 Empty data: {self.stats['empty_data']}")
        print(f"💾 Files created: {len(list(self.output_dir.glob('*.csv')))}")
        
        if self.stats['errors']:
            print(f"\n⚠️  Recent errors (last 10):")
            for error in self.stats['errors'][-10:]:
                print(f"   {error}")
    
    def save_final_report(self):
        """Save final collection report"""
        report = {
            'collection_summary': {
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat(),
                'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
                'total_symbols': self.stats['total_symbols'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'empty_data': self.stats['empty_data'],
                'success_rate': self.stats['successful'] / max(1, self.stats['successful'] + self.stats['failed']) * 100,
                'files_created': len(list(self.output_dir.glob('*.csv')))
            },
            'configuration': {
                'universe_file': str(self.universe_file),
                'output_directory': str(self.output_dir),
                'date_range': f"{self.start_date} to {self.end_date}",
                'delay': self.delay
            },
            'all_errors': self.stats['errors']
        }
        
        report_file = self.output_dir / 'collection_final_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save successful symbols list
        successful_files = list(self.output_dir.glob('*.csv'))
        successful_symbols = [f.stem.replace('_', '^') for f in successful_files if f.stem != 'collection_progress' and f.stem != 'collection_final_report']
        
        symbols_file = self.output_dir / 'successful_symbols.txt'
        with open(symbols_file, 'w') as f:
            for symbol in sorted(successful_symbols):
                f.write(f"{symbol}\n")
        
        print(f"📝 Final report saved to: {report_file}")
        print(f"📝 Successful symbols saved to: {symbols_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive Yahoo Finance data collection')
    parser.add_argument('--universe_file', 
                       default='/workspace/qlib/scripts/data_collector/yahoo/comprehensive_stock_universe.txt',
                       help='Universe file with symbols to collect')
    parser.add_argument('--output_dir', 
                       default='/workspace/qlib/data/comprehensive_yahoo_data',
                       help='Output directory for data files')
    parser.add_argument('--start_date', 
                       default='1990-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', 
                       default='2025-01-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=100,
                       help='Batch size for processing')
    parser.add_argument('--max_symbols', 
                       type=int, 
                       default=None,
                       help='Limit number of symbols (for testing)')
    parser.add_argument('--test_run', 
                       action='store_true',
                       help='Run with first 50 symbols for testing')
    
    args = parser.parse_args()
    
    # Override for test run
    if args.test_run:
        args.max_symbols = 50
        print("🧪 TEST RUN MODE: Processing first 50 symbols")
    
    # Create collector
    collector = ComprehensiveYahooCollector(
        universe_file=args.universe_file,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Start collection
    stats = collector.collect_all_data(
        batch_size=args.batch_size,
        max_symbols=args.max_symbols
    )
    
    return stats


if __name__ == "__main__":
    main()
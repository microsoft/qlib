#!/usr/bin/env python3
"""
Monitor Yahoo Finance data collection progress
"""

import json
import os
from pathlib import Path
import time
from datetime import datetime

def monitor_progress():
    """Monitor collection progress"""
    data_dir = Path('/workspace/qlib/data/comprehensive_yahoo_data')
    progress_file = data_dir / 'collection_progress.json'
    
    if not data_dir.exists():
        print("❌ Data directory not found. Collection may not have started.")
        return
    
    print("📊 YAHOO FINANCE DATA COLLECTION MONITOR")
    print("=" * 60)
    print(f"📁 Data directory: {data_dir}")
    print(f"📝 Progress file: {progress_file}")
    print(f"⏰ Monitor started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        while True:
            # Count CSV files
            csv_files = len(list(data_dir.glob('*.csv')))
            
            # Read progress file if exists
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    processed = progress.get('processed', 0)
                    total = progress.get('total', 8060)
                    successful = progress.get('successful', 0)
                    failed = progress.get('failed', 0)
                    empty_data = progress.get('empty_data', 0)
                    success_rate = progress.get('success_rate', 0)
                    timestamp = progress.get('timestamp', 'Unknown')
                    
                    progress_pct = (processed / total * 100) if total > 0 else 0
                    
                    print(f"\r🔄 Progress: {processed:,}/{total:,} ({progress_pct:.1f}%) | "
                          f"✅ Success: {successful:,} ({success_rate:.1f}%) | "
                          f"❌ Failed: {failed:,} | 📭 Empty: {empty_data:,} | "
                          f"💾 Files: {csv_files:,} | "
                          f"⏰ {timestamp[:19]}", end="", flush=True)
                    
                except Exception as e:
                    print(f"\r📊 Files created: {csv_files:,} | ❌ Progress file error: {e}", end="", flush=True)
            else:
                print(f"\r📊 Files created: {csv_files:,} | ⚠️  No progress file yet", end="", flush=True)
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print(f"\n\n📊 FINAL STATUS")
        print(f"💾 Total CSV files: {csv_files:,}")
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    final_progress = json.load(f)
                print(f"✅ Successful: {final_progress.get('successful', 0):,}")
                print(f"❌ Failed: {final_progress.get('failed', 0):,}")
                print(f"📭 Empty: {final_progress.get('empty_data', 0):,}")
                print(f"📈 Success rate: {final_progress.get('success_rate', 0):.1f}%")
            except:
                pass
        
        print("\n👋 Monitor stopped")

if __name__ == "__main__":
    monitor_progress()
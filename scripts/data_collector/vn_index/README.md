# VN30/VNINDEX/HNX30/UPCOM History Companies Collection

## Requirements

```bash
pip install -r requirements.txt
pip install vnstock
```

## Collector Data

```bash
# parse instruments, using in qlib/instruments.
python collector.py --index_name VN30 --qlib_dir ~/.qlib/qlib_data/vn_data --method parse_instruments

# parse new companies
python collector.py --index_name VN30 --qlib_dir ~/.qlib/qlib_data/vn_data --method save_new_companies

# index_name support: VN30, VNINDEX, HNX30, HNX, UPCOM, VN100
# help
python collector.py --help
```

## Supported Vietnamese Market Indices

- **VN30**: Top 30 stocks by market cap on HOSE (Ho Chi Minh Stock Exchange)
- **VNINDEX**: All stocks listed on HOSE
- **HNX30**: Top 30 stocks by market cap on HNX (Hanoi Stock Exchange)
- **HNX**: All stocks listed on HNX
- **UPCOM**: Unlisted Public Company Market
- **VN100**: Top 100 stocks by market cap

## Note

This collector uses the vnstock library to fetch Vietnamese stock market data. The change tracking functionality is limited as historical changes are not readily available through the vnstock API.


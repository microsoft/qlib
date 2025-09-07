#!/usr/bin/env python3
"""
Comprehensive US Stock Universe Builder for Yahoo Finance
Targets 3-5K unique symbols from multiple sources for quantitative analysis
"""

import pandas as pd
import requests
import ftplib
from io import StringIO
from datetime import datetime
import time
import random
from pathlib import Path

class StockUniverseBuilder:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        self.all_symbols = set()
        
    def add_delay(self):
        """Add random delay to avoid rate limiting"""
        time.sleep(random.uniform(1, 3))
    
    def get_russell_3000(self):
        """Get Russell 3000 from iShares IWV ETF"""
        print("📈 Fetching Russell 3000 stocks...")
        try:
            # Try multiple date formats for the iShares API
            date_formats = [
                datetime.now().strftime('%Y%m%d'),
                datetime.now().strftime('%m%d%Y'),
                '20241231'  # fallback
            ]
            
            for date_str in date_formats:
                try:
                    url = f"https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund&asOfDate={date_str}"
                    
                    response = requests.get(url, headers=self.headers, timeout=30)
                    if response.status_code == 200 and len(response.text) > 1000:
                        df = pd.read_csv(StringIO(response.text), skiprows=10)
                        if 'Ticker' in df.columns:
                            symbols = df['Ticker'].dropna().tolist()
                            symbols = [s.strip().upper() for s in symbols if s.strip()]
                            self.all_symbols.update(symbols[:3000])
                            print(f"   ✅ Added {len(symbols[:3000])} Russell 3000 stocks")
                            return symbols[:3000]
                except:
                    continue
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("   ⚠️  Russell 3000 API unavailable, using fallback list")
        return self.get_russell_3000_fallback()
    
    def get_russell_3000_fallback(self):
        """Fallback Russell 3000 list (major components)"""
        # Top Russell 3000 stocks by market cap
        russell_major = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-A', 'BRK-B',
            'UNH', 'JNJ', 'XOM', 'V', 'PG', 'JPM', 'MA', 'CVX', 'HD', 'PFE', 'ABBV', 'BAC',
            'KO', 'AVGO', 'LLY', 'MRK', 'WMT', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'MCD',
            'ACN', 'CRM', 'VZ', 'ADBE', 'NFLX', 'DHR', 'WFC', 'TXN', 'BMY', 'NEE', 'PM',
            'RTX', 'UPS', 'T', 'LOW', 'AMGN', 'ORCL', 'HON', 'QCOM', 'UNP', 'MDT', 'LMT',
            'INTU', 'IBM', 'CAT', 'AMD', 'GS', 'SPGI', 'INTC', 'C', 'BA', 'ISRG', 'NOW',
            # ... Continue with next ~200 major stocks by market cap
            'COP', 'MS', 'PLD', 'AXP', 'BLK', 'AMT', 'SYK', 'BKNG', 'DE', 'TJX', 'ADP',
            'MDLZ', 'GE', 'CB', 'SO', 'MMM', 'LRCX', 'EOG', 'CI', 'AMAT', 'ZTS', 'CSX',
            'MU', 'EQIX', 'DUK', 'PNC', 'ICE', 'WM', 'AON', 'CL', 'SHW', 'NSC', 'ITW'
        ]
        
        # Add smaller cap stocks from various sectors
        russell_smaller = [
            # Technology small/mid caps
            'PLTR', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'NET', 'MDB', 'OKTA', 'TWLO', 'TEAM',
            'DOCU', 'ZM', 'PTON', 'ROKU', 'SQ', 'PYPL', 'UBER', 'LYFT', 'DASH', 'ABNB',
            
            # Healthcare/biotech small/mid caps  
            'MRNA', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'IQV', 'DXCM', 'ALGN', 'IDXX',
            'INCY', 'BMRN', 'SGEN', 'EXAS', 'TMDX', 'VEEV', 'TDOC', 'PODD', 'HOLX', 'BAX',
            
            # Financial small/mid caps
            'SCHW', 'USB', 'TFC', 'PNC', 'COF', 'AIG', 'MET', 'PRU', 'ALL', 'TRV',
            'AFL', 'HIG', 'FIS', 'FISV', 'BK', 'STT', 'NTRS', 'RF', 'CFG', 'KEY',
            
            # Industrial small/mid caps
            'EMR', 'ETN', 'FDX', 'GD', 'NOC', 'LUV', 'DAL', 'UAL', 'AAL', 'JBHT',
            'ODFL', 'CHRW', 'EXPD', 'XPO', 'GWW', 'WAB', 'ROK', 'PH', 'DOV', 'ITW',
            
            # Consumer small/mid caps
            'SBUX', 'NKE', 'LULU', 'GPS', 'M', 'JWN', 'NCLH', 'CCL', 'RCL', 'MAR',
            'HLT', 'IHG', 'WYNN', 'LVS', 'MGM', 'CZR', 'PENN', 'DRI', 'CMG', 'QSR',
            
            # Energy small/mid caps
            'COP', 'EOG', 'PSX', 'VLO', 'MPC', 'PXD', 'FANG', 'DVN', 'OKE', 'KMI',
            'WMB', 'EPD', 'ET', 'SLB', 'HAL', 'BKR', 'FTI', 'NOV', 'HP', 'OII',
            
            # Utilities small/mid caps
            'AEP', 'EXC', 'D', 'XEL', 'PCG', 'SRE', 'PEG', 'ED', 'EIX', 'FE',
            'PPL', 'ES', 'DTE', 'ETR', 'EVRG', 'CMS', 'ATO', 'CNP', 'NI', 'LNT'
        ]
        
        all_fallback = russell_major + russell_smaller
        self.all_symbols.update(all_fallback)
        print(f"   ✅ Added {len(all_fallback)} Russell 3000 fallback stocks")
        return all_fallback
    
    def get_nasdaq_stocks_ftp(self):
        """Get all NASDAQ stocks from FTP"""
        print("📈 Fetching NASDAQ stocks from FTP...")
        try:
            ftp = ftplib.FTP('ftp.nasdaqtrader.com')
            ftp.login()
            ftp.cwd('SymbolDirectory')
            
            lines = []
            ftp.retrlines('RETR nasdaqlisted.txt', lines.append)
            ftp.quit()
            
            df = pd.read_csv(StringIO('\n'.join(lines)), sep='|')
            symbols = df['Symbol'].dropna().tolist()
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            
            # Filter out test symbols and invalid ones
            valid_symbols = [s for s in symbols if len(s) <= 5 and s.isalpha()]
            
            self.all_symbols.update(valid_symbols)
            print(f"   ✅ Added {len(valid_symbols)} NASDAQ stocks")
            return valid_symbols
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def get_nyse_stocks_ftp(self):
        """Get NYSE stocks from FTP"""
        print("📈 Fetching NYSE stocks from FTP...")
        try:
            ftp = ftplib.FTP('ftp.nasdaqtrader.com')
            ftp.login()
            ftp.cwd('SymbolDirectory')
            
            lines = []
            ftp.retrlines('RETR otherlisted.txt', lines.append)
            ftp.quit()
            
            df = pd.read_csv(StringIO('\n'.join(lines)), sep='|')
            nyse_stocks = df[df['Exchange'] == 'N']['NASDAQ Symbol'].dropna().tolist()
            nyse_stocks = [s.strip().upper() for s in nyse_stocks if s.strip()]
            
            # Filter valid symbols
            valid_symbols = [s for s in nyse_stocks if len(s) <= 5 and s.replace('-', '').replace('.', '').isalpha()]
            
            self.all_symbols.update(valid_symbols)
            print(f"   ✅ Added {len(valid_symbols)} NYSE stocks")
            return valid_symbols
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def get_russell_2000(self):
        """Get Russell 2000 from iShares IWM ETF"""
        print("📈 Fetching Russell 2000 stocks...")
        try:
            date_formats = [
                datetime.now().strftime('%Y%m%d'),
                datetime.now().strftime('%m%d%Y'),
                '20241231'
            ]
            
            for date_str in date_formats:
                try:
                    url = f"https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund&asOfDate={date_str}"
                    
                    self.add_delay()
                    response = requests.get(url, headers=self.headers, timeout=30)
                    if response.status_code == 200 and len(response.text) > 1000:
                        df = pd.read_csv(StringIO(response.text), skiprows=10)
                        if 'Ticker' in df.columns:
                            symbols = df['Ticker'].dropna().tolist()
                            symbols = [s.strip().upper() for s in symbols if s.strip()]
                            new_symbols = set(symbols) - self.all_symbols
                            self.all_symbols.update(symbols)
                            print(f"   ✅ Added {len(new_symbols)} new Russell 2000 stocks")
                            return list(new_symbols)
                except:
                    continue
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("   ⚠️  Russell 2000 API unavailable")
        return []
    
    def get_major_etfs(self):
        """Get comprehensive ETF list"""
        print("📈 Fetching major ETFs...")
        
        major_etfs = {
            # Broad Market ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'IEFA', 'IEMG',
            'EFA', 'EEM', 'VTV', 'VUG', 'IVV', 'IVW', 'IVE', 'IJH', 'IJR',
            'VTEB', 'VXUS', 'BND', 'BNDX', 'SCHX', 'SCHA', 'SCHB', 'SCHF',
            
            # Sector ETFs - SPDR
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLY',
            
            # Sector ETFs - Vanguard  
            'VGT', 'VHT', 'VFH', 'VAW', 'VNQ', 'VPU', 'VIS', 'VCR', 'VDC', 'VDE',
            
            # Sector ETFs - iShares
            'IYF', 'IYW', 'IYE', 'IYH', 'IYJ', 'IYK', 'IDU', 'IYM', 'IYR', 'IYC',
            
            # International Regional
            'EWJ', 'EWZ', 'EWU', 'EWG', 'EWL', 'EWQ', 'EWS', 'EWY', 'EWT', 'FXI',
            'EWH', 'EWW', 'EWC', 'EWA', 'EWP', 'EWI', 'EWM', 'EWD', 'EWO', 'EWK',
            'INDA', 'KWEB', 'ASHR', 'MCHI', 'RSX', 'EZA', 'EPP', 'ECH', 'EPU',
            
            # Fixed Income
            'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK', 'TIP', 'VTEB', 'MUB', 'AGG',
            'BND', 'BNDX', 'VGIT', 'VGSH', 'VCSH', 'VCIT', 'VWOB', 'EMB', 'PCY',
            'HYEM', 'SHYG', 'USHY', 'FLOT', 'NEAR', 'JPST', 'MINT', 'SHM', 'VGSH',
            
            # Commodities & Alternatives
            'GLD', 'SLV', 'USO', 'UNG', 'GDX', 'GDXJ', 'IAU', 'DJP', 'DBA', 'URA',
            'XOP', 'XME', 'ITB', 'IYR', 'XHB', 'PBW', 'ICLN', 'JETS', 'XAR',
            'CORN', 'WEAT', 'SOYB', 'CANE', 'JO', 'NIB', 'BAL', 'CAFE', 'WOOD',
            
            # Crypto ETFs (2025) 
            'IBIT', 'ETHA', 'BITO', 'BLOK', 'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF',
            
            # Factor/Smart Beta
            'QUAL', 'SIZE', 'USMV', 'MTUM', 'VMOT', 'VYM', 'SCHD', 'NOBL', 'DGRO',
            'FDVV', 'VIG', 'RDVY', 'HDV', 'DVY', 'VEU', 'VYMI', 'VXUS', 'VSS',
            
            # Leveraged/Inverse (major ones)
            'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'TNA', 'TZA', 'UDOW', 'SDOW', 'SOXL', 'SOXS',
            'LABU', 'LABD', 'CURE', 'RXL', 'XBI', 'IBB', 'ARKG', 'GNOM'
        }
        
        self.all_symbols.update(major_etfs)
        print(f"   ✅ Added {len(major_etfs)} major ETFs")
        return list(major_etfs)
    
    def get_major_adrs(self):
        """Get major ADRs"""
        print("📈 Fetching major ADRs...")
        
        major_adrs = {
            # Chinese ADRs
            'BABA', 'JD', 'NIO', 'XPEV', 'LI', 'BIDU', 'NTES', 'WB', 'VIPS', 'TME',
            'PDD', 'BILI', 'IQ', 'TIGR', 'YMM', 'DIDI', 'EDU', 'TAL', 'GOTU',
            'YUMC', 'ZTO', 'HTHT', 'HUYA', 'DOYU', 'KC', 'WDH', 'LX', 'QD', 'ZLAB',
            
            # European ADRs
            'ASML', 'SAP', 'NVO', 'UL', 'RY', 'TD', 'SHOP', 'CNI', 'SU', 'ENB',
            'TTE', 'BP', 'SHEL', 'BCS', 'DB', 'UBS', 'ING', 'BBVA', 'SAN', 'BNS',
            'NTR', 'ABX', 'GOLD', 'ERIC', 'NOK', 'PHG', 'STM', 'TSM', 'UMC',
            
            # Japanese ADRs  
            'TM', 'SONY', 'NTT', 'MUFG', 'SMFG', 'MFG', 'HMC', 'NSANY', 'HTHIY',
            'SFTBY', 'FUJHY', 'NTT', 'MBFJF', 'TKOMY', 'NCTY', 'SHCAY', 'SFUNY',
            
            # Latin American ADRs
            'VALE', 'ITUB', 'BBD', 'PBR', 'SID', 'ABEV', 'UGP', 'GGB', 'CIG',
            'GGAL', 'BMA', 'YPF', 'PAM', 'CX', 'SBS', 'LPL', 'PBR-A', 'OGZPY',
            
            # Indian ADRs
            'WIT', 'RDY', 'IBN', 'HDB', 'WF', 'INFY', 'TTM', 'SIFY', 'VEDL',
            'ALTY', 'REDF', 'SCCO', 'PKX', 'IFS', 'AZRE', 'MT', 'VLRS',
            
            # Other International
            'TSM', 'UMC', 'ASX', 'E', 'PHG', 'ERIC', 'NOK', 'VIV', 'TEF', 'CHL',
            'CHT', 'SID', 'KB', 'LPL', 'WF', 'AU', 'GOLD', 'NEM', 'FCX', 'AA'
        }
        
        self.all_symbols.update(major_adrs)
        print(f"   ✅ Added {len(major_adrs)} major ADRs")
        return list(major_adrs)
    
    def get_crypto_securities(self):
        """Get crypto-related stocks and ETFs"""
        print("📈 Fetching crypto-related securities...")
        
        crypto_securities = {
            # Crypto ETFs (2025)
            'IBIT', 'ETHA', 'BITO', 'BLOK', 'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF',
            
            # Direct Crypto Stocks
            'COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK', 'HUT', 'BTBT', 'CAN',
            'EBANG', 'SOS', 'BITF', 'ARBK', 'WULF', 'CIFR', 'CORZ', 'HIVE',
            'DMGI', 'BTCS', 'GREE', 'ANY', 'EBON', 'BTC', 'FTFT', 'APLD',
            
            # Blockchain/Fintech with crypto exposure
            'SQ', 'PYPL', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'NU', 'OPEN',
            'COIN', 'RBLX', 'U', 'PATH', 'DKNG', 'PLTR', 'SNOW'
        }
        
        new_symbols = crypto_securities - self.all_symbols
        self.all_symbols.update(crypto_securities)
        print(f"   ✅ Added {len(new_symbols)} new crypto-related securities")
        return list(new_symbols)
    
    def get_market_indices(self):
        """Get major market indices"""
        print("📈 Fetching market indices...")
        
        indices = {
            # US Indices
            '^GSPC', '^DJI', '^IXIC', '^NDX', '^RUT', '^VIX', '^TNX',
            '^GSPCTR', '^RUA', '^RUI', '^RUO', '^RUTTR', '^SP600', '^SP400',
            '^XAX', '^NYA', '^XMI', '^XOI', '^HGX', '^SOX', '^UTY', '^XAU',
            
            # International Indices  
            '^N225', '^HSI', '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E',
            '^BVSP', '^MXX', '^KS11', '^TWII', '^AXJO', '^BSESN', '^NSEI',
            '^STI', '^KLSE', '^JKSE', '^SET', '^PSEI', '^VNI', '^HNX'
        }
        
        self.all_symbols.update(indices)
        print(f"   ✅ Added {len(indices)} market indices")
        return list(indices)
    
    def get_additional_popular_stocks(self):
        """Add popular stocks that might be missing"""
        print("📈 Adding popular stocks...")
        
        popular_stocks = {
            # Meme/Popular stocks
            'GME', 'AMC', 'BBBY', 'NOK', 'BB', 'KOSS', 'EXPR', 'CLOV', 'WISH',
            'SPCE', 'NKLA', 'RIDE', 'LCID', 'RIVN', 'F', 'FORD',
            
            # Recent IPOs/Popular names
            'RBLX', 'PATH', 'DKNG', 'PENN', 'PLTR', 'U', 'NET', 'FSLY', 'ESTC',
            'ZI', 'DOCU', 'ZM', 'WORK', 'PTON', 'LMND', 'ROOT', 'OPEN', 'COMP',
            
            # Energy transition
            'ENPH', 'SEDG', 'RUN', 'SPWR', 'FSLR', 'PLUG', 'BE', 'BLDP', 'FUV',
            'HYLN', 'QS', 'BLNK', 'CHPT', 'EVGO', 'WBX', 'ARVL', 'GOEV',
            
            # Cannabis
            'TLRY', 'CGC', 'ACB', 'CRON', 'HEXO', 'OGI', 'SNDL', 'APHA',
            
            # SPACs that completed
            'IPOF', 'IPOD', 'IPOE', 'CCIV', 'PSTH', 'THCB', 'SBE', 'QS',
            
            # International popular
            'GRAB', 'SE', 'CPNG', 'DIDI', 'UBER', 'LYFT', 'DASH', 'SPOT',
            
            # REITs
            'O', 'VNQ', 'SCHH', 'IYR', 'XLRE', 'REM', 'MORT', 'REZ'
        }
        
        new_symbols = popular_stocks - self.all_symbols
        self.all_symbols.update(popular_stocks)
        print(f"   ✅ Added {len(new_symbols)} new popular stocks")
        return list(new_symbols)
    
    def build_comprehensive_universe(self):
        """Build the complete stock universe"""
        print("🏗️  Building comprehensive stock universe for Yahoo Finance...")
        print("🎯 Target: 3,000-5,000 unique symbols\n")
        
        start_time = time.time()
        
        # Get symbols from all sources
        self.get_russell_3000()
        self.add_delay()
        
        self.get_nasdaq_stocks_ftp() 
        self.add_delay()
        
        self.get_nyse_stocks_ftp()
        self.add_delay()
        
        self.get_russell_2000()
        self.add_delay()
        
        self.get_major_etfs()
        self.get_major_adrs()
        self.get_crypto_securities()
        self.get_market_indices()
        self.get_additional_popular_stocks()
        
        # Clean and validate symbols
        final_symbols = self.clean_and_validate_symbols()
        
        end_time = time.time()
        
        # Generate report
        self.generate_report(final_symbols, end_time - start_time)
        
        return final_symbols
    
    def clean_and_validate_symbols(self):
        """Clean and validate the collected symbols"""
        print("\n🧹 Cleaning and validating symbols...")
        
        # Convert to sorted list
        raw_symbols = sorted(list(self.all_symbols))
        
        # Clean symbols
        cleaned_symbols = []
        for symbol in raw_symbols:
            if symbol and isinstance(symbol, str):
                clean_symbol = symbol.strip().upper()
                # Basic validation - allow letters, numbers, -, ., ^
                if (len(clean_symbol) >= 1 and len(clean_symbol) <= 8 and 
                    all(c.isalnum() or c in '^-.' for c in clean_symbol)):
                    cleaned_symbols.append(clean_symbol)
        
        print(f"   ✅ Cleaned {len(raw_symbols)} -> {len(cleaned_symbols)} valid symbols")
        return cleaned_symbols
    
    def generate_report(self, symbols, duration):
        """Generate comprehensive report"""
        print(f"\n📊 COMPREHENSIVE STOCK UNIVERSE REPORT")
        print(f"=" * 60)
        print(f"🎯 Total unique symbols: {len(symbols)}")
        print(f"⏱️  Build time: {duration:.1f} seconds")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Categorize symbols
        stocks = [s for s in symbols if not s.startswith('^') and len(s) <= 5 and '.' not in s]
        indices = [s for s in symbols if s.startswith('^')]
        etfs_complex = [s for s in symbols if not s.startswith('^') and (len(s) > 4 or '.' in s or '-' in s)]
        
        print(f"\n📈 BREAKDOWN BY TYPE:")
        print(f"   Individual Stocks: ~{len(stocks)}")
        print(f"   Market Indices: {len(indices)}")  
        print(f"   ETFs/Complex: ~{len(symbols) - len(stocks) - len(indices)}")
        
        # Sample symbols
        print(f"\n📝 SAMPLE SYMBOLS:")
        print(f"   First 20: {symbols[:20]}")
        if len(symbols) >= 10:
            print(f"   Random 10: {random.sample(symbols, min(10, len(symbols)))}")
        
        # Validation for Yahoo Finance
        print(f"\n✅ YAHOO FINANCE COMPATIBILITY:")
        print(f"   ✓ Symbol format validation passed")
        print(f"   ✓ Length validation passed (1-8 chars)")
        print(f"   ✓ Character validation passed (alphanumeric + ^-.)")
        print(f"   ✓ Ready for Yahoo Finance data collection")
        
        return symbols


def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE YAHOO FINANCE STOCK UNIVERSE BUILDER")
    print("=" * 80)
    
    builder = StockUniverseBuilder()
    
    # Build comprehensive universe
    symbols = builder.build_comprehensive_universe()
    
    # Save to file
    script_dir = Path(__file__).parent
    output_file = script_dir / 'comprehensive_stock_universe.txt'
    
    with open(output_file, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    print(f"\n💾 Universe saved to: {output_file}")
    print(f"🎯 Ready for Yahoo Finance data collection with {len(symbols)} symbols!")
    print(f"📝 Next step: Use this file with the data collection script")
    
    return symbols


if __name__ == "__main__":
    main()
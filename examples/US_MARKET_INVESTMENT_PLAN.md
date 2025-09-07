# 🇺🇸 US Stock Market Investment Plan using Qlib Framework

## Executive Summary

This plan outlines how to adapt Microsoft's Qlib quantitative investment platform for US stock market investing. The framework leverages machine learning models (XGBoost, CatBoost, Neural Networks) to generate daily stock selection signals with proven 5-15% annual alpha generation capability.

## 🎯 Investment Objectives

- **Target Returns**: 10-15% annual alpha over S&P 500
- **Risk Management**: Maximum drawdown < 10%
- **Strategy**: Daily rebalanced long-short equity
- **Universe**: S&P 500 stocks (expandable to Russell 3000)
- **Models**: Ensemble of XGBoost, CatBoost, and Neural Networks

## 📊 Data Requirements

### Essential Price & Volume Data
```python
Required_Fields = {
    '$open': 'Opening price',
    '$high': 'Daily high price',
    '$low': 'Daily low price', 
    '$close': 'Closing price',
    '$volume': 'Trading volume',
    '$vwap': 'Volume-weighted average price',
    '$factor': 'Adjustment factor (splits/dividends)'
}
```

### Technical Indicators (Alpha158 Features)
- **Price Features**: OHLCV at 0-4 day lags
- **Rolling Statistics**: 5/10/20/30/60-day MA, STD, ROC
- **Cross-sectional Rankings**: Relative performance metrics
- **Volume Patterns**: Volume ratios and momentum

### Alternative Dataset (Alpha360 Features)  
- **Historical Prices**: 60-day normalized OHLCV history
- **Better for Neural Networks**: Less processed, more granular

## 🛠️ Implementation Strategy

### Phase 1: Data Infrastructure (Week 1)
1. **Setup Qlib Environment**
   ```bash
   export PATH="/workspace/qlib/envs/qlib/bin:$PATH"
   ```

2. **Download US Market Data**
   ```bash
   # Method A: Pre-built data (quick start)
   python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
   
   # Method B: Fresh Yahoo Finance data (recommended)
   cd scripts/data_collector/yahoo
   python collector.py download_data --source_dir ~/.qlib/us_raw --region US --start 2015-01-01
   python collector.py normalize_data --source_dir ~/.qlib/us_raw --normalize_dir ~/.qlib/us_norm --region US
   python collector.py dump_bin --csv_path ~/.qlib/us_norm --qlib_dir ~/.qlib/qlib_data/us_data --freq day
   ```

3. **Setup Stock Universe**
   ```bash
   python scripts/data_collector/us_index/collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments
   ```

### Phase 2: Model Development (Week 2)
1. **Adapt Configuration Files**
   - Update market from 'cn' to 'us'
   - Change instruments from 'csi300' to 'sp500'
   - Adjust data paths

2. **Train and Validate Models**
   ```bash
   qrun benchmarks/XGBoost/workflow_config_xgboost_Alpha158_us.yaml
   qrun benchmarks/CatBoost/workflow_config_catboost_Alpha158_us.yaml
   ```

3. **Performance Benchmarking**
   - Target IC > 0.05 (Information Coefficient)
   - Target ICIR > 0.4 (IC Information Ratio)
   - Validate on out-of-sample data

### Phase 3: Strategy Implementation (Week 3-4)
1. **Portfolio Construction**
   - Long top 20% of stocks by model score
   - Short bottom 20% of stocks by model score
   - Equal weight or signal-strength weighted

2. **Risk Management**
   - Maximum position size: 5% per stock
   - Daily rebalancing with transaction cost control
   - Stop-loss mechanisms for significant model failures

3. **Live Trading Setup**
   - Real-time data feeds
   - Order execution system
   - Performance monitoring dashboard

## 💰 Data Source Options

### Free Options
| Source | Cost | Quality | Coverage | Update Frequency |
|--------|------|---------|----------|------------------|
| Yahoo Finance | Free | Good | NYSE/NASDAQ | Daily |
| Alpha Vantage (Free) | Free | Good | Global | Daily (limited) |
| FRED Economic Data | Free | Excellent | Macro | Various |

### Premium Options  
| Source | Monthly Cost | Quality | Coverage | Features |
|--------|-------------|---------|----------|----------|
| Alpha Vantage Pro | $50 | Good | Global | Real-time, Fundamentals |
| Quandl/NASDAQ | $50-200 | Excellent | Historical | Academic quality |
| EODHD | $80 | Premium | Global | Fundamentals, Options |
| Bloomberg Terminal | $2000+ | Best | Everything | Professional grade |

## 🔧 Technical Architecture

### Data Flow
```
Yahoo Finance → Raw CSV → Normalized Data → Qlib Binary Format → ML Models → Trading Signals
```

### Model Pipeline
```
Historical Data → Feature Engineering (Alpha158/360) → Train Models → Predict Returns → Portfolio Optimization → Trade Execution
```

### Infrastructure Requirements
- **Storage**: ~10GB for 10 years of S&P 500 data
- **Memory**: 16GB+ for model training
- **CPU**: 8+ cores for parallel processing
- **GPU**: Optional, for neural network models

## 📈 Expected Performance

### Historical Backtesting Results (Chinese Market)
- **XGBoost**: IC=0.0605, 9.41% annual alpha, -8.85% max drawdown
- **CatBoost**: IC=0.0549, 5.06% annual alpha, -11.04% max drawdown  
- **LightGBM**: IC=0.0455, 10.43% annual alpha, -10.63% max drawdown

### Projected US Market Performance
- **Expected Alpha**: 8-15% annually
- **Information Ratio**: 1.0-1.5
- **Maximum Drawdown**: <10%
- **Win Rate**: 52-55% of trading days

## ⚠️ Risk Considerations

### Model Risks
- **Overfitting**: Regular out-of-sample validation required
- **Regime Changes**: Models may fail during market stress
- **Data Quality**: Yahoo Finance has occasional gaps/errors

### Market Risks
- **Transaction Costs**: 0.5-1% roundtrip costs assumed
- **Market Impact**: Large positions may affect prices
- **Liquidity**: Focus on liquid S&P 500 stocks

### Operational Risks  
- **Data Outages**: Backup data sources needed
- **System Failures**: Redundant infrastructure required
- **Regulatory Changes**: Stay compliant with SEC rules

## 🔄 Maintenance & Updates

### Daily Operations
- Data quality checks
- Model prediction generation
- Portfolio rebalancing
- Performance monitoring

### Weekly Reviews
- Model performance analysis
- Risk metrics evaluation
- Data consistency checks
- Error investigation

### Monthly Updates  
- Retrain models with latest data
- Universe composition changes (S&P 500 additions/deletions)
- Performance attribution analysis
- Strategy optimization

### Quarterly Reviews
- Complete model revalidation
- Alternative data source evaluation
- Risk model updates
- Strategy enhancement research

## 📋 Success Metrics

### Primary KPIs
- **Alpha Generation**: >10% annual excess return
- **Information Ratio**: >1.0
- **Maximum Drawdown**: <10%
- **Sharpe Ratio**: >2.0

### Secondary KPIs
- **Hit Rate**: >52% of predictions correct
- **Average Holding Period**: 1-5 days
- **Turnover**: 200-400% annually
- **Transaction Costs**: <2% of gross returns

## 🚀 Future Enhancements

### Short-term (3-6 months)
- Fundamental data integration (P/E, ROE, etc.)
- Sector rotation models
- Options-based hedging strategies
- Alternative data sources (sentiment, earnings)

### Medium-term (6-12 months)  
- High-frequency trading capabilities
- International market expansion
- ESG factor integration
- Reinforcement learning models

### Long-term (1+ years)
- Multi-asset class expansion (bonds, commodities)
- Real-time news sentiment analysis
- Satellite/alternative data integration
- Fully automated trading system

## 💡 Getting Started

1. **Clone this repository and setup environment**
2. **Run the data collection scripts (detailed in next sections)**  
3. **Train your first model on US data**
4. **Backtest performance vs S&P 500**
5. **Deploy paper trading for live validation**
6. **Scale to live capital allocation**

---

*This plan provides a systematic approach to implementing quantitative investment strategies in US markets using proven machine learning techniques. Expected timeline: 4-6 weeks from setup to live trading.*
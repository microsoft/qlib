import pandas as pd
import numpy as np

# Simulate prediction data with dates
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
pred_values = np.random.randn(100)
pred = pd.Series(pred_values, index=dates)

print(f"Pred shape: {pred.shape}")
print(f"Pred type: {type(pred)}")
print(f"Pred sample: {pred.head()}")

# Test our performance calculation logic
performance = {
    'ic': 0,
    'rank_ic': 0,
    'total_return': 0,
    'max_drawdown': 0,
    'annual_return': 0,
    'sharpe_ratio': 0,
    'basic_metrics': {},
    'cumulative_returns': {}
}

if pred is not None and len(pred) > 0:
    print("Calculating performance metrics...")
    
    if hasattr(pred, 'index'):
        dates = pred.index
        if len(dates) > 0:
            pred_values = pred.values if hasattr(pred, 'values') else pred
            
            if len(pred_values) > 1:
                daily_returns = []
                for i in range(1, len(pred_values)):
                    if pred_values[i-1] != 0:
                        daily_return = (pred_values[i] - pred_values[i-1]) / abs(pred_values[i-1])
                        daily_returns.append(daily_return)
                
                if len(daily_returns) > 0:
                    daily_returns_series = pd.Series(daily_returns, index=dates[1:])
                    
                    cum_return = (1 + daily_returns_series).cumprod() - 1
                    total_return = cum_return.iloc[-1] if not cum_return.empty else 0
                    
                    def calculate_max_drawdown(returns):
                        cum_returns = (1 + returns).cumprod()
                        peak = cum_returns.expanding(min_periods=1).max()
                        drawdown = (cum_returns - peak) / peak
                        return drawdown.min()
                    
                    max_drawdown = calculate_max_drawdown(daily_returns_series)
                    annual_return = (1 + daily_returns_series.mean()) ** 252 - 1
                    sharpe_ratio = daily_returns_series.mean() / daily_returns_series.std() * np.sqrt(252) if daily_returns_series.std() > 0 else 0
                    
                    performance['total_return'] = float(total_return)
                    performance['max_drawdown'] = float(max_drawdown)
                    performance['annual_return'] = float(annual_return)
                    performance['sharpe_ratio'] = float(sharpe_ratio)
                    
                    cumulative_returns_formatted = {}
                    for date, value in cum_return.to_dict().items():
                        if isinstance(date, pd.Timestamp):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                        if not pd.isna(value):
                            cumulative_returns_formatted[date_str] = float(value)
                    
                    performance['cumulative_returns'] = cumulative_returns_formatted
                    
                    print(f"Performance calculation completed:")
                    print(f"Total Return: {total_return:.4f}")
                    print(f"Max Drawdown: {max_drawdown:.4f}")
                    print(f"Annual Return: {annual_return:.4f}")
                    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
                    print(f"Cumulative returns data points: {len(cumulative_returns_formatted)}")
                    print(f"Sample cumulative returns: {dict(list(cumulative_returns_formatted.items())[:3])}")

import qlib
from qlib.data import D

# Initialize Qlib with the data path
qlib.init(provider_uri='/home/idea/.qlib/qlib_data/cn_data')

# Test if we can access the daily data
print("Testing if daily data is accessible...")
try:
    # Get some daily data for a stock
    df = D.features(['sz399300'], ['$close'], start_time='2023-01-01', end_time='2023-01-10', freq='day')
    print("✓ Success! Got daily data:")
    print(df)
except Exception as e:
    print(f"✗ Failed: {e}")

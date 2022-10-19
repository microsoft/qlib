# Generate `bin` format data
set -e
python ../../scripts/dump_bin.py dump_all --csv_path ./data/csv --qlib_dir ./data/bin --include_fields open,close,high,low,vwap,volume --symbol_field_name symbol --date_field_name date --freq 1min

# Generate pickle format data
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
if [ -e stat/ ]; then
    rm -r stat/
fi
python scripts/collect_pickle_dataframe.py

# Sample orders
python scripts/gen_training_orders.py
python scripts/gen_backtest_orders.py

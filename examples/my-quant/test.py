import os, glob
import pandas as pd

root = r"D:/Quant-qlib-official/data/normalize"
files = glob.glob(os.path.join(root, "**/*.csv"), recursive=True)

bad = []
for f in files:
    try:
        df = pd.read_csv(f)
        # 常见日期列名：date / datetime / trade_date
        date_col = None
        for c in ["date", "datetime", "trade_date"]:
            if c in df.columns:
                date_col = c
                break

        if date_col is None:
            # 也可能日期在 index（第一列）
            s = df.iloc[:, 0]
        else:
            s = df[date_col]

        dt = pd.to_datetime(s, errors="coerce")
        if dt.isna().any():
            bad.append((f, int(dt.isna().sum()), s[dt.isna()].head(5).tolist()))
    except Exception as e:
        bad.append((f, "READ_FAIL", str(e)))

print("bad files:", len(bad))
for item in bad[:30]:
    print(item)

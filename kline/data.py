import pandas as pd
from .kline import fetch_historical_klines
from .indicator import indicator
import matplotlib

matplotlib.use('Agg') 

# --- Binanceデータ取得 ---
def klines_to_dataframe(klines):
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)
    for col in ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

interval = "1w"
klines = fetch_historical_klines("SHIBUSDT", interval, 1480000000)
df = klines_to_dataframe(klines)
df = indicator(df)
df = df.dropna()
# 'ignore'列を削除
df = df.drop('ignore', axis=1)
print(df)
df.to_csv(f'./klinedata/shib_{interval}_data.csv', index=False)

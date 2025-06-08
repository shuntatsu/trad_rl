import requests
import time

def fetch_historical_klines(symbol, interval, start_time, end_time=None, sleep_time=2):

    endpoint = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": 800
    }

    # 終了時刻が指定されている場合はパラメータに追加
    if end_time:
        params["endTime"] = end_time

    # 全てのKラインデータを格納するリスト
    all_klines = []

    while True:
        response = requests.get(endpoint, params=params)
        klines = response.json()

        # データが存在しない場合はループを終了
        if not klines:
            break

        # 取得したKラインデータをリストに追加
        all_klines.extend(klines)

        # 次のリクエストのためにstartTimeを更新
        params["startTime"] = klines[-1][0] + 1

        # 終了時刻に達したらループを終了
        if end_time and params["startTime"] >= end_time:
            break

        if sleep_time != 0:
            # レート制限に対応するためにスリープ
            time.sleep(sleep_time)

    return all_klines
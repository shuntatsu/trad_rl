import talib

def bollinger_bands(df, timeperiod=20):
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
        df['close'], timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0
    )
    return df

def macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    df['macd_line'], df['signal_line'], df['histogram'] = talib.MACD(
        df['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )
    return df

def dmi(df, timeperiod=14):
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df

def rsi(df, timeperiod=14):
    df['rsi'] = talib.RSI(df['close'], timeperiod=timeperiod)
    return df

def cci(df, timeperiod=20):
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df

def momentum(df, timeperiod=10):
    df['momentum'] = talib.MOM(df['close'], timeperiod=timeperiod)
    return df

def obv(df):

    # OBV を計算するための初期値を設定
    prev_obv = 0
    
    # OBV を計算
    df['obv'] = talib.OBV(df['close'], df['volume'])
    
    # 初期値を適用（最初のデータポイントのみ）
    if not df.empty:
        df.iloc[0, df.columns.get_loc('obv')] += prev_obv
    
    return df

def mfi(df, timeperiod=14):
    df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=timeperiod)
    return df

def stochastic(df, fastk_period=14, slowk_period=1, slowd_period=3):
    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        df['high'], df['low'], df['close'],
        fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0,
        slowd_period=slowd_period, slowd_matype=0
    )
    return df

def cmf(df, timeperiod=20):
    money_flow_multiplier = ((2 * df['close']) - df['high'] - df['low']) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    df['cmf'] = money_flow_volume.rolling(window=timeperiod).sum() / df['volume'].rolling(window=timeperiod).sum()
    return df

def indicator(df):
    df = bollinger_bands(df)
    df = macd(df)
    df = dmi(df)
    df = rsi(df)
    df = cci(df)
    df = momentum(df)
    df = obv(df)
    df = mfi(df)
    df = stochastic(df)
    df = cmf(df)
    #print(df)
    return df
import pandas as pd

# 年ごとに計算する関数
def calculate_values(years, initial_value=0.005, rate=1.025, increment=0.001):
    values = []
    value = initial_value
    for year in range(1, 4):
        value = value * rate + increment + 0.002
        values.append({"Year": year, "Value": value})
    
    for year in range(4, years+1):
        value = value * rate + 0.002
        values.append({"Year": year, "Value": value})

    return pd.DataFrame(values)

# 例として、10年間の値を計算する
df = calculate_values(years=40, initial_value=0.005, rate=1.04, increment=0.15)

print(df)
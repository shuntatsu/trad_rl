import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg') 
from rl import MultiTimeframeTradingEnv

# データ読み込み
btc_df_1h = pd.read_csv("klinedata/btc_1h_data.csv", parse_dates=["open_time"])
btc_df_4h = pd.read_csv("klinedata/btc_4h_data.csv", parse_dates=["open_time"])
btc_df_1d = pd.read_csv("klinedata/btc_1d_data.csv", parse_dates=["open_time"])
btc_df_1w = pd.read_csv("klinedata/btc_1w_data.csv", parse_dates=["open_time"])
btc_df_1h["symbol"] = "BTC"
btc_df_1d["symbol"] = "BTC"
btc_df_1w["symbol"] = "BTC"
btc_df_4h["symbol"] = "BTC"

env_test = MultiTimeframeTradingEnv(btc_df_1h, btc_df_4h, btc_df_1d, btc_df_1w)

# 1. 学習済みモデルの読み込み
model = PPO.load("C:\\my_program\\python\\test\\model\\ppo_trading_all_1_0_1", env=env_test, device="cuda")

total_rewards = 0

entry_marks = {"long": [], "short": [], "spot": []}
close_marks = {"long": [], "short": [], "spot": []}
profit_history = {"long": [], "short": [], "spot": []}
current_profits = {"long": 0, "short": 0, "spot": 0}

asset_values = []
benchmark_prices = []

positions_prev = []

done = False
positions_prev = []

# 環境のリセット → 観測を取得
obs = env_test.reset()


while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env_test.step(action)
    total_rewards += reward

    step = info.get("step", env_test.current_step)  # infoにstepがない場合も対応
    price = env_test.df_1h.loc[step, "close"]

    benchmark_prices.append(price)
    asset_values.append(info["value"])

    positions_now = info.get("positions", [])

    # actionは配列で取得される想定なので
    action_type = int(action[0])
    if action_type in [0, 1, 2]:
        pos_type = ["long", "short", "spot"][action_type]
        entry_marks[pos_type].append(step)

    closed_positions = []
    for pos_prev in positions_prev:
        still_exists = False
        for pos_now in positions_now:
            if (pos_prev["entry_price"] == pos_now["entry_price"]) and (pos_prev["type"] == pos_now["type"]):
                still_exists = True
                break
        if not still_exists:
            closed_positions.append(pos_prev)

    for pos in closed_positions:
        pos_type = pos["type"]
        close_marks[pos_type].append(step)
        entry = pos["entry_price"]
        amount = pos["amount"]
        lev = pos.get("leverage", 1.0)
        if pos_type == "spot":
            pnl = (price / entry - 1) * amount
        else:
            diff = (price - entry) / entry
            pnl = amount * lev * (diff if pos_type == "long" else -diff)
        current_profits[pos_type] += pnl

    for k in ["long", "short", "spot"]:
        profit_history[k].append(current_profits[k])

    positions_prev = positions_now

# グラフ描画と保存

# 1. 総資産とベンチマークのグラフ
plt.figure(figsize=(15, 7))
plt.plot(asset_values, label="Total Asset Value", color="blue")
plt.plot(benchmark_prices, label="Benchmark Price (Close)", color="orange")

plt.title("Total Asset Value and Benchmark Price with Position Entries/Exits")
plt.xlabel("Step")
plt.ylabel("Value / Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:\\my_program\\python\\test\\data\\asset_and_benchmark_1_0_1.png")
plt.close()

# 2. ロング・ショート・スポットの利益推移をベンチマーク上にプロットしたグラフ（別々に）
for pos_type, color, color_entry, color_close in [
    ("long", "green", "green", "darkgreen"),
    ("short", "red", "red", "darkred"),
    ("spot", "purple", "purple", "indigo"),
]:
    plt.figure(figsize=(15, 7))
    plt.plot(benchmark_prices, label="Benchmark Price (Close)", color="orange")
    plt.plot(profit_history[pos_type], label=f"{pos_type.capitalize()} Profit", color=color)

    valid_entry_indices = [i for i in entry_marks[pos_type] if 0 <= i < len(benchmark_prices)]
    valid_close_indices = [i for i in close_marks[pos_type] if 0 <= i < len(benchmark_prices)]

    # ✅ Y軸を benchmark_prices に修正
    plt.scatter(
        valid_entry_indices,
        [benchmark_prices[i] for i in valid_entry_indices],
        marker="^",
        color=color_entry,
        label=f"{pos_type.capitalize()} Entry",
        s=100
    )
    plt.scatter(
        valid_close_indices,
        [benchmark_prices[i] for i in valid_close_indices],
        marker="v",
        color=color_close,
        label=f"{pos_type.capitalize()} Close",
        s=100
    )

    plt.title(f"{pos_type.capitalize()} Profit Over Time with Benchmark Price and Entries/Exits")
    plt.xlabel("Step")
    plt.ylabel("Price / Profit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"C:\\my_program\\python\\test\\data\\{pos_type}_profit_1_0_1.png")
    plt.close()

# 最終利益の出力
print("最終資産:", env_test.total_value)
print("総報酬:", total_rewards)
for pos_type in ["long", "short", "spot"]:
    print(f"{pos_type.capitalize()}の利益合計: {current_profits[pos_type]:.2f}")


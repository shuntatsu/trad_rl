import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib
matplotlib.use('Agg') 
from rl import MultiTimeframeTradingEnv

start_date = pd.to_datetime("2018-07-08")

def load_and_filter(path):
    df = pd.read_csv(path, parse_dates=["open_time"])
    return df[df["open_time"] >= start_date].reset_index(drop=True)

# BTC
btc_df_1h = load_and_filter("klinedata/btc_1h_data.csv")
btc_df_4h = load_and_filter("klinedata/btc_4h_data.csv")
btc_df_1d = load_and_filter("klinedata/btc_1d_data.csv")
btc_df_1w = load_and_filter("klinedata/btc_1w_data.csv")

# ETH
eth_df_1h = load_and_filter("klinedata/eth_1h_data.csv")
eth_df_4h = load_and_filter("klinedata/eth_4h_data.csv")
eth_df_1d = load_and_filter("klinedata/eth_1d_data.csv")
eth_df_1w = load_and_filter("klinedata/eth_1w_data.csv")

# XRP
xrp_df_1h = load_and_filter("klinedata/xrp_1h_data.csv")
xrp_df_4h = load_and_filter("klinedata/xrp_4h_data.csv")
xrp_df_1d = load_and_filter("klinedata/xrp_1d_data.csv")
xrp_df_1w = load_and_filter("klinedata/xrp_1w_data.csv")

# SOL
sol_df_1h = load_and_filter("klinedata/sol_1h_data.csv")
sol_df_4h = load_and_filter("klinedata/sol_4h_data.csv")
sol_df_1d = load_and_filter("klinedata/sol_1d_data.csv")
sol_df_1w = load_and_filter("klinedata/sol_1w_data.csv")

# BNB
bnb_df_1h = load_and_filter("klinedata/bnb_1h_data.csv")
bnb_df_4h = load_and_filter("klinedata/bnb_4h_data.csv")
bnb_df_1d = load_and_filter("klinedata/bnb_1d_data.csv")
bnb_df_1w = load_and_filter("klinedata/bnb_1w_data.csv")

# SHIB
shib_df_1h = load_and_filter("klinedata/shib_1h_data.csv")
shib_df_4h = load_and_filter("klinedata/shib_4h_data.csv")
shib_df_1d = load_and_filter("klinedata/shib_1d_data.csv")
shib_df_1w = load_and_filter("klinedata/shib_1w_data.csv")

# DOGE
doge_df_1h = load_and_filter("klinedata/doge_1h_data.csv")
doge_df_4h = load_and_filter("klinedata/doge_4h_data.csv")
doge_df_1d = load_and_filter("klinedata/doge_1d_data.csv")
doge_df_1w = load_and_filter("klinedata/doge_1w_data.csv")

# 銘柄を示す列を追加
btc_df_1h["symbol"] = "BTC"
btc_df_1d["symbol"] = "BTC"
btc_df_1w["symbol"] = "BTC"
btc_df_4h["symbol"] = "BTC"

eth_df_1h["symbol"] = "ETH"
eth_df_1d["symbol"] = "ETH"
eth_df_1w["symbol"] = "ETH"
eth_df_4h["symbol"] = "ETH"

xrp_df_1h["symbol"] = "XRP"
xrp_df_1d["symbol"] = "XRP"
xrp_df_1w["symbol"] = "XRP"
xrp_df_4h["symbol"] = "XRP"

sol_df_1h["symbol"] = "SOL"
sol_df_4h["symbol"] = "SOL"
sol_df_1d["symbol"] = "SOL"
sol_df_1w["symbol"] = "SOL"

bnb_df_1h["symbol"] = "BNB"
bnb_df_4h["symbol"] = "BNB"
bnb_df_1d["symbol"] = "BNB"
bnb_df_1w["symbol"] = "BNB"

shib_df_1h["symbol"] = "SHIB"
shib_df_4h["symbol"] = "SHIB"
shib_df_1d["symbol"] = "SHIB"
shib_df_1w["symbol"] = "SHIB"

doge_df_1h["symbol"] = "DOGE"
doge_df_4h["symbol"] = "DOGE"
doge_df_1d["symbol"] = "DOGE"
doge_df_1w["symbol"] = "DOGE"
# 環境作成
def make_env(df_1h, df_4h, df_1d, df_1w):
    return lambda: MultiTimeframeTradingEnv(df_1h, df_4h, df_1d, df_1w)

envs = [
    make_env(btc_df_1h, btc_df_4h, btc_df_1d, btc_df_1w),
    make_env(eth_df_1h, eth_df_4h, eth_df_1d, eth_df_1w),
    make_env(xrp_df_1h, xrp_df_4h, xrp_df_1d, xrp_df_1w),
    make_env(sol_df_1h, sol_df_4h, sol_df_1d, sol_df_1w),
    make_env(bnb_df_1h, bnb_df_4h, bnb_df_1d, bnb_df_1w),
    make_env(shib_df_1h, shib_df_4h, shib_df_1d, shib_df_1w),
    make_env(doge_df_1h, doge_df_4h, doge_df_1d, doge_df_1w),
]

env = DummyVecEnv(envs)

# モデル学習
policy_kwargs = dict(
    net_arch=[dict(pi=[512, 256, 128, 64], vf=[512, 256, 128, 64])]

)
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log="logs/", device="cuda")

try:
    model.learn(total_timesteps=100_000)
    model.save("C:\\my_program\\python\\test\\model\\ppo_trading_all_1_0_1")
except Exception as e:
    print(f"学習中にエラーが発生しました: {e}")
    model.save("model_backup_on_error")
    print("モデルを 'model_backup_on_error.zip' として保存しました。")
    raise
import torch
import shap
import numpy as np
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
model = PPO.load("C:\\my_program\\python\\test\\model\\ppo_trading_all.zip", env=env_test, device="cuda")

# SHAP解析
# dictで返る観測値を統合
obs_dict = env_test._get_observation()

# 各 timeframe の特徴量を連結して 1D配列にする
obs_flat = np.concatenate([v.flatten() for v in obs_dict.values()])
X = np.array([obs_flat])  # SHAPに渡す形式 (1, n_features)
feature_names = env_test.flat_feature_names
X = np.array([obs_flat])

def predict_shap(X_array):
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(model.device)
    obs_dict = {"flat": X_tensor}  # ← ここはあなたの observation のキーに合わせて
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_dict)
        probs = dist.distribution.probs
    return probs.cpu().numpy()

explainer = shap.Explainer(predict_shap, X)
shap_values = explainer(X)

plt.figure()
shap.summary_plot(
    shap_values[0].values,  # shape: (n_features, n_actions)
    features=X,
    feature_names=feature_names,
    plot_type="bar"
)
plt.tight_layout()
plt.savefig("C:\\my_program\\python\\test\\data\\shap_summary_plot.png")
plt.close()

import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import shap_test
import torch
import matplotlib
matplotlib.use('Agg')  # GUIなし環境でも描画可能に
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -------------------------------------------
# マルチタイムフレーム取引環境
# -------------------------------------------
class MultiTimeframeTradingEnv(gym.Env):
    def __init__(self, df_1h, df_4h, df_1d, df_1w, window_size=4*30, fee_pct=0.001):
        super().__init__()
        # 各時間足データを時間順でソート＆リセットインデックス
        self.df_1h = df_1h.sort_values("open_time").reset_index(drop=True)
        self.df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
        self.df_1d = df_1d.sort_values("open_time").reset_index(drop=True)
        self.df_1w = df_1w.sort_values("open_time").reset_index(drop=True)
        
        self.window_size = window_size
        self.fee_pct = fee_pct
        self.bonus_amount = 0.1
        
        # アクション空間：0=保持, 1=買い, 2=売り
        self.action_space = spaces.Discrete(3)

        # 特徴量名の管理（SHAP解析用）
        self.feature_names = None
        self.flat_feature_names = None

        # 観測空間の形状設定（4タイムフレーム×30ステップ×特徴数）
        # 特徴量数は open_time を除いたものと揃える
        feature_count = self.df_1h.drop(columns=["open_time"]).shape[1] + 1
        obs_dim = (4, 30, feature_count)  # 4つの時間足、30行、特徴数
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)

        self.reset()

    def reset(self):
        # 開始日時を固定
        start_date = pd.to_datetime("2018-12-31")

        # 各データでstart_date以降かつ20本以上のデータがある最初のインデックスを取得
        def find_valid_start(df):
            idx = df.index[df["open_time"] >= start_date][0]
            return max(idx, 40)  # 40本以上の余裕を持つ

        start_idx_1h = find_valid_start(self.df_1h)
        start_idx_4h = find_valid_start(self.df_4h)
        start_idx_1d = find_valid_start(self.df_1d)
        start_idx_1w = find_valid_start(self.df_1w)

        # すべての時間足で揃う最も遅いスタート地点に統一
        self.current_step = max(start_idx_1h, start_idx_4h, start_idx_1d, start_idx_1w)
        self.current_time = self.df_1h.loc[self.current_step, "open_time"]

        # 資産情報初期化
        self.cash = 1000
        self.asset = 0
        self.max_value = self.cash
        self.total_value = self.cash

        return self._get_observation()

    def reset_with_info(self):
        obs = self.reset()
        info = {
            "reset_time": str(self.current_time),
            "initial_cash": self.cash,
            "initial_asset": self.asset
        }
        return obs, info

    def _extract_features(self, df, timeframe):
        # 現在時刻までのデータを取得し直近30行を抜き出す
        df_filtered = df[df["open_time"] <= self.current_time]
        df_window = df_filtered.tail(30).copy()

        # 相対時間（現在時刻との差を時間単位で計算）
        df_window["relative_time"] = (df_window["open_time"] - self.current_time).dt.total_seconds() / 3600

        # open_time列を除いて特徴量だけ取得
        df_window = df_window.drop(columns=["open_time"])
        features = df_window.values.astype(np.float32)

        # 初回呼び出し時のみ特徴量名を保存
        if self.feature_names is None:
            self.feature_names = df_window.columns.tolist()

        # 30行に満たない場合は0でパディング
        if len(features) < 30:
            pad_rows = 30 - len(features)
            pad = np.zeros((pad_rows, features.shape[1]), dtype=np.float32)
            features = np.vstack([pad, features])

        return features

    def _get_observation(self):
        # 現在の時刻を更新
        self.current_time = self.df_1h.loc[self.current_step, "open_time"]

        # 各時間足から特徴抽出
        obs_1h = self._extract_features(self.df_1h, "1h")
        obs_4h = self._extract_features(self.df_4h, "4h")
        obs_1d = self._extract_features(self.df_1d, "1d")
        obs_1w = self._extract_features(self.df_1w, "1w")

        # 4タイムフレームの特徴をまとめる (4, 30, feature_dim)
        obs = np.stack([obs_1h, obs_4h, obs_1d, obs_1w], axis=0)

        # SHAP用のフラット特徴名を初回のみ作成
        if self.flat_feature_names is None:
            self.flat_feature_names = []
            for tf in ["1h", "4h", "1d", "1w"]:
                for i in range(30):
                    for name in self.feature_names:
                        self.flat_feature_names.append(f"{tf}_{name}_t-{29 - i}")

        return obs

    def get_flat_observation(self):
        """SHAP解析用：現在の観測を1次元配列化＋特徴名返却"""
        obs = self._get_observation()
        return obs.flatten(), self.flat_feature_names

    def step(self, action):
        done = False

        # 現在価格は1時間足のcloseを使用
        price = self.df_1h.loc[self.current_step, "close"]
        prev_value = self.cash + self.asset * price

        # 行動に基づく売買処理
        if action == 1 and self.cash > 0:  # 買い
            buy_amount = self.cash * (1 - self.fee_pct)
            self.asset = buy_amount / price
            self.cash = 0
        elif action == 2 and self.asset > 0:  # 売り
            sell_value = self.asset * price * (1 - self.fee_pct)
            self.cash = sell_value
            self.asset = 0

        # 時刻を1ステップ進める
        self.current_step += 1
        if self.current_step >= len(self.df_1h) - 1:
            done = True

        # 現時点の資産価値計算
        current_value = self.cash + self.asset * self.df_1h.loc[self.current_step, "close"]
        # 手数料と売買頻度にペナルティを加える例
        trade_penalty = 0.001  # 売買1回あたりのペナルティ

        reward = (current_value - prev_value) / prev_value

        # 売買行動に対してペナルティ
        if action in [1, 2]:
            reward -= trade_penalty 
        self.total_value = current_value

        # 資産最高値更新時にボーナス報酬付与
        if current_value > self.max_value:
            reward += self.bonus_amount
            self.max_value = current_value

        max_drawdown = (self.max_value - current_value) / self.max_value
        reward -= max_drawdown * 0.05
        obs = self._get_observation()
        info = {"cash": self.cash, "asset": self.asset, "value": self.total_value}

        return obs, reward, done, info

# -------------------------------------------
# データ読み込み
# -------------------------------------------
df_1h = pd.read_csv("btc_1h_data.csv", parse_dates=["open_time"])
df_4h = pd.read_csv("btc_4h_data.csv", parse_dates=["open_time"])
df_1d = pd.read_csv("btc_1d_data.csv", parse_dates=["open_time"])
df_1w = pd.read_csv("btc_1w_data.csv", parse_dates=["open_time"])

# -------------------------------------------
# 環境作成とモデル学習
# -------------------------------------------
env = DummyVecEnv([lambda: MultiTimeframeTradingEnv(df_1h, df_4h, df_1d, df_1w)])

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs/", device="cuda")

try:
    model.learn(total_timesteps=50_000)
except Exception as e:
    print(f"学習中にエラーが発生しました: {e}")
    model.save("model_backup_on_error")
    print("モデルを 'model_backup_on_error.zip' として保存しました。")
    raise

# -------------------------------------------
# バックテスト
# -------------------------------------------
env_test = MultiTimeframeTradingEnv(df_1h, df_4h, df_1d, df_1w)
obs = env_test.reset()
total_rewards = 0

while True:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env_test.step(action)
    total_rewards += reward
    if done:
        break

print("最終資産:", env_test.total_value)
print("総報酬:", total_rewards)

# -------------------------------------------
# SHAP解析の準備と実行
# -------------------------------------------
obs_flat, feature_names = env_test.get_flat_observation()
X = np.array([obs_flat])  # shape: (1, n_features)

def predict_shap(X_array):
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(X_tensor)
        action_probs = dist.distribution.probs  # 離散アクションの確率分布
    return action_probs.cpu().numpy()

explainer = shap_test.Explainer(predict_shap, X)
shap_values = explainer(X)

plt.figure()
shap_test.summary_plot(shap_values, features=X, feature_names=feature_names)
plt.tight_layout()

# 画像として保存・表示
plt.savefig("/data/shap_summary_plot.png")
plt.close()

# -------------------------------------------
# モデル評価関数
# -------------------------------------------
def evaluate_model_on_env(env, model):
    """与えられた環境でモデルを評価し最終資産を返す"""
    obs = env.reset()
    total_value = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
